#!/usr/bin/env python3

import os
import sys
import gc
import pickle
import argparse
import configparser
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from bin.generate_ifgs_from_epochs import generate_ifg_pairs
from bin.resample import resample_geotiff
from bin.ifg_cc_totif import ifg_cc_2tif


HELP_TEXT = """
PhaseBias_04_Apply_Phase_Bias_Correction_Chunked_To_GEOC.py

This script applies phase-bias correction using chunked inputs and chunked
inversion results, but writes the final outputs in the same GEOC directory
structure as the old script.

It uses:
1. Chunked wrapped IFGs from Data/ifgs/
2. Chunked coherence from Data/coh/
3. Chunked base bias terms from Data/X_chunks/
4. a_n values/maps from:
   - config.txt
   - Data/an_per_long_baseline.txt / an_chunks/global(_clean)
   - Data/an_chunks/pixel_maps

Output:
GEOC/
  yyyymmdd_yyyymmdd/
    yyyymmdd_yyyymmdd.geo.diff_pha.tif
    yyyymmdd_yyyymmdd.geo.cc.tif

Parallelism:
- --n_workers: chunk correction stage
- --n_write_workers: GEOC writing stage (parallel by category)
"""

if "--help" in sys.argv:
    print(HELP_TEXT)
    sys.exit(0)


# ======================================================================================
# Config / IO helpers
# ======================================================================================

def read_config(config_file="config.txt"):
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Error: The configuration file '{config_file}' was not found."
        )

    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}
    try:
        params["root_path"] = config.get("DEFAULT", "root_path")
        params["output_path"] = config.get("DEFAULT", "output_path")
        params["frame"] = config.get("DEFAULT", "frame")
        params["start"] = config.get("DEFAULT", "start")
        params["end"] = config.get("DEFAULT", "end")
        params["interval"] = config.getint("DEFAULT", "interval")
        params["nlook"] = config.getint("DEFAULT", "nlook")
        params["LiCSAR_data"] = config.get("DEFAULT", "LiCSAR_data")

        params["a1_6_day"] = config.getfloat("DEFAULT", "a1_6_day")
        params["a2_6_day"] = config.getfloat("DEFAULT", "a2_6_day")
        params["a3_6_day"] = config.getfloat("DEFAULT", "a3_6_day")
        params["a4_6_day"] = config.getfloat("DEFAULT", "a4_6_day")

        params["a1_12_day"] = config.getfloat("DEFAULT", "a1_12_day")
        params["a2_12_day"] = config.getfloat("DEFAULT", "a2_12_day")
        params["a3_12_day"] = config.getfloat("DEFAULT", "a3_12_day")
        params["a4_12_day"] = config.getfloat("DEFAULT", "a4_12_day")

        params["estimate_an_mode"] = config.get("DEFAULT", "estimate_an_mode").strip().lower()

    except configparser.NoOptionError as e:
        raise ValueError(f"Missing required config parameter: {e}")

    allowed_modes = {"config", "global", "pixel"}
    if params["estimate_an_mode"] not in allowed_modes:
        raise ValueError(
            f"estimate_an_mode must be one of {sorted(allowed_modes)}, "
            f"got '{params['estimate_an_mode']}'"
        )

    return params


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_chunk_index(output_path):
    chunk_index_file = os.path.join(output_path, "Data", "chunk_index.pkl")
    if not os.path.exists(chunk_index_file):
        raise FileNotFoundError(
            f"Chunk index not found: {chunk_index_file}\n"
            f"Please run the chunked PhaseBias_01_Read_Data script first."
        )
    return load_pickle(chunk_index_file)


def get_track_from_frame(frame_id):
    track = frame_id[0:3]
    if track[0] == "0":
        track = track[1:3]
    if track[0] == "0":
        track = track[1:2]
    return track


def get_root_directory(root_path, frame, LiCSAR_data):
    if LiCSAR_data == "yes":
        track = get_track_from_frame(frame)
        return f"/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/{track}/{frame}"
    return root_path


def get_config_an_values(params, max_con):
    interval = params["interval"]
    an_values = []

    for i in range(1, max_con):
        key = f"a{i}_{interval}_day"
        if key not in params:
            raise ValueError(f"Missing config parameter: {key}")
        an_values.append(params[key])

    return an_values


def load_global_a_values_from_text(output_path, fallback_an, max_con):
    file_path = os.path.join(output_path, "Data", "an_per_long_baseline.txt")
    if not os.path.exists(file_path):
        return fallback_an.copy()

    an_values = fallback_an.copy()

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key.startswith("a"):
                try:
                    idx = int(key[1:])
                except ValueError:
                    continue
                if 1 <= idx < max_con:
                    an_values[idx - 1] = float(value)

    return an_values


def chunk_file_path(output_path, kind, category_or_name, chunk_id):
    cr, cc = chunk_id

    if kind == "ifgs":
        return os.path.join(
            output_path, "Data", "ifgs", f"{category_or_name}d", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    if kind == "coh":
        return os.path.join(
            output_path, "Data", "coh", f"{category_or_name}d", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    if kind == "x":
        return os.path.join(
            output_path, "Data", "X_chunks", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    if kind == "an_global":
        return os.path.join(
            output_path, "Data", "an_chunks", "global", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    if kind == "an_global_clean":
        return os.path.join(
            output_path, "Data", "an_chunks", "global_clean", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    if kind == "an_pixel":
        return os.path.join(
            output_path, "Data", "an_chunks", "pixel_maps", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    if kind == "tmp_corrected_ifgs":
        return os.path.join(
            output_path, "Data", "corrected_tmp", "ifgs", f"{category_or_name}d", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    if kind == "tmp_corrected_coh":
        return os.path.join(
            output_path, "Data", "corrected_tmp", "coh", f"{category_or_name}d", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )

    raise ValueError(f"Unknown kind: {kind}")


def load_ifg_chunk(output_path, category, chunk_id):
    path = chunk_file_path(output_path, "ifgs", category, chunk_id)
    if not os.path.exists(path):
        return None
    return load_pickle(path)


def load_coh_chunk(output_path, category, chunk_id):
    path = chunk_file_path(output_path, "coh", category, chunk_id)
    if not os.path.exists(path):
        return None
    return load_pickle(path)


def load_x_chunk(output_path, chunk_id):
    path = chunk_file_path(output_path, "x", None, chunk_id)
    if not os.path.exists(path):
        return None
    return load_pickle(path)


def load_chunk_global_a_values(output_path, chunk_id, fallback_an, max_con):
    an_values = fallback_an.copy()

    path_clean = chunk_file_path(output_path, "an_global_clean", None, chunk_id)
    path_raw = chunk_file_path(output_path, "an_global", None, chunk_id)

    payload = None
    if os.path.exists(path_clean):
        payload = load_pickle(path_clean)
    elif os.path.exists(path_raw):
        payload = load_pickle(path_raw)

    if payload is not None and "values" in payload:
        vals = payload["values"]
        for i in range(1, max_con):
            key = f"a{i}"
            if key in vals and "regression" in vals[key] and np.isfinite(vals[key]["regression"]):
                an_values[i - 1] = float(vals[key]["regression"])

    return an_values


def load_chunk_pixel_a_maps(output_path, chunk_id, frame_shape, fallback_an, max_con):
    global_fallback_an = load_chunk_global_a_values(output_path, chunk_id, fallback_an, max_con)

    path = chunk_file_path(output_path, "an_pixel", None, chunk_id)
    pixel_maps = {}

    if not os.path.exists(path):
        return pixel_maps, global_fallback_an

    payload = load_pickle(path)
    maps = payload.get("maps", payload)

    for i in range(1, max_con):
        key = f"a{i}"
        if key in maps:
            arr = np.array(maps[key], dtype=np.float32)
            if arr.shape != frame_shape:
                raise ValueError(
                    f"Shape mismatch in pixel-wise map {key} for chunk {chunk_id}. "
                    f"Expected {frame_shape}, got {arr.shape}."
                )
            arr = np.where(np.isnan(arr), global_fallback_an[i - 1], arr)
            pixel_maps[key] = arr

    return pixel_maps, global_fallback_an


def wrap_to_pi_inplace(arr):
    np.remainder(arr + np.pi, 2 * np.pi, out=arr)
    arr -= np.pi
    return arr


# ======================================================================================
# Bias extension
# ======================================================================================

def extend_bias_stack_chunk(X_base, max_con, an_mode, an_values, pixel_a_maps=None):
    len_base = X_base.shape[0]
    X_extended = [X_base[i].astype(np.float32) for i in range(len_base)]

    for n in range(2, max_con + 1):
        for i in range(len_base - (n - 1)):
            X_sum = np.sum(X_base[i:i + n], axis=0).astype(np.float32)

            if an_mode == "pixel":
                key = f"a{n-1}"
                if pixel_a_maps is not None and key in pixel_a_maps:
                    Xn = pixel_a_maps[key] * X_sum
                else:
                    Xn = an_values[n - 2] * X_sum
            else:
                Xn = an_values[n - 2] * X_sum

            X_extended.append(Xn.astype(np.float32))

    X = np.stack(X_extended, axis=0).astype(np.float32)
    return X


# ======================================================================================
# Stage 1: chunk correction worker
# ======================================================================================

def process_one_chunk(chunk_id, params, args_dict):
    output_path = params["output_path"]
    an_mode = params["estimate_an_mode"]
    max_con = args_dict["max_con"]
    uncor_only = args_dict["uncor_only"]
    desired_lengths = args_dict["desired_lengths"]

    config_an = get_config_an_values(params, max_con)

    x_payload = load_x_chunk(output_path, chunk_id)
    if x_payload is None:
        return {
            "chunk_id": chunk_id,
            "status": "skipped",
            "reason": "missing X chunk",
        }

    X_base = np.array(x_payload["X"], dtype=np.float32)
    if X_base.shape[0] < 1:
        return {
            "chunk_id": chunk_id,
            "status": "skipped",
            "reason": "empty X chunk",
        }

    frame_shape = X_base.shape[1:]

    pixel_a_maps = None
    if an_mode == "config":
        an_values = config_an
        a_source = {"mode": "config", "scalar_values": an_values}

    elif an_mode == "global":
        text_global_an = load_global_a_values_from_text(output_path, config_an, max_con)
        an_values = load_chunk_global_a_values(output_path, chunk_id, text_global_an, max_con)
        a_source = {"mode": "global", "scalar_values": an_values}

    elif an_mode == "pixel":
        text_global_an = load_global_a_values_from_text(output_path, config_an, max_con)
        pixel_a_maps, an_values = load_chunk_pixel_a_maps(
            output_path, chunk_id, frame_shape, text_global_an, max_con
        )
        a_source = {
            "mode": "pixel",
            "scalar_fallback_values": an_values,
            "available_pixel_maps": sorted(pixel_a_maps.keys()),
        }
    else:
        return {
            "chunk_id": chunk_id,
            "status": "failed",
            "reason": f"unknown estimate_an_mode: {an_mode}",
        }

    X = extend_bias_stack_chunk(
        X_base=X_base,
        max_con=max_con,
        an_mode=an_mode,
        an_values=an_values,
        pixel_a_maps=pixel_a_maps,
    )

    len_base = X_base.shape[0]
    x_start_by_n = {1: 0}
    running = len_base
    for n in range(2, max_con + 1):
        x_start_by_n[n] = running
        running += len_base - (n - 1)

    corrected_categories = []
    corrected_counts = {}

    for category in desired_lengths:
        ifg_payload = load_ifg_chunk(output_path, category, chunk_id)
        coh_payload = load_coh_chunk(output_path, category, chunk_id)

        if ifg_payload is None or coh_payload is None:
            continue

        all_ifgs = ifg_payload["data"]
        all_coh = coh_payload["data"]

        n = category // params["interval"]
        if n < 1:
            continue

        expected_count = len_base - (n - 1)
        if expected_count <= 0:
            continue

        out_ifgs = []
        out_coh = []

        x_start = x_start_by_n[n]

        for t in range(len(all_ifgs)):
            if t >= expected_count:
                out_ifgs.append(None)
                out_coh.append(None)
                continue

            if all_ifgs[t] is None:
                out_ifgs.append(None)
                out_coh.append(all_coh[t] if t < len(all_coh) else None)
                continue

            if uncor_only == 1:
                corrected = np.array(all_ifgs[t], dtype=np.float32, copy=True)
            else:
                bias = X[x_start + t]
                corrected = np.angle(np.exp(1j * (np.array(all_ifgs[t], dtype=np.float32) - bias)))
                wrap_to_pi_inplace(corrected)

            out_ifgs.append(corrected.astype(np.float32))
            out_coh.append(all_coh[t] if t < len(all_coh) else None)

        save_pickle(
            chunk_file_path(output_path, "tmp_corrected_ifgs", category, chunk_id),
            {
                "data": out_ifgs,
                "category": category,
                "chunk_id": chunk_id,
                "row_range": ifg_payload["row_range"],
                "col_range": ifg_payload["col_range"],
                "valid_fraction": ifg_payload["valid_fraction"],
                "estimate_an_mode": an_mode,
                "a_source": a_source,
                "uncor_only": uncor_only,
            },
        )

        save_pickle(
            chunk_file_path(output_path, "tmp_corrected_coh", category, chunk_id),
            {
                "data": out_coh,
                "category": category,
                "chunk_id": chunk_id,
                "row_range": coh_payload["row_range"],
                "col_range": coh_payload["col_range"],
                "valid_fraction": coh_payload["valid_fraction"],
            },
        )

        corrected_categories.append(category)
        corrected_counts[category] = sum(1 for item in out_ifgs if item is not None)

    del X_base, X
    gc.collect()

    return {
        "chunk_id": chunk_id,
        "status": "ok",
        "corrected_categories": corrected_categories,
        "corrected_counts": corrected_counts,
        "a_source": a_source,
        "uncor_only": uncor_only,
    }


# ======================================================================================
# Stage 2: streaming GEOC writing helpers
# ======================================================================================

def load_tmp_corrected_chunk(output_path, category, chunk_id, kind):
    if kind == "ifgs":
        path = chunk_file_path(output_path, "tmp_corrected_ifgs", category, chunk_id)
    elif kind == "coh":
        path = chunk_file_path(output_path, "tmp_corrected_coh", category, chunk_id)
    else:
        raise ValueError("kind must be 'ifgs' or 'coh'")

    if not os.path.exists(path):
        return None
    return load_pickle(path)


def get_n_times_for_category(output_path, category, valid_chunks):
    for chunk_id in valid_chunks:
        payload = load_tmp_corrected_chunk(output_path, category, chunk_id, "ifgs")
        if payload is not None:
            return len(payload["data"])
    return 0


def write_one_category_to_geoc(
    category,
    output_path,
    valid_chunks,
    frame_shape,
    ifg_names_for_category,
    output_wrap_dir,
    template_tif_path,
):
    """
    Worker for one category:
    - stream stitch one epoch at a time
    - write final GEOC files for this category only
    """
    n_times = get_n_times_for_category(output_path, category, valid_chunks)
    if n_times == 0:
        return {
            "category": category,
            "status": "skipped",
            "written": 0,
            "n_times": 0,
        }

    written = 0

    for t in range(n_times):
        full_ifg = np.full(frame_shape, np.nan, dtype=np.float32)
        full_coh = np.full(frame_shape, np.nan, dtype=np.float32)

        has_ifg = False
        has_coh = False

        for chunk_id in valid_chunks:
            ifg_payload = load_tmp_corrected_chunk(output_path, category, chunk_id, "ifgs")
            coh_payload = load_tmp_corrected_chunk(output_path, category, chunk_id, "coh")

            if ifg_payload is None or coh_payload is None:
                continue

            if t >= len(ifg_payload["data"]) or t >= len(coh_payload["data"]):
                continue

            r0, r1 = ifg_payload["row_range"]
            c0, c1 = ifg_payload["col_range"]

            arr_ifg = ifg_payload["data"][t]
            arr_coh = coh_payload["data"][t]

            if arr_ifg is not None:
                full_ifg[r0:r1, c0:c1] = arr_ifg
                has_ifg = True

            if arr_coh is not None:
                full_coh[r0:r1, c0:c1] = arr_coh
                has_coh = True

        if not has_ifg:
            del full_ifg, full_coh
            gc.collect()
            continue

        ifg_name = ifg_names_for_category[t]

        output_phase_path = os.path.join(output_wrap_dir, ifg_name)
        os.makedirs(output_phase_path, exist_ok=True)

        output_phase_file = os.path.join(
            output_phase_path, ifg_name + ".geo.diff_pha.tif"
        )
        output_cc_file = os.path.join(
            output_phase_path, ifg_name + ".geo.cc.tif"
        )

        ifg_cc_2tif(full_ifg, output_phase_file, template_tif_path)
        if has_coh:
            ifg_cc_2tif(full_coh, output_cc_file, template_tif_path)

        written += 1

        del full_ifg, full_coh
        gc.collect()

    return {
        "category": category,
        "status": "ok",
        "written": written,
        "n_times": n_times,
    }


# ======================================================================================
# Main
# ======================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=48, help="Number of parallel workers across chunks.")
    parser.add_argument("--n_write_workers", type=int, default=48, help="Number of parallel workers for final GEOC writing by category.")
    parser.add_argument("--max_con", type=int, default=4, help="Maximum connection length in multiples of interval.")
    parser.add_argument("--uncor_only", type=int, default=0, help="If 1, skip correction and just copy original IFGs.")
    parser.add_argument("--keep_tmp_chunks", type=int, default=0, help="If 1, keep temporary corrected chunk files.")
    args = parser.parse_args()

    params = read_config()

    root_path = params["root_path"]
    output_path = params["output_path"]
    frame = params["frame"]
    start = params["start"]
    end = params["end"]
    interval = params["interval"]
    nlook = params["nlook"]
    LiCSAR_data = params["LiCSAR_data"]
    an_mode = params["estimate_an_mode"]

    start_date = datetime.strptime(start, "%Y%m%d")
    end_date = datetime.strptime(end, "%Y%m%d")

    desired_lengths = [interval * i for i in range(1, args.max_con + 1)]

    print(f"estimate_an_mode = {an_mode}")
    print(f"desired_lengths = {desired_lengths}")
    print(f"n_workers = {args.n_workers}")
    print(f"n_write_workers = {args.n_write_workers}")
    print(f"uncor_only = {args.uncor_only}")

    root_directory = get_root_directory(root_path, frame, LiCSAR_data)

    chunk_index = load_chunk_index(output_path)
    valid_chunks = chunk_index["valid_chunks"]
    frame_shape = tuple(chunk_index["frame_shape"])

    print(f"Number of valid chunks to process: {len(valid_chunks)}")

    args_dict = {
        "max_con": args.max_con,
        "uncor_only": args.uncor_only,
        "desired_lengths": desired_lengths,
    }

    results = []

    # ------------------------------------------------------------------
    # Stage 1: correct chunkwise
    # ------------------------------------------------------------------
    if args.n_workers == 1:
        for chunk_id in valid_chunks:
            cr, cc = chunk_id
            print(f"Processing chunk r{cr:03d}, c{cc:03d} ...")
            res = process_one_chunk(chunk_id, params, args_dict)
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = {
                ex.submit(process_one_chunk, chunk_id, params, args_dict): chunk_id
                for chunk_id in valid_chunks
            }

            completed = 0
            total = len(futures)

            for fut in as_completed(futures):
                completed += 1
                chunk_id = futures[fut]
                cr, cc = chunk_id

                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "chunk_id": chunk_id,
                        "status": "failed",
                        "reason": str(e),
                    }

                results.append(res)

                pct = 100.0 * completed / total
                print(
                    f"[Correction {completed}/{total}] {pct:5.1f}% - chunk r{cr:03d}, c{cc:03d}: {res['status']}"
                )

    ok_results = [r for r in results if r["status"] == "ok"]
    failed_results = [r for r in results if r["status"] != "ok"]

    # ------------------------------------------------------------------
    # Stage 2: prepare GEOC writing
    # ------------------------------------------------------------------
    GEOC_dir = "GEOC"
    metadata_dir = "metadata"

    output_wrap_dir = os.path.join(output_path, GEOC_dir)
    os.makedirs(output_wrap_dir, exist_ok=True)

    template_tif_path = os.path.join(root_directory, metadata_dir, frame + ".geo.hgt.tif")
    if nlook != 1:
        resampled_tif_path = os.path.join(output_wrap_dir, frame + ".geo.hgt.tif")
        resample_geotiff(template_tif_path, resampled_tif_path, nlook)
        template_tif_path = resampled_tif_path

    all_ifgs_string = generate_ifg_pairs(start_date, end_date, interval, desired_lengths)

    print("\nPreparing GEOC writing by category...")

    category_epoch_counts = {}
    for category in desired_lengths:
        category_epoch_counts[category] = get_n_times_for_category(output_path, category, valid_chunks)

    ifg_names_by_category = {}
    flat_index = 0
    for category in desired_lengths:
        ncat = category_epoch_counts[category]
        ifg_names_by_category[category] = all_ifgs_string[flat_index:flat_index + ncat]
        flat_index += ncat

    # ------------------------------------------------------------------
    # Stage 3: GEOC writing by category, optionally parallel
    # ------------------------------------------------------------------
    print("Writing GEOC outputs by category...")

    write_results = []

    if args.n_write_workers == 1:
        for k, category in enumerate(desired_lengths, start=1):
            print(f"Writing category {category} days ({k}/{len(desired_lengths)}) ...")
            res = write_one_category_to_geoc(
                category=category,
                output_path=output_path,
                valid_chunks=valid_chunks,
                frame_shape=frame_shape,
                ifg_names_for_category=ifg_names_by_category.get(category, []),
                output_wrap_dir=output_wrap_dir,
                template_tif_path=template_tif_path,
            )
            write_results.append(res)
            print(
                f"  category {category}: {res['status']}, wrote {res['written']} of {res['n_times']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.n_write_workers) as ex:
            futures = {
                ex.submit(
                    write_one_category_to_geoc,
                    category,
                    output_path,
                    valid_chunks,
                    frame_shape,
                    ifg_names_by_category.get(category, []),
                    output_wrap_dir,
                    template_tif_path,
                ): category
                for category in desired_lengths
            }

            completed = 0
            total = len(futures)

            for fut in as_completed(futures):
                completed += 1
                category = futures[fut]

                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "category": category,
                        "status": "failed",
                        "written": 0,
                        "n_times": 0,
                        "reason": str(e),
                    }

                write_results.append(res)

                pct = 100.0 * completed / total
                msg = (
                    f"[Writing {completed}/{total}] {pct:5.1f}% - "
                    f"category {category}: {res['status']}, wrote {res.get('written', 0)}"
                )
                if "reason" in res:
                    msg += f" ({res['reason']})"
                print(msg)

    print("\nAll corrected interferograms have been written to disk.")

    # ------------------------------------------------------------------
    # Stage 4: save summary
    # ------------------------------------------------------------------
    correction_dir = os.path.join(output_path, "Data", "corrected_chunks")
    os.makedirs(correction_dir, exist_ok=True)

    summary_counts = {cat: 0 for cat in desired_lengths}
    for r in ok_results:
        for cat, cnt in r.get("corrected_counts", {}).items():
            summary_counts[cat] += cnt

    written_total = sum(r.get("written", 0) for r in write_results if r.get("status") == "ok")

    correction_chunk_index = {
        "estimate_an_mode": an_mode,
        "n_workers": args.n_workers,
        "n_write_workers": args.n_write_workers,
        "max_con": args.max_con,
        "desired_lengths": desired_lengths,
        "uncor_only": args.uncor_only,
        "start": start,
        "end": end,
        "interval": interval,
        "n_total_chunks": len(valid_chunks),
        "n_ok_chunks": len(ok_results),
        "n_failed_chunks": len(failed_results),
        "results": results,
        "write_results": write_results,
        "summary_counts": summary_counts,
        "written_total": written_total,
        "output_geoc_dir": output_wrap_dir,
    }

    correction_chunk_index_file = os.path.join(correction_dir, "correction_chunk_index.pkl")
    save_pickle(correction_chunk_index_file, correction_chunk_index)

    # ------------------------------------------------------------------
    # Stage 5: optionally clean temporary chunk files
    # ------------------------------------------------------------------
    if args.keep_tmp_chunks == 0:
        tmp_dir = os.path.join(output_path, "Data", "corrected_tmp")
        if os.path.exists(tmp_dir):
            for root, dirs, files in os.walk(tmp_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except OSError:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    print("\nChunked correction completed.")
    print(f"Successful chunks: {len(ok_results)}")
    print(f"Failed/skipped chunks: {len(failed_results)}")
    print(f"Total GEOC files written: {written_total}")
    print(f"Saved correction summary to: {correction_chunk_index_file}")
    print(f"Final GEOC output directory: {output_wrap_dir}")


if __name__ == "__main__":
    main()
