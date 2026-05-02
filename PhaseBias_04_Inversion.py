#!/usr/bin/env python3

import os
import sys
import gc
import pickle
import warnings
import configparser
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import lsqr

from bin.circular_mean_var import circular_mean_and_variance_over_epochs

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

HELP_TEXT = """
PhaseBias_03_Phase_Bias_Estimation_Chunked.py

This script estimates the phase-bias terms for the base interferograms using
short-term loop closures as observations, chunk by chunk.

It uses only:
- loop closures: (2*interval, interval) and (3*interval, interval)
- coherence: interval, 2*interval, 3*interval

The source of the calibration parameters a1 and a2 is controlled by:
    estimate_an_mode = config | global | pixel

Modes:
- config : use scalar a1/a2 values from config.txt
- global : use chunkwise scalar a1/a2 values from Data/an_chunks/global_clean/
           (fallback to Data/an_chunks/global/, then config values)
- pixel  : use chunkwise pixel-wise a1/a2 maps from Data/an_chunks/pixel_maps/
           (fallback to chunkwise global, then config values)

Outputs:
- Data/X_chunks/chunk_rXXX_cXXX.pkl
- Data/X_chunks/x_chunk_index.pkl

This script parallelises naturally over chunks.
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
            f"Error: configuration file '{config_file}' not found."
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
        params["a1_12_day"] = config.getfloat("DEFAULT", "a1_12_day")
        params["a2_12_day"] = config.getfloat("DEFAULT", "a2_12_day")

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


def get_config_a_values(params):
    interval = params["interval"]
    if interval == 6:
        return params["a1_6_day"], params["a2_6_day"]
    else:
        return params["a1_12_day"], params["a2_12_day"]


def load_chunk_index(output_path):
    chunk_index_file = os.path.join(output_path, "Data", "chunk_index.pkl")
    if not os.path.exists(chunk_index_file):
        raise FileNotFoundError(
            f"Chunk index not found: {chunk_index_file}\n"
            f"Please run the chunked PhaseBias_01_Read_Data script first."
        )
    return load_pickle(chunk_index_file)


def chunk_file_path(output_path, kind, category_or_name, chunk_id, short_baseline=None):
    cr, cc = chunk_id

    if kind == "coh":
        return os.path.join(
            output_path, "Data", "coh", f"{category_or_name}d", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    elif kind == "loops":
        return os.path.join(
            output_path, "Data", "loops", f"{category_or_name}_{short_baseline}",
            f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    elif kind == "an_global":
        return os.path.join(
            output_path, "Data", "an_chunks", "global", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    elif kind == "an_global_clean":
        return os.path.join(
            output_path, "Data", "an_chunks", "global_clean", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    elif kind == "an_pixel":
        return os.path.join(
            output_path, "Data", "an_chunks", "pixel_maps", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")


def load_coh_chunk(output_path, category, chunk_id):
    path = chunk_file_path(output_path, "coh", category, chunk_id)
    if not os.path.exists(path):
        return None
    return load_pickle(path)


def load_loop_chunk(output_path, long_baseline, short_baseline, chunk_id):
    path = chunk_file_path(output_path, "loops", long_baseline, chunk_id, short_baseline=short_baseline)
    if not os.path.exists(path):
        return None
    return load_pickle(path)


def load_chunk_global_a_values(output_path, chunk_id, fallback_a1, fallback_a2):
    """
    Prefer cleaned chunkwise global values, then raw chunkwise global values, then config fallback.
    """
    a1 = fallback_a1
    a2 = fallback_a2

    path_clean = chunk_file_path(output_path, "an_global_clean", None, chunk_id)
    path_raw = chunk_file_path(output_path, "an_global", None, chunk_id)

    payload = None
    if os.path.exists(path_clean):
        payload = load_pickle(path_clean)
    elif os.path.exists(path_raw):
        payload = load_pickle(path_raw)

    if payload is not None and "values" in payload:
        vals = payload["values"]
        if "a1" in vals and "regression" in vals["a1"] and np.isfinite(vals["a1"]["regression"]):
            a1 = float(vals["a1"]["regression"])
        if "a2" in vals and "regression" in vals["a2"] and np.isfinite(vals["a2"]["regression"]):
            a2 = float(vals["a2"]["regression"])

    return a1, a2


def load_chunk_pixel_a_maps(output_path, chunk_id, frame_shape, fallback_a1, fallback_a2):
    """
    Loads chunkwise pixel maps from Data/an_chunks/pixel_maps/chunk_*.pkl
    Falls back to scalar values if maps or NaNs are missing.
    """
    path = chunk_file_path(output_path, "an_pixel", None, chunk_id)

    # first get chunkwise global fallback if available
    global_fallback_a1, global_fallback_a2 = load_chunk_global_a_values(
        output_path, chunk_id, fallback_a1, fallback_a2
    )

    if not os.path.exists(path):
        a1_map = np.full(frame_shape, global_fallback_a1, dtype=np.float32)
        a2_map = np.full(frame_shape, global_fallback_a2, dtype=np.float32)
        return a1_map, a2_map

    payload = load_pickle(path)
    maps = payload.get("maps", payload)

    if "a1" in maps:
        a1_map = np.array(maps["a1"], dtype=np.float32)
    else:
        a1_map = np.full(frame_shape, global_fallback_a1, dtype=np.float32)

    if "a2" in maps:
        a2_map = np.array(maps["a2"], dtype=np.float32)
    else:
        a2_map = np.full(frame_shape, global_fallback_a2, dtype=np.float32)

    if a1_map.shape != frame_shape or a2_map.shape != frame_shape:
        raise ValueError(
            f"Shape mismatch in pixel-wise a maps for chunk {chunk_id}. "
            f"Expected {frame_shape}, got a1 {a1_map.shape}, a2 {a2_map.shape}."
        )

    a1_map = np.where(np.isnan(a1_map), global_fallback_a1, a1_map)
    a2_map = np.where(np.isnan(a2_map), global_fallback_a2, a2_map)

    return a1_map, a2_map


# ======================================================================================
# Preprocessing helpers
# ======================================================================================

def coherence_mask_loops(loop_arr, long_coh, base_coh, n_base, coh_thresh):
    total_pixels = 0
    masked_pixels = 0

    for i, arr in enumerate(loop_arr):
        if arr is None:
            continue

        long_coh_i = long_coh[i]
        if long_coh_i is None:
            continue

        contributing = []
        valid = True
        for j in range(n_base):
            idx = i + j
            if idx >= len(base_coh) or base_coh[idx] is None:
                valid = False
                break
            contributing.append(base_coh[idx])

        if not valid:
            continue

        mask = long_coh_i < coh_thresh
        for c in contributing:
            mask |= c < coh_thresh

        total_pixels += mask.size
        masked_pixels += np.count_nonzero(mask)

        loop_arr[i] = np.where(mask, np.nan, arr)

    pct = (100.0 * masked_pixels / total_pixels) if total_pixels > 0 else np.nan
    return loop_arr, pct


def object_array_to_dense_with_none_indices(arr_obj):
    dense = []
    none_indices = []

    for i, arr in enumerate(arr_obj):
        if arr is None:
            none_indices.append(i)
        else:
            dense.append(arr)

    if len(dense) == 0:
        raise ValueError("No valid loop closures found after preprocessing.")

    return np.array(dense, dtype=np.float32), none_indices


def restore_none_structure(dense_arr, none_indices, total_length):
    restored = []
    dense_idx = 0
    none_set = set(none_indices)

    for i in range(total_length):
        if i in none_set:
            restored.append(None)
        else:
            restored.append(dense_arr[dense_idx])
            dense_idx += 1

    return np.array(restored, dtype=object)


def circular_moving_average(arr, window_size):
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    half_window = window_size // 2
    moving_averages = np.zeros_like(arr, dtype=np.float32)

    mirrored_shape = (arr.shape[0] + window_size, arr.shape[1], arr.shape[2])
    mirrored_arr = np.empty(mirrored_shape, dtype=arr.dtype)

    mirrored_arr[half_window:-half_window] = arr

    for i in range(half_window):
        mirrored_arr[i] = arr[min(half_window - i, arr.shape[0] - 1)]

    for i in range(half_window):
        mirrored_arr[-(i + 1)] = arr[max(-1 - (half_window - i), -arr.shape[0])]

    for t in range(half_window, mirrored_arr.shape[0] - half_window):
        moving_averages[t - half_window] = np.angle(
            np.nanmean(
                np.exp(1j * mirrored_arr[t - half_window:t + half_window]),
                axis=0,
            )
        )

    return moving_averages


def mask_noisy_loops(loop_dense, label, window_size=30, sigma_factor=2.0):
    _, circular_var = circular_mean_and_variance_over_epochs(loop_dense, axis=0)
    thresh = np.expand_dims(circular_var, axis=0)

    moving_avg = circular_moving_average(loop_dense, window_size)
    distances = np.abs(np.angle(np.exp(1j * (loop_dense - moving_avg))))
    mask = distances > sigma_factor * thresh

    loop_dense_masked = np.where(mask, np.nan, loop_dense)

    total = np.prod(mask.shape)
    n_masked = np.count_nonzero(mask)
    pct = 100.0 * n_masked / total if total > 0 else np.nan

    return loop_dense_masked, pct


def build_observation_stack(loop_12_obj, loop_18_obj):
    b = []
    for arr in loop_12_obj:
        b.append(arr)
    for arr in loop_18_obj:
        b.append(arr)
    return np.array(b, dtype=object)


def remove_none_rows(b):
    valid_mask = np.array([arr is not None for arr in b])
    b_valid = b[valid_mask]
    return b_valid, valid_mask


# ======================================================================================
# Design matrices
# ======================================================================================

def build_global_design_matrix(len12, len18, a1, a2):
    n_unk = len12 + 1
    rows = []

    for i in range(len12):
        row = np.zeros(n_unk, dtype=np.float32)
        row[i:i + 2] = a1 - 1.0
        rows.append(row)

    for i in range(len18):
        row = np.zeros(n_unk, dtype=np.float32)
        row[i:i + 3] = a2 - 1.0
        rows.append(row)

    return np.array(rows, dtype=np.float32)


def build_pixelwise_design_matrix(len12, len18, a1_pix, a2_pix):
    n_unk = len12 + 1
    rows = []

    for i in range(len12):
        row = np.zeros(n_unk, dtype=np.float32)
        row[i:i + 2] = a1_pix - 1.0
        rows.append(row)

    for i in range(len18):
        row = np.zeros(n_unk, dtype=np.float32)
        row[i:i + 3] = a2_pix - 1.0
        rows.append(row)

    return np.array(rows, dtype=np.float32)


def add_temporal_constraints_dense(A, n_unk, w):
    temporal_rows = []
    for i in range(n_unk - 2):
        row = np.zeros(n_unk, dtype=np.float32)
        row[i] = -w
        row[i + 1] = 2.0 * w
        row[i + 2] = -w
        temporal_rows.append(row)

    if len(temporal_rows) == 0:
        A_temporal = np.zeros((0, n_unk), dtype=np.float32)
    else:
        A_temporal = np.array(temporal_rows, dtype=np.float32)

    A_full = np.vstack((A, A_temporal))
    n_temporal = A_temporal.shape[0]
    return A_full, n_temporal


# ======================================================================================
# Inversion kernels
# ======================================================================================

def invert_with_global_a(
    b_obs,
    valid_mask_rows,
    len12,
    len18,
    a1,
    a2,
    frame_row,
    frame_col,
    w,
    min_num_eq=10,
    damp=0.0,
):
    n_unk = len12 + 1
    X = np.full((n_unk, frame_row, frame_col), np.nan, dtype=np.float32)

    A_all = build_global_design_matrix(len12, len18, a1, a2)
    A_obs = A_all[valid_mask_rows, :]
    A_with_temporal, n_temporal = add_temporal_constraints_dense(A_obs, n_unk, w)
    A_temporal = A_with_temporal[A_obs.shape[0]:, :]

    total_misfit = 0.0
    total_equations = 0

    for row in range(frame_row):
        for col in range(frame_col):
            bb_obs = b_obs[:, row, col]
            non_nan_mask = ~np.isnan(bb_obs)

            bb = bb_obs[non_nan_mask]
            AA_obs = A_obs[non_nan_mask, :]

            if len(bb) <= min_num_eq:
                continue

            bb_temporal = np.zeros(n_temporal, dtype=np.float32)
            AA = np.vstack((AA_obs, A_temporal))
            bb_full = np.concatenate((bb, bb_temporal))

            if len(bb_full) > max(min_num_eq, n_temporal):
                x, istop, itn, normr, normr2 = lsqr(
                    csc_array(AA), bb_full, damp=damp
                )[:5]

                X[:, row, col] = x.astype(np.float32)
                total_misfit += normr ** 2
                total_equations += len(bb_full)

    rms_residual = np.sqrt(total_misfit / total_equations) if total_equations > 0 else np.nan
    return X, rms_residual


def invert_pixelwise_with_pixel_a(
    b_obs,
    valid_mask_rows,
    len12,
    len18,
    a1_map,
    a2_map,
    frame_row,
    frame_col,
    w,
    min_num_eq=10,
    damp=0.0,
):
    n_unk = len12 + 1
    X = np.full((n_unk, frame_row, frame_col), np.nan, dtype=np.float32)

    total_misfit = 0.0
    total_equations = 0

    for row in range(frame_row):
        for col in range(frame_col):
            a1_pix = a1_map[row, col]
            a2_pix = a2_map[row, col]

            if np.isnan(a1_pix) or np.isnan(a2_pix):
                continue

            A_local_all = build_pixelwise_design_matrix(len12, len18, a1_pix, a2_pix)
            A_local_obs = A_local_all[valid_mask_rows, :]

            A_with_temporal, n_temporal = add_temporal_constraints_dense(A_local_obs, n_unk, w)

            bb_obs = b_obs[:, row, col]
            non_nan_mask = ~np.isnan(bb_obs)

            bb = bb_obs[non_nan_mask]
            AA_obs = A_local_obs[non_nan_mask, :]

            if len(bb) <= min_num_eq:
                continue

            A_temporal = A_with_temporal[A_local_obs.shape[0]:, :]
            bb_temporal = np.zeros(n_temporal, dtype=np.float32)

            AA = np.vstack((AA_obs, A_temporal))
            bb_full = np.concatenate((bb, bb_temporal))

            if len(bb_full) > max(min_num_eq, n_temporal):
                x, istop, itn, normr, normr2 = lsqr(
                    csc_array(AA), bb_full, damp=damp
                )[:5]

                X[:, row, col] = x.astype(np.float32)
                total_misfit += normr ** 2
                total_equations += len(bb_full)

    rms_residual = np.sqrt(total_misfit / total_equations) if total_equations > 0 else np.nan
    return X, rms_residual


# ======================================================================================
# Per-chunk worker
# ======================================================================================

def process_one_chunk(chunk_id, params, args_dict):
    output_path = params["output_path"]
    interval = params["interval"]
    an_mode = params["estimate_an_mode"]

    config_a1, config_a2 = get_config_a_values(params)

    cr, cc = chunk_id

    # required chunks
    loop_12_payload = load_loop_chunk(output_path, 2 * interval, interval, chunk_id)
    loop_18_payload = load_loop_chunk(output_path, 3 * interval, interval, chunk_id)

    coh_6_payload = load_coh_chunk(output_path, interval, chunk_id)
    coh_12_payload = load_coh_chunk(output_path, 2 * interval, chunk_id)
    coh_18_payload = load_coh_chunk(output_path, 3 * interval, chunk_id)

    if any(x is None for x in [loop_12_payload, loop_18_payload, coh_6_payload, coh_12_payload, coh_18_payload]):
        return {
            "chunk_id": chunk_id,
            "status": "skipped",
            "reason": "missing required loop/coherence chunk files",
        }

    loop_12 = np.array(loop_12_payload["data"], dtype=object)
    loop_18 = np.array(loop_18_payload["data"], dtype=object)

    coh_6 = np.array(coh_6_payload["data"], dtype=object)
    coh_12 = np.array(coh_12_payload["data"], dtype=object)
    coh_18 = np.array(coh_18_payload["data"], dtype=object)

    frame_row, frame_col = None, None
    for arr in loop_12:
        if arr is not None:
            frame_row, frame_col = arr.shape
            break
    if frame_row is None:
        for arr in loop_18:
            if arr is not None:
                frame_row, frame_col = arr.shape
                break
    if frame_row is None:
        return {
            "chunk_id": chunk_id,
            "status": "skipped",
            "reason": "could not determine chunk shape from loop closures",
        }

    # coherence mask
    coh_mask_stats = {}
    if args_dict["coh_thresh"] > 0:
        loop_12, pct12 = coherence_mask_loops(
            loop_12, coh_12, coh_6, n_base=2, coh_thresh=args_dict["coh_thresh"]
        )
        loop_18, pct18 = coherence_mask_loops(
            loop_18, coh_18, coh_6, n_base=3, coh_thresh=args_dict["coh_thresh"]
        )
        coh_mask_stats["loop12_pct"] = pct12
        coh_mask_stats["loop18_pct"] = pct18
    else:
        coh_mask_stats["loop12_pct"] = np.nan
        coh_mask_stats["loop18_pct"] = np.nan

    # remove None for noisy-loop masking
    loop_12_dense, none_indices12 = object_array_to_dense_with_none_indices(loop_12)
    loop_18_dense, none_indices18 = object_array_to_dense_with_none_indices(loop_18)

    loop_12_dense, noisy12_pct = mask_noisy_loops(
        loop_12_dense,
        label=f"{2 * interval}-day",
        window_size=args_dict["window_size"],
        sigma_factor=2.0,
    )
    loop_18_dense, noisy18_pct = mask_noisy_loops(
        loop_18_dense,
        label=f"{3 * interval}-day",
        window_size=args_dict["window_size"],
        sigma_factor=2.0,
    )

    loop_12_obj = restore_none_structure(loop_12_dense, none_indices12, len(loop_12))
    loop_18_obj = restore_none_structure(loop_18_dense, none_indices18, len(loop_18))

    len12 = len(loop_12_obj)
    len18 = len(loop_18_obj)

    b = build_observation_stack(loop_12_obj, loop_18_obj)
    b_valid, valid_mask_rows = remove_none_rows(b)
    b_valid = np.array(b_valid.tolist(), dtype=np.float32)

    if an_mode == "config":
        X, rms_residual = invert_with_global_a(
            b_obs=b_valid,
            valid_mask_rows=valid_mask_rows,
            len12=len12,
            len18=len18,
            a1=config_a1,
            a2=config_a2,
            frame_row=frame_row,
            frame_col=frame_col,
            w=args_dict["w"],
            min_num_eq=args_dict["min_num_eq"],
            damp=0.0,
        )
        a_source = {"mode": "config", "a1": config_a1, "a2": config_a2}

    elif an_mode == "global":
        global_a1, global_a2 = load_chunk_global_a_values(
            output_path, chunk_id, config_a1, config_a2
        )
        X, rms_residual = invert_with_global_a(
            b_obs=b_valid,
            valid_mask_rows=valid_mask_rows,
            len12=len12,
            len18=len18,
            a1=global_a1,
            a2=global_a2,
            frame_row=frame_row,
            frame_col=frame_col,
            w=args_dict["w"],
            min_num_eq=args_dict["min_num_eq"],
            damp=0.0,
        )
        a_source = {"mode": "global", "a1": global_a1, "a2": global_a2}

    elif an_mode == "pixel":
        a1_map, a2_map = load_chunk_pixel_a_maps(
            output_path=output_path,
            chunk_id=chunk_id,
            frame_shape=(frame_row, frame_col),
            fallback_a1=config_a1,
            fallback_a2=config_a2,
        )

        X, rms_residual = invert_pixelwise_with_pixel_a(
            b_obs=b_valid,
            valid_mask_rows=valid_mask_rows,
            len12=len12,
            len18=len18,
            a1_map=a1_map,
            a2_map=a2_map,
            frame_row=frame_row,
            frame_col=frame_col,
            w=args_dict["w"],
            min_num_eq=args_dict["min_num_eq"],
            damp=0.0,
        )
        a_source = {
            "mode": "pixel",
            "a1_min": float(np.nanmin(a1_map)),
            "a1_max": float(np.nanmax(a1_map)),
            "a2_min": float(np.nanmin(a2_map)),
            "a2_max": float(np.nanmax(a2_map)),
        }

    else:
        return {
            "chunk_id": chunk_id,
            "status": "failed",
            "reason": f"Unknown estimate_an_mode: {an_mode}",
        }

    # save chunk result
    x_chunk_dir = os.path.join(output_path, "Data", "X_chunks")
    x_chunk_file = os.path.join(x_chunk_dir, f"chunk_r{cr:03d}_c{cc:03d}.pkl")

    payload = {
        "chunk_id": chunk_id,
        "row_range": loop_12_payload["row_range"],
        "col_range": loop_12_payload["col_range"],
        "valid_fraction": loop_12_payload["valid_fraction"],
        "estimate_an_mode": an_mode,
        "a_source": a_source,
        "w": args_dict["w"],
        "coh_thresh": args_dict["coh_thresh"],
        "min_num_eq": args_dict["min_num_eq"],
        "window_size": args_dict["window_size"],
        "frame_shape": (frame_row, frame_col),
        "X": X,
        "rms_residual": rms_residual,
        "coh_mask_stats": coh_mask_stats,
        "noisy_loop_pct": {
            "loop12_pct": noisy12_pct,
            "loop18_pct": noisy18_pct,
        },
        "len12": len12,
        "len18": len18,
    }

    save_pickle(x_chunk_file, payload)

    # free some memory
    del loop_12, loop_18, coh_6, coh_12, coh_18, b_valid, X
    gc.collect()

    return {
        "chunk_id": chunk_id,
        "status": "ok",
        "x_chunk_file": x_chunk_file,
        "rms_residual": rms_residual,
        "valid_fraction": loop_12_payload["valid_fraction"],
        "a_source": a_source,
    }


# ======================================================================================
# Main
# ======================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=float, default=0.1, help="Temporal smoothing weight.")
    parser.add_argument("--window_size", type=int, default=30, help="Moving-average window size.")
    parser.add_argument("--coh_thresh", type=float, default=0, help="Coherence threshold.")
    parser.add_argument("--min_num_eq", type=int, default=10, help="Minimum equations per pixel.")
    parser.add_argument("--n_workers", type=int, default=48, help="Number of parallel workers across chunks.")
    args = parser.parse_args()

    params = read_config()
    output_path = params["output_path"]
    an_mode = params["estimate_an_mode"]

    config_a1, config_a2 = get_config_a_values(params)

    print(f"estimate_an_mode = {an_mode}")
    print(f"Config scalar values: a1={config_a1}, a2={config_a2}, w={args.w}")

    chunk_index = load_chunk_index(output_path)
    valid_chunks = chunk_index["valid_chunks"]

    print(f"Number of valid chunks to process: {len(valid_chunks)}")
    print(f"Running with n_workers = {args.n_workers}")

    args_dict = {
        "w": args.w,
        "window_size": args.window_size,
        "coh_thresh": args.coh_thresh,
        "min_num_eq": args.min_num_eq,
    }

    results = []

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
                    f"[{completed}/{total}] {pct:5.1f}% finished - chunk r{cr:03d}, c{cc:03d}: {res['status']}"
                )

    # write summary index
    x_chunk_dir = os.path.join(output_path, "Data", "X_chunks")
    os.makedirs(x_chunk_dir, exist_ok=True)

    ok_results = [r for r in results if r["status"] == "ok"]
    failed_results = [r for r in results if r["status"] != "ok"]

    rms_vals = [r["rms_residual"] for r in ok_results if np.isfinite(r["rms_residual"])]
    rms_mean = float(np.mean(rms_vals)) if len(rms_vals) > 0 else np.nan

    x_chunk_index = {
        "estimate_an_mode": an_mode,
        "w": args.w,
        "window_size": args.window_size,
        "coh_thresh": args.coh_thresh,
        "min_num_eq": args.min_num_eq,
        "n_workers": args.n_workers,
        "n_total_chunks": len(valid_chunks),
        "n_ok_chunks": len(ok_results),
        "n_failed_chunks": len(failed_results),
        "mean_rms_residual": rms_mean,
        "results": results,
    }

    x_chunk_index_file = os.path.join(x_chunk_dir, "x_chunk_index.pkl")
    save_pickle(x_chunk_index_file, x_chunk_index)

    print("\nChunked inversion completed.")
    print(f"Successful chunks: {len(ok_results)}")
    print(f"Failed/skipped chunks: {len(failed_results)}")
    print(f"Mean RMS residual over successful chunks: {rms_mean:.6f}" if np.isfinite(rms_mean) else "Mean RMS residual: nan")
    print(f"Saved chunk summary to: {x_chunk_index_file}")


if __name__ == "__main__":
    main()
