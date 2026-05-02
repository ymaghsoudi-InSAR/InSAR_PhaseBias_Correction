#!/usr/bin/env python3

import os
import sys
import glob
import pickle
import warnings
import configparser

import numpy as np

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

################################################################################################

HELP_TEXT = """
PhaseBias_02_Loop_Closures_Chunked.py

This script calculates loop closures (Δφ) using the chunked interferograms
written by the chunked version of PhaseBias_01_Read_Data.py.

Definition of Loop Closures:
    Δφ_(i,k) = | φ_(i,k) - Σ φ_(t,t+Δt) |_2π

Outputs:
1. Chunked loop-closure files:
   Data/loops/<long>_<short>/chunk_rXXX_cXXX.pkl

2. Chunked missing-index files:
   Data/missing/<category>d/chunk_rXXX_cXXX.pkl

3. Data/loop_index.pkl
   Metadata describing the generated loop products.

Each loop chunk file contains:
- data           : list of arrays / None
- long_baseline  : e.g. 12
- short_baseline : e.g. 6
- chunk_id       : (row_chunk_index, col_chunk_index)
- row_range      : (r0, r1)
- col_range      : (c0, c1)
- valid_fraction : valid land fraction in this chunk
"""

if "--help" in sys.argv:
    print(HELP_TEXT)
    sys.exit(0)

################################################################################################
# Config
################################################################################################

config_file = "config.txt"

if not os.path.exists(config_file):
    raise FileNotFoundError(
        f"Error: The configuration file '{config_file}' was not found in the script's directory."
    )

config = configparser.ConfigParser()
config.read(config_file)

parameters = {}

try:
    parameters["root_path"] = config.get("DEFAULT", "root_path")
    parameters["output_path"] = config.get("DEFAULT", "output_path")
    parameters["frame"] = config.get("DEFAULT", "frame")
    parameters["start"] = config.get("DEFAULT", "start")
    parameters["end"] = config.get("DEFAULT", "end")
    parameters["interval"] = config.getint("DEFAULT", "interval")
    parameters["nlook"] = config.getint("DEFAULT", "nlook")
    parameters["LiCSAR_data"] = config.get("DEFAULT", "LiCSAR_data")
except configparser.NoOptionError as e:
    missing_param = str(e).split(": ")[1]
    raise ValueError(
        f"Error: Required parameter '{missing_param}' is missing in '{config_file}'. Please check the file contents."
    )
except configparser.Error as e:
    raise ValueError(f"Error while reading '{config_file}': {e}")

root_path = parameters["root_path"]
output_path = parameters["output_path"]
frame = parameters["frame"]
start = parameters["start"]
end = parameters["end"]
interval = parameters["interval"]
nlook = parameters["nlook"]
LiCSAR_data = parameters["LiCSAR_data"]

# hardcoded parameters
landmask = 1
filtered_ifgs = "yes"
max_loop = 5

print("Root Path:", root_path)
print("Output Path:", output_path)
print("Frame:", frame)
print("Start Date:", start)
print("End Date:", end)
print("Interval:", interval)
print("nlook:", nlook)
print("LiCSAR_data:", LiCSAR_data)
print("max_loop:", max_loop)

################################################################################################
# Helpers
################################################################################################

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_chunk_index(data_dir):
    chunk_index_file = os.path.join(data_dir, "chunk_index.pkl")
    if not os.path.exists(chunk_index_file):
        raise FileNotFoundError(
            f"Chunk index not found: {chunk_index_file}\n"
            f"Please run the chunked PhaseBias_01_Read_Data.py first."
        )
    return load_pickle(chunk_index_file)


def load_ifg_chunk(data_dir, category, chunk_id):
    cr, cc = chunk_id
    path = os.path.join(
        data_dir,
        "ifgs",
        f"{category}d",
        f"chunk_r{cr:03d}_c{cc:03d}.pkl"
    )
    if not os.path.exists(path):
        return None
    return load_pickle(path)


def load_all_ifg_categories_for_chunk(data_dir, categories, chunk_id):
    all_ifgs = {}
    chunk_meta = None

    for category in categories:
        payload = load_ifg_chunk(data_dir, category, chunk_id)
        if payload is not None:
            all_ifgs[category] = payload["data"]
            if chunk_meta is None:
                chunk_meta = {
                    "chunk_id": payload["chunk_id"],
                    "row_range": payload["row_range"],
                    "col_range": payload["col_range"],
                    "valid_fraction": payload["valid_fraction"],
                }

    return all_ifgs, chunk_meta


def initialise_loop_dict(cat, max_loop):
    loop = {}
    missing = {}

    for category in cat:
        loop[category] = {}
        missing[category] = []

        max_l = min(max_loop, len(cat))
        for l in range(max_l):
            loop[category][cat[l]] = []

    return loop, missing


def calculate_total_iterations(all_ifgs, cat, max_loop):
    total_iterations = 0
    max_l = min(max_loop, len(cat))

    for l in range(max_l):
        for i in range(1, len(cat)):
            if cat[i] % cat[l] == 0 and cat[i] != cat[l]:
                total_iterations += len(all_ifgs[cat[i]])

    return total_iterations


################################################################################################
# Core loop calculator for one chunk
################################################################################################

def loop_calc_chunk(all_ifgs, max_loop, interval, show_progress=False):
    """
    all_ifgs:
        dict like {6: [arr/None, arr/None, ...], 12: [...], ...}
        for one chunk only
    """
    if len(all_ifgs) == 0:
        return {}, {}, {}, {}

    cat = sorted(all_ifgs.keys())
    loop, missing = initialise_loop_dict(cat, max_loop)

    loop_counts = {
        category: {
            cat[l]: 0
            for l in range(min(max_loop, len(cat)))
        }
        for category in cat
    }

    missing_counts = {
        category: {
            cat[l]: 0
            for l in range(min(max_loop, len(cat)))
        }
        for category in cat
    }

    total_iterations = calculate_total_iterations(all_ifgs, cat, max_loop)
    completed_iterations = 0

    if show_progress:
        print(
            f"Calculating all {interval}, {2 * interval}, and {3 * interval} loop closures for this chunk."
        )

    max_l = min(max_loop, len(cat))

    for l in range(max_l):
        for i in range(1, len(cat)):
            if cat[i] % cat[l] == 0 and cat[i] != cat[l]:
                for t in range(len(all_ifgs[cat[i]])):
                    completed_iterations += 1

                    if show_progress and total_iterations > 0:
                        percentage = (completed_iterations / total_iterations) * 100
                        sys.stdout.write(
                            "\rProgress: [{:<50}] {}%".format(
                                "=" * int(percentage // 2), int(percentage)
                            )
                        )
                        sys.stdout.flush()

                    if cat[l] == interval:
                        end_index = t + int(cat[i] / cat[l])
                    else:
                        end_index = t + int(cat[i] / cat[l]) * int(cat[l] / interval)

                    # record missing long IFG
                    if np.any(np.array(all_ifgs[cat[i]], dtype=object)[t] == None):
                        missing[cat[i]].append(t)

                    # overshoot check
                    if end_index > len(all_ifgs[cat[l]]):
                        loop[cat[i]][cat[l]].append(None)
                        missing_counts[cat[i]][cat[l]] += 1
                        continue

                    # record missing short IFGs contributing to this loop
                    for e in range(t, end_index, int(cat[l] / interval)):
                        elem = all_ifgs[cat[l]][e]
                        if elem is None:
                            missing[cat[l]].append(e)

                    if (
                        all_ifgs[cat[i]][t] is None
                        or any(
                            all_ifgs[cat[l]][elem] is None
                            for elem in range(t, end_index, int(cat[l] / interval))
                        )
                    ):
                        loop[cat[i]][cat[l]].append(None)
                        missing_counts[cat[i]][cat[l]] += 1
                    else:
                        closure = np.angle(
                            np.exp(
                                1j
                                * (
                                    np.array(all_ifgs[cat[i]][t], dtype=np.float32)
                                    - np.sum(
                                        np.array(
                                            all_ifgs[cat[l]][t:end_index:int(cat[l] / interval)],
                                            dtype=np.float32
                                        ),
                                        axis=0,
                                    )
                                )
                            )
                        )

                        loop[cat[i]][cat[l]].append(closure)
                        loop_counts[cat[i]][cat[l]] += 1

    if show_progress:
        print("\nFinished calculating the loop closures for this chunk.")

    return loop, missing, loop_counts, missing_counts


################################################################################################
# Writers
################################################################################################

def save_loop_chunk(data_dir, long_baseline, short_baseline, chunk_meta, data_list):
    cr, cc = chunk_meta["chunk_id"]
    out_path = os.path.join(
        data_dir,
        "loops",
        f"{long_baseline}_{short_baseline}",
        f"chunk_r{cr:03d}_c{cc:03d}.pkl",
    )

    payload = {
        "data": data_list,
        "long_baseline": long_baseline,
        "short_baseline": short_baseline,
        "chunk_id": chunk_meta["chunk_id"],
        "row_range": chunk_meta["row_range"],
        "col_range": chunk_meta["col_range"],
        "valid_fraction": chunk_meta["valid_fraction"],
    }

    save_pickle(out_path, payload)


def save_missing_chunk(data_dir, category, chunk_meta, missing_list):
    cr, cc = chunk_meta["chunk_id"]
    out_path = os.path.join(
        data_dir,
        "missing",
        f"{category}d",
        f"chunk_r{cr:03d}_c{cc:03d}.pkl",
    )

    payload = {
        "missing_indices": sorted(list(set(missing_list))),
        "category": category,
        "chunk_id": chunk_meta["chunk_id"],
        "row_range": chunk_meta["row_range"],
        "col_range": chunk_meta["col_range"],
        "valid_fraction": chunk_meta["valid_fraction"],
    }

    save_pickle(out_path, payload)


################################################################################################
# Main
################################################################################################

def main():
    data_dir = os.path.join(output_path, "Data")
    os.makedirs(data_dir, exist_ok=True)

    chunk_index = load_chunk_index(data_dir)
    valid_chunks = chunk_index["valid_chunks"]
    categories_ifgs = sorted(chunk_index["categories_ifgs"])

    print(f"Number of valid chunks: {len(valid_chunks)}")
    print(f"IFG categories available: {categories_ifgs}")

    overall_loop_counts = {}
    overall_missing_counts = {}

    for category in categories_ifgs:
        overall_loop_counts[category] = {}
        overall_missing_counts[category] = {}
        for short_cat in categories_ifgs[:min(max_loop, len(categories_ifgs))]:
            overall_loop_counts[category][short_cat] = 0
            overall_missing_counts[category][short_cat] = 0

    total_chunks = len(valid_chunks)
    processed_chunks = 0

    print("\nCalculating loop closures chunk by chunk...")

    for chunk_id in valid_chunks:
        processed_chunks += 1
        cr, cc = chunk_id

        pct = 100.0 * processed_chunks / total_chunks
        sys.stdout.write(
            f"\rChunk progress: [{'=' * int(pct // 2):<50}] {int(pct)}% "
            f"(chunk r{cr:03d}, c{cc:03d})"
        )
        sys.stdout.flush()

        all_ifgs, chunk_meta = load_all_ifg_categories_for_chunk(
            data_dir, categories_ifgs, chunk_id
        )

        if len(all_ifgs) == 0 or chunk_meta is None:
            continue

        loop, missing, loop_counts, missing_counts = loop_calc_chunk(
            all_ifgs=all_ifgs,
            max_loop=max_loop,
            interval=interval,
            show_progress=False,
        )

        # save loops
        for long_baseline in sorted(loop.keys()):
            for short_baseline in sorted(loop[long_baseline].keys()):
                save_loop_chunk(
                    data_dir=data_dir,
                    long_baseline=long_baseline,
                    short_baseline=short_baseline,
                    chunk_meta=chunk_meta,
                    data_list=loop[long_baseline][short_baseline],
                )

                overall_loop_counts[long_baseline][short_baseline] += sum(
                    1 for item in loop[long_baseline][short_baseline] if item is not None
                )
                overall_missing_counts[long_baseline][short_baseline] += sum(
                    1 for item in loop[long_baseline][short_baseline] if item is None
                )

        # save missing
        for category in sorted(missing.keys()):
            save_missing_chunk(
                data_dir=data_dir,
                category=category,
                chunk_meta=chunk_meta,
                missing_list=missing[category],
            )

    print("\nFinished calculating chunked loop closures.")

    ################################################################################################
    # Reports
    ################################################################################################

    print("\n=== Loop Closure Report ===")
    for category in [2 * interval, 3 * interval]:
        if category not in overall_loop_counts:
            continue
        short_category = interval
        if short_category in overall_loop_counts[category]:
            generated = overall_loop_counts[category][short_category]
            missing_val = overall_missing_counts[category][short_category]
            print(
                f"Loop Closure {category} - {short_category}: Generated = {generated}, Missing = {missing_val}"
            )

    print("\n=== Loop Closure Report for Long-interval Loop Closures ===")
    for category in categories_ifgs:
        if category > 200:
            for short_category in categories_ifgs[:min(max_loop, len(categories_ifgs))]:
                if short_category in overall_loop_counts[category]:
                    generated = overall_loop_counts[category][short_category]
                    if generated >= 1:
                        print(
                            f"Loop Closure {category} - {short_category}: Generated = {generated}"
                        )

    ################################################################################################
    # loop index
    ################################################################################################

    loop_products = []
    for long_baseline in categories_ifgs:
        for short_baseline in categories_ifgs[:min(max_loop, len(categories_ifgs))]:
            if long_baseline % short_baseline == 0 and long_baseline != short_baseline:
                loop_products.append((long_baseline, short_baseline))

    loop_index = {
        "frame": frame,
        "start": start,
        "end": end,
        "interval": interval,
        "nlook": nlook,
        "LiCSAR_data": LiCSAR_data,
        "max_loop": max_loop,
        "valid_chunks": valid_chunks,
        "categories_ifgs": categories_ifgs,
        "loop_products": loop_products,
        "overall_loop_counts": overall_loop_counts,
        "overall_missing_counts": overall_missing_counts,
    }

    loop_index_file = os.path.join(data_dir, "loop_index.pkl")
    save_pickle(loop_index_file, loop_index)

    print(f"\nLoop index written to: {loop_index_file}")


if __name__ == "__main__":
    main()
