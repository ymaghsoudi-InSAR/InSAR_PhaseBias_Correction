#!/usr/bin/env python3

import os
import re
import glob
import sys
import pickle
import warnings
import configparser
from datetime import datetime, timedelta

import numpy as np
from osgeo import gdal

from bin.multilook_w import multilook_w
from bin.multilook import multilook

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

#######################################################################

HELP_TEXT = """
PhaseBias_01_Read_Data_Chunked.py

This script reads wrapped interferograms and coherence images and stores them
in spatial chunks instead of one large pickle file.

Assumptions:
- Interferograms are organized in folders named as `yyyymmdd_yyyymmdd`.
- Missing epochs inside a temporal category are still represented by `None`.
- Data are stored separately for:
    - IFGs / coherence
    - each temporal category
    - each spatial chunk

Outputs:
1. Data/chunk_index.pkl
   Contains metadata about chunk layout, categories, and valid chunks.

2. Chunked IFG files:
   Data/ifgs/<category>d/chunk_rXXX_cXXX.pkl

3. Chunked coherence files:
   Data/coh/<category>d/chunk_rXXX_cXXX.pkl

Each chunk file contains a dictionary with:
- data           : list of arrays / None, preserving temporal alignment
- category       : temporal baseline in days
- chunk_id       : (row_chunk_index, col_chunk_index)
- row_range      : (r0, r1)
- col_range      : (c0, c1)
- valid_fraction : valid land fraction in this chunk

Chunk filtering:
- Chunks with very small valid fraction are skipped completely.
- Valid fraction is computed from the multilooked landmask.

Additional Output:
- Report on number of IFGs and missing IFGs per category
- Report on chunk layout and number of valid chunks
"""

if "--help" in sys.argv:
    print(HELP_TEXT)
    sys.exit(0)

#######################################################################
# Config
#######################################################################

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

    # optional chunk parameters
    parameters["chunk_rows"] = config.getint("DEFAULT", "chunk_rows", fallback=256)
    parameters["chunk_cols"] = config.getint("DEFAULT", "chunk_cols", fallback=256)
    parameters["min_valid_fraction"] = config.getfloat("DEFAULT", "min_valid_fraction", fallback=0.02)

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

chunk_rows = parameters["chunk_rows"]
chunk_cols = parameters["chunk_cols"]
min_valid_fraction = parameters["min_valid_fraction"]

# hardcoded parameters
landmask = 1
filtered_ifgs = "yes"
min_baseline = 5
max_baseline = 366

print("Root Path:", root_path)
print("Output Path:", output_path)
print("Frame:", frame)
print("Start Date:", start)
print("End Date:", end)
print("Interval:", interval)
print("nlook:", nlook)
print("LiCSAR_data:", LiCSAR_data)
print("chunk_rows:", chunk_rows)
print("chunk_cols:", chunk_cols)
print("min_valid_fraction:", min_valid_fraction)

#######################################################################
# Helpers
#######################################################################

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
        return f"/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/{track}/{frame}/interferograms/"
    else:
        return os.path.join(root_path, "interferograms")


def get_landmask_array(root_path, frame, LiCSAR_data, nlook):
    if LiCSAR_data == "yes":
        track = get_track_from_frame(frame)
        landmask_path = (
            f"/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products.public/"
            f"{track}/{frame}/metadata/{frame}.geo.landmask.tif"
        )
    else:
        metadata_dir = os.path.join(root_path, "metadata")
        landmask_files = glob.glob(os.path.join(metadata_dir, "*.geo.landmask.tif"))
        if landmask_files:
            landmask_path = landmask_files[0]
        else:
            raise FileNotFoundError(
                "No file ending with '.geo.landmask.tif' found in the metadata directory."
            )

    ds = gdal.Open(landmask_path)
    if ds is None:
        raise FileNotFoundError(f"Could not open landmask file: {landmask_path}")

    arr = ds.ReadAsArray().astype(np.float32)
    del ds

    if nlook is not None and nlook != 1:
        arr = multilook(arr, nlook)

    return arr


def build_chunk_layout(array_landmask, chunk_rows, chunk_cols, min_valid_fraction):
    nrows, ncols = array_landmask.shape
    n_chunk_rows = int(np.ceil(nrows / chunk_rows))
    n_chunk_cols = int(np.ceil(ncols / chunk_cols))

    chunk_defs = {}
    valid_chunks = []

    for cr in range(n_chunk_rows):
        for cc in range(n_chunk_cols):
            r0 = cr * chunk_rows
            r1 = min((cr + 1) * chunk_rows, nrows)
            c0 = cc * chunk_cols
            c1 = min((cc + 1) * chunk_cols, ncols)

            tile_landmask = array_landmask[r0:r1, c0:c1]
            valid_mask = np.isfinite(tile_landmask) & (tile_landmask == 1)
            valid_fraction = np.count_nonzero(valid_mask) / valid_mask.size

            chunk_id = (cr, cc)
            chunk_defs[chunk_id] = {
                "row_range": (r0, r1),
                "col_range": (c0, c1),
                "valid_fraction": valid_fraction,
                "is_valid_chunk": valid_fraction >= min_valid_fraction,
            }

            if valid_fraction >= min_valid_fraction:
                valid_chunks.append(chunk_id)

    return chunk_defs, valid_chunks, (nrows, ncols), (n_chunk_rows, n_chunk_cols)


def initialise_chunk_store(valid_chunks):
    return {chunk_id: [] for chunk_id in valid_chunks}


def append_none_to_all_chunks(category_store):
    for chunk_id in category_store:
        category_store[chunk_id].append(None)


def append_array_to_chunks(array_data, category_store, chunk_defs):
    for chunk_id in category_store:
        r0, r1 = chunk_defs[chunk_id]["row_range"]
        c0, c1 = chunk_defs[chunk_id]["col_range"]
        tile = array_data[r0:r1, c0:c1]
        category_store[chunk_id].append(tile)


def save_chunked_category_store(base_dir, kind, category_store, category, chunk_defs):
    out_dir = os.path.join(base_dir, kind, f"{category}d")
    os.makedirs(out_dir, exist_ok=True)

    for chunk_id, data_list in category_store.items():
        cr, cc = chunk_id
        chunk_file = os.path.join(out_dir, f"chunk_r{cr:03d}_c{cc:03d}.pkl")

        payload = {
            "data": data_list,
            "category": category,
            "chunk_id": chunk_id,
            "row_range": chunk_defs[chunk_id]["row_range"],
            "col_range": chunk_defs[chunk_id]["col_range"],
            "valid_fraction": chunk_defs[chunk_id]["valid_fraction"],
        }

        with open(chunk_file, "wb") as f:
            pickle.dump(payload, f)


def save_chunk_index(base_dir, metadata):
    chunk_index_file = os.path.join(base_dir, "chunk_index.pkl")
    with open(chunk_index_file, "wb") as f:
        pickle.dump(metadata, f)


#######################################################################
# Core reader
#######################################################################

def read_data_chunked(
    start,
    end,
    min_baseline,
    max_baseline,
    landmask,
    nlook,
    interval,
    LiCSAR_data,
    filtered_ifgs,
    read_mode,   # "ifgs" or "coh"
    array_landmask,
    chunk_defs,
    valid_chunks,
):
    start_date = datetime.strptime(start, "%Y%m%d")
    end_date = datetime.strptime(end, "%Y%m%d")

    root_directory = get_root_directory(root_path, frame, LiCSAR_data)

    if read_mode == "ifgs":
        req_file_name = "geo.diff_pha.tif" if filtered_ifgs == "yes" else "geo.diff_unfiltered_pha.tif"
        read_label = "wrapped interferograms"
    elif read_mode == "coh":
        req_file_name = "geo.cc.tif"
        read_label = "coherence data"
    else:
        raise ValueError("read_mode must be 'ifgs' or 'coh'")

    print(f"Reading {read_label} from the path: {root_directory}")

    total_files = sum(
        1
        for dirpath, _, filenames in os.walk(root_directory)
        for filename in filenames
        if filename.endswith(req_file_name)
    )
    processed_files = 0

    category_arrays = {}
    last_date = {}
    n = 1
    first_acq = None
    last_acq = None

    for dirpath, dirnames, filenames in os.walk(root_directory):
        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            if not filename.endswith(req_file_name):
                continue

            file_path = os.path.join(dirpath, filename)

            processed_files += 1
            percentage = (processed_files / total_files) * 100 if total_files > 0 else 100

            sys.stdout.write(
                f"\rReading {read_label}: [{'=' * int(percentage // 2):<50}] {int(percentage)}%"
            )
            sys.stdout.flush()

            match = re.search(r"(\d+)_(\d+)", filename)
            if not match:
                continue

            date1 = match.group(1)
            date2 = match.group(2)

            date1_obj = datetime.strptime(date1, "%Y%m%d")
            date2_obj = datetime.strptime(date2, "%Y%m%d")
            category = (date2_obj - date1_obj).days

            if not (
                start_date <= date1_obj <= end_date
                and start_date <= date2_obj <= end_date
                and category < max_baseline
                and category > min_baseline
            ):
                continue

            if n == 1:
                first_acq = date1_obj
            else:
                last_acq = date2_obj
            n += 1

            ds = gdal.Open(file_path)

            if ds is None:
                print(f"\n⚠️ Could not open {file_path}. Appending None.")
                array_data = None
            else:
                try:
                    array_data = ds.ReadAsArray().astype(np.float32)
                except Exception as e:
                    print(f"\n⚠️ Error reading {file_path}: {e}. Appending None instead.")
                    array_data = None
                finally:
                    del ds

            if category not in category_arrays:
                category_arrays[category] = initialise_chunk_store(valid_chunks)

            if category not in last_date:
                last_date[category] = date1_obj

            diff_ifgs = (date1_obj - last_date[category]).days
            if diff_ifgs > interval:
                n_no_acq = int(diff_ifgs / interval - 1)
                for _ in range(n_no_acq):
                    append_none_to_all_chunks(category_arrays[category])

            last_date[category] = date1_obj

            if array_data is None:
                append_none_to_all_chunks(category_arrays[category])
                continue

            if read_mode == "ifgs":
                array_data[array_data == 0] = np.nan

            if landmask is not None:
                array_data[array_landmask != 1] = np.nan

            if nlook is not None and nlook != 1:
                if read_mode == "ifgs":
                    array_data = multilook_w(array_data, nlook)
                else:
                    array_data = multilook(array_data, nlook)

            append_array_to_chunks(array_data, category_arrays[category], chunk_defs)

    if first_acq is None or last_acq is None:
        print(f"\nNo valid {read_label} found in requested date/baseline range.")
        return {}, {}, None, None

    for category in sorted(category_arrays):
        while (last_date[category] + timedelta(category)) < last_acq:
            append_none_to_all_chunks(category_arrays[category])
            last_date[category] = last_date[category] + timedelta(interval)

    max_ifg_number = abs(int((first_acq - last_acq).days / interval))

    cat = sorted(category_arrays.keys())
    for i in range(interval, max(cat) + 1, interval):
        if i in cat:
            target_len = int(max_ifg_number + 1 - i / interval)
            for chunk_id in category_arrays[i]:
                while len(category_arrays[i][chunk_id]) < target_len:
                    category_arrays[i][chunk_id].insert(0, None)

    print(f"\nFinished reading all {read_label}.")
    return category_arrays, last_date, first_acq, last_acq


#######################################################################
# Main
#######################################################################

def main():
    directory_path = os.path.join(output_path, "Data")
    os.makedirs(directory_path, exist_ok=True)

    # Clear existing Data directory contents
    if os.path.exists(directory_path):
        files = glob.glob(os.path.join(directory_path, "*"))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)

    print("\nPreparing landmask and chunk layout...")
    array_landmask = get_landmask_array(root_path, frame, LiCSAR_data, nlook)
    chunk_defs, valid_chunks, frame_shape, chunk_grid_shape = build_chunk_layout(
        array_landmask, chunk_rows, chunk_cols, min_valid_fraction
    )

    print(f"Frame shape after multilooking: {frame_shape}")
    print(f"Chunk grid shape: {chunk_grid_shape}")
    print(f"Number of valid chunks: {len(valid_chunks)}")

    # IFGs
    all_ifgs_chunked, _, first_acq_ifg, last_acq_ifg = read_data_chunked(
        start=start,
        end=end,
        min_baseline=min_baseline,
        max_baseline=max_baseline,
        landmask=landmask,
        nlook=nlook,
        interval=interval,
        LiCSAR_data=LiCSAR_data,
        filtered_ifgs=filtered_ifgs,
        read_mode="ifgs",
        array_landmask=array_landmask,
        chunk_defs=chunk_defs,
        valid_chunks=valid_chunks,
    )

    for category in sorted(all_ifgs_chunked):
        save_chunked_category_store(directory_path, "ifgs", all_ifgs_chunked[category], category, chunk_defs)

    print("\nReport on the number of IFGs and missing IFGs per category:")
    for category in sorted(all_ifgs_chunked):
        # report using first valid chunk as representative, since None alignment is same for all chunks
        first_chunk = next(iter(all_ifgs_chunked[category]))
        seq = all_ifgs_chunked[category][first_chunk]
        num_ifgs = sum(1 for item in seq if item is not None)
        num_missing_ifgs = sum(1 for item in seq if item is None)
        print(
            f"{category}-days: Number of IFGs: {num_ifgs}   Number of missing IFGs: {num_missing_ifgs}"
        )

    del all_ifgs_chunked

    # COH
    all_coh_chunked, _, first_acq_coh, last_acq_coh = read_data_chunked(
        start=start,
        end=end,
        min_baseline=min_baseline,
        max_baseline=max_baseline,
        landmask=landmask,
        nlook=nlook,
        interval=interval,
        LiCSAR_data=LiCSAR_data,
        filtered_ifgs=filtered_ifgs,
        read_mode="coh",
        array_landmask=array_landmask,
        chunk_defs=chunk_defs,
        valid_chunks=valid_chunks,
    )

    for category in sorted(all_coh_chunked):
        save_chunked_category_store(directory_path, "coh", all_coh_chunked[category], category, chunk_defs)

    del all_coh_chunked

    # metadata
    chunk_index = {
        "frame": frame,
        "start": start,
        "end": end,
        "interval": interval,
        "nlook": nlook,
        "LiCSAR_data": LiCSAR_data,
        "filtered_ifgs": filtered_ifgs,
        "min_baseline": min_baseline,
        "max_baseline": max_baseline,
        "chunk_rows": chunk_rows,
        "chunk_cols": chunk_cols,
        "min_valid_fraction": min_valid_fraction,
        "frame_shape": frame_shape,
        "chunk_grid_shape": chunk_grid_shape,
        "valid_chunks": valid_chunks,
        "chunk_defs": chunk_defs,
        "categories_ifgs": sorted(list(getattr(locals().get("all_ifgs_chunked", {}), "keys", lambda: [])())) if False else None,
        "categories_coh": sorted(list(getattr(locals().get("all_coh_chunked", {}), "keys", lambda: [])())) if False else None,
        "landmask_shape": array_landmask.shape,
        "first_acq_ifg": first_acq_ifg.strftime("%Y%m%d") if first_acq_ifg is not None else None,
        "last_acq_ifg": last_acq_ifg.strftime("%Y%m%d") if last_acq_ifg is not None else None,
        "first_acq_coh": first_acq_coh.strftime("%Y%m%d") if first_acq_coh is not None else None,
        "last_acq_coh": last_acq_coh.strftime("%Y%m%d") if last_acq_coh is not None else None,
    }

    # re-scan saved category folders to record actual categories
    ifgs_dir = os.path.join(directory_path, "ifgs")
    coh_dir = os.path.join(directory_path, "coh")

    categories_ifgs = []
    categories_coh = []

    if os.path.exists(ifgs_dir):
        for name in sorted(os.listdir(ifgs_dir)):
            if name.endswith("d"):
                try:
                    categories_ifgs.append(int(name[:-1]))
                except ValueError:
                    pass

    if os.path.exists(coh_dir):
        for name in sorted(os.listdir(coh_dir)):
            if name.endswith("d"):
                try:
                    categories_coh.append(int(name[:-1]))
                except ValueError:
                    pass

    chunk_index["categories_ifgs"] = categories_ifgs
    chunk_index["categories_coh"] = categories_coh

    save_chunk_index(directory_path, chunk_index)

    print("\nChunked data writing completed.")
    print(f"Chunk index written to: {os.path.join(directory_path, 'chunk_index.pkl')}")
    print(f"Valid chunks kept: {len(valid_chunks)}")
    print(f"IFG categories written: {categories_ifgs}")
    print(f"Coherence categories written: {categories_coh}")


if __name__ == "__main__":
    main()

