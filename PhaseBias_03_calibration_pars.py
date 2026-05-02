#!/usr/bin/env python3

import os
import pickle
import configparser
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

# ============================================================
# 1️⃣ Load configuration
# ============================================================

config_file = "config.txt"
if not os.path.exists(config_file):
    raise FileNotFoundError("config.txt not found.")

config = configparser.ConfigParser()
config.read(config_file)

output_path = config.get("DEFAULT", "output_path")
interval = config.getint("DEFAULT", "interval")
num_a = config.getint("DEFAULT", "num_a")
selected_baseline = config.getint("DEFAULT", "selected_baseline")

# fixed as requested
coh_thresh = 20

# noisy-pixel removal parameters
min_count_for_cleaning = config.getint("DEFAULT", "min_count_for_cleaning", fallback=1)
local_window_size = config.getint("DEFAULT", "local_window_size", fallback=7)
local_thresh = config.getfloat("DEFAULT", "local_thresh", fallback=3.0)
min_mad = config.getfloat("DEFAULT", "min_mad", fallback=0.15)

# optional plotting
plot_chunk_results = config.getboolean("DEFAULT", "plot_an_chunks", fallback=False)
plot_first_n_chunks = config.getint("DEFAULT", "plot_first_n_chunks", fallback=3)
plot_global_chunk_maps = config.getboolean("DEFAULT", "plot_global_chunk_maps", fallback=True)
save_global_chunk_maps = config.getboolean("DEFAULT", "save_global_chunk_maps", fallback=True)

# robust regression parameters
robust_sigma = config.getfloat("DEFAULT", "robust_sigma", fallback=3.0)
robust_min_mad = config.getfloat("DEFAULT", "robust_min_mad", fallback=0.05)

# ============================================================
# 2️⃣ Paths and loading helpers
# ============================================================

data_dir = os.path.join(output_path, "Data")

chunk_index_file = os.path.join(data_dir, "chunk_index.pkl")
if not os.path.exists(chunk_index_file):
    raise FileNotFoundError(f"Missing chunk index: {chunk_index_file}")

with open(chunk_index_file, "rb") as f:
    chunk_index = pickle.load(f)

valid_chunks = chunk_index["valid_chunks"]
available_coh_categories = sorted(chunk_index["categories_coh"])
frame_shape = tuple(chunk_index["frame_shape"])
chunk_defs = chunk_index["chunk_defs"]

if selected_baseline not in available_coh_categories:
    raise ValueError(
        f"selected_baseline={selected_baseline} not found in available categories: {available_coh_categories}"
    )


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def chunk_file_path(kind, category_or_name, chunk_id, short_baseline=None):
    cr, cc = chunk_id

    if kind == "coh":
        return os.path.join(
            data_dir, "coh", f"{category_or_name}d", f"chunk_r{cr:03d}_c{cc:03d}.pkl"
        )
    elif kind == "loops":
        return os.path.join(
            data_dir,
            "loops",
            f"{category_or_name}_{short_baseline}",
            f"chunk_r{cr:03d}_c{cc:03d}.pkl",
        )
    else:
        raise ValueError("Unknown kind")


def load_coh_chunk(category, chunk_id):
    path = chunk_file_path("coh", category, chunk_id)
    if not os.path.exists(path):
        return None
    return load_pickle(path)


def load_loop_chunk(long_baseline, short_baseline, chunk_id):
    path = chunk_file_path("loops", long_baseline, chunk_id, short_baseline=short_baseline)
    if not os.path.exists(path):
        return None
    return load_pickle(path)


# ============================================================
# 3️⃣ Utility: determine which a_n are estimable
# ============================================================

def get_estimable_n_values(selected_baseline, interval, available_loop_products, available_coh_categories, max_n=None):
    estimable_n = []

    max_possible_n = selected_baseline // interval
    if max_n is not None:
        max_possible_n = min(max_possible_n, max_n)

    for n in range(2, max_possible_n + 1):
        short_len = n * interval

        if short_len > selected_baseline:
            break
        if selected_baseline % short_len != 0:
            break
        if (selected_baseline, short_len) not in available_loop_products:
            break
        if short_len not in available_coh_categories:
            break

        estimable_n.append(n)

    return estimable_n


# ============================================================
# 4️⃣ Loop masking using coherence
# ============================================================

def coherence_mask_loops_for_an(loop_arr, long_coh, short_coh, n_short, coh_thresh):
    out = []

    for i, arr in enumerate(loop_arr):
        if not isinstance(arr, np.ndarray):
            out.append(None)
            continue

        if i >= len(long_coh) or not isinstance(long_coh[i], np.ndarray):
            out.append(None)
            continue

        valid = True
        contributing = []

        for j in range(n_short):
            idx = i + j
            if idx >= len(short_coh) or not isinstance(short_coh[idx], np.ndarray):
                valid = False
                break
            contributing.append(short_coh[idx])

        if not valid:
            out.append(None)
            continue

        mask = long_coh[i] < coh_thresh
        for c in contributing:
            mask |= c < coh_thresh

        out.append(np.where(mask, np.nan, arr))

    return out


# ============================================================
# 5️⃣ Scalar/global estimation
# ============================================================

def calc_an(loop_n_list, loop_base_list):
    sum_xy = 0.0
    sum_x2 = 0.0
    ratio_means = []

    for x_arr, y_arr in zip(loop_base_list, loop_n_list):
        if not isinstance(x_arr, np.ndarray) or not isinstance(y_arr, np.ndarray):
            continue

        x = x_arr.astype(float)
        y = y_arr.astype(float)

        valid = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0)

        if not np.any(valid):
            continue

        ratio = y[valid] / x[valid]
        ratio_means.append(np.nanmean(ratio))

        x_valid = x[valid]
        y_valid = y[valid]

        sum_xy += np.sum(x_valid * y_valid)
        sum_x2 += np.sum(x_valid * x_valid)

    regression_estimate = sum_xy / sum_x2 if sum_x2 > 0 else np.nan
    ratio_mean = np.nanmean(ratio_means) if len(ratio_means) > 0 else np.nan
    return ratio_mean, regression_estimate


def robust_regression_two_pass(x_valid, y_valid, sigma=3.0, min_mad_local=0.05):
    if x_valid.size == 0:
        return np.nan, 0, 0

    sum_x2 = np.sum(x_valid * x_valid)
    if sum_x2 <= 0:
        return np.nan, x_valid.size, 0

    a0 = np.sum(x_valid * y_valid) / sum_x2
    residuals = y_valid - a0 * x_valid

    med_res = np.median(residuals)
    mad_res = np.median(np.abs(residuals - med_res))
    thresh = sigma * max(mad_res, min_mad_local)

    keep = np.abs(residuals - med_res) <= thresh

    x2 = x_valid[keep]
    y2 = y_valid[keep]

    if x2.size == 0:
        return np.nan, x_valid.size, 0

    sum_x2_2 = np.sum(x2 * x2)
    if sum_x2_2 <= 0:
        return np.nan, x_valid.size, x2.size

    a1 = np.sum(x2 * y2) / sum_x2_2
    return a1, x_valid.size, x2.size


def calc_an_with_bad_pixel_mask(
    loop_n_list,
    loop_base_list,
    bad_pixel_mask,
    min_total_valid=0,
    sigma=3.0,
    min_mad_local=0.05,
):
    ratio_means = []
    x_all = []
    y_all = []
    total_valid_count = 0

    for x_arr, y_arr in zip(loop_base_list, loop_n_list):
        if not isinstance(x_arr, np.ndarray) or not isinstance(y_arr, np.ndarray):
            continue

        x = x_arr.astype(float)
        y = y_arr.astype(float)

        valid = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0) & (~bad_pixel_mask)

        n_valid = np.count_nonzero(valid)
        total_valid_count += n_valid

        if n_valid == 0:
            continue

        ratio = y[valid] / x[valid]
        ratio_means.append(np.nanmean(ratio))

        x_all.append(x[valid])
        y_all.append(y[valid])

    if total_valid_count < min_total_valid or len(x_all) == 0:
        return np.nan, np.nan, total_valid_count, 0

    x_valid = np.concatenate(x_all)
    y_valid = np.concatenate(y_all)

    reg_clean, _, n_after_robust = robust_regression_two_pass(
        x_valid, y_valid, sigma=sigma, min_mad_local=min_mad_local
    )

    ratio_mean = np.nanmean(ratio_means) if len(ratio_means) > 0 else np.nan
    return ratio_mean, reg_clean, total_valid_count, n_after_robust


# ============================================================
# 6️⃣ Pixel-wise regression
# ============================================================

def pixelwise_regression(loop_base_masked, loop_n_masked, min_abs_x=0.02):
    sum_xy = None
    sum_x2 = None
    count_valid = None

    for x_arr, y_arr in zip(loop_base_masked, loop_n_masked):
        if not isinstance(x_arr, np.ndarray) or not isinstance(y_arr, np.ndarray):
            continue

        x = x_arr.astype(float)
        y = y_arr.astype(float)

        valid = (~np.isnan(x)) & (~np.isnan(y)) & (np.abs(x) > min_abs_x)

        if sum_xy is None:
            sum_xy = np.zeros_like(x, dtype=float)
            sum_x2 = np.zeros_like(x, dtype=float)
            count_valid = np.zeros_like(x, dtype=np.int32)

        sum_xy[valid] += x[valid] * y[valid]
        sum_x2[valid] += x[valid] * x[valid]
        count_valid[valid] += 1

    if sum_xy is None:
        raise ValueError("No valid data found for pixelwise regression.")

    a_map = np.full_like(sum_xy, np.nan, dtype=float)
    valid_final = sum_x2 > 0
    a_map[valid_final] = sum_xy[valid_final] / sum_x2[valid_final]

    return a_map, count_valid


# ============================================================
# 7️⃣ Identify noisy pixels in raw pixel-wise a maps
# ============================================================

def make_noisy_pixel_mask(a_map, count_map, min_count=2, win=7, local_thresh=3.0, min_mad=0.15):
    a_work = a_map.copy()
    bad_mask = np.zeros(a_map.shape, dtype=bool)

    # remove values outside [0, 1] before cleaning
    out_of_range_mask = np.isfinite(a_work) & ((a_work < 0) | (a_work > 1))
    bad_mask |= out_of_range_mask

    # weakly supported pixels
    bad_mask |= (count_map < min_count)

    a_work[bad_mask] = np.nan

    def nanmedian_filter(values):
        values = values[~np.isnan(values)]
        return np.nanmedian(values) if len(values) > 0 else np.nan

    local_median = generic_filter(
        a_work, nanmedian_filter, size=win, mode="constant", cval=np.nan
    )

    def local_mad(values):
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return np.nan
        med = np.nanmedian(values)
        return np.nanmedian(np.abs(values - med))

    local_mad_map = generic_filter(
        a_work, local_mad, size=win, mode="constant", cval=np.nan
    )

    diff = np.abs(a_work - local_median)
    thresh = local_thresh * np.maximum(local_mad_map, min_mad)

    outlier_mask = np.isfinite(a_work) & np.isfinite(local_median) & (diff > thresh)
    bad_mask |= outlier_mask

    a_map_clean = a_work.copy()
    a_map_clean[bad_mask] = np.nan

    return bad_mask, a_map_clean


# ============================================================
# 8️⃣ Global chunk-map helpers
# ============================================================

def build_chunk_scalar_map(frame_shape, chunk_defs, scalar_dict):
    out = np.full(frame_shape, np.nan, dtype=float)

    for chunk_id, value in scalar_dict.items():
        if chunk_id not in chunk_defs:
            continue
        r0, r1 = chunk_defs[chunk_id]["row_range"]
        c0, c1 = chunk_defs[chunk_id]["col_range"]
        out[r0:r1, c0:c1] = value

    return out


def plot_scalar_chunk_map(arr, title, save_path=None):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(arr, cmap="RdBu", vmin=-2, vmax=2)
    plt.colorbar(im)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")


def get_chunk_centres(chunk_defs):
    centres = {}
    for chunk_id, meta in chunk_defs.items():
        r0, r1 = meta["row_range"]
        c0, c1 = meta["col_range"]
        centres[chunk_id] = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
    return centres


def fill_unreliable_chunks_with_neighbours(original_global_dict, cleaned_global_dict, chunk_defs):
    """
    Fill only chunks where:
      - original global value is finite
      - cleaned global value is NaN
    Do NOT fill chunks that were already NaN in the original global map.
    """
    filled = cleaned_global_dict.copy()
    centres = get_chunk_centres(chunk_defs)

    # chunks eligible for replacement: finite in original, NaN in cleaned
    target_chunk_ids = [
        cid for cid in cleaned_global_dict
        if np.isfinite(original_global_dict.get(cid, np.nan)) and np.isnan(cleaned_global_dict.get(cid, np.nan))
    ]

    # valid neighbours must also come from chunks finite in cleaned map
    valid_ids = [cid for cid, val in cleaned_global_dict.items() if np.isfinite(val)]

    for cid in target_chunk_ids:
        if cid not in centres:
            continue

        rc, cc = centres[cid]
        neigh = []

        for vid in valid_ids:
            if vid == cid or vid not in centres:
                continue

            rv, cv = centres[vid]
            dist = np.hypot(rc - rv, cc - cv)
            if dist == 0:
                continue

            neigh.append((dist, cleaned_global_dict[vid]))

        if len(neigh) == 0:
            filled[cid] = np.nan
            continue

        neigh.sort(key=lambda x: x[0])
        neigh = neigh[:8]

        dists = np.array([x[0] for x in neigh], dtype=float)
        vals = np.array([x[1] for x in neigh], dtype=float)
        weights = 1.0 / dists

        filled[cid] = np.sum(weights * vals) / np.sum(weights)

    return filled


# ============================================================
# 9️⃣ Available loop products
# ============================================================

loop_root = os.path.join(data_dir, "loops")
available_loop_products = set()

if os.path.exists(loop_root):
    for name in os.listdir(loop_root):
        if "_" in name:
            try:
                long_baseline, short_baseline = map(int, name.split("_"))
                available_loop_products.add((long_baseline, short_baseline))
            except ValueError:
                pass

valid_n_values = get_estimable_n_values(
    selected_baseline,
    interval,
    available_loop_products,
    available_coh_categories,
    max_n=1 + num_a,
)

if not valid_n_values:
    raise ValueError(
        f"No estimable a_n values for selected_baseline={selected_baseline} "
        f"with available loop products {sorted(available_loop_products)}"
    )

print(f"Selected baseline: {selected_baseline}")
print(f"Estimable n values: {valid_n_values}")
print(f"Using coh_thresh={coh_thresh}")

# ============================================================
# 🔟 Output dirs
# ============================================================

an_root = os.path.join(data_dir, "an_chunks")
global_dir = os.path.join(an_root, "global")
global_clean_dir = os.path.join(an_root, "global_clean")
pixel_dir = os.path.join(an_root, "pixel_maps")
global_maps_dir = os.path.join(an_root, "global_maps")

os.makedirs(global_dir, exist_ok=True)
os.makedirs(global_clean_dir, exist_ok=True)
os.makedirs(pixel_dir, exist_ok=True)
os.makedirs(global_maps_dir, exist_ok=True)

global_values_by_param = {f"a{n-1}": {} for n in valid_n_values}
global_clean_values_by_param = {f"a{n-1}": {} for n in valid_n_values}
global_clean_values_by_param_filled = {f"a{n-1}": {} for n in valid_n_values}

# ============================================================
# 1️⃣1️⃣ Chunkwise processing
# ============================================================

processed_chunk_summaries = []
plotted = 0

for chunk_idx, chunk_id in enumerate(valid_chunks, start=1):
    cr, cc = chunk_id
    print(f"\nProcessing chunk {chunk_idx}/{len(valid_chunks)}: r{cr:03d}, c{cc:03d}")

    coh_long_payload = load_coh_chunk(selected_baseline, chunk_id)
    coh_base_payload = load_coh_chunk(interval, chunk_id)

    if coh_long_payload is None or coh_base_payload is None:
        print("  Skipping chunk: missing required coherence chunk.")
        continue

    coh_long = coh_long_payload["data"]
    coh_short_base = coh_base_payload["data"]

    chunk_meta = {
        "chunk_id": chunk_id,
        "row_range": coh_long_payload["row_range"],
        "col_range": coh_long_payload["col_range"],
        "valid_fraction": coh_long_payload["valid_fraction"],
    }

    chunk_rows = chunk_meta["row_range"][1] - chunk_meta["row_range"][0]
    chunk_cols = chunk_meta["col_range"][1] - chunk_meta["col_range"][0]
    chunk_npix = chunk_rows * chunk_cols
    min_total_valid_pixels = max(1, int(np.ceil(0.03 * chunk_npix)))

    loop_base_payload = load_loop_chunk(selected_baseline, interval, chunk_id)
    if loop_base_payload is None:
        print("  Skipping chunk: missing required base loop chunk.")
        continue

    loop_base_all = loop_base_payload["data"]
    n_short_base = selected_baseline // interval

    loop_base = coherence_mask_loops_for_an(
        loop_base_all, coh_long, coh_short_base, n_short_base, coh_thresh
    )

    loop_base = [
        arr.astype(float) if isinstance(arr, np.ndarray) else None
        for arr in loop_base
    ]

    chunk_global_values = {}
    chunk_global_clean_values = {}
    chunk_pixel_dict = {}

    for n in valid_n_values:
        short_len_n = n * interval
        param_name = f"a{n-1}"

        print(f"  Computing {param_name} in chunk using short baseline {short_len_n}")

        loop_n_payload = load_loop_chunk(selected_baseline, short_len_n, chunk_id)
        coh_short_n_payload = load_coh_chunk(short_len_n, chunk_id)

        if loop_n_payload is None or coh_short_n_payload is None:
            print(f"    Missing loop/coh chunk for {param_name}; skipping.")
            continue

        loop_n_all = loop_n_payload["data"]
        coh_short_n = coh_short_n_payload["data"]

        n_short_n = selected_baseline // short_len_n
        loop_n_masked = coherence_mask_loops_for_an(
            loop_n_all, coh_long, coh_short_n, n_short_n, coh_thresh
        )

        loop_n_masked = [
            arr.astype(float) if isinstance(arr, np.ndarray) else None
            for arr in loop_n_masked
        ]

        ratio_val, reg_val = calc_an(loop_n_masked, loop_base)
        chunk_global_values[param_name] = {
            "ratio": ratio_val,
            "regression": reg_val,
        }
        global_values_by_param[param_name][chunk_id] = reg_val

        print(
            f"    original global regression={reg_val:.6f}"
            if not np.isnan(reg_val) else "    original global regression=nan"
        )

        try:
            a_map, a_count = pixelwise_regression(loop_base, loop_n_masked)
        except ValueError:
            print(f"    No valid data found for pixelwise regression of {param_name}")
            continue

        bad_mask, a_map_clean = make_noisy_pixel_mask(
            a_map,
            a_count,
            min_count=min_count_for_cleaning,
            win=local_window_size,
            local_thresh=local_thresh,
            min_mad=min_mad,
        )

        ratio_clean, reg_clean, total_valid_count, n_after_robust = calc_an_with_bad_pixel_mask(
            loop_n_masked,
            loop_base,
            bad_mask,
            min_total_valid=min_total_valid_pixels,
            sigma=robust_sigma,
            min_mad_local=robust_min_mad,
        )

        unreliable = np.isnan(reg_clean)

        chunk_global_clean_values[param_name] = {
            "ratio": ratio_clean,
            "regression": reg_clean,
            "total_valid_count": total_valid_count,
            "min_required_valid_count": min_total_valid_pixels,
            "n_after_robust": n_after_robust,
            "unreliable": unreliable,
        }
        global_clean_values_by_param[param_name][chunk_id] = reg_clean

        if not np.isnan(reg_clean):
            print(
                f"    cleaned global regression={reg_clean:.6f} "
                f"(valid={total_valid_count}, robust_keep={n_after_robust})"
            )
        else:
            print(
                f"    cleaned global regression=nan "
                f"(valid={total_valid_count}, min_required={min_total_valid_pixels})"
            )

        if np.isnan(reg_clean):
            a_map_filled = a_map_clean.copy()
        else:
            a_map_filled = np.where(np.isnan(a_map_clean), reg_clean, a_map_clean)

        chunk_pixel_dict[param_name] = a_map_filled
        chunk_pixel_dict[f"{param_name}_raw"] = a_map
        chunk_pixel_dict[f"{param_name}_clean"] = a_map_clean
        chunk_pixel_dict[f"{param_name}_count"] = a_count
        chunk_pixel_dict[f"{param_name}_bad_mask"] = bad_mask
        chunk_pixel_dict[f"{param_name}_global"] = reg_val
        chunk_pixel_dict[f"{param_name}_global_clean"] = reg_clean

        if plot_chunk_results and plotted < plot_first_n_chunks:
            def plot_map(arr, title):
                plt.figure(figsize=(5, 4))
                im = plt.imshow(arr, cmap="RdBu", vmin=-2, vmax=2)
                plt.colorbar(im)
                plt.title(title)
                plt.axis("off")
                plt.tight_layout()

            plot_map(a_map, f"{param_name} raw\nchunk r{cr:03d} c{cc:03d}")
            plot_map(bad_mask.astype(float), f"{param_name} bad mask\nchunk r{cr:03d} c{cc:03d}")
            plot_map(a_map_clean, f"{param_name} clean\nchunk r{cr:03d} c{cc:03d}")
            plot_map(a_map_filled, f"{param_name} filled\nchunk r{cr:03d} c{cc:03d}")

    if plot_chunk_results and plotted < plot_first_n_chunks:
        plt.show()
        plotted += 1

    save_pickle(
        os.path.join(global_dir, f"chunk_r{cr:03d}_c{cc:03d}.pkl"),
        {
            **chunk_meta,
            "selected_baseline": selected_baseline,
            "values": chunk_global_values,
        },
    )

    save_pickle(
        os.path.join(global_clean_dir, f"chunk_r{cr:03d}_c{cc:03d}.pkl"),
        {
            **chunk_meta,
            "selected_baseline": selected_baseline,
            "values": chunk_global_clean_values,
        },
    )

    save_pickle(
        os.path.join(pixel_dir, f"chunk_r{cr:03d}_c{cc:03d}.pkl"),
        {
            **chunk_meta,
            "selected_baseline": selected_baseline,
            "maps": chunk_pixel_dict,
        },
    )

    processed_chunk_summaries.append(
        {
            **chunk_meta,
            "selected_baseline": selected_baseline,
            "global_keys": sorted(chunk_global_values.keys()),
            "global_clean_keys": sorted(chunk_global_clean_values.keys()),
            "pixel_keys": sorted([
                k for k in chunk_pixel_dict.keys()
                if not k.endswith(("_raw", "_clean", "_count", "_bad_mask", "_global", "_global_clean"))
            ]),
        }
    )

# ============================================================
# 1️⃣2️⃣ Fill unreliable chunk-global values from neighbours
# ============================================================

unreliable_chunk_counts = {}
replaced_chunk_ids_by_param = {}

for n in valid_n_values:
    param_name = f"a{n-1}"

    original_global_vals = global_values_by_param[param_name]
    cleaned_vals = global_clean_values_by_param[param_name]

    filled_vals = fill_unreliable_chunks_with_neighbours(
        original_global_vals,
        cleaned_vals,
        chunk_defs,
    )
    global_clean_values_by_param_filled[param_name] = filled_vals

    replaced_ids = []
    for chunk_id in cleaned_vals:
        orig_global_val = original_global_vals.get(chunk_id, np.nan)
        clean_val = cleaned_vals.get(chunk_id, np.nan)
        fill_val = filled_vals.get(chunk_id, np.nan)

        if np.isfinite(orig_global_val) and np.isnan(clean_val) and np.isfinite(fill_val):
            replaced_ids.append(chunk_id)

    unreliable_chunk_counts[param_name] = len(replaced_ids)
    replaced_chunk_ids_by_param[param_name] = replaced_ids

for chunk_id in valid_chunks:
    cr, cc = chunk_id
    clean_file = os.path.join(global_clean_dir, f"chunk_r{cr:03d}_c{cc:03d}.pkl")
    if not os.path.exists(clean_file):
        continue

    payload = load_pickle(clean_file)
    vals = payload.get("values", {})

    for n in valid_n_values:
        param_name = f"a{n-1}"
        if param_name not in vals:
            continue

        orig_global_val = global_values_by_param[param_name].get(chunk_id, np.nan)
        clean_val = vals[param_name].get("regression", np.nan)
        filled_val = global_clean_values_by_param_filled[param_name].get(chunk_id, np.nan)

        used_fill = bool(
            np.isfinite(orig_global_val) and np.isnan(clean_val) and np.isfinite(filled_val)
        )

        vals[param_name]["regression_filled_from_neighbours"] = filled_val
        vals[param_name]["used_neighbour_fill"] = used_fill

    save_pickle(clean_file, payload)

# ============================================================
# 1️⃣3️⃣ Build and plot/save global chunk maps
# ============================================================

global_map_files = {}
global_clean_map_files = {}
global_clean_filled_map_files = {}

for n in valid_n_values:
    param_name = f"a{n-1}"

    global_map = build_chunk_scalar_map(
        frame_shape=frame_shape,
        chunk_defs=chunk_defs,
        scalar_dict=global_values_by_param[param_name],
    )

    global_clean_map = build_chunk_scalar_map(
        frame_shape=frame_shape,
        chunk_defs=chunk_defs,
        scalar_dict=global_clean_values_by_param[param_name],
    )

    global_clean_filled_map = build_chunk_scalar_map(
        frame_shape=frame_shape,
        chunk_defs=chunk_defs,
        scalar_dict=global_clean_values_by_param_filled[param_name],
    )

    if save_global_chunk_maps:
        global_map_path = os.path.join(global_maps_dir, f"{param_name}_global_chunk_map.png")
        global_clean_map_path = os.path.join(global_maps_dir, f"{param_name}_global_clean_chunk_map.png")
        global_clean_filled_map_path = os.path.join(global_maps_dir, f"{param_name}_global_clean_filled_chunk_map.png")

        plot_scalar_chunk_map(
            global_map,
            f"{param_name} original global values over all chunks",
            save_path=global_map_path,
        )
        plt.close()

        plot_scalar_chunk_map(
            global_clean_map,
            f"{param_name} cleaned global values over all chunks",
            save_path=global_clean_map_path,
        )
        plt.close()

        plot_scalar_chunk_map(
            global_clean_filled_map,
            f"{param_name} cleaned global values filled from neighbours",
            save_path=global_clean_filled_map_path,
        )
        plt.close()

        global_map_files[param_name] = global_map_path
        global_clean_map_files[param_name] = global_clean_map_path
        global_clean_filled_map_files[param_name] = global_clean_filled_map_path

    if plot_global_chunk_maps:
        plot_scalar_chunk_map(
            global_map,
            f"{param_name} original global values over all chunks",
            save_path=None,
        )
        plot_scalar_chunk_map(
            global_clean_map,
            f"{param_name} cleaned global values over all chunks",
            save_path=None,
        )
        plot_scalar_chunk_map(
            global_clean_filled_map,
            f"{param_name} cleaned global values filled from neighbours",
            save_path=None,
        )

if plot_global_chunk_maps:
    plt.show()

# ============================================================
# 1️⃣4️⃣ Save unreliable chunk report
# ============================================================

report_file = os.path.join(an_root, "unreliable_chunk_report.txt")
with open(report_file, "w") as f:
    f.write("Report on unreliable chunks replaced by neighbour values\n")
    for n in valid_n_values:
        param_name = f"a{n-1}"
        count = unreliable_chunk_counts.get(param_name, 0)
        f.write(f"{param_name}: {count} chunks replaced\n")

print("\nReport on unreliable chunks replaced by neighbour values:")
for n in valid_n_values:
    param_name = f"a{n-1}"
    count = unreliable_chunk_counts.get(param_name, 0)
    print(f"  {param_name}: {count} chunks replaced")

# ============================================================
# 1️⃣5️⃣ Save index
# ============================================================

an_chunk_index = {
    "selected_baseline": selected_baseline,
    "interval": interval,
    "num_a": num_a,
    "coh_thresh": coh_thresh,
    "valid_n_values": valid_n_values,
    "min_count_for_cleaning": min_count_for_cleaning,
    "local_window_size": local_window_size,
    "local_thresh": local_thresh,
    "min_mad": min_mad,
    "robust_sigma": robust_sigma,
    "robust_min_mad": robust_min_mad,
    "processed_chunks": processed_chunk_summaries,
    "global_map_files": global_map_files,
    "global_clean_map_files": global_clean_map_files,
    "global_clean_filled_map_files": global_clean_filled_map_files,
    "unreliable_chunk_counts": unreliable_chunk_counts,
    "replaced_chunk_ids_by_param": replaced_chunk_ids_by_param,
}

save_pickle(os.path.join(an_root, "an_chunk_index.pkl"), an_chunk_index)

print("\nChunkwise a_n estimation completed.")
print(f"Saved global chunk values to: {global_dir}")
print(f"Saved cleaned global chunk values to: {global_clean_dir}")
print(f"Saved pixel-wise chunk maps to: {pixel_dir}")
print(f"Saved global chunk plots to: {global_maps_dir}")
print(f"Saved unreliable chunk report to: {report_file}")
print(f"Saved chunk index to: {os.path.join(an_root, 'an_chunk_index.pkl')}")
