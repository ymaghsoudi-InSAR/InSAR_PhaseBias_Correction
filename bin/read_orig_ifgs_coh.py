#!/usr/bin/env python3

import numpy as np

def read_orig_ifgs_coh(ifgs, coh, desired_categories):
    print('ifg stack started.....')
    existing_ifgs_index = []

    # Collect categories in order
    categories = [cat for cat in sorted(ifgs.keys()) if cat in desired_categories]
    print(f"Number of categories to process: {len(categories)}")

    # --- Find first non-None image shape safely ---
    first_shape = None
    for cat in categories:
        print('cat is ', cat)
        for img in ifgs[cat]:
            if img is not None:
                first_shape = img.shape
                break
        if first_shape is not None:
            break

    if first_shape is None:
        raise ValueError("No non-None images found in ifgs for desired categories")

    # Total number of images (including None placeholders)
    n_total = sum(len(ifgs[cat]) for cat in categories)
    print(f"Total images to stack: {n_total}, shape={first_shape}")

    # Preallocate arrays with NaN
    all_ifgs = np.full((n_total, *first_shape), np.nan, dtype=np.float32)
    all_coh  = np.full((n_total, *first_shape), np.nan, dtype=np.float32)

    # Fill arrays
    counter = 0
    for category in categories:
        print(f'Processing category: {category} with {len(ifgs[category])} images')
        for img_ifg, img_coh in zip(ifgs[category], coh[category]):
            if img_ifg is not None:
                all_ifgs[counter] = img_ifg
                existing_ifgs_index.append(counter)
            if img_coh is not None:
                all_coh[counter] = img_coh
            counter += 1

    print(f"Finished stacking. Final shape: {all_ifgs.shape}")
    return all_ifgs, all_coh, existing_ifgs_index

