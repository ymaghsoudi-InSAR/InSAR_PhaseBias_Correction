#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os
import sys
import glob
import configparser
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)

################################################################################

HELP_TEXT = """
PhaseBias_03_calibration_pars.py

This script estimates the calibration parameters a_n using the loop closures calculated in the previous step. 

The calibration parameters a_n are estimated by calculating the ratio of the long-term loop closure for the interferogram of interest to the long-term loop closure for the base interferogram. Assuming negligible bias in long-term interferograms, a_n is calculated as follows:

    a_n = |φ(i,i+l) - Σ(t=i to i+l)φ(t,t+1+n)|_2π / |φ(i,i+l) - Σ(t=i to i+l)φ(t,t+1)|_2π

where:
- φ(i,i+l) is a long-term interferogram connecting epoch i to i+l.
- φ(t,t+1) represents the base interferogram.
- φ(t,t+1+n) is the short interferogram to be corrected using a_n.

### Input:
1. Loop closures from the previous step (e.g., 216-6, 216-12, and 216-18 for 6-day acquisition intervals).
2. The long-term interferogram length (e.g., 216 days) must be specified at the beginning.
3. The number of a_n parameters to estimate, specified using `num_a`. For example:
   - Setting `num_a=2` indicates that a1 and a2 will be estimated (used to correct 12- and 18-day interferograms).
   - For correcting longer interferograms, `num_a` should be set accordingly.

### Output:
1. All possible a_n values are calculated.
2. Visualizations:
   - Example plots of a_n arrays and their corresponding histograms, illustrating spatial variations.
   - Time-series plots showing how a_n values evolve over time.
3. Mean a_n values over time are output for further use.

### File Outputs:
- Results are stored in the output directory defined in the `config.txt` file under `/Data/` as `Mean_an_values.pkl`.

### Additional Notes:
- Users should ensure that `num_a` aligns with the desired temporal baselines for correction.

"""
if "--help" in sys.argv:
    print(HELP_TEXT)
    sys.exit(0)


###############################################################################
# Config file path
config_file = "config.txt"

# Check if config file exists
if not os.path.exists(config_file):
    raise FileNotFoundError(
        f"Error: The configuration file '{config_file}' was not found in the script's directory."
    )

# Initialize the configparser
config = configparser.ConfigParser()


# Read the config.txt file
config.read(config_file)

# Define required parameters
required_parameters = [
    "root_path",
    "frame",
    "start",
    "end",
    "interval",
    "nlook",
    "LiCSAR_data",
]

# Initialize a dictionary to store parameters
parameters = {}

# Attempt to retrieve each parameter and catch errors
try:
    parameters["root_path"] = config.get("DEFAULT", "root_path")
    parameters["output_path"] = config.get("DEFAULT", "output_path")
    parameters["frame"] = config.get("DEFAULT", "frame")
    parameters["start"] = config.get("DEFAULT", "start")
    parameters["end"] = config.get("DEFAULT", "end")
    parameters["interval"] = config.getint("DEFAULT", "interval")
    parameters["nlook"] = config.getint(
        "DEFAULT", "nlook"
    )  # Assuming 'nlook' should be an integer
    parameters["num_a"] = config.getint("DEFAULT", "num_a")
    parameters["LiCSAR_data"] = config.get("DEFAULT", "LiCSAR_data")

except configparser.NoOptionError as e:
    # Error message if a required parameter is missing
    missing_param = str(e).split(": ")[1]
    raise ValueError(
        f"Error: Required parameter '{missing_param}' is missing in '{config_file}'. Please check the file contents."
    )
except configparser.Error as e:
    # General error for any configparser-related issues
    raise ValueError(f"Error while reading '{config_file}': {e}")

# Assign to individual variables for easy access
root_path = parameters["root_path"]
output_path = parameters["output_path"]
frame = parameters["frame"]
start = parameters["start"]
end = parameters["end"]
interval = parameters["interval"]
nlook = parameters["nlook"]
num_a = parameters["num_a"]
LiCSAR_data = parameters["LiCSAR_data"]

# hardcoded parameters, not included in the config file
landmask = 1
filtered_ifgs = "yes"
max_loop = 5  # in case of 3 all loops with 6,12,18-day will be calculated (e.g. (60,6), (60,12) and (60,18))


##################################################### Loop Closure calculation #####################################################
####################################################################################################################################
track = frame[0:3]
if track[0] == "0":
    track = track[1:3]
if track[0] == "0":
    track = track[1:2]


###################### Read coh:
print("\nReading all coherence data...")
# Define the path components
sub_dir = "Data"
filename_pattern = "All_coh_*"  # Pattern to match files starting with "All_ifgs"

# Construct the full file path with the refined pattern
file_path_pattern = os.path.join(output_path, sub_dir, filename_pattern)
# Use glob to find the file and ensure it exists
file_list = glob.glob(file_path_pattern)
if file_list:
    file_path = file_list[0]  # Get the single file path directly

    # Load the data from the file
    with open(file_path, "rb") as file:
        coh = pickle.load(file)
else:
    raise FileNotFoundError(
        f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'."
    )
print("Reading coherence data completed.")


##################### Read Loops

print("\nReading all loop closures...")
filename_pattern = "All_loops_*"  # Pattern to match files starting with "All_ifgs"

# Construct the full file path with the refined pattern
file_path_pattern = os.path.join(output_path, sub_dir, filename_pattern)
# Use glob to find the file and ensure it exists
file_list = glob.glob(file_path_pattern)
if file_list:
    file_path = file_list[0]  # Get the single file path directly

    # Load the data from the file
    with open(file_path, "rb") as file:
        all_loops = pickle.load(file)
else:
    raise FileNotFoundError(
        f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'."
    )

print("Reading loop closures completed.")


##################################################### Loop Closure calculation #####################################################
####################################################################################################################################


long_baseline = 216  # 36,72,108,144,180,216,252,288,324
coh_thresh = 25


def calc_an(loop_360_n, loop_360_6):
    """
    Calculates the a_n values from loop closures.

    Parameters:
        loop_360_n (ndarray): Array of loop closures for n-day intervals.
        loop_360_6 (ndarray): Array of loop closures for 6-day intervals.

    Returns:
        ndarray: Mean a_n values for all loop closures.
        list: List of all a_n arrays for further processing or plotting.
    """
    ratios = []

    # Loop through indices
    for i in range(len(loop_360_6)):
        if loop_360_6[i] is not None and loop_360_n[i] is not None:
            # Calculate the ratio for non-NaN elements
            ratio_i = np.divide(
                loop_360_n[i],
                loop_360_6[i],
                out=np.full_like(loop_360_n[i], np.nan),
                where=~np.isnan(loop_360_6[i]),
            )
            ratios.append(ratio_i)

    # Convert the list of ratios to a numpy array
    an = np.array(ratios, dtype=object)
    mean_an_long_baseline = []

    # Calculate mean values of each a_n
    for arr in an:
        mask = (arr > 0) & (arr < 1)  # Apply mask to filter invalid values
        arr[~mask] = np.nan
        if not np.isnan(arr).all():  # Check if all elements are NaN
            mean_an = np.nanmean(arr)
        else:
            mean_an = np.nan  # Assign NaN for completely empty slices

        mean_an_long_baseline.append(mean_an)

    return np.array(mean_an_long_baseline), an


##################################################################################
############################
def plot_an(an_arrays, mean_a_long_baseline, k):
    """
    Plots the a_n arrays with the largest coverage and their histograms in a two-row layout,
    as well as the mean values of calibration parameters a_n over time.

    Parameters:
        an_arrays (list): List of a_n arrays.
        mean_a_long_baseline (ndarray): Array of mean calibration parameter values for a_1, a_2, ..., a_n.
        k (int): The current value of k (e.g., 2 for a1, 3 for a2).
    """

    # Calculate the coverage (number of non-NaN values) for each array
    coverage = []
    valid_an_arrays = []  # Store only valid arrays for plotting

    for arr in an_arrays:
        if (
            arr is not None and arr.ndim == 2 and not np.all(np.isnan(arr))
        ):  # Check for valid 2D arrays
            coverage.append(np.sum(~np.isnan(arr)))  # Count non-NaN values
            valid_an_arrays.append(arr)

    # Sort arrays by coverage in descending order
    sorted_indices = np.argsort(coverage)[::-1]
    selected_arrays = [
        valid_an_arrays[idx] for idx in sorted_indices[:6]
    ]  # Select top 6 arrays

    # Plot the a_n arrays and their histograms in a two-row layout
    if selected_arrays:
        num_arrays = len(selected_arrays)
        fig = plt.figure(figsize=(4 * num_arrays, 8))
        gs = GridSpec(
            2, num_arrays + 1, width_ratios=[1] * num_arrays + [0.1]
        )  # Add space for colorbar

        # Plot selected arrays
        axes_img = []
        for i, arr in enumerate(selected_arrays):
            ax_img = fig.add_subplot(gs[0, i])
            im = ax_img.imshow(arr, cmap="RdYlBu", vmin=-0.8, vmax=0.8)
            ax_img.set_title(f"{i+1}th $a_{{{k-1}}}$ Array")  # Dynamically set title
            ax_img.axis("off")
            axes_img.append(ax_img)

        # Add a single colorbar for all array plots
        cbar_ax = fig.add_subplot(gs[0, -1])
        fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="Color Scale")

        # Plot histograms for the selected arrays
        for i, arr in enumerate(selected_arrays):
            ax_hist = fig.add_subplot(gs[1, i])
            ax_hist.hist(arr.flatten(), bins=100, color="blue", alpha=0.7, density=True)
            ax_hist.set_title(
                f"{i+1}th $a_{{{k-1}}}$ Histogram"
            )  # Dynamically set title
            ax_hist.set_xlim([-10, 10])  # Adjust histogram range as needed

        plt.tight_layout()
        plt.show()
    else:
        print("No valid a_n arrays available for plotting.")

    # Plot the mean values of a_n over time
    x_values = np.arange(len(mean_a_long_baseline))  # Sequential indices for x-axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Remove NaN values for trendline fitting
    valid_indices = ~np.isnan(mean_a_long_baseline)  # Mask for valid (non-NaN) entries
    if (
        valid_indices.sum() > 1
    ):  # Ensure there are at least two valid points for fitting
        x_valid = x_values[valid_indices]
        y_valid = np.array(mean_a_long_baseline)[valid_indices]

        # Fit and plot a trendline
        coefficients = np.polyfit(x_valid, y_valid, 1)
        trendline = np.polyval(coefficients, x_valid)
        ax.plot(x_valid, trendline, color="red", label="Trendline")

    # Scatter plot for all points (including NaN if any)
    ax.scatter(
        x_values,
        mean_a_long_baseline,
        color="blue",
        marker="o",
        label="Mean Values",
        s=9,
    )

    # Set labels and limits
    ax.set_ylim(-2, 2)  # Adjust limits as needed
    ax.set_xlabel("Time Step")  # Sequential index as x-axis
    ax.set_ylabel("Mean Values", fontsize=12)
    ax.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


##
######### forming 360-6

loop_360_6_all = np.array(all_loops[long_baseline][interval], dtype=object)
coh_360 = np.array(coh[long_baseline][:], dtype=object)

loop_360_6 = np.full_like(
    loop_360_6_all, np.nan
)  # create a new array as the same size of loop_360_6_all with nan values


for i, arr in enumerate(loop_360_6_all):
    if arr is not None:
        (frame_row, frame_col) = np.shape(arr)
        loop_360_6[i] = np.where((coh_360[i] > coh_thresh), arr, np.nan)

###### forming 360-n (n=12,18,24 etc)
# mean_a_long_baseline = []
final_mean_values = []
an_labels = []

# Loop through the values of k corresponding to the number of calibration parameters
for k in range(2, 2 + num_a):  # k=2 for a1, k=3 for a2, etc.
    mean_an_long_baseline = []

    # Forming 360-n (n=12, 18, 24, etc.)
    loop_360_n_all = np.array(all_loops[long_baseline][k * interval], dtype=object)
    loop_360_n = np.full_like(loop_360_n_all, np.nan)

    for i, arr in enumerate(loop_360_n_all):
        if arr is not None:
            # Apply coherence threshold filtering
            loop_360_n[i] = np.where((coh_360[i] > coh_thresh), arr, np.nan)

    # Calculate a_n values
    mean_an_long_baseline, an_arrays = calc_an(loop_360_n, loop_360_6)

    # Calculate the mean of all mean values for this k and store it
    overall_mean = np.nanmean(mean_an_long_baseline)
    final_mean_values.append(overall_mean)
    an_labels.append(f"a{k-1}")

    for idx, mean_a in enumerate(mean_an_long_baseline, start=1):
        if not np.isnan(mean_a):  # Check if the value is not NaN
            print(f"Mean of the {idx}th a({k-1}) is {mean_a:.6f}")

    plot_an(an_arrays, mean_an_long_baseline, k)


# Print the final mean values across all k
for k_idx, final_mean in enumerate(final_mean_values, start=2):
    print(f"Final mean value for a({k_idx-1}) is {final_mean:.6f}")

# Write final mean values to a text file
output_path = os.path.join(parameters["output_path"], "Data")
os.makedirs(output_path, exist_ok=True)  # Create directory if it doesn't exist
output_file = os.path.join(output_path, "an.txt")

with open(output_file, "w") as file:
    for label, mean_value in zip(an_labels, final_mean_values):
        file.write(f"{label}={mean_value:.6f}\n")

print(f"Final mean values written to {output_file}.")
