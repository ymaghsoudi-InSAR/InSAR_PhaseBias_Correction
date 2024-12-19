#!/usr/bin/env python3

import numpy as np
import pickle
import glob
import configparser
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

################################################################################################

HELP_TEXT = """
PhaseBias_02_Loop_Closures.py

This script calculates loop closures (Δφ) using the interferograms imported in the first step. 

Definition of Loop Closures:
Loop closures, Δφ, are calculated for the epochs between `i` and `k`, and are defined as:
    Δφ_(i,k) = |φ_(i,k) - ∑_(t=i)^k φ_(t,t+1)|_2π 

Where:
- φ_(i,j): Represents the phase difference for a pixel in the interferogram formed between epochs `i` and `j`.
- |.|_2π: Indicates that the result is wrapped modulo 2π (i.e., values range from -π to π).

Key Information:
- Nonzero closure phase is a by-product of spatial filtering/multilooking and is primarily associated with changes in the scattering and electrical properties of the ground surface.
- The calculated loop closures are based on the minimum temporal-baseline interferograms defined in the configuration file (parameter: `interval`) and are referred to as base interferograms. These may represent:
  - 12- and 6-day closures (or 24- and 12-day closures), Δφ_(i,i+2)
  - 18- and 6-day closures (or 36- and 12-day closures), Δφ_(i,i+3)
  The distinction depends on whether the base interferograms are 6-day or 12-day intervals.

Outputs:
- The calculated loop closures are stored as a dictionary in a file named:
  `All_loops_start_end.pkl`
- The output is saved in the directory `Data/` under the `output_path` defined in the configuration file.

Loop Closure Report:
At the end of the script, a loop closure report is displayed. This report includes:
- The number of generated and missing loop closures for short-interval closures, such as 12-6 and 18-6.
- The number of generated and missing long-interval closures, obtained using long-term interferograms, such as 204-6 and 204-12.

Usage:
1. Ensure the interferograms from the first script are available.
2. Run the script to generate loop closures for the specified temporal baselines.

Example:
```bash
python PhaseBias_02_Loop_Closures.py


"""
if "--help" in sys.argv:
    print(HELP_TEXT)
    sys.exit(0)

################################################################################################
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
        "DEFAULT", "nlook")  # Assuming 'nlook' should be an integer
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
LiCSAR_data = parameters["LiCSAR_data"]

# hardcoded parameters, not included in the config file
landmask = 1
filtered_ifgs = "yes"
max_loop = 5  # in case of 3 all loops with 6,12,18-day will be calculated (e.g. (60,6), (60,12) and (60,18))


#############################################################################################
#################################################################################################
def loop_calc(max_loop):

    ## Read ifgs:

    # Define the path components
    sub_dir = "Data"
    filename_pattern = "All_ifgs_*"  # Pattern to match files starting with "All_ifgs"

    # Construct the full file path with the refined pattern
    file_path_pattern = os.path.join(output_path, sub_dir, filename_pattern)
    # Use glob to find the file and ensure it exists
    file_list = glob.glob(file_path_pattern)
    if file_list:
        file_path = file_list[0]  # Get the single file path directly

        # Load the data from the file
        with open(file_path, "rb") as file:
            all_ifgs = pickle.load(file)
    else:
        raise FileNotFoundError(
            f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'."
        )

    loop = {}
    missing = {}

    cat = sorted(all_ifgs.keys())  # Get sorted categories
    # print('cat = ', cat)

    # Initialize counters for reporting
    loop_counts = {
        category: {
            cat[l]: 0
            for l in range(max_loop)
        }
        for category in sorted(all_ifgs.keys())
    }
    missing_counts = {
        category: {
            cat[l]: 0
            for l in range(max_loop)
        }
        for category in sorted(all_ifgs.keys())
    }

    for category in cat:
        loop[category] = {
        }  # Initialize the key in the loop dictionary with an inner dictionary
        missing[category] = []  # second ver
        for l in range(
                max_loop
        ):  # it was up to 3 in the first version. but this allows to go for longer loops e.g. 288-72 (cat[11] is 72)
            loop[category][cat[l]] = [
            ]  # Initialize the key in the inner dictionary
    #            missing[cat[l]] = [] # first ver
    # Calculate total iterations for progress percentage
    total_iterations = sum(
        len(all_ifgs[cat[i]]) for l in range(max_loop)
        for i in range(1, len(cat))
        if cat[i] % cat[l] == 0 and cat[i] != cat[l])
    completed_iterations = 0

    # Display the message before starting the progress bar
    print(
        f"Calculating all {interval}, {2 * interval}, and {3 * interval} loop closures."
    )

    for l in range(
            0, max_loop
    ):  # l could be 6, 12 and 18 to calculate e.g. 36-6 or 36-12 or 36-18
        for i in range(1, len(cat)):  # index for the category e.g. 6, 12, 18
            if cat[i] % cat[l] == 0 and cat[i] != cat[
                    l]:  # such as 12/6 or 24/12
                for t in range(len(all_ifgs[
                        cat[i]])):  # index for the epochs in each category

                    ####################progress bar
                    completed_iterations += 1
                    percentage = (completed_iterations /
                                  total_iterations) * 100

                    # Display progress percentage
                    sys.stdout.write("\rProgress: [{:<50}] {}%".format(
                        "=" * int(percentage // 2), int(percentage)))
                    sys.stdout.flush()
                    ##################################

                    if cat[l] == interval:
                        end_index = t + int(cat[i] / cat[l])
                        # print('end_index = ', end_index)
                    else:
                        end_index = t + int(cat[i] / cat[l]) * int(
                            cat[l] / interval)

                    # recording the missing ifgs in each cat e.g. 6,12,18
                    if np.any(np.array(all_ifgs[cat[i]])[t] == None):
                        missing[cat[i]].append(
                            t
                        )  # to record the missing ifgs index for each cateogory e.g. 18

                    for e in range(
                            t, end_index, int(cat[l] / interval)
                    ):  # to record the missing ifgs index for the period of each loop
                        elem = np.array(all_ifgs[cat[l]])[e]
                        if elem is None:
                            missing[cat[l]].append(e)

                    if np.any(np.array(all_ifgs[cat[i]])[t] == None) or np.any(
                        [
                            elem is None for elem in np.array(all_ifgs[cat[l]])
                            [t:end_index:int(cat[l] / interval)]
                        ]):
                        loop[cat[i]][cat[l]].append(None)
                        missing_counts[cat[i]][
                            cat[l]] += 1  # Increment missing loop counter

                    else:
                        closure = np.angle(
                            np.exp(1j *
                                   (np.array(all_ifgs[cat[i]])[t] - (np.sum(
                                       np.array(all_ifgs[cat[l]])
                                       [t:end_index:int(cat[l] / interval)],
                                       0,
                                   )))))

                        if cat[i] in loop:
                            loop[cat[i]][cat[l]].append(closure)
                            loop_counts[cat[i]][cat[
                                l]] += 1  # Increment successful loop counter

                        else:
                            loop[cat[i]][cat[l]] = [closure]
    print("\n Finished calculating the loop closures.")

    # Generate report
    print("\n=== Loop Closure Report ===")
    for category in [
            2 * interval,
            3 * interval,
    ]:  # Restrict to relevant categories e.g. 18-6 and 12-6 as the main observations
        for l in [interval]:  # Restrict to the loop levels of interest
            if category in loop_counts and l in loop_counts[category]:
                generated = loop_counts[category].get(l, 0)
                missing = missing_counts[category].get(l, 0)
                print(
                    f"Loop Closure {category} - {l}: Generated = {generated}, Missing = {missing}"
                )

    print("\n=== Loop Closure Report for Long-interval Loop Closures ===")

    # Iterate over categories greater than 200
    for category in cat:
        if category > 200:  # Check for long categories
            for l in range(
                    1,
                    5 * interval):  # Check loop levels less than 5 * interval
                if category in loop_counts and l in loop_counts[category]:
                    generated = loop_counts[category].get(l, 0)
                    if generated >= 1:  # Only report if at least one loop is generated
                        print(
                            f"Loop Closure {category} - {l}: Generated = {generated}"
                        )

    return loop, missing


################################################################################################

all_loops, missing = loop_calc(max_loop)

# Define the directory path
directory_path = os.path.join(output_path, "Data")

# Create the directory if it doesn't exist
os.makedirs(directory_path, exist_ok=True)

# Define the file path
file_path = os.path.join(directory_path, f"All_loops_{start}_{end}.pkl")

with open(file_path, "wb") as file:  # for writting a dictionary to a file
    pickle.dump(all_loops, file)

###

# Define the file path
file_path = os.path.join(directory_path, f"missing_loops")

with open(file_path, "wb") as file:  # for writting a dictionary to a file
    pickle.dump(missing, file)
