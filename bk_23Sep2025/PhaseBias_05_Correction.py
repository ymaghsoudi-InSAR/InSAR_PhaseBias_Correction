#!/usr/bin/env python3

import numpy as np
from datetime import datetime
import pickle
import time
import os
from bin.generate_ifgs_from_epochs import generate_ifg_pairs
import configparser
import glob
from bin.resample import resample_geotiff
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import bin
from bin.read_orig_ifgs_coh import read_orig_ifgs_coh

from bin.ifg_cc_totif import ifg_cc_2tif
import gc

###############################################################################

HELP_TEXT = """
PhaseBias_05_Correction.py

This script applies phase bias correction to desired interferograms using the estimated bias terms for the base interferograms. The corrected interferograms are saved in GeoTIFF format for further analysis.

### Workflow:
1. **Inputs:**
   - **Original Wrapped Interferograms, and Coherence Data:** Obtained from step 1 (`PhaseBias_01_Read_Data.py`).
   - **Estimated Bias Terms for Base Intierferograms:** Obtained from step 4 (`PhaseBias_04_Inversion.py`).
   - **Calibration Parameters (*a_n*):** Estimated from step 3 (`PhaseBias_03_calibration_pars.py`).
   - **max_con:** Defines the maximum number of connections to be corrected. For instance:
     - `max_con=5` corrects interferograms up to 30-day intervals for a 6-day acquisition interval.
     - Adjust `max_con` as needed.

2. **Bias Term Estimation for Desired Interferograms:**
   - Bias terms for desired interferograms, δ_(i,i+n+1), are estimated using the relationship:
     δ_(i,i+n+1) = a_n * (∑_(t=i)^(i+n) δ_(t,t+1) )

3. **Correction of Interferograms:**
   - Using the estimated bias terms, the interferograms are corrected as follows:
     φ_(i,i+n+1)^c = φ_(i,i+n+1) - δ_(i,i+n+1)

   - Here, φ_(i,i+n+1) is the original interferogram, and φ_(i,i+n+1)^c is the corrected interferogram.

4. **Outputs:**
   - Corrected interferograms are saved in GeoTIFF format in the `GEOC` directory under the `output_path` specified in `config.txt`.

### Output File Format:
- **Corrected Interferograms:** Saved in GeoTIFF format, organized by temporal baseline.

### Input Requirements:
- Original wrapped interferograms from step 1.
- Bias terms for base interferograms from step 4.
- Configuration parameters (e.g., `output_path`, `interval`) defined in `config.txt`.

### Output Directory:
- `GEOC` directory under `output_path`: Contains the corrected interferograms in GeoTIFF format, organized by temporal baseline.

"""
if "--help" in sys.argv:
    print(HELP_TEXT)
    sys.exit(0)


################################################################################


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
    "a1_6_day",
    "a2_6_day",
    "a3_6_day",
    "a4_6_day",
    "a1_12_day",
    "a2_12_day",
    "a3_12_day",
    "a4_12_day",
    "estimate_an_values",
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
    parameters["LiCSAR_data"] = config.get("DEFAULT", "LiCSAR_data")
    parameters["a1_6_day"] = config.getfloat("DEFAULT", "a1_6_day")
    parameters["a2_6_day"] = config.getfloat("DEFAULT", "a2_6_day")
    parameters["a3_6_day"] = config.getfloat("DEFAULT", "a3_6_day")
    parameters["a4_6_day"] = config.getfloat("DEFAULT", "a4_6_day")
    parameters["a1_12_day"] = config.getfloat("DEFAULT", "a1_12_day")
    parameters["a2_12_day"] = config.getfloat("DEFAULT", "a2_12_day")
    parameters["a3_12_day"] = config.getfloat("DEFAULT", "a3_12_day")
    parameters["estimate_an_values"] = config.get("DEFAULT", "estimate_an_values")

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
estimate_an_values = parameters["estimate_an_values"]


start_date = datetime.strptime(start, "%Y%m%d")
end_date = datetime.strptime(end, "%Y%m%d")

uncor_only = 0  # in case of 1 only uses 6-12-18 uncorrected ifgs. it is based on the desired_lengths ifgs
max_con = 5  # in case of 5, it will correct up to 6*5=30-day ifgs.


#####################################################
# Root directory where the category directories are located

if LiCSAR_data == "yes":
    track = frame[0:3]  # extracting track number from frame id
    if track[0] == "0":
        track = track[1:3]
    if track[0] == "0":
        track = track[1:2]

    root_directory = (
        f"/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/{track}/{frame}"
    )
else:
    root_directory = root_path


############################################

# Initialize the an values with None for safety
an_values = [None] * (
    max_con - 1
)  # For max_con=5, this will create [None, None, None, None]

print("estimate_an_values is ", estimate_an_values)

# Check if estimate_an_values is 'yes'
if estimate_an_values == "yes":
    an_file_path = os.path.join(output_path, "Data", "an.txt")

    # Try reading the file
    try:
        if os.path.exists(an_file_path):
            print(f"Reading 'a' values from {an_file_path}...")

            # Read the file and load the an values dynamically
            with open(an_file_path, "r") as file:
                for line in file:
                    line = line.strip()
                    for i in range(
                        1, max_con
                    ):  # Loop through a1, a2, ..., a(max_con-1)
                        param_name = f"a{i}"
                        if line.startswith(f"{param_name}="):
                            an_values[i - 1] = float(line.split("=")[1])

            # Check if all required an values were found
            if any(value is None for value in an_values):
                print(
                    "Warning: Missing one or more an values in an.txt. Switching to default values from config file."
                )
                raise ValueError("Incomplete an values.")
        else:
            print(
                f"Warning: {an_file_path} does not exist. Switching to default values from config file."
            )
            raise FileNotFoundError

    except (FileNotFoundError, ValueError):
        # Fallback to default values from config file
        print("Loading default 'an' values from config file...")
        for i in range(1, max_con):
            if interval == 6:
                an_values[i - 1] = parameters[f"a{i}_6_day"]
            else:
                an_values[i - 1] = parameters[f"a{i}_12_day"]
else:
    # Use default values from the config file
    print("Using default 'an' values from the config file.")
    for i in range(1, max_con):
        if interval == 6:
            an_values[i - 1] = parameters[f"a{i}_6_day"]
        else:
            an_values[i - 1] = parameters[f"a{i}_12_day"]

# Print the chosen an values
for i, value in enumerate(an_values, start=1):
    print(f"Final value - a{i}: {value}")


############## generating ifg pairs between two epochs

# desired_lengths = [interval, 2*interval, 3*interval, 4*interval, 5*interval]
desired_lengths = [interval * i for i in range(1, max_con + 1)]  # replacing the above

all_ifgs_string = generate_ifg_pairs(start_date, end_date, interval, desired_lengths)


##### reading original ifgs/coh  #############################
####################################################################################

############################### Reading input data #############################################
#################### Read ifgs:
print("Reading all ifgs...")
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
        ifgs = pickle.load(file)
else:
    raise FileNotFoundError(
        f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'."
    )


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
print("\nReading data completed.")


######## stacking all the ifgs 6/12/18 in a numpy var all_ifgs
desired_categories = desired_lengths

all_ifgs, all_coh, existing_ifgs_index = read_orig_ifgs_coh(
    ifgs, coh, desired_categories
)


del ifgs, coh
gc.collect()

##### Reading the estimated bias terms as well as the indices of unknowns that can be corrected obtained in step 03  ##############
##############################################################################################
print("Correcting the interferograms...")
# Define the path components
sub_dir = "Data"
filename_pattern_X = "X_base*"  # Pattern to match files starting with "X_base"
filename_pattern_indices = "indices*"

# Construct the full file path with the refined pattern
file_path_pattern_X = os.path.join(output_path, sub_dir, filename_pattern_X)
file_path_pattern_indices = os.path.join(output_path, sub_dir, filename_pattern_indices)

# Use glob to find the file and ensure it exists
file_list = glob.glob(file_path_pattern_X)
if file_list:
    file_path = file_list[0]  # Get the single file path directly

    # Load the data from the file
    with open(file_path, "rb") as file:
        X = np.load(file)
else:
    raise FileNotFoundError(
        f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'."
    )
# print('Reading ifgs completed.')

# Use glob to find the file and ensure it exists
file_list = glob.glob(file_path_pattern_indices)
if file_list:
    file_path = file_list[0]  # Get the single file path directly

    # Load the data from the file
    with open(file_path, "rb") as file:
        indices_unknown_tobe_corrected = np.load(file)
else:
    raise FileNotFoundError(
        f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'."
    )


#### loading indices of unknown that can be corrected and the estimated unknowns

indices_unknown_tobe_corrected_orig = indices_unknown_tobe_corrected.copy()

len_6_biases = indices_unknown_tobe_corrected[-1] + 1  # number of 6-day biases

all_column_indices = np.array(
    range(len_6_biases)
)  # generating all indices in case we want to esimate all 6-day biases with smoothing constraints.
indices_unknown_tobe_corrected = all_column_indices  # in this case we had provided the corrections for all 6-day biases


# extending the indices_unknown_tobe_corrected in the original approach to have the indices of 12/18 correctable ifgs:
extended_indices = (
    indices_unknown_tobe_corrected.copy()
)  # because in this case indices_unknown_tobe_corrected referes to all_column_indices
X_extended = (
    X.copy()
)  # because in this case we have estimated all 6-day biases but we don't need them for the correction. we just need them to correct the existing 12/18 days based on the 6 days

extended_indices = extended_indices.tolist()
X_extended = X_extended.tolist()


# Generalized loop to handle any max_con
for n in range(2, max_con + 1):  # Start from 2*interval and go up to (max_con)*interval
    current_length = n * interval  # Calculate the length (e.g., 12, 18, 24, ...)

    if current_length in desired_lengths:  # Check if this length is required
        for i in range(len(indices_unknown_tobe_corrected) - (n - 1)):
            last_item = extended_indices[-1]  # Get the last index
            extended_indices.append(last_item + 1)  # Append the new index

            # Sum up X values for the current length (n)
            X_sum = sum(
                X[i + j, :, :] for j in range(n)
            )  # Sum over n consecutive indices

            # Append the result to X_extended using the corresponding an_values
            X_extended.append(
                an_values[n - 2]  # an_values is 0-based, so use n-2
                * np.angle(np.exp(complex(0 + 1j) * X_sum))
            )


X = np.array(X_extended)
indices_unknown_tobe_corrected = extended_indices
del X_extended


#### correcting the ifgs (6/12/18) ########################
#################################################################
all_ifgs_cor = np.full_like(all_ifgs, None)
all_ifgs_cor[:, :, :] = np.nan

if uncor_only == 0:  # if we want to use the corrected ifgs (using wrapped phases)
    all_ifgs_cor[indices_unknown_tobe_corrected, :, :] = np.angle(
        np.exp(
            complex(0 + 1j)
            * (all_ifgs[indices_unknown_tobe_corrected, :, :] - X[:, :, :])
        )
    )
else:  # no need for correction
    all_ifgs_cor[indices_unknown_tobe_corrected, :, :] = all_ifgs[
        indices_unknown_tobe_corrected, :, :
    ]


### Outputting to disk  ################
#############################################################################


# Create the directory (and parent directories) if they don't exist
GEOC_dir = "GEOC"
metadata_dir = "metadata"

output_wrap_dir = os.path.join(output_path, GEOC_dir)
os.makedirs(
    output_wrap_dir, exist_ok=True
)  # create the GEOC directory if it doesn't exist

template_tif_path = os.path.join(root_directory, metadata_dir, frame + ".geo.hgt.tif")

if nlook != 1:
    resampled_tif_path = os.path.join(output_wrap_dir, frame + ".geo.hgt.tif")
    resample_geotiff(template_tif_path, resampled_tif_path, nlook)
    template_tif_path = resampled_tif_path


total_files = len(all_ifgs)  # Total number of interferograms
processed_files = 0  # Counter for processed files

print(f"Writing the corrected interferograms to {output_wrap_dir}")
print("Progress: [", end="", flush=True)
time.sleep(3)


for i in range(len(all_ifgs)):
    if i in indices_unknown_tobe_corrected:

        # Writing progress bar
        processed_files += 1
        percentage = (processed_files / total_files) * 100
        if processed_files % (total_files // 100) == 0:  # Update progress every 1%
            sys.stdout.write(
                f"\rProgress: ["
                + "=" * (int(percentage) // 2)
                + " " * (50 - int(percentage) // 2)
                + f"] {int(percentage)}%"
            )
            sys.stdout.flush()

        inpha = all_ifgs_cor[i, :, :]  # your phase input, float32
        incoh = all_coh[i, :, :]  # your coh input. It can be either 0-255 or 0-1

        output_phase_path = os.path.join(output_wrap_dir, all_ifgs_string[i])
        output_cc_path = os.path.join(output_wrap_dir, all_ifgs_string[i])

        # output_phase_path = output_wrap_dir + all_ifgs_string[i]
        # output_cc_path = output_wrap_dir + all_ifgs_string[i]

        os.makedirs(output_phase_path, exist_ok=True)
        os.makedirs(output_cc_path, exist_ok=True)

        output_phase_file = (
            output_phase_path + "/" + all_ifgs_string[i] + ".geo.diff_pha.tif"
        )

        output_cc_file = output_cc_path + "/" + all_ifgs_string[i] + ".geo.cc.tif"

        ifg_cc_2tif(inpha, output_phase_file, template_tif_path)
        ifg_cc_2tif(incoh, output_cc_file, template_tif_path)

print("\nAll corrected interferograms have been written to disk.")
