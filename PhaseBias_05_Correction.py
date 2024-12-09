#!/usr/bin/env python3

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pickle
import math
import time
import os
#from bin.read_orig_ifgs_coh import read_orig_ifgs_coh
#from bin.ifg_cc_totif import ifg_cc_2tif
import rasterio
from bin.generate_ifgs_from_epochs import generate_ifg_pairs
import configparser
import glob
from bin.resample import resample_geotiff
import sys
# Add the script's directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import bin
from bin.read_orig_ifgs_coh import read_orig_ifgs_coh

from bin.ifg_cc_totif import ifg_cc_2tif

#################### initial set up b#################

################################################################################


# Config file path
config_file = 'config.txt'

# Check if config file exists
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Error: The configuration file '{config_file}' was not found in the script's directory.")

# Initialize the configparser
config = configparser.ConfigParser()


# Read the config.txt file
config.read(config_file)

# Define required parameters
required_parameters = ['root_path', 'frame', 'start', 'end', 'interval', 'nlook', 'LiCSAR_data']

# Initialize a dictionary to store parameters
parameters = {}

# Attempt to retrieve each parameter and catch errors
try:
    parameters['root_path'] = config.get('DEFAULT', 'root_path')
    parameters['output_path'] = config.get('DEFAULT', 'output_path')
    parameters['frame'] = config.get('DEFAULT', 'frame')
    parameters['start'] = config.get('DEFAULT', 'start')
    parameters['end'] = config.get('DEFAULT', 'end')
    parameters['interval'] = config.getint('DEFAULT', 'interval')
    parameters['nlook'] = config.getint('DEFAULT', 'nlook')  # Assuming 'nlook' should be an integer
    parameters['LiCSAR_data'] = config.get('DEFAULT', 'LiCSAR_data')

except configparser.NoOptionError as e:
    # Error message if a required parameter is missing
    missing_param = str(e).split(": ")[1]
    raise ValueError(f"Error: Required parameter '{missing_param}' is missing in '{config_file}'. Please check the file contents.")
except configparser.Error as e:
    # General error for any configparser-related issues
    raise ValueError(f"Error while reading '{config_file}': {e}")

# Assign to individual variables for easy access
root_path = parameters['root_path']
output_path = parameters['output_path']
frame = parameters['frame']
start = parameters['start']
end = parameters['end']
interval = parameters['interval']
nlook = parameters['nlook']
LiCSAR_data = parameters['LiCSAR_data']

start_date = datetime.strptime(start, "%Y%m%d")
end_date = datetime.strptime(end, "%Y%m%d")

#hardcoded parameters, not included in the config file
landmask=1
filtered_ifgs='yes'
max_loop=5 # in case of 3 all loops with 6,12,18-day will be calculated (e.g. (60,6), (60,12) and (60,18))








#####################################################


track=frame[0:3]
if track[0]=='0':
    track=track[1:3]
if track[0]=='0':
    track=track[1:2]


# Root directory where the category directories are located
if LiCSAR_data == 'yes':
   root_directory = '/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/' + track + '/' + frame
else:
   root_directory = root_path


uncor_only = 0 # in case of 1 only uses 6-12-18 uncorrected ifgs. it is based on the desired_lengths ifgs
long_ifgs = 'no' # if you want to include the long term ifgs (more than 18 days) this has to be 'yes'. It is currently uncorrected

max_con = 5 # in case of 5, it will correct up to 6*5=30-day ifgs. 
max_long_con = 20 # in case of long_ifgs='yes', what is the length of the max long connectoin, e.g .in case of 8, it will go until 48


#a1 = 0.7614027# using the wrapped phases for 6-day
#a2 = 0.57838875

#a1 = 0.50 # old 0.540  # mean of all a1 values in 022 frame obtained by wrapped data
#a2 = 0.36 # old 0.395
#a3 = 0.299  #(using 324)
#a4 = 0.2476  #(Using 330)  

#a1 = 0.447  # mean of all a1 values in 022 frame obtained by unwrapped data
#a2 = 0.378

#a1 = 0.538 # old: 0.535 # mean of all a1 values in 082 frame obatined by wrapped data
#a2 = 0.336 # old:  0.388

#a1 = 0.58 # mean of all a1 values in 050 frame obatined by wrapped data
#a2 = 0.51

###########
# using 12-days
a1 = 0.494 # using 022 wrapped
a2 = 0.297
a3 = 0.24
a4 = 0.22

############## generating ifg pairs between two epochs

#desired_lengths = [interval, 2*interval, 3*interval, 4*interval, 5*interval]
desired_lengths = [interval * i for i in range(1, max_con + 1)] # replacing the above

all_ifgs_string = generate_ifg_pairs(start_date, end_date, interval, desired_lengths)

#desired_lengths_long = [4*interval, 5*interval, 6*interval, 7*interval, 8*interval, 9*interval, 10*interval] 
desired_lengths_long = [interval * i for i in range(max_con+1, max_long_con + 1)] # replacing the above

all_ifgs_string_long = generate_ifg_pairs(start_date, end_date, interval, desired_lengths_long)



##### reading original ifgs/coh  #############################
####################################################################################

############################### Reading input data #############################################
#################### Read ifgs:
print('Reading all ifgs...')
# Define the path components
sub_dir = "01_Data"
filename_pattern = "All_ifgs_*"  # Pattern to match files starting with "All_ifgs"

# Construct the full file path with the refined pattern
file_path_pattern = os.path.join(output_path, sub_dir, filename_pattern)
# Use glob to find the file and ensure it exists
file_list = glob.glob(file_path_pattern)
if file_list:
    file_path = file_list[0]  # Get the single file path directly

    # Load the data from the file
    with open(file_path, 'rb') as file:
        ifgs = pickle.load(file)
else:
    raise FileNotFoundError(f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'.")


###################### Read coh:
print('\nReading all coherence data...')
# Define the path components
sub_dir = "01_Data"
filename_pattern = "All_coh_*"  # Pattern to match files starting with "All_ifgs"

# Construct the full file path with the refined pattern
file_path_pattern = os.path.join(output_path, sub_dir, filename_pattern)
# Use glob to find the file and ensure it exists
file_list = glob.glob(file_path_pattern)
if file_list:
    file_path = file_list[0]  # Get the single file path directly

    # Load the data from the file
    with open(file_path, 'rb') as file:
        coh = pickle.load(file)
else:
    raise FileNotFoundError(f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'.")
print('\nReading data completed.')


######## stacking all the ifgs 6/12/18 in a numpy var all_ifgs
desired_categories = desired_lengths
#print('desired_categories', desired_categories)

all_ifgs, all_coh, existing_ifgs_index = read_orig_ifgs_coh(ifgs, coh, desired_categories)

desired_categories = desired_lengths_long
#print('desired_categories_long', desired_categories)

if long_ifgs=='yes':
    all_ifgs_long, all_coh_long, existing_ifgs_index_long = read_orig_ifgs_coh(ifgs, coh, desired_categories)

del ifgs, coh


##### Reading the estimated bias terms as well as the indices of unknowns that can be corrected obtained in step 03  ##############
##############################################################################################
print('Correcting the interferograms...')
# Define the path components
sub_dir = "01_Data"
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
    with open(file_path, 'rb') as file:
        X = np.load(file)
else:
    raise FileNotFoundError(f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'.")
#print('Reading ifgs completed.')

# Use glob to find the file and ensure it exists
file_list = glob.glob(file_path_pattern_indices)
if file_list:
    file_path = file_list[0]  # Get the single file path directly

    # Load the data from the file
    with open(file_path, 'rb') as file:
        indices_unknown_tobe_corrected = np.load(file)
else:
    raise FileNotFoundError(f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'.")


#### loading indices of unknown that can be corrected and the estimated unknowns

indices_unknown_tobe_corrected_orig = indices_unknown_tobe_corrected.copy()

len_6_biases = indices_unknown_tobe_corrected[-1] + 1 # number of 6-day biases 

all_column_indices = np.array(range(len_6_biases)) # generating all indices in case we want to esimate all 6-day biases with smoothing constraints. 
indices_unknown_tobe_corrected = all_column_indices # in this case we had provided the corrections for all 6-day biases


#print('indices_unk_tobe_cor', indices_unknown_tobe_corrected)
#print('np.shape(indices_unknown_tobe_corrected)', np.shape(indices_unknown_tobe_corrected))



#print('shape(X)= ', np.shape(X))
#print('len_6_biases = ', len_6_biases)

# extending the indices_unknown_tobe_corrected in the original approach to have the indices of 12/18 correctable ifgs:
extended_indices = indices_unknown_tobe_corrected.copy() # because in this case indices_unknown_tobe_corrected referes to all_column_indices 
X_extended = X.copy() # because in this case we have estimated all 6-day biases but we don't need them for the correction. we just need them to correct the existing 12/18 days based on the 6 days

extended_indices = extended_indices.tolist()
X_extended = X_extended.tolist()

# for 12 days
for i in range(len(indices_unknown_tobe_corrected) - 1):
    last_item = extended_indices[-1] # is -1 ok in case of having a missing 6-day ifg at the end of the time-series?
    extended_indices.append(last_item + 1)
    X_extended.append(a1*np.angle(np.exp(complex(0+1j)*((X[i,:,:] + X[i+1,:,:])))))


#fo 18 days
for i in range(len(indices_unknown_tobe_corrected) - 2):
    last_item = extended_indices[-1] # is -1 ok in case of having a missing 6-day ifg at the end of the -series?
    extended_indices.append(last_item + 1)
    X_extended.append(a2*np.angle(np.exp(complex(0+1j)*((X[i,:,:] + X[i+1,:,:] + X[i+2,:,:])))))


if 4*interval in desired_lengths: 
    # for 24 days
    for i in range(len(indices_unknown_tobe_corrected) - 3):
        last_item = extended_indices[-1] # is -1 ok in case of having a missing 6-day ifg at the end of the time-series?
        extended_indices.append(last_item + 1)
        X_extended.append(a3*np.angle(np.exp(complex(0+1j)*((X[i,:,:] + X[i+1,:,:] + X[i+2,:,:] + X[i+3,:,:])))))

if 5*interval in desired_lengths:
    # for 30 days
    for i in range(len(indices_unknown_tobe_corrected) - 4):
        last_item = extended_indices[-1] # is -1 ok in case of having a missing 6-day ifg at the end of the time-series?
        extended_indices.append(last_item + 1)
        X_extended.append(a4*np.angle(np.exp(complex(0+1j)*((X[i,:,:] + X[i+1,:,:] + X[i+2,:,:] + X[i+3,:,:] + X[i+4,:,:])))))


X = np.array(X_extended)
indices_unknown_tobe_corrected = extended_indices
del X_extended

#print('np.shape(indices_unknown_tobe_corrected) = ', np.shape(indices_unknown_tobe_corrected) )
#print('shape X = ', np.shape(X))

#### correcting the ifgs (6/12/18) ########################
#################################################################
all_ifgs_cor = np.full_like(all_ifgs, None)
all_ifgs_cor[:,:,:] = np.nan

if uncor_only == 0: # if we want to use the corrected ifgs (using wrapped phases)
    all_ifgs_cor[indices_unknown_tobe_corrected, :, :] = np.angle(np.exp(complex(0+1j)*(all_ifgs[indices_unknown_tobe_corrected,:,:] - X[:,:,:]))) 
else: # no need for correction
    all_ifgs_cor[indices_unknown_tobe_corrected, :, :] = all_ifgs[indices_unknown_tobe_corrected, :, :]
    


### Outputting to disk  ################
#############################################################################



# Create the directory (and parent directories) if they don't exist
GEOC_dir = "GEOC"
metadata_dir = "metadata"

output_wrap_dir = os.path.join(output_path, GEOC_dir)
os.makedirs(output_wrap_dir, exist_ok=True) # create the GEOC directory if it doesn't exist

template_tif_path = os.path.join(root_directory, metadata_dir, frame + '.geo.hgt.tif')

if nlook != 1:
    resampled_tif_path = os.path.join(output_wrap_dir, frame + '.geo.hgt.tif')
    resample_geotiff(template_tif_path, resampled_tif_path, nlook)
    template_tif_path = resampled_tif_path


total_files = len(all_ifgs)  # Total number of interferograms
processed_files = 0  # Counter for processed files

print(f'Writing the corrected interferograms to {output_wrap_dir}')
print("Progress: [", end="", flush=True)
time.sleep(3)



for i in range(len(all_ifgs)):
    if i in indices_unknown_tobe_corrected:

        # Writing progress bar
        processed_files += 1
        percentage = (processed_files / total_files) * 100
        if processed_files % (total_files // 100) == 0:  # Update progress every 1%
            sys.stdout.write(f"\rProgress: [" + "=" * (int(percentage) // 2) + " " * (50 - int(percentage) // 2) + f"] {int(percentage)}%")
            sys.stdout.flush()


        inpha = all_ifgs_cor[i,:,:] # your phase input, float32
        incoh= all_coh[i,:,:] # your coh input. It can be either 0-255 or 0-1

        output_phase_path = os.path.join(output_wrap_dir, all_ifgs_string[i])
        output_cc_path = os.path.join(output_wrap_dir, all_ifgs_string[i])
    
        #output_phase_path = output_wrap_dir + all_ifgs_string[i] 
        #output_cc_path = output_wrap_dir + all_ifgs_string[i] 

        os.makedirs(output_phase_path, exist_ok=True)
        os.makedirs(output_cc_path, exist_ok=True)

        output_phase_file = output_phase_path + '/' + all_ifgs_string[i] + '.geo.diff_pha.tif'

  
        output_cc_file = output_cc_path + '/' + all_ifgs_string[i] + '.geo.cc.tif'

        ifg_cc_2tif(inpha, output_phase_file, template_tif_path)
        ifg_cc_2tif(incoh, output_cc_file, template_tif_path)

print("\nAll corrected interferograms have been written to disk.")
                        
if long_ifgs == 'yes': # in case you want to include the long term uncorrected ifgs in addition to short terms.
    for i in range(len(all_ifgs_string_long)):
        if i in existing_ifgs_index_long:

            print('Writting long ifgs to file ... ')
            print(all_ifgs_string_long[i])
            inpha = all_ifgs_long[i,:,:] # your phase input, float32
            incoh= all_coh_long[i,:,:] # your coh input. It can be either 0-255 or 0-1

            output_phase_path = output_wrap_dir + all_ifgs_string_long[i]
            output_cc_path = output_wrap_dir + all_ifgs_string_long[i]

            os.makedirs(output_phase_path, exist_ok=True)
            os.makedirs(output_cc_path, exist_ok=True)


            output_phase_file = output_phase_path + '/' + all_ifgs_string_long[i] + '.geo.diff_pha.tif'

            output_cc_file = output_cc_path + '/' + all_ifgs_string_long[i] + '.geo.cc.tif'

            ifg_cc_2tif(inpha, output_phase_file, template_tif_path)
            ifg_cc_2tif(incoh, output_cc_file, template_tif_path)





