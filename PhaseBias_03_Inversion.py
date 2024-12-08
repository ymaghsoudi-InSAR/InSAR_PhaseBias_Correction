#!/usr/bin/env python3

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
import matplotlib.dates as mdates
#from statsmodels.tsa.seasonal import STL
import pickle
from scipy.stats import pearsonr
import time
import os
#import pandas as pd
from scipy.signal import savgol_filter
#from collections import OrderedDict
#sys.path.append('/home/users/yma')
from scipy.sparse import csc_array
#from scipy.sparse.linalg import inv
from scipy.sparse.linalg import lsqr
#from scipy.sparse.linalg import lsmr

#from Tikhonov_inversion import solve_inversion
#from joblib import Parallel, delayed
#from numpy.linalg import LinAlgError
import math
from bin.read_orig_ifgs_coh import read_orig_ifgs_coh
from bin.circular_mean_var import circular_mean_and_variance_over_epochs
from scipy.signal import savgol_filter
#from scipy.interpolate import interp1d
import glob
import configparser
import sys
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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

#hardcoded parameters, not included in the config file
landmask=1
filtered_ifgs='yes'
max_loop=5 # in case of 3 all loops with 6,12,18-day will be calculated (e.g. (60,6), (60,12) and (60,18))




##################################################### Loop Closure calculation #####################################################
####################################################################################################################################
####################################################################################################################################


track=frame[0:3]
if track[0]=='0':
    track=track[1:3]
if track[0]=='0':
    track=track[1:2]

 
#w = float(sys.argv[1]) # weight of the temporal constraints
with_temporals = 'yes'
apply_to_all='no' # in case of with_temporal='yes', we can decide if you want to use the temporal to all ifgs (i.e. apply_to_all='yes'), or just to those that can not be corrected (i.e. apply_to_all='no')
w = 0.1
#######

## using 6-days

#a1 = 0.7614027# using the wrapped phases for 6-day in the paper
#a2 = 0.57838875

#a1 = 0.50  # old 0.540  # mean of all a1 values in 022 frame obtained by wrapped data
#a2 = 0.36 # 0.395


a1 = 0.538 # old: 0.535 # mean of all a1 values in 082 frame obatined by wrapped data
a2 = 0.336 # old:  0.388

#a1 = 0.58 # mean of all a1 values in 050 frame obatined by wrapped data
#a2 = 0.51
###########
# using 12-days
#a1 = 0.494 # using 022 wrapped 
#a2 = 0.297

#######
# using unfilt 
# 6-day: 082
#a1 = 0.4890869160493215
#a2 = 0.46132062872250873


coh_thresh = 11
threshold = 5*math.pi/3  # for wrap data it is not needed i.e. 3.14 is ok.
threshold = 1 # for wrap data in order to remove the noisy points (or those exceeding 2pi) with dynamic thresholding it is not needed
useunwrap = 'no'
std_num = 4 # the number of std to keep the loops using mean + (std_num)*std as the threshold. mean could be a single value from circular threshodling or moving average sisusoidal therhsolding
sinusoidal_thresholding = 'yes' # using mean+std where mean is the moving average not a single mean value. if this is yes the below should be no
circular_thresholding = 'no' # using the thresholds from mean and std from mean of complex values and/or von mises distribution
min_num_eq = 10# I used 10 in all my experiments
unfilt='no' # for imporing and correcting unfiltered interferograms


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
        all_ifgs = pickle.load(file)
else:
    raise FileNotFoundError(f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'.")
print('Reading ifgs completed.')


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
        all_coh = pickle.load(file)
else:
    raise FileNotFoundError(f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'.")
print('Reading coherence data completed.')


##################### Read Loops

print('\nReading all loop closures...')
filename_pattern = "All_loops_*"  # Pattern to match files starting with "All_ifgs"

# Construct the full file path with the refined pattern
file_path_pattern = os.path.join(output_path, sub_dir, filename_pattern)
# Use glob to find the file and ensure it exists
file_list = glob.glob(file_path_pattern)
if file_list:
    file_path = file_list[0]  # Get the single file path directly

    # Load the data from the file
    with open(file_path, 'rb') as file:
        all_loops = pickle.load(file)
else:
    raise FileNotFoundError(f"No file matching '{filename_pattern}' was found in '{os.path.join(output_dir, sub_dir)}'.")

print('Reading loop closures completed.')


all_cat=[]
for cat in all_loops:
    all_cat.append(cat)


### stacking all the ifgs 6/12/18 in a numpy var all_ifgs
print('Extracting the required interferograms(up to 3 epochs), and the corresponding loop closures for inversion...')
desired_categories = [interval, 2*interval, 3*interval]
all_ifgs, _, existing_ifgs_index = read_orig_ifgs_coh(all_ifgs, all_coh, desired_categories)


######### forming 12-6 and 18-6 loop closures from the imported data
#################################################################
loop_12 = np.array(all_loops[2*interval][interval])
coh_12 = np.array(all_coh[2*interval][:])

loop_18 = np.array(all_loops[3*interval][interval])
coh_18 = np.array(all_coh[3*interval][:])



############## finding dynamic pixel-based thresholds to remove the noisy loops
loop_12_orig = []
loop_18_orig = []
none_indices12 = []
none_indices18 = []

# refine loop_12 to exclude the none indices and create loop_12_orig 
for i, arr in enumerate(loop_12):
    if arr is not None:
        (frame_row, frame_col) = np.shape(arr)
        loop_12_orig.append(arr)
    else:
        none_indices12.append(i)

# refine loop_18 to exclude the none indices and create loop_18_orig
for i, arr in enumerate(loop_18):
    if arr is not None:
        loop_18_orig.append(arr)
    else:
        none_indices18.append(i)

loop_12_orig = np.array(loop_12_orig)        
loop_18_orig = np.array(loop_18_orig)

#nan_counts = np.sum(np.isnan(loop_12_orig), axis=0)
#all_nan_pixels = np.where(nan_counts == loop_12_orig.shape[0])
#print(f"Number of pixels with all NaN values: {len(all_nan_pixels[0])}")

### removing noisy loops
print('Masking noisy loop closures using circular moving average and standard deviaiton of the loop closures in time:')

#using circular_mean to calculate the mean phase value using complex numbers, and von_mises_variance to calculate the second moment of the Von Mises distribution
_, thresh_loop12 = circular_mean_and_variance_over_epochs(loop_12_orig, axis=0)
_, thresh_loop18 = circular_mean_and_variance_over_epochs(loop_18_orig, axis=0)



#thresh_loop12 = 1*thresh_loop12 # multiply the std by an arbitrary factor
#thresh_loop18 = 1*thresh_loop18 # multiply the std by a arbitrary factor


#  Add a new axis at the beginning (axis=0) of these arrays to have shape (1, ...)
# These are the threshold vlaues obtained for each pixel that will be used for masking later
thresh_loop12 = np.expand_dims(thresh_loop12, axis=0)
thresh_loop18 = np.expand_dims(thresh_loop18, axis=0)


#calculate a temporal moving average for wrapped data
def moving_average(arr, window_size):
    # Initialize an empty array to store the temporal averages
    moving_averages = np.zeros_like(arr)
    half_window = window_size // 2
    ### miroring data at both ends for window_size/2 at each end
    mirrored_shape = (arr.shape[0] + window_size, arr.shape[1], arr.shape[2])

    # Create an empty array to hold the mirrored data
    mirrored_arr = np.empty(mirrored_shape, dtype=arr.dtype)

    # Fill the mirrored array with the original data
    mirrored_arr[window_size//2:-window_size//2] = arr

    # Mirror the data at the beginning
    for i in range(window_size//2):
        mirrored_arr[i] = arr[window_size//2 - i]

    # Mirror the data at the end
    for i in range(window_size//2):
        mirrored_arr[-(i+1)] = arr[-(i+1) - window_size//2]
     ####

    # Calculate temporal averages within each window
    for t in range(half_window, mirrored_arr.shape[0] - half_window):
        moving_averages[t-half_window,:,:] = np.angle(np.nanmean(np.exp(1j *(mirrored_arr[t-half_window:t+half_window, :, :])), axis=0))
    return moving_averages



# for loop 12
# Calculate distance between each epoch and the moving average for each pixel
print(f'Calculating the moving average for {2 * interval} loop closures, followed by masking noisy loop closures...')
print("This may take a while, as it depends on the number of loops and the size of your dataset.\n")

window_size = 30
moving_avg = moving_average(loop_12_orig, window_size)
distances = np.abs(np.angle(np.exp(1j*(loop_12_orig - moving_avg))))
mask = distances > std_num*thresh_loop12 


# Replace values with NaN where mask is True
loop_12_bk = loop_12_orig
loop_12_orig = np.where(mask, np.nan, loop_12_orig)
loop_12 = loop_12_orig.tolist()


for idx in none_indices12:
    loop_12.insert(idx, None)
loop_12 = np.array(loop_12)

# for loop 18
# Calculate distance between each epoch and the moving average for each pixel
print(f'Calculating the moving average for {3 * interval} loop closures, followed by masking noisy loop closures...')
print("This may take a while, as it depends on the number of loops and the size of your dataset.\n")

moving_avg = moving_average(loop_18_orig, window_size)
distances = np.abs(np.angle(np.exp(1j*(loop_18_orig - moving_avg))))
mask = distances > std_num*thresh_loop18 #[:,:,np.newaxis]

# Replace values with NaN where mask is True
loop_18_bk = loop_18_orig
loop_18_orig = np.where(mask, np.nan, loop_18_orig)
loop_18 = loop_18_orig.tolist()

for idx in none_indices18:
    loop_18.insert(idx, None)
loop_18 = np.array(loop_18)

print("Masking of noisy loop closures is completed.")


#   ################ Apply coherece thresholding   
#    print('shape loop_12 = ', np.shape(loop_12))
#    print('shape loop_12_orig = ', np.shape(loop_12_orig))
#    print('shape coh_12 = ', np.shape(coh_12))
#    for i, arr in enumerate(loop_12):
#        if arr is not None:
#            (frame_row, frame_col) = np.shape(arr)
#            loop_12[i] = np.where((coh_12[i] > coh_thresh), arr, np.nan)
#
#
#    for i, arr in enumerate(loop_18):
#        if arr is not None:
#            loop_18[i] = np.where((coh_18[i] > coh_thresh), arr, np.nan)
#
#
#

len12 = len(loop_12) #number of 12 day loops including None
len18 = len(loop_18)  # number of 18 day loops including None


######################## Forming the design matrix A ##############################
####################################################################################

print("Preparing the design matrix and observation vector for the inversion step...")

b=[]
A=[]
n_unk= (len12 + 1) # this in an initial value(i.e. the num of 6 biases). will be changed by the actual number of unk after considering the None values/missing
#print('n_unk = ', n_unk)

row = np.zeros(n_unk , dtype=np.float32) # each row of the design matrix A for the 12-day
for i in range(len12):
    b.append(loop_12[i])
    #print('np.shape(loop_12[i] = ', np.shape(loop_12[i]))
    row[i:i+2] = a1 - 1
    A.append(row)
    row = np.zeros(n_unk , dtype=np.float32)

row = np.zeros(n_unk, dtype=np.float32) # each row of the design matrix A for the 18-day
for i in range(len18):
    b.append(loop_18[i])
    row[i:i+3] = a2 - 1
    A.append(row)
    row = np.zeros(n_unk, dtype=np.float32)

A=np.array(A)
b = np.array(b)

#print('shape A (initial) =', np.shape(A))
#print('shape b (initial) =', np.shape(b))

####################### removing the rows from b and A where b is None
mask = []
mask = np.array([arr is not None for arr in b])
# Convert the boolean mask to an integer mask
mask = np.nonzero(mask)

#print('mask ', mask)
#print('shape mask ', np.shape(mask))

A = A[mask]  # Filter rows of A based on the mask
b = b[mask]

#print('shape A after removing Nones =', np.shape(A))
#print('shape b after removing Nones =', np.shape(b))
b=b.tolist()

############  removing the columns of A where are values are zero (these are the unkonws which doesn't fall in any equations nor 12 neither 18 and thus cannot be corrected)

# Find the column indices where all values are zero
zero_columns = np.all(A == 0, axis=0)

## Get the indices of the zero columns
zero_column_indices = np.where(zero_columns)[0]

# Get the indices of the non-zero columns
non_zero_column_indices = np.where(~zero_columns)[0]  # the indices of the unknowns that can be corrected
#print('non_zero_column_indices=', non_zero_column_indices)

# Save the file
directory_path = os.path.join(output_path, '01_Data')
file_path = os.path.join(directory_path, f"indices_unknown_tobe_corrected.npy")
np.save(file_path, non_zero_column_indices)

#np.save('results/' + frame + '/indices_unknown_tobe_corrected_orig_approach.npy', non_zero_column_indices)

### Adding all the existing 6-day ifgs indices to non_zero_column_indices. We want to use estimate their biases in the second step.  


## Remove the zero columns from A
#if with_temporals=='no':

A_bk = A # to keep a copy of A in case we want to esimate all 6-day biases with smoothing constraints
b_bk = b
all_column_indices = np.array(range(A_bk.shape[1])) # generating all indices in case we want to esimate all 6-day biases with smoothing constraints


A = A[:, non_zero_column_indices]



# finding the last column with value -1. this gives the number of 6-day biases that can be corrected
last_column_with_minus_1 = None

# Iterate through the columns from right to left
for col in range(len(A[0]) - 1, -1, -1):
    if -1 in [row[col] for row in A]:
        last_column_with_minus_1 = col
        break


num_rows_before_temporals = A.shape[0]




################################# Least Square inversion ###############################
########################################################################################
# First Inversion: Without Temporal Smoothing Constraints
print("Starting the first inversion (without temporal smoothing constraints).")
print("This step estimates the bias terms that can be corrected based on observed loop closures.")


damp_factor = 0

if apply_to_all=='no': # this is without using any temporal constraints

    X1 = np.zeros((np.shape(A_bk)[1], frame_row, frame_col), dtype=np.float32)
    #X1[:,:,:] = np.nan # after checking noticed this doesn't have any effect on the final vel
    b_bk=np.array(b_bk)

    total_pixels = frame_row * frame_col  # Total number of pixels for progress bar
    processed_pixels = 0  # Counter for processed pixels
    print("Progress: [", end="", flush=True)



    for row in range(frame_row):
        for col in range(frame_col):
            processed_pixels += 1
            # Update progress bar
            percentage = (processed_pixels / total_pixels) * 100
            if processed_pixels % (total_pixels // 100) == 0:  # Update every 1%
                print(f"{int(percentage)}%", end="", flush=True)
                sys.stdout.write("\rProgress: [" + "=" * (int(percentage) // 2) + " " * (50 - int(percentage) // 2) + "]")


            non_nan_mask = ~np.isnan(b_bk[:,row,col])
            bb = b_bk[non_nan_mask,row,col]  # Remove NaN values
            AA = A_bk[non_nan_mask, :]  # Remove corresponding rows
            if (len(bb) > min_num_eq): # what is the min number of equation?

                x, istop, itn, normr, normr2 = lsqr(csc_array(AA), bb, damp = damp_factor)[:5] # Using sparse matrix representation scipy.sparse.linalg
                X1[:,row, col] = x 

print("\nFirst inversion (without temporal smoothing constraints) completed.")

###########################
# Second Inversion: With Temporal Smoothing Constraints
print("Starting the second inversion (with temporal smoothing constraints).")
print("This step estimates all bias terms, including those that cannot be corrected, using temporal smoothing constraints.")




if with_temporals == 'yes':  # This is using the temporal constranints on all unknowns
    X2 = np.zeros((np.shape(A_bk)[1], frame_row, frame_col), dtype=np.float32)
    #X2[:,:,:] = np.nan # after checking noticed this doesn't have any effect on the final vel
    b_bk= np.array(b_bk)

    #### Adding temporal constraints
    row = np.zeros(A_bk.shape[1], dtype=np.float32) # additional rows the design matrix A for temporal constraints, setting that to zero again
    for i in range(len(all_column_indices) - 2): # for all unknowns
        row[i] = -1 * w
        row[i + 1] = 2 * w
        row[i + 2] = -1 * w
        A_bk = np.vstack((A_bk, row))
        row = np.zeros(A_bk.shape[1], dtype=np.float32) # additional rows the design matrix A for temporal constraints, setting that to zero again
    num_temporals = len(A_bk) - len(b_bk)
    #print('num_temporals is ', num_temporals)

    ######### Appending zeroes to b according to the added number of temporal constraints obtained by np.shape(A)[0] - np.shape(b)[0]
    b_0=np.zeros((np.shape(A_bk)[0] - np.shape(b)[0],frame_row, frame_col), dtype=np.float32)
    b_bk = np.vstack((b_bk, b_0))

    thresh_n_eq = num_temporals # in case of using temporals, we don't want the nan pixels with zero equations with only using temporals

    processed_pixels = 0  # Reset counter for second inversion
    print("Progress: [", end="", flush=True)

    for row in range(frame_row):
        for col in range(frame_col):
            processed_pixels += 1
            # Update progress bar
            percentage = (processed_pixels / total_pixels) * 100
            if processed_pixels % (total_pixels // 100) == 0:  # Update every 1%
                print(f"{int(percentage)}%", end="", flush=True)
                sys.stdout.write("\rProgress: [" + "=" * (int(percentage) // 2) + " " * (50 - int(percentage) // 2) + "]")

            non_nan_mask = ~np.isnan(b_bk[:,row,col])
            bb = b_bk[non_nan_mask,row,col]  # Remove NaN values
            AA = A_bk[non_nan_mask, :]  # Remove corresponding rows
            if (len(bb) > thresh_n_eq):

                x, istop, itn, normr, normr2 = lsqr(csc_array(AA), bb, damp = damp_factor)[:5] # Using sparse matrix representation scipy.sparse.linalg
                X2[:,row, col] = x

    print("\nSecond inversion (with temporal smoothing constraints) completed.")


    ####### combining X1 and X2 into X

    X =  np.zeros((np.shape(A_bk)[1], frame_row, frame_col), dtype=np.float32)
    X[:,:,:] = np.nan 

    # Fill X with values from X1 according to non_zero_column_indices
    for i, idx in enumerate(non_zero_column_indices):
        X[idx, :, :] = X1[i, :, :]

    # Fill in the missing values from X2
    for i, idx in enumerate(all_column_indices):
        if idx not in non_zero_column_indices:
            X[idx, :, :] = X2[i, :, :]

    # Ensure the dtype is appropriate for your data
    X = X.astype(np.float32)




directory_path = os.path.join(output_path, '01_Data')
file_path = os.path.join(directory_path, f"X_base_ifgs_biases.npy")
np.save(file_path, X)

print("Inversion process completed. Results saved to:", file_path)





##########################################################################################################
   
