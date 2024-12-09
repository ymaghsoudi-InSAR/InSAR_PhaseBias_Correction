#!/usr/bin/env python3

import os
import re
from datetime import datetime, timedelta
import numpy as np
from osgeo import gdal
from bin.multilook_w import multilook_w
import time
import configparser
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import pickle

#######################################################################

HELP_TEXT = """
PhaseBias_01_Read_Data.py

This script automates the process of downloading GeoTIFF files for wrapped interferograms and coherence images. 
The files can be retrieved either from the COMET-LiCS web portal or from a root path specified in the configuration file.

Assumptions:
- Interferograms are organized in folders named as `yyyymmdd_yyyymmdd` (e.g., `20230101_20230107`).

Files Downloaded:
1. `yyyymmdd_yyyymmdd.geo.diff_pha.tif`: 
   - Contains the wrapped phase image in radians.
   - Values range from -3.14 to 3.14.

2. `yyyymmdd_yyyymmdd.geo.cc.tif`: 
   - Contains the coherence image of the interferometric pair.
   - Values range from 0 to 255, where:
       - 0 represents the lowest coherence.
       - 255 represents the highest coherence.

Outputs:
1. `All_ifgs_start_end` (Pickle file, saved as `.pkl`):
   - A dictionary containing all interferograms available between the specified start and end dates.
2. `All_coh_start_end` (Pickle file, saved as `.pkl`):
   - A dictionary containing all coherence data available between the specified start and end dates.

Storage Information:
- All output files are saved in the directory `Data/` under the `output_path` defined in the configuration file.

Additional Output:
- A summary report is generated that includes:
   - The number of available interferograms for each temporal baseline.
   - The number of missing interferograms.

"""

if "--help" in sys.argv:
    print(HELP_TEXT)
    sys.exit(0)



###################

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
min_baseline=5
max_baseline=366

# Print the values to confirm they're being read correctly
print("Root Path:", root_path)
print("output Path:", output_path)
print("Frame:", frame)
print("Start Date:", start)
print("End Date:", end)
print("Interval:", interval)
print("nlook:", nlook)
print('LiCSAR_data', LiCSAR_data)

################
#all_ifgs = read_ifgs(frame, start, end, min_baseline, max_baseline, landmask, nlook, interval, LiCSAR_data, filtered_ifgs)


def read_ifgs(start, end, min_baseline, max_baseline, landmask, nlook , interval, LiCSAR_data, filtered_ifgs):


    start_date = datetime.strptime(start, "%Y%m%d")
    end_date = datetime.strptime(end, "%Y%m%d")


    # Check if the file ends with "geo.diff_pha.tif"
    #if filename.endswith("geo.diff_unfiltered_pha.tif"): # if unfiltered data are used
    if filtered_ifgs == 'yes': # yes for the filtered ifgs, no for the unfiltered ifgs
        req_file_name = "geo.diff_pha.tif"
    else:
        req_file_name = "geo.diff_unfiltered_pha.tif"



    ############ reading

    # Root directory where the category directories are located
    if LiCSAR_data == 'yes':

        track=frame[0:3] # extracting track number from frame id
        if track[0]=='0':
            track=track[1:3]
        if track[0]=='0':
            track=track[1:2]

        root_directory = '/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/' + track + '/' + frame + '/interferograms/'

        
    else:
        root_directory = root_path 
        #root_directory = '/work/scratch-pw3/earyma/022D_04826_121209_subset/GEOC/' # one-off

    ##### Count total files matching the criteria for accurate progress calculation
    print(f"Reading data from the path: {root_directory}")

    total_files = sum(1 for dirpath, _, filenames in os.walk(root_directory) for filename in filenames if filename.endswith(req_file_name))
    processed_files = 0

    print("Reading wrapped interferograms: [", end="", flush=True)
    #####

    # Initialize an empty dictionary to store the arrays for each category
    category_arrays = {}
    last_date = {}
    n=0

    if landmask != None and LiCSAR_data == 'yes': # in case of wrapped data the original landmask data is used
        landmask_path = '/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/' + track + '/' + frame + '/metadata/' + frame + '.geo.landmask.tif'
        #one off     
        #landmask_path = '/work/scratch-pw3/earyma/022D_04826_121209_subset/022D_04826_121209.geo.landmask.tif'

        landmask_dataset = gdal.Open(landmask_path)
        landmask_image = landmask_dataset.ReadAsArray()
        array_landmask = np.array(landmask_image)

  

    n = 1
    # Traverse the root directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        dirnames.sort()  # This ensures dirnames are processed in alphabetical order
        # Iterate through the files in the current directory
        for filename in filenames:
            # Check if the file ends with "geo.diff_pha.tif"
            #if filename.endswith("geo.diff_unfiltered_pha.tif"): # if unfiltered data are used
            if filtered_ifgs == 'yes': # yes for the filtered ifgs, no for the unfiltered ifgs
                req_file_name = "geo.diff_pha.tif"
            else:
                req_file_name = "geo.diff_unfiltered_pha.tif"



            if filename.endswith(req_file_name):
                file_path = os.path.join(dirpath, filename)

                #### for printing into the console
                processed_files += 1
                percentage = (processed_files / total_files) * 100

                # Clear previous percentage and print updated one
                sys.stdout.write("\rReading wrapped phases: [{:<50}] {}%".format(
                    "=" * int(percentage // 2), int(percentage)
                ))
                sys.stdout.flush()

               #####


                # Extract the dates from the file name
                match = re.search(r"(\d+)_(\d+)", filename)
                if match:
                    date1 = match.group(1)
                    date2 = match.group(2)



                    # Convert the dates to datetime objects
                    date1_obj = datetime.strptime(date1, "%Y%m%d")
                    date2_obj = datetime.strptime(date2, "%Y%m%d")

                    # Check if the dates fall within the specified range
                    category = (date2_obj - date1_obj).days
                    #print(category)
#                    if start_date <= date1_obj <= end_date and start_date <= date2_obj <= end_date and category > max_baseline and category < (max_baseline + min_baseline) or category < min_baseline:
                    if start_date <= date1_obj <= end_date and start_date <= date2_obj <= end_date and category < max_baseline and category > min_baseline:
                        if n==1: # finding the first and last acquisiton in the time-series
                            first_acq = date1_obj
                        else:
                            last_acq = date2_obj
                        n = n + 1
#                        print(file_path)

                        # Calculate the difference in days between the two dates
                        

                        
                        # Open the TIFF file
                        tiff_dataset = gdal.Open(file_path)
                        # Read the image data as a NumPy array
                        tiff_image = tiff_dataset.ReadAsArray()

                        # Convert the image to a numpy array
                        array_data = np.array(tiff_image)
                        array_data[array_data==0]=np.nan ## converting the zerso values in wrapped phases into nan. This is helpful when calculating loop closures



                        if landmask != None:
                            array_data[array_landmask != 1]=np.nan

                        if nlook != None and nlook != 1 : # incase of unwrap data because they are already multilooked to 10 we don't multilook them here
                            array_data = multilook_w(array_data,nlook) 

 
                        if category not in last_date:
                            last_date[category] = date1_obj
                            #if (date1_obj - date2_obj) > interval

                        # appending None to the category_arrays where there are missing ifgs in the middle of time-series
                        diff_ifgs = (date1_obj - last_date[category]).days #/category #difference between the first epochs of two consecutive ifg
                        if diff_ifgs > interval:
                            n_no_acq = int(diff_ifgs/interval -1) ## number of missing ifgs
                            for i in range(n_no_acq):# the number of none depends on the number of missing ifgs
                                # Append the none to the existing array
                                category_arrays[category].append(None)
                        last_date[category] = date1_obj

                        # Check if the category is already in the dictionary
                        if category in category_arrays:
                            #If the category already exists, append the array to the existing array
                            category_arrays[category].append(array_data)
                        else:
                            #If the category doesn't exist, create a new list with the array
                            category_arrays[category] = [array_data]

    # appending None to the category_arrays where there are missing ifgs in the end of time-series
    for category in sorted(category_arrays):
        while((last_date[category]+timedelta(category)) < last_acq):
            category_arrays[category].append(None)
            last_date[category] = last_date[category] + timedelta(interval)

   # appending None to the category_arrays where there are missing ifgs in the begining of time-series
   # max number of expected 6-day(i.e. interval) interferograms in the full time-series
    max_ifg_number =  abs(int((first_acq - last_acq).days/interval))

    # all the categories in the data e.g. 6, 12, 18 etc
    cat=[]
    for category in category_arrays:
        cat.append(category)

    for i in range(interval, max(cat)+1, interval):
        if i in cat:
            #print('the ith category is ', i)
            while len(category_arrays[i]) < (max_ifg_number + 1 - i/interval):
                category_arrays[i] = np.insert(category_arrays[i], 0, None, axis=0)



#    # After processing all interferograms, generate the report
#    print("\nReport on the number of IFGs and missing IFGs per category:")
#    for category in sorted(category_arrays):
#        num_ifgs = sum(1 for item in category_arrays[category] if item is not None)  # Count valid IFGs
#        num_missing_ifgs = sum(1 for item in category_arrays[category] if item is None)  # Count missing IFGs
        
#        print(f"{category}-days: Number of IFGs: {num_ifgs}   Number of missing IFGs: {num_missing_ifgs}")


    print("\nFinished reading all interferograms.")

    return category_arrays
###########################################################################################################################
##########################################################################################################################

def read_coh(start, end, min_baseline, max_baseline, nlook, landmask, interval, LiCSAR_data):
    # frame = '131A_05336_121011'
    # start = '20200522'
    # end = '20201226'

    start_date = datetime.strptime(start, "%Y%m%d")
    end_date = datetime.strptime(end, "%Y%m%d")

    ############ reading

    # Root directory where the category directories are located
    if LiCSAR_data == 'yes':
        track=frame[0:3]

        if track[0]=='0':
            track=track[1:3]
        if track[0]=='0':
            track=track[1:2]

        root_directory = '/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/' + track + '/' + frame + '/interferograms/'

    else:
        root_directory = root_path 
       #root_directory = '/work/scratch-pw3/earyma/022D_04826_121209_subset/GEOC/' # one off

    req_file_name="geo.cc.tif"

    ##### Count total files matching the criteria for accurate progress calculation

    total_files = sum(1 for dirpath, _, filenames in os.walk(root_directory) for filename in filenames if filename.endswith(req_file_name))
    processed_files = 0

    print("Reading coherence data: [", end="", flush=True)
    #####



    # Initialize an empty dictionary to store the arrays for each category
    category_arrays = {}
    last_date = {}
    n=0

    if landmask != None and LiCSAR_data == 'yes':
        landmask_path = '/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/' + track + '/' + frame + '/metadata/' + frame + '.geo.landmask.tif'
        #one off
        #landmask_path = '/work/scratch-pw3/earyma/022D_04826_121209_subset/022D_04826_121209.geo.landmask.tif'

        landmask_dataset = gdal.Open(landmask_path)
        landmask_image = landmask_dataset.ReadAsArray()
        array_landmask = np.array(landmask_image)



    n = 1
    # Traverse the root directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Iterate through the files in the current directory
        for filename in filenames:
            # Check if the file ends with "geo.diff_pha.tif"




            if filename.endswith(req_file_name):
                file_path = os.path.join(dirpath, filename)


                #### for printing into the console
                processed_files += 1
                percentage = (processed_files / total_files) * 100

                # Clear previous percentage and print updated one
                sys.stdout.write("\rReading coherence data: [{:<50}] {}%".format(
                    "=" * int(percentage // 2), int(percentage)
                ))
                sys.stdout.flush()

               #####




                # Extract the dates from the file name
                match = re.search(r"(\d+)_(\d+)", filename)
                if match:
                    date1 = match.group(1)
                    date2 = match.group(2)

                    # Convert the dates to datetime objects
                    date1_obj = datetime.strptime(date1, "%Y%m%d")
                    date2_obj = datetime.strptime(date2, "%Y%m%d")

                    # Check if the dates fall within the specified range
                    category = (date2_obj - date1_obj).days

                    #if start_date <= date1_obj <= end_date and start_date <= date2_obj <= end_date and category > max_baseline and category < (max_baseline + min_baseline) or category < min_baseline:
                    if start_date <= date1_obj <= end_date and start_date <= date2_obj <= end_date and category < max_baseline and category > min_baseline:
                        if n==1: # finding the first and last acquisiton in the time-series
                            first_acq = date1_obj
                        else:
                            last_acq = date2_obj
                        n = n + 1
                        #print(file_path)

                        # Calculate the difference in days between the two dates


                        # Open the TIFF file
                        tiff_dataset = gdal.Open(file_path)

                        # Read the image data as a NumPy array
                        tiff_image = tiff_dataset.ReadAsArray()

                        # Convert the image to a numpy array
                        array_data = np.array(tiff_image)
                        # Convert array_data to a float type that can accommodate np.nan
                        array_data = array_data.astype(np.float32)

                        if landmask != 0:
                            array_data[array_landmask != 1]=np.nan

                        if nlook != None and nlook != 1: # incase of unwrap data because they are already multilooked to 10 we don't multilook them here
                            array_data = multilook(array_data,nlook)



                        if category not in last_date:
                            last_date[category] = date1_obj
                            #if (date1_obj - date2_obj) > interval

                        # appending None to the category_arrays where there are missing ifgs in the middle of time-series
                        diff_ifgs = (date1_obj - last_date[category]).days #/category #difference between the first epochs of two consecutive ifg
                        if diff_ifgs > interval:
                            n_no_acq = int(diff_ifgs/interval -1) ## number of missing ifgs
                            for i in range(n_no_acq):# the number of none depends on the number of missing ifgs
                                # Append the none to the existing array
                                category_arrays[category].append(None)
                        last_date[category] = date1_obj

                        # Check if the category is already in the dictionary
                        if category in category_arrays:
                            #If the category already exists, append the array to the existing array
                            category_arrays[category].append(array_data)
                        else:
                            #If the category doesn't exist, create a new list with the array
                            category_arrays[category] = [array_data]

    # appending None to the category_arrays where there are missing ifgs in the end of time-series
    for category in sorted(category_arrays):
        while((last_date[category]+timedelta(category)) < last_acq):
            category_arrays[category].append(None)
            last_date[category] = last_date[category] + timedelta(interval)

   # appending None to the category_arrays where there are missing ifgs in the begining of time-series
   # max number of expected 6-day(i.e. interval) interferograms in the full time-series
    max_ifg_number =  abs(int((first_acq - last_acq).days/interval))

    # all the categories in the data e.g. 6, 12, 18 etc
    cat=[]
    for category in category_arrays:
        cat.append(category)

    for i in range(interval, max(cat)+1, interval):
        if i in cat:
            while len(category_arrays[i]) < (max_ifg_number + 1 - i/interval):
                category_arrays[i] = np.insert(category_arrays[i], 0, None, axis=0)
    print("\nFinished reading all coherence data.")

    return category_arrays

##########################################################################################################################
##########################################################################################################################

all_ifgs = read_ifgs(start, end, min_baseline, max_baseline, landmask, nlook, interval, LiCSAR_data, filtered_ifgs)
# Define the directory path
directory_path = os.path.join(output_path, 'Data')

# Create the directory if it doesn't exist
os.makedirs(directory_path, exist_ok=True)

# Define the file path
file_path = os.path.join(directory_path, f'_all_ifgs_{start}_{end}')

# for writing the all_loops
file_path = output_path + '/Data/' + 'All_ifgs_' + start + '_' + end + '.pkl'
with open(file_path, 'wb') as file:  # for writting a dictionary to a file
    pickle.dump(all_ifgs, file)
################################################################

all_coh = read_coh(start, end, min_baseline, max_baseline, landmask, nlook, interval, LiCSAR_data)


## for writting the all_coh
file_path = output_path + '/Data/' + 'All_coh_' + start + '_' + end + '.pkl'
with open(file_path, 'wb') as file:  # for writting a dictionary to a file
    pickle.dump(all_coh, file)

################################################

# After processing all interferograms, generate the report
print("\nReport on the number of IFGs and missing IFGs per category:")
for category in sorted(all_ifgs):
    num_ifgs = sum(1 for item in all_ifgs[category] if item is not None)  # Count valid IFGs
    num_missing_ifgs = sum(1 for item in all_ifgs[category] if item is None)  # Count missing IFGs
    print(f"{category}-days: Number of IFGs: {num_ifgs}   Number of missing IFGs: {num_missing_ifgs}")



