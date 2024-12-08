#!/usr/bin/env python3

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pickle
import math
import time
#from read_orig_ifgs_coh import read_orig_ifgs_coh
import rasterio

def ifg_cc_2tif(in_file, out_file, template_tif ):

    # Open the existing GeoTIFF file using rasterio
    with rasterio.open(template_tif) as src:
        # Get the geospatial information from the existing GeoTIFF file
        crs = src.crs  # Coordinate Reference System (CRS)
        transform = src.transform  # Affine transform

        # Get the shape (rows and columns) from the existing GeoTIFF
        rows, cols = src.shape
        # Resample rows and cols by a factor of 10

        # Create the output GeoTIFF file and write the data
        with rasterio.open(out_file, 'w', driver='GTiff', width=cols, height=rows, count=1, dtype=in_file.dtype, crs=crs, transform=transform) as dst:
            dst.write(in_file, 1)


