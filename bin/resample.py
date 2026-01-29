#!/usr/bin/env python3


import rasterio
from rasterio.enums import Resampling
import os
frame = '022D_04826_121209'
frame ='145A_05047_000706'
frame ='082D_05128_030500'
frame = '050D_05049_060600'
frame = '027A_04532_191920'
frame='130A_05394_131213' # jess iran

track=frame[0:3]
if track[0]=='0':
    track=track[1:3]
if track[0]=='0':
    track=track[1:2]



def resample_geotiff(input_path, output_path, scale_factor):


    with rasterio.open(input_path) as src:


        # Get the original geotransform and metadata
        transform = src.transform
        metadata = src.meta.copy()

        # Update the metadata with the new dimensions
        metadata['width'] = int(src.width / scale_factor)
        metadata['height'] = int(src.height / scale_factor)

        # Update the transform with the new scale
        metadata['transform'] = rasterio.Affine(transform.a * scale_factor, transform.b, transform.c,
                                               transform.d, transform.e * scale_factor, transform.f)

        # Resample the data
        data = src.read(
            out_shape=(metadata['count'], metadata['height'], metadata['width']),
            resampling=Resampling.bilinear
        )

    # Save the resampled GeoTIFF
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(data)

# Example usage
#input_file = ".geo.landmask.tif"
#frame_tif = "/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/" + track + "/" + frame + "/" + "/metadata/" + frame + input_file
#resampled_tif = frame + ".resampled" + input_file
#resample_factor = 10

# one off
#frame_tif='022D_04826_121209_subset.geo.hgt.tif'
#resampled_tif='022D_04826_121209_subset.resampled.geo.hgt.tif'
#resample_factor = 8

#resample_geotiff(frame_tif, resampled_tif, resample_factor)

