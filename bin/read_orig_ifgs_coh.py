#!/usr/bin/env python3

import numpy as np



#desired_lengths_long = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360]

def read_orig_ifgs_coh(ifgs, coh, desired_categories):
    all_images = []
    existing_ifgs_index=[]
    counter=0
    # Sort the items of the ifgs dictionary based on keys
    sorted_ifgs_items = sorted(ifgs.items(), key=lambda x: x[0]) 

    # Iterate over categories of ifgs
    for category, images in sorted_ifgs_items: #ifgs.items():
        #print('ifg categories =', category)
        if category in desired_categories:

            #print('category in desired category: ', category)
            #print('images = ', images)
            # Get the shape of the first non-None image
            non_none_image_shape = next(img for img in images if img is not None).shape
            ### this part is for finding the indices of the existng ifgs
            for img in images:
                #print('counter = ', counter)
                if img is not None:
                    #print('counter exist')
                    existing_ifgs_index.append(counter)
                    counter += 1
                else:
                    counter += 1
            ####
            # Replace None values with NaN-filled images
            images_with_nan = [img if img is not None else np.full(non_none_image_shape, np.nan) for img in images]

            # Stack the images into a numpy array
            stacked_images = np.stack(images_with_nan, axis=0)

            # Append the stacked images to the list
            all_images.append(stacked_images)

    # Stack the list of arrays into a single numpy array
    all_ifgs = np.concatenate(all_images, axis=0)
    #print('existing_ifgs_index = ', existing_ifgs_index)
    #print('shape_existing_ifgs_index = ', np.shape(existing_ifgs_index))



    all_coh = []

    # Sort the items of the coh dictionary based on keys
    sorted_coh_items = sorted(coh.items(), key=lambda x: x[0])


    # Iterate over categories of coh
    for category, images in sorted_coh_items: #coh.items():
        #print('coh categories =', category)

        if category in desired_categories:

            # Get the shape of the first non-None image
            non_none_coh_shape = next(img for img in images if img is not None).shape

            # Replace None values with NaN-filled images
            coh_with_nan = [img if img is not None else np.full(non_none_coh_shape, np.nan) for img in images]

            # Stack the images into a numpy array
            stacked_coh = np.stack(coh_with_nan, axis=0)

            # Append the stacked images to the list
            all_coh.append(stacked_coh)

    # Stack the list of arrays into a single numpy array
    all_coh = np.concatenate(all_coh, axis=0)

    return all_ifgs, all_coh, existing_ifgs_index

