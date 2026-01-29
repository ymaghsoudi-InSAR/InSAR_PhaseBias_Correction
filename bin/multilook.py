#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import warnings



def multilook(array, nlook, n_valid_thre=0.5):
        
    length, width = array.shape
    
    length_ml = int(np.floor(length/nlook))
    width_ml = int(np.floor(width/nlook))
    
    array_reshape = array[:length_ml*nlook,:width_ml*nlook].reshape(length_ml, nlook, width_ml, nlook)
    
    with warnings.catch_warnings(): ## To silence RuntimeWarning: Mean of empty slice
        warnings.simplefilter('ignore', RuntimeWarning)
        array_ml = np.nanmean(array_reshape, axis=(1, 3))
    
    n_valid = np.sum(~np.isnan(array_reshape), axis=(1, 3))
    bool_invalid = n_valid < n_valid_thre*nlook*nlook
    
    array_ml[bool_invalid] = np.nan
    
    return array_ml

