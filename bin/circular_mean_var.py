#!/usr/bin/env python3

import numpy as np
from scipy.special import i0, i1
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#def circular_mean_and_variance_over_epochs(phase_values, axis=0):
#    # Convert phase values to complex numbers
#    complex_values = np.exp(1j * phase_values)
#    
#    # Calculate the mean of complex numbers along the specified axis, ignoring NaN values
#    mean_complex = np.nanmean(complex_values, axis=axis)
#    
#    # Check if all values along the axis are NaN
#    if np.all(np.isnan(mean_complex)):
#        mean_phase = np.nan
#        variance = np.nan
#        std_dev = np.nan
#    else:
#        # Calculate the phase of the mean complex value
#        mean_phase = np.angle(mean_complex)
#        
#        # Ensure the phase is between -pi and pi
#        mean_phase = (mean_phase + np.pi) % (2 * np.pi) - np.pi
#        
#        # Calculate the magnitude of the mean complex value
#        r_bar = np.abs(mean_complex)
#        
#        # Calculate the concentration parameter
#        kappa = 1 / (1 - r_bar)
#        
#        # Calculate the variance using the 2nd moment of the Von Mises distribution
#        variance = 1 - i1(kappa) / i0(kappa)
#        
#        # Calculate the standard deviation from the variance
#        std_dev = np.sqrt(variance)
#    
#    return mean_phase, std_dev



def circular_mean_and_variance_over_epochs(phase_values, axis=0):
    # Convert phase values to complex numbers
    complex_values = np.exp(1j * phase_values)

    
    # Calculate the mean of complex numbers along the specified axis, ignoring NaN values
    mean_complex = np.nanmean(complex_values, axis=axis)

    
    mean_phase = np.angle(mean_complex)
    # Ensure the phase is between -pi and pi
    mean_phase = (mean_phase + np.pi) % (2 * np.pi) - np.pi
    
    # Calculate the magnitude of the mean complex value
    r_bar = np.abs(mean_complex)
    
    # Calculate the concentration parameter
    kappa = 1 / (1 - r_bar)

#    nan_count = np.sum(np.isnan(kappa))

    # Count the number of non-NaN values in kappa
#    non_nan_count = np.sum(~np.isnan(kappa))

#    print("Number of NaN values in kappa:", nan_count)
#    print("Number of non-NaN values in kappa:", non_nan_count)

    # Calculate the variance using the 2nd moment of the Von Mises distribution
    #variance = (1 - i1(kappa)) / i0(kappa)
    variance = 1 - i1(kappa) / i0(kappa) 
    std_dev = 2*np.sqrt(variance)

    return mean_phase, std_dev

