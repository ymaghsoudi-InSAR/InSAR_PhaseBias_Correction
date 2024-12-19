# InSAR Phase Bias Correction
This repository contains scripts for mitigating phase bias in InSAR data. It includes tools for reading data, calculating loop closures, estimating calibration parameters, performing inversion, and correcting interferograms.

## Directory Structure
- `PhaseBias_01_Read_Data.py`: Reads input wrapped ifgs and coherence data.
- `PhaseBias_02_Loop_Closures.py`: Calculates loop closures.
- `PhaseBias_03_calibration_pars.py`: Estimates the calibration parameters $a_n$.
- `PhaseBias_04_Inversion.py`: Performs inversion for phase bias estimation.
- `PhaseBias_05_Correction.py`: Corrects interferograms using estimated biases.
- `bin/`: Contains auxiliary functions used by the scripts.
- `config.txt`: Configuration file for setting parameters.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/InSAR_PhaseBias_Correction.git
   cd InSAR_PhaseBias_Correction

## Test Dataset

The test dataset used in this repository originates from one of the islands in the Azores. The full dataset can be accessed through the COMET LiCSAR frame **082D_05125_020000**, available at the following link:

[https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/82/082D_05125_020000/](https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/82/082D_05125_020000/)

Additionally, we provide a **sample dataset**, including example interferograms, which can be downloaded from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14525360.svg)](https://doi.org/10.5281/zenodo.14525360)


# Citation
If you use this repository for your research, please cite the following paper:

Maghsoudi, Y., Hooper, A.J., Wright, T.J., Lazecky, M., & Ansari, H. (2022). Characterizing and correcting phase biases in short-term, multilooked interferograms. Remote Sensing of Environment, 275, 113022.
https://doi.org/10.1016/j.rse.2022.113022


# Acknowledgment
This work was funded by the European Space Agency (ESA) as part of the Phase Bias Correction Project. We gratefully acknowledge their support.

