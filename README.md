# InSAR_PhaseBias_Correction

# InSAR Phase Bias Correction
This repository contains scripts for mitigating phase bias in InSAR data. It includes tools for reading data, calculating loop closures, estimating calibration parameters, performing inversion, and correcting interferograms.

## Directory Structure
- `PhaseBias_01_Read_Data.py`: Reads input wrapped ifgs and coherence data.
- `PhaseBias_02_Loop_Closures.py`: Calculates loop closures.
- `PhaseBias_03_calibration_pars.py`: Estimates the calibration parameters an.
- `PhaseBias_03_Inversion.py`: Performs inversion for phase bias estimation.
- `PhaseBias_04_Correction.py`: Corrects interferograms using estimated biases.
- `bin/`: Contains auxiliary functions used by the scripts.
- `config.txt`: Configuration file for setting parameters.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/InSAR_PhaseBias_Correction.git
   cd InSAR_PhaseBias_Correction
