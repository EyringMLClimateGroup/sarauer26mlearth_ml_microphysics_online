# Data

## License
This data is licensed under the Creative Commons Attribution 4.0 International License (CC-BY 4.0).
See: http://creativecommons.org/licenses/by/4.0/

## Description
This directory contains a preprocessed subset of ICON r2b9 AMIP simulation output used to train,
validate, and test the ML microphysics parameterization. The data is stored in HDF5 format.

## File
- `df_mig_subset.h5` — contains three datasets: `train` (800k samples), `val` (100k samples), `test` (100k samples)

## Variables
### Inputs (8 features)
| Variable | Description |
|----------|-------------|
| `pf_mig` | Pressure (Pa) |
| `ta_mig` | Air temperature (K) |
| `qv_mig` | Specific humidity (kg/kg) |
| `qc_mig` | Cloud liquid water (kg/kg) |
| `qi_mig` | Cloud ice (kg/kg) |
| `qr_mig` | Rain water (kg/kg) |
| `qs_mig` | Snow (kg/kg) |
| `qg_mig` | Graupel (kg/kg) |

### Outputs (7 targets)
| Variable | Description |
|----------|-------------|
| `tend_ta_mig` | Temperature tendency (K/s) |
| `tend_qv_mig` | Specific humidity tendency (kg/kg/s) |
| `tend_qc_mig` | Cloud liquid water tendency (kg/kg/s) |
| `tend_qi_mig` | Cloud ice tendency (kg/kg/s) |
| `tend_qr_mig` | Rain water tendency (kg/kg/s) |
| `tend_qs_mig` | Snow tendency (kg/kg/s) |
| `tend_qg_mig` | Graupel tendency (kg/kg/s) |

## Preprocessing
The data was generated using `preprocessing/preprocess_combined.py`

## Source
ICON r2b9 AMIP simulation, coarse-grained to R02B05, 12 days in January–February 2020.
For details see the Methodology section of the accompanying paper:

Sarauer, Ellen, et al. "From stable online coupling to decade-long climate simulations with ICON:
A machine learning parameterization for cloud microphysics." Machine Learning: Earth (under review).

## Contact
Ellen Sarauer, German Aerospace Center (DLR), Institut für Physik der Atmosphäre
