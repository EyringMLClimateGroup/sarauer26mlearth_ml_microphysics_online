# From stable online coupling to decade-long climate simulations with ICON: A machine learning parameterization for cloud microphysics

This repository contains the code for an online-coupled machine learning parameterization of cloud
microphysics in ICON to enable stable, physically consistent, decade-long climate simulations.
By replacing the traditional microphysics scheme with a trained neural network, we achieve
stable coupling and preserve the physical behavior of the atmosphere.

The corresponding paper is currently under review in *Machine Learning: Earth*:

> Sarauer, Ellen, et al. "From stable online coupling to decade-long climate simulations:
> A machine learning parameterization for cloud microphysics in ICON."

[![DOI](https://zenodo.org/badge/1242257476.svg)](https://doi.org/10.5281/zenodo.20442527)

---

## Repository content

| Directory | Description |
|-----------|-------------|
| [`data/`](data) | Preprocessed training dataset in HDF5 format (`df_mig_subset.h5`), split into training, validation, and test subsets. See [`data/Readme.md`](data/Readme.md) for details on variables and preprocessing steps. |
| [`model_training/`](model_training) | ML model definition, training pipeline, trained checkpoints, evaluation metrics, and inference plots |
| [`preprocessing/`](preprocessing) | Scripts for vertical coarse-graining and data preprocessing, plus output distribution histograms |

## Script usage

### `preprocessing/`

- **`preprocess_combined.py`** - Main preprocessing script. Loads raw ICON NetCDF output, applies
  physical consistency filters (e.g. non-negative mixing ratios, phase-transition corrections),
  and splits the data into training, validation, and test samples using an
  outlier-aware 90/10 sampling strategy. Output is saved as `data/df_mig_subset.h5`.
- **`vertical_coarse.py`** - Performs vertical coarse-graining of the raw ICON simulation output
  prior to preprocessing.

### `model_training/`

- **`fix_loss_mask_model.py`** - Defines the constrained regression model architecture and runs
  the training pipeline, including a physics-informed loss mask to enforce non-negative
  hydrometeor tendencies. To start training, adapt the data path and submit via `submit_training.sh`.
- **`submit_training.sh`** - SLURM job submission script for running the training on an HPC cluster.
- **`read_checkpoint.py`** - Utility script to inspect a saved model checkpoint, e.g. to verify
  layer shapes or confirm the model was saved correctly.

## Environment Setup

The conda environment can be recreated from the provided `environment.yml` file. This will install
all required dependencies including PyTorch, scikit-learn, and FTorch.

```bash
conda env create --name ftorch_env --file environment.yml
conda activate ftorch_env
```

## Figures
Figures presented in the paper were generated using
[ICONEval](https://github.com/EyringMLClimateGroup/ICONEval).

## Key dependencies

This project relies on the following core packages:

- **[PyTorch](https://github.com/pytorch/pytorch)** - all ML models are defined and trained in PyTorch
- **[scikit-learn](https://github.com/scikit-learn/scikit-learn)** - used for preprocessing and evaluation utilities
- **[FTorch](https://github.com/Cambridge-ICCS/FTorch)** - provides the Python–Fortran bridge for online coupling with ICON

## License

The code in this repository is licensed under the **Apache License 2.0**.
The sample data is licensed under **CC-BY 4.0**.
See the [License](License) file for details.
