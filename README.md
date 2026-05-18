# From stable online coupling to decade-long climate simulations with ICON: A machine learning parameterization for cloud microphysics
This repository contains the code for the online-coupled machine learning parameterization of cloud microphysics in ICON, enabling stable long-term climate simulations as presented in the paper.

The corresponding paper is currently under Review in Machine Learning: Earth
> Sarauer, Ellen, et al. "From stable online coupling to decade-long climate simulations with ICON: A machine learning parameterization for cloud microphysics."

## Repository content
- [data](data): contains the preprocessed training dataset in HDF5 format (`df_mig_subset.h5`), split into training, validation, and test subsets.
- [model_training](model_training): contains the ML model training and inference pipeline, including the model definition and training script (`fix_loss_mask_model.py`), trained model checkpoints (`checkpoint_fix.pth`, `constrained_regression_model_final_cpu_scripted.pt`), a checkpoint inspection utility (`read_checkpoint.py`), a job submission script (`submit_training.sh`), and evaluation outputs including metrics (`evaluation_log.json`) and inference plots (`inference_plots/`).
- [preprocessing](preprocessing): contains the data preprocessing pipeline, including scripts for vertical coarse-graining (`vertical_coarse.py`) and combined data preprocessing (`preprocess_combined.py`), and output distribution histograms (`data_histos/`).

## Environment Setup

The conda environment can be recreated from the provided `environment.yml` file. This will install all required dependencies including FTorch and its related packages.

```bash
conda env create --name ftorch_env --file environment.yml
conda activate ftorch_env
```

## Important packages
- all models are trained with Pytorch (https://github.com/pytorch/pytorch) and using Sklearn (https://github.com/scikit-learn/scikit-learn)
- we use FTorch (https://github.com/Cambridge-ICCS/FTorch) to build the python-fortran-bridge
