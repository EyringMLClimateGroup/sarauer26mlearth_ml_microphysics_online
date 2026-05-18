# From stable online coupling to decade-long climate simulations with ICON: A machine learning parameterization for cloud microphysics

This repository contains the code for an online-coupled machine learning parameterization of cloud
microphysics in ICON — enabling stable, physically consistent, decade-long climate simulations.
By replacing the traditional microphysics scheme with a trained neural network, we achieve
significant speedups while preserving the physical behavior of the atmosphere.

The corresponding paper is currently under review in *Machine Learning: Earth*:

> Sarauer, Ellen, et al. "From stable online coupling to decade-long climate simulations with ICON:
> A machine learning parameterization for cloud microphysics."

---

## Repository content

| Directory | Description |
|-----------|-------------|
| [`data/`](data) | Preprocessed training dataset in HDF5 format (`df_mig_subset.h5`), split into training, validation, and test subsets |
| [`model_training/`](model_training) | ML model definition, training pipeline, trained checkpoints, evaluation metrics, and inference plots |
| [`preprocessing/`](preprocessing) | Scripts for vertical coarse-graining and data preprocessing, plus output distribution histograms |

## Environment Setup

The conda environment can be recreated from the provided `environment.yml` file. This will install
all required dependencies including PyTorch, scikit-learn, and FTorch.

```bash
conda env create --name ftorch_env --file environment.yml
conda activate ftorch_env
```

## Key dependencies

This project relies on the following core packages:

- **[PyTorch](https://github.com/pytorch/pytorch)** — all ML models are defined and trained in PyTorch
- **[scikit-learn](https://github.com/scikit-learn/scikit-learn)** — used for preprocessing and evaluation utilities
- **[FTorch](https://github.com/Cambridge-ICCS/FTorch)** — provides the Python–Fortran bridge for online coupling with ICON

## License

The code in this repository is licensed under the **Apache License 2.0**.
The sample data is licensed under **CC-BY 4.0**.
See the [License](License) file for details.
