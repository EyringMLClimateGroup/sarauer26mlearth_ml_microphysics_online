######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: preprocess_combined.py                                                                   #
######################################################################################################
# In this script we preprocess our data for the combined ML microphysics Model.                      #
# We load our netcdf simulation file and apply preselection criteria.                                #
# We split data in test, train and validation sets and save them.                                    #
# For more information, please check Methodology section in our paper.                               #
######################################################################################################

# Import
import xarray as xr
import h5py
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

# Load Data
data_path = ".../experiments/r2b9_amip/coarse-grained-data/"
dates = [
    "20200126",
    "20200127",
    "20200128",
    "20200129",
    "20200131",
    "20200201",
    "20200202",
    "20200203",
    "20200205",
    "20200206",
    "20200207",
    "20200208"
]
atm_3d_general_vars_path = []
atm_mig_inputs_path = []
atm_mig_tendencies_path = []

for date in dates:
    atm_3d_general_vars_path += glob.glob(data_path + f"*atm_cl_ml_{date}*")
    atm_mig_inputs_path += glob.glob(data_path + f"*mig_inputs_ml_{date}*")
    atm_mig_tendencies_path += glob.glob(data_path + f"*mig_tendencies_ml_{date}*")


def create_data_array(path_to_files, varname):
    """Create data arrays from nc files, using all vertical levels"""
    joined_arr = np.zeros((0,), dtype=np.float32)  # Start with empty array

    for path in path_to_files:
        file = xr.open_dataset(path)
        raw_arr = file[varname]
        arr_np = raw_arr.values      # Convert to NumPy array
        flat_arr = arr_np.reshape(-1)  # Flatten all dimensions into 1D
        joined_arr = np.concatenate((joined_arr, flat_arr))

    print(f"Final array shape of {varname}: {joined_arr.shape}")
    return joined_arr


# Load the original tendency file
ds_tendencies = xr.open_dataset(".../r2b9_amip_atm_mig_tendencies_ml_20200203T000000Z_R02B05.nc")

# Print diagnostic information about tend_qr_mig
print("\nOriginal tendency file diagnostics:")
print("Variable attributes:", ds_tendencies['tend_qr_mig'].attrs)
print("Dimensions:", ds_tendencies['tend_qr_mig'].dims)
print("Shape:", ds_tendencies['tend_qr_mig'].shape)

# Basic statistics for different dimensions
print("\nStatistics across different dimensions:")

# Create input and output arrays with correct variable names
input_vars = ["pf_mig", "ta_mig", "qv_mig", "qc_mig", "qi_mig", "qr_mig", "qs_mig", "qg_mig"]
output_vars = ["tend_ta_mig", "tend_qv_mig", "tend_qc_mig", "tend_qi_mig", "tend_qr_mig", "tend_qs_mig", "tend_qg_mig"]

# Create variable name mapping for tendencies
tendency_mapping = {
    "tend_qv_mig": "tend_qhus_mig",
    "tend_qc_mig": "tend_qclw_mig",
    "tend_qi_mig": "tend_qcli_mig",
    "tend_qr_mig": "tend_qr_mig",
    "tend_qs_mig": "tend_qs_mig",
    "tend_qg_mig": "tend_qg_mig",
    "tend_ta_mig": "tend_ta_mig"
}

# Create DataFrames with correct variable names
df_data = {}
for var in input_vars:
    df_data[var] = create_data_array(atm_mig_inputs_path, var)
for var in output_vars:
    nc_var_name = tendency_mapping[var]  # Get the correct NetCDF variable name
    df_data[var] = create_data_array(atm_mig_tendencies_path, nc_var_name)

df_mig = pd.DataFrame(df_data)

def analyze_and_preprocess_data(df, output_vars):
    """Preprocess data and correct tendencies to ensure physical consistency."""
    print("\nChecking temperature-based phase transitions...")
    
    # Temperature thresholds (Kelvin)
    T_ice = 233.15    # -40°C - ice formation dominant
    T_mixed = 278.15  # 5°C - mixed phase possible
    
    # Create phase regime masks
    ice_mask = df['ta_mig'] <= T_ice
    mixed_mask = (df['ta_mig'] > T_ice) & (df['ta_mig'] <= T_mixed)
    liquid_mask = df['ta_mig'] > T_mixed
    
    violations = {'ice': 0, 'mixed': 0, 'liquid': 0}
    total_samples = len(df)
    
    # Ice regime checks (T ≤ -40°C)
    ice_violations = (ice_mask & 
                     ((df['tend_qc_mig'] > 0) |  # No liquid water formation
                      (df['tend_qr_mig'] > 0)))  # No rain formation
    
    df.loc[ice_violations, ['tend_qc_mig', 'tend_qr_mig']] = 0.0
    violations['ice'] += ice_violations.sum()
    # Print phase transition violation statistics
    print("\nPhase transition violation statistics:")
    for phase, count in violations.items():
        print(f"{phase.capitalize()} phase violations: {count} ({count/total_samples*100:.4f}%)")
    
    print("\nPhase distribution in dataset:")
    print(f"Ice phase samples: {ice_mask.sum()} ({ice_mask.sum()/total_samples*100:.2f}%)")
    print(f"Mixed phase samples: {mixed_mask.sum()} ({mixed_mask.sum()/total_samples*100:.2f}%)")
    print(f"Liquid phase samples: {liquid_mask.sum()} ({liquid_mask.sum()/total_samples*100:.2f}%)")

    # Continue with mass conservation checks
    print("\nAnalyzing mass conservation...")
    
    # Map tendencies to their corresponding mass variables
    tendency_to_mass = {
        'tend_qv_mig': 'qv_mig',
        'tend_qc_mig': 'qc_mig',
        'tend_qi_mig': 'qi_mig',
        'tend_qr_mig': 'qr_mig',
        'tend_qs_mig': 'qs_mig',
        'tend_qg_mig': 'qg_mig'
    }
    
    total_samples = len(df)
    negative_corrections = 0
    
    for tend_var, mass_var in tendency_to_mass.items():
        if tend_var in output_vars:
            min_mass = 0
            # Calculate final masses
            current_mass = df[mass_var].values
            tendency = df[tend_var].values
            final_mass = current_mass + tendency
            
            # First correction: Ensure non-negative masses
            violations = final_mass < min_mass
            correction_indices = df.index[np.where(violations)[0]]
            
            if len(correction_indices) > 0:
                df.loc[correction_indices, tend_var] = min_mass - df.loc[correction_indices, mass_var]
                negative_corrections += len(correction_indices)
                print(f"\n{tend_var}:")
                print(f"Corrected {len(correction_indices)} negative mass violations ({len(correction_indices)/total_samples*100:.4f}%)")

    # # Second correction: Ensure mass conservation (we do not apply this correction in the study, but it is implemented here for completeness)
    # mass_vars = list(tendency_to_mass.values())
    # tend_vars = list(tendency_to_mass.keys())

    # initial_total_mass = df[mass_vars].sum(axis=1)
    # final_masses = df[mass_vars].values + df[tend_vars].values
    # final_total_mass = final_masses.sum(axis=1)

    # # Find where mass is not conserved
    # mass_diff = initial_total_mass - final_total_mass
    # mass_violation_mask = abs(mass_diff) > 1e-10  # Tolerance for floating point errors
    # mass_violation_indices = df.index[mass_violation_mask]
    # print(f"\nFound {len(mass_violation_indices)} mass conservation violations "
    #     f"({len(mass_violation_indices)/total_samples*100:.4f}%)")

    # if len(mass_violation_indices) > 0:
    #     conservation_corrections = len(mass_violation_indices)

    #     print("\nDiagnostics before corrections:")
    #     violation_magnitudes = np.abs(mass_diff[mass_violation_mask])
    #     print(f"Mean violation magnitude: {np.mean(violation_magnitudes):e}")
    #     print(f"Max violation magnitude: {np.max(violation_magnitudes):e}")
    #     print(f"Min violation magnitude: {np.min(violation_magnitudes):e}")

    #     # Plot histogram
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(violation_magnitudes, bins=50, log=True)
    #     plt.title("Mass Conservation Violation Magnitudes")
    #     plt.xlabel("Absolute Difference in Total Mass")
    #     plt.ylabel("Count (log scale)")
    #     plt.savefig("mass_violation_histogram.png")
    #     plt.close()

    #     print("\nCorrecting mass conservation violations...")

    #     #  Weighted Redistribution 
    #     tendency_stds = df[tend_vars].std().values  # (num_tend_vars,)
    #     base_weights = 1.0 / tendency_stds          # (num_tend_vars,)

    #     # Mass difference for violating samples
    #     mass_diff_vals = mass_diff[mass_violation_mask].values  # (N,)

    #     # Extract tendencies for violating samples
    #     tendencies = df.loc[mass_violation_indices, tend_vars].values  # (N, T)

    #     # Determine which tendencies are active (nonzero)
    #     active_mask = np.abs(tendencies) > 0  # (N, T)

    #     # Expand weights for batch
    #     weights_batched = np.broadcast_to(base_weights, tendencies.shape)  # (N, T)

    #     # Zero out inactive tracers
    #     weights_active = weights_batched * active_mask  # (N, T)

    #     # Normalize per sample (avoid division by zero)
    #     weights_sum = weights_active.sum(axis=1, keepdims=True)  # (N, 1)
    #     weights_sum[weights_sum == 0] = 1.0  # prevent div/0 for all-inactive
    #     norm_weights = weights_active / weights_sum  # (N, T)

    #     # Compute per-sample corrections
    #     corrections = mass_diff_vals[:, np.newaxis] * norm_weights  # (N, T)

    #     # Apply corrections
    #     df.loc[mass_violation_indices, tend_vars] += corrections

    #     # Verify conservation
    #     final_masses_after = df[mass_vars].values + df[tend_vars].values
    #     final_total_mass_after = final_masses_after.sum(axis=1)
    #     still_violated = np.abs(final_total_mass_after - initial_total_mass) > 1e-10
    #     if np.any(still_violated):
    #         print(f"Warning: {np.sum(still_violated)} violations remain after correction")

    # # Summary
    # print(f"\nTotal corrections summary:")
    # print(f"Negative mass corrections: {negative_corrections} ({negative_corrections/total_samples*100:.4f}%)")
    # print(f"Mass conservation corrections: {conservation_corrections} ({conservation_corrections/total_samples*100:.4f}%)")

    # Final max conservation error
    #final_total_mass = df[mass_vars].values + df[tend_vars].values
    # max_conservation_error = np.max(np.abs(final_total_mass.sum(axis=1) - initial_total_mass))
    # print(f"Maximum conservation error after corrections: {max_conservation_error:e}")

    return df



# Main preprocessing pipeline
print("Starting preprocessing...")

# Filter basic conditions and remove NaN
df_mig = df_mig.dropna()
df_mig = df_mig[(df_mig["ta_mig"] > 10.) & (df_mig["pf_mig"] > 10.)
                & (df_mig["qv_mig"] >= 0.) & (df_mig["qc_mig"] >= 0.)
                & (df_mig["qi_mig"] >= 0.) & (df_mig["qr_mig"] >= 0.)
                & (df_mig["qs_mig"] >= 0.) & (df_mig["qg_mig"] >= 0.)]
print(f"Samples after basic filtering: {len(df_mig)}")
df_mig = df_mig[(df_mig["ta_mig"] > 150.) & (df_mig["pf_mig"] > 10000.)]
print(f"Samples after pressure level filtering: {len(df_mig)}")

# Apply preprocessing
df_processed = analyze_and_preprocess_data(df_mig, output_vars)

# Dataset splitting strategy
train_size = int(20e6)
val_size   = int(1e6)
test_size  = int(1e6)

# # Outlier budget for training
# Outlier sampling: 90% normal, 10% outliers (tail samples across all output vars)
tail_quantile = 0.1  # 10th and 90th percentile thresholds

# Build outlier mask across all output variables
outlier_mask = np.zeros(len(df_processed), dtype=bool)
for var in output_vars:
    lower, upper = df_processed[var].quantile([tail_quantile, 1 - tail_quantile])
    outlier_mask |= (df_processed[var] < lower) | (df_processed[var] > upper)

normal_mask = ~outlier_mask

outlier_df = df_processed[outlier_mask]
normal_df  = df_processed[normal_mask]

print("\nDiagnostics:")
print(f"Outliers : {len(outlier_df):,}")
print(f"Normals  : {len(normal_df):,}")

# Sample 90/10 split
n_outliers = min(len(outlier_df), int(train_size * 0.1))
n_normal   = train_size - n_outliers

print(f"\nSampling {n_outliers:,} outliers and {n_normal:,} normal samples")

train_outliers = outlier_df.sample(n=n_outliers, random_state=42, replace=False)
train_normal   = normal_df.sample(n=n_normal,   random_state=42, replace=False)

train_df = pd.concat([train_outliers, train_normal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Validation + test from remaining
remaining_df = df_processed.drop(train_outliers.index.union(train_normal.index))
val_df = remaining_df.sample(n=val_size, random_state=42, replace=False)
remaining_df = remaining_df.drop(val_df.index)
test_df = remaining_df.sample(n=test_size, random_state=42, replace=False)

# Convert to numpy arrays and save
train_array = train_df.to_numpy()
val_array = val_df.to_numpy()
test_array = test_df.to_numpy()

print("\nFinal dataset sizes:")
print(f"Training: {len(train_array)}")
print(f"Validation: {len(val_array)}")
print(f"Test: {len(test_array)}")


# Generate histograms
def plot_histogram(data, var_name, bins=100):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', alpha=0.7, log=True)
    plt.title(f"Histogram of {var_name}")
    plt.xlabel(var_name)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)

# Save histograms
for var in input_vars:
    plot_histogram(df_processed[var], var)
    plt.savefig(f"data_histos/histogram_{var}.png")
for var in output_vars:
    plot_histogram(df_processed[var], var)
    plt.savefig(f"data_histos/histogram_{var}.png")


# Save to HDF5
save_path = ".../data/"

with h5py.File(f"{save_path}df_mig.h5", "w") as h5f:
    h5f.create_dataset("train", data=train_array, compression="gzip")
    h5f.create_dataset("val", data=val_array, compression="gzip")
    h5f.create_dataset("test", data=test_array, compression="gzip")

print(f"\nSaved HDF5 file to: {save_path}df_mig.h5")