######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: fix_loss_mask_model.py                                                                   #
######################################################################################################
# This script includes the whole ML model definition and training pipeline                           #
######################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, confusion_matrix
import seaborn as sns
import os
import time
import h5py
import json
import torch.nn.functional as F

#####################################
# Useful functions

def weighted_mse_loss(y_pred, y_true):
    abs_error = torch.abs(y_pred - y_true)
    weight = torch.where(abs_error > 0.1, 20.0, 1.0)
    return torch.mean(weight * (y_pred - y_true) ** 2)

def unwrap_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def prepare_data(data):
    # Define per-feature thresholds
    thresholds = np.full(data.shape[1], 1e-10)
    thresholds[0] = 0
    thresholds[1:4] = 0

    # Binary mask: 1 where data is considered nonzero (above threshold)
    class_targets = (np.abs(data) > thresholds).astype(int)

    scaled_data = np.zeros_like(data)

    for i in range(data.shape[1]):
        nonzero_mask = np.abs(data[:, i]) > thresholds[i]

        if nonzero_mask.any():
            mean = np.mean(data[nonzero_mask, i])
            std = np.std(data[nonzero_mask, i])
            scaled_feature = (data[:, i] - mean) / std

            # Zero out entries not considered active
            scaled_feature[class_targets[:, i] == 0] = 0.0
            scaled_data[:, i] = scaled_feature
        else:
            scaled_data[:, i] = 0.0

    return class_targets, scaled_data


# Compute stats for inputs and outputs
def compute_nonzero_stats(data):
    """Compute mean and std of non-zero values"""
    # Convert to numpy if tensor
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    stats = []
    for i in range(data.shape[1]):
        nonzero_mask = (data[:, i] != 0)
        if nonzero_mask.any():
            mean = np.mean(data[nonzero_mask, i])
            std = np.std(data[nonzero_mask, i])
            stats.append((mean, std))
    return stats

def negative_mass_loss(reg_pred, x_input, model, y_class_batch):
    """Compute loss for negative mass predictions in unscaled space"""
    loss = torch.tensor(0.0, device=device)
    core_model = unwrap_model(model)
    # Unscale predictions and inputs to physical space
    reg_unscaled = reg_pred * core_model.output_stds_buf + core_model.output_means_buf
    x_unscaled = x_input * core_model.input_stds_buf + core_model.input_means_buf
    
    for i in range(1, 8):  # For each feature with residual connection
        input_idx = i
        output_idx = i-1
        
        # Only consider samples where the feature is active
        active_mask = y_class_batch[:, output_idx] == 1
        
        if active_mask.any():
            # Calculate total mass (input + tendency)
            total_mass = x_unscaled[active_mask, input_idx] + reg_unscaled[active_mask, output_idx]
            
            # Strong penalty for negative masses
            neg_mass_penalty = torch.relu(-total_mass) * 1e3
            
            if neg_mass_penalty.any():
                loss = loss + neg_mass_penalty.mean()
    
    return loss

def train_epoch(model, train_loader, optimizer, scheduler, phase='classification'):
    model.train()
    total_loss = class_losses = reg_losses = neg_mass_losses = 0
    n_batches = 0
    total_class_acc = torch.zeros(7, device=device)  # Assuming 7 features as the output
    
    for X_batch, y_class_batch, y_reg_batch in train_loader:
        X_batch = X_batch.to(device)
        y_class_batch = y_class_batch.to(device)
        y_reg_batch = y_reg_batch.to(device)
        
        optimizer.zero_grad()
        class_pred, reg_pred = model(X_batch, scale_input=True, unscale_output=False)
        core_model = unwrap_model(model)
        y_reg_scaled = (y_reg_batch - core_model.output_means_buf) / core_model.output_stds_buf
        
        # Classification loss and accuracy
        class_loss = 0
        for i in range(y_class_batch.shape[1]):
            feature_loss = nn.CrossEntropyLoss(reduction='none')(class_pred[:, i], y_class_batch[:, i])
            class_loss += feature_loss.mean()
            
            # Calculate accuracy per feature
            pred = torch.argmax(class_pred[:, i], dim=1)
            total_class_acc[i] += (pred == y_class_batch[:, i]).float().mean()
        
        class_loss /= y_class_batch.shape[1]
        
        # Regression loss with mask: only compute loss where y_class_batch != 0 (non-zero)
        reg_losses_per_feature = []
        for i in range(y_reg_batch.shape[1]):
            nonzero_mask = y_class_batch[:, i] != 0

            if nonzero_mask.any():
                masked_reg_pred = reg_pred[nonzero_mask, i]
                masked_y_reg_scaled = y_reg_scaled[nonzero_mask, i]

                # Calculate the feature loss (MSE loss)
                feature_loss = weighted_mse_loss(masked_reg_pred, masked_y_reg_scaled)

                # Apply scaling: normalize the loss based on the standard deviation of the target values
                feature_std = torch.std(masked_y_reg_scaled)
                scaled_feature_loss = feature_loss / (feature_std)
                # Append the scaled loss
                reg_losses_per_feature.append(scaled_feature_loss)

        # Calculate the final regression loss, averaging over all features
        reg_loss = torch.stack(reg_losses_per_feature).mean() if reg_losses_per_feature else torch.tensor(0.0, device=device)
        
        # Add negative mass loss
        neg_mass_loss_val = negative_mass_loss(reg_pred, X_batch, model, y_class_batch)
        
        loss = class_loss + reg_loss + neg_mass_loss_val
        
        loss.backward()
        optimizer.step()
        if hasattr(scheduler, 'total_steps') and scheduler._step_count < scheduler.total_steps:
            scheduler.step()

        
        # Accumulate losses
        total_loss += loss.item()
        class_losses += class_loss.item()
        reg_losses += reg_loss.item()
        neg_mass_losses += neg_mass_loss_val.item()
        n_batches += 1
    
    # Calculate average accuracies
    avg_acc_per_feature = total_class_acc / n_batches
    avg_acc = avg_acc_per_feature.mean().item()
    
    return (total_loss / n_batches, class_losses / n_batches, 
            reg_losses / n_batches, neg_mass_losses / n_batches, avg_acc)


def compute_r2_scores(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for X_batch, y_class_batch, y_reg_batch in data_loader:
            X_batch = X_batch.to(device)
            y_class_batch = y_class_batch.to(device)
            y_reg_batch = y_reg_batch.to(device)
            core_model = unwrap_model(model)
            # Get predictions in scaled space
            _, reg_pred = model(X_batch, scale_input=True, unscale_output=False)
            y_reg_scaled = (y_reg_batch - core_model.output_means_buf) / core_model.output_stds_buf
            
            # Move to CPU and convert to numpy immediately
            all_preds.append(reg_pred.cpu().numpy())
            all_targets.append(y_reg_scaled.cpu().numpy())
            all_masks.append(y_class_batch.cpu().numpy())
    
    # Concatenate numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    r2_scores = []
    for i in range(all_targets.shape[1]):
        mask = all_masks[:, i] == 1
        if mask.any():
            r2 = r2_score(all_targets[mask, i], all_preds[mask, i])
            r2_scores.append(r2)
    
    return r2_scores

# Evaluation function
def evaluate_model(model, X_test, y_test, y_test_class, output_stats, device="cuda"):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    results = {"features": []}  # container for JSON
    with torch.no_grad():
        # Convert test data to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        # Get predictions in scaled space
        class_pred, reg_pred = model(X_test_tensor, scale_input=True, unscale_output=False)
        
        # Convert to numpy for evaluation
        class_pred = torch.argmax(class_pred.reshape(-1, 2), dim=1).reshape(-1, y_test.shape[1])
        class_pred = class_pred.cpu().numpy()
        reg_pred = reg_pred.cpu().numpy()
        
        # Scale test targets using output_stats (not input_stats)
        y_test_scaled = np.zeros_like(y_test)
        for i in range(y_test.shape[1]):
            mean, std = output_stats[i]  # Use output_stats for regression targets
            y_test_scaled[:, i] = (y_test[:, i] - mean) / std
        
        print("\nDetailed evaluation:")
        for i in range(y_test.shape[1]):
            nonzero_mask = y_test_class[:, i] == 1
            if nonzero_mask.any():
                # Calculate metrics using scaled values
                r2 = r2_score(y_test_scaled[nonzero_mask, i], reg_pred[nonzero_mask, i])
                acc = (class_pred[:, i] == y_test_class[:, i]).mean()
                
                # Get statistics
                true_mean = np.mean(y_test_scaled[nonzero_mask, i])
                true_std = np.std(y_test_scaled[nonzero_mask, i])
                pred_mean = np.mean(reg_pred[nonzero_mask, i])
                pred_std = np.std(reg_pred[nonzero_mask, i])
                
                print(f"\nFeature {i}:")
                print(f"R² Score: {r2:.4f}")
                print(f"Binary Classification Accuracy: {acc:.2%}")
                print(f"True stats - Mean: {true_mean:.4f}, Std: {true_std:.4f}")
                print(f"Pred stats - Mean: {pred_mean:.4f}, Std: {pred_std:.4f}")
                residuals = reg_pred[nonzero_mask, i] - y_test_scaled[nonzero_mask, i]

                # High-event mask (top 5% of true values)
                high_mask = y_test_scaled[nonzero_mask, i] > np.quantile(y_test_scaled[nonzero_mask, i], 0.95)
                high_residuals = residuals[high_mask]

                # Print statistics
                print(f"Feature {i} statistics:")
                print(f"  True mean: {true_mean:.4e}, True std: {true_std:.4e}")
                print(f"  Pred mean: {pred_mean:.4e}, Pred std: {pred_std:.4e}")
                print(f"  Residuals mean: {residuals.mean():.4e}, std: {residuals.std():.4e}")
                print(f"  High-event (top 5%) residuals mean: {high_residuals.mean():.4e}, std: {high_residuals.std():.4e}")
                print(f"  High-event min/max residual: {high_residuals.min():.4e} / {high_residuals.max():.4e}")
                
                # Save stats for JSON
                results["features"].append({
                    "feature_idx": int(i),
                    "r2_score": float(r2),
                    "accuracy": float(acc),
                    "true_mean": float(true_mean),
                    "true_std": float(true_std),
                    "pred_mean": float(pred_mean),
                    "pred_std": float(pred_std),
                    "residual_mean": float(residuals.mean()),
                    "residual_std": float(residuals.std()),
                    "high_event": {
                        "mean": float(high_residuals.mean()),
                        "std": float(high_residuals.std()),
                        "min": float(high_residuals.min()),
                        "max": float(high_residuals.max())
                    }
                })
                
                
                # Create histogram comparison
                plt.figure(figsize=(10, 6))
                
                # Define consistent bins for both histograms
                min_val = min(np.min(y_test_scaled[nonzero_mask, i]), 
                            np.min(reg_pred[nonzero_mask, i]))
                max_val = max(np.max(y_test_scaled[nonzero_mask, i]), 
                            np.max(reg_pred[nonzero_mask, i]))
                bins = np.linspace(min_val, max_val, 50)
                
                # Plot true values in grey
                plt.hist(y_test_scaled[nonzero_mask, i], bins=bins, alpha=0.5, 
                        color='grey', density=True, label='Test set')
                
                # Plot predicted values as red line
                plt.hist(reg_pred[nonzero_mask, i], bins=bins, alpha=0.7, 
                        color='red', density=True, histtype='step', linewidth=2,
                        label='ML prediction')
                
                plt.yscale('log')  # Set logarithmic y-axis
                
                plt.title(f'Feature {i} Distribution Comparison\n'
                         f'R²: {r2:.4f}, Acc: {acc:.2%}')
                plt.xlabel('Scaled Values')
                plt.ylabel('Density (log scale)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add statistics as text box
                stats_text = f'True μ={true_mean:.3f}, σ={true_std:.3f}\n' \
                           f'Pred μ={pred_mean:.3f}, σ={pred_std:.3f}'
                plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.8))
                
                plt.savefig(f'feature_{i}_distribution.png', dpi=200, bbox_inches='tight')
                plt.close()
        # Save JSON log
    json_path = os.path.join(checkpoint_dir, "evaluation_log.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation results to {json_path}")
    return reg_pred

######################################
# Define neural network model
class ScaleLayer(nn.Module):
    def __init__(self, feature_means, feature_stds):
        super().__init__()
        self.register_buffer('means', torch.FloatTensor(feature_means))
        self.register_buffer('stds', torch.FloatTensor(feature_stds))
    
    def forward(self, x):
        return (x - self.means) / self.stds

class UnscaleLayer(nn.Module):
    def __init__(self, feature_means, feature_stds):
        super().__init__()
        self.register_buffer('means', torch.FloatTensor(feature_means))
        self.register_buffer('stds', torch.FloatTensor(feature_stds))
    
    def forward(self, x):
        return x * self.stds + self.means

class ConstrainedRegressor(nn.Module):
    """Regressor with separate training and inference paths"""
    def __init__(self, original_regressor: nn.Module):
        super().__init__()
        self.regressor = original_regressor

    def forward(
        self,
        x: torch.Tensor,
        x_orig: torch.Tensor,
        output_means: torch.Tensor,
        output_stds: torch.Tensor,
        input_means: torch.Tensor,
        input_stds: torch.Tensor
    ) -> torch.Tensor:
    
        reg_pred = self.regressor(x)
        del x

        if self.training:
            return reg_pred

        # Unscale predictions and inputs
        reg_unscaled = reg_pred * output_stds.unsqueeze(0) + output_means.unsqueeze(0)
        x_unscaled = x_orig[:, :input_means.size(0)] * input_stds.unsqueeze(0) + input_means.unsqueeze(0)
        constrained = reg_unscaled.clone()
        invalid_mask = (x_unscaled[:, 0] < 10000) | (x_unscaled[:, 1] < 150) | (x_unscaled[:, 2] > 0.2) | (x_unscaled[:, 3] < 0) | (x_unscaled[:, 4] < 0) | (x_unscaled[:, 5] < 0) | (x_unscaled[:, 6] < 0) | (x_unscaled[:, 7] < 0)
        constrained[invalid_mask, :7] = 0.0  # Zero all tendencies (ta, qv, qc, qi, qr, qs, qg)
        # Temperature-based phase constraints
        T_ice = 233.15    # -40°C
        T_mixed = 278.15  # 5°C
        
        # Create phase masks using unscaled temperature (ta_mig is at index 1)
        ice_mask = x_unscaled[:, 1] <= T_ice

        # Ice regime (T ≤ -40°C): No liquid water or rain formation
        constrained[ice_mask, 2] = torch.minimum(constrained[ice_mask, 2], torch.zeros_like(constrained[ice_mask, 2]))  # tend_qc_mig
        constrained[ice_mask, 4] = torch.minimum(constrained[ice_mask, 4], torch.zeros_like(constrained[ice_mask, 4]))  # tend_qr_mig
        
        # Negativity constraints
        constrained[:, 0] = torch.maximum(-x_unscaled[:, 1], constrained[:, 0])  # Temperature
        constrained[:, 1] = torch.maximum(-x_unscaled[:, 2], constrained[:, 1])  # qv
        constrained[:, 2] = torch.maximum(-x_unscaled[:, 3], constrained[:, 2])  # qc
        constrained[:, 3] = torch.maximum(-x_unscaled[:, 4], constrained[:, 3])  # qi
        constrained[:, 4] = torch.maximum(-x_unscaled[:, 5], constrained[:, 4])  # qr
        constrained[:, 5] = torch.maximum(-x_unscaled[:, 6], constrained[:, 5])  # qs
        constrained[:, 6] = torch.maximum(-x_unscaled[:, 7], constrained[:, 6])  # qg

        # Proportional constraints
        max_frac_ta = 0.3
        max_frac_qv = 0.5
        max_frac_tracers = 2 # try to increase probably
        constrained[:, 0] = torch.clamp(constrained[:, 0], -max_frac_ta * x_unscaled[:, 1].abs(), max_frac_ta * x_unscaled[:, 1].abs())  # Temperature
        constrained[:, 1] = torch.clamp(constrained[:, 1], -max_frac_qv * x_unscaled[:, 2].abs(), max_frac_qv * x_unscaled[:, 2].abs())  # qv
        constrained[:, 2] = torch.clamp(constrained[:, 2], -max_frac_tracers * x_unscaled[:, 3].abs(), max_frac_tracers * x_unscaled[:, 3].abs())  # qc
        constrained[:, 3] = torch.clamp(constrained[:, 3], -max_frac_tracers * x_unscaled[:, 4].abs(), max_frac_tracers * x_unscaled[:, 4].abs())  # qi
        constrained[:, 4] = torch.clamp(constrained[:, 4], -max_frac_tracers * x_unscaled[:, 5].abs(), max_frac_tracers * x_unscaled[:, 5].abs())  # qr
        constrained[:, 5] = torch.clamp(constrained[:, 5], -max_frac_tracers * x_unscaled[:, 6].abs(), max_frac_tracers * x_unscaled[:, 6].abs())  # qs
        constrained[:, 6] = torch.clamp(constrained[:, 6], -max_frac_tracers * x_unscaled[:, 7].abs(), max_frac_tracers * x_unscaled[:, 7].abs())  # qg        

        # Mass conservation constraint (not used in the study)
        # mass_vars = [1, 2, 3, 4, 5, 6]  # qv–qg
        # tend_vars = [1, 2, 3, 4, 5, 6]  # same order

        # # Total initial and final mass
        # initial_mass = x_unscaled[:, [v + 1 for v in mass_vars]].sum(dim=1)
        # final_masses = x_unscaled[:, [v + 1 for v in mass_vars]] + constrained[:, tend_vars]
        # final_mass = final_masses.sum(dim=1)

        # # Total mass difference per sample
        # mass_diff = initial_mass - final_mass  # (batch_size,)

        # # Find active tendencies (nonzero)
        # active_mask = (constrained[:, tend_vars].abs() > 0)  # (batch_size, 6)

        # # Use inverse std as weights (broadcasted)
        # correction_stds = output_stds[tend_vars]  # shape (6,)
        # weights = 1.0 / correction_stds           # inverse std, shape (6,)
        # # Expand weights to batch
        # weights_batched = weights.unsqueeze(0).expand(constrained.size(0), -1)  # (batch_size, 6)
        # weights_active = weights_batched * active_mask  # Zero out inactive
        # # Normalize weights per sample (avoid division by zero)
        # weights_sum = weights_active.sum(dim=1, keepdim=True)  # (batch_size, 1)
        # # To avoid division by zero, set sum to 1 where all inactive (no redistribution)
        # weights_sum = torch.where(weights_sum == 0, torch.ones_like(weights_sum), weights_sum)
        # norm_weights = weights_active / weights_sum  # (batch_size, 6)
        # # Compute mass adjustment only for active tendencies
        # mass_adjustment = mass_diff.unsqueeze(1) * norm_weights  # (batch_size, 6)
        # # Apply correction
        # constrained[:, tend_vars] += mass_adjustment
        # Rescale to model space
        reg_pred = (constrained - output_means.unsqueeze(0)) / output_stds.unsqueeze(0)
        return reg_pred


# Updated model architecture with LayerNorm
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, output_dim, input_stats, output_stats):
        super().__init__()

        self.input_means = torch.FloatTensor([stat[0] for stat in input_stats])
        self.input_stds = torch.FloatTensor([stat[1] for stat in input_stats])
        self.output_means = torch.FloatTensor([stat[0] for stat in output_stats])
        self.output_stds = torch.FloatTensor([stat[1] for stat in output_stats])
        
        self.register_buffer('input_means_buf', self.input_means)
        self.register_buffer('input_stds_buf', self.input_stds)
        self.register_buffer('output_means_buf', self.output_means)
        self.register_buffer('output_stds_buf', self.output_stds)
        
        self.input_scaler = ScaleLayer(self.input_means, self.input_stds)
        self.output_unscaler = UnscaleLayer(self.output_means, self.output_stds)
        nodes = 256
        
        # Wider shared layers with dropout
        self.shared = nn.Sequential(
            nn.Linear(input_dim, nodes),
            nn.LayerNorm(nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes),
            nn.LayerNorm(nodes),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(nodes, nodes),
            nn.LayerNorm(nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes), 
            nn.LayerNorm(nodes), 
            nn.ReLU(),
            nn.Linear(nodes, output_dim * 2)
        )

        self.regressor = ConstrainedRegressor(nn.Sequential(
            nn.Linear(nodes, nodes),
            nn.LayerNorm(nodes),
            nn.ReLU(),
            nn.Linear(nodes, nodes), 
            nn.LayerNorm(nodes),  
            nn.ReLU(),
            nn.Linear(nodes, output_dim)
        ))

        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor, scale_input: bool = True, unscale_output: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if scale_input:
            x = (x - self.input_means_buf) / self.input_stds_buf
        
        shared_features = self.shared(x)
        original_input = x.clone()
        del x  # Free memory
        
        class_logits = self.classifier(shared_features).view(-1, self.output_dim, 2)
        predicted_class = torch.argmax(class_logits[:, :, 0], dim=1)

        reg_out = self.regressor(
            shared_features,
            original_input,
            self.output_means_buf, self.output_stds_buf,
            self.input_means_buf, self.input_stds_buf
        )
        reg_out[predicted_class == 0] = 0
        del shared_features  # Free memory
        del original_input  # Free memory
        if unscale_output:
            reg_out = reg_out * self.output_stds_buf + self.output_means_buf        
        return class_logits, reg_out

    
##########################################

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# Checkpoint path
checkpoint_dir = "..."
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_fix.pth")

# Start timer
start_time = time.time()

# Change sample size
subset_size = 20*10**6 # 20 million samples
batch_size = 4096

# Path to the input HDF5 file
hdf5_path = ".../data/df_mig.h5"


# Load datasets from HDF5
with h5py.File(hdf5_path, "r") as h5f:
    train_array = h5f["train"][:]
    val_array = h5f["val"][:]
    test_array = h5f["test"][:]

# Reduce dataset size
np.random.seed(42)
train_array = train_array[np.random.choice(len(train_array), size=subset_size, replace=False)]

# Split features and targets
X_train, y_train = train_array[:, :8], train_array[:, 8:15]
X_val, y_val = val_array[:, :8], val_array[:, 8:15]
X_test, y_test = test_array[:, :8], test_array[:, 8:15]

# Prepare all data for training
y_train_class, y_train_reg = prepare_data(y_train)
y_val_class, y_val_reg = prepare_data(y_val)
y_test_class, y_test_reg = prepare_data(y_test)

# Convert to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_class_tensor = torch.LongTensor((y_train_class != 0).astype(int)).to(device)
y_train_reg_tensor = torch.FloatTensor(y_train).to(device)

# Create dataset
train_dataset = TensorDataset(X_train_tensor, y_train_class_tensor, y_train_reg_tensor)
torch.cuda.synchronize()  # Ensures that all GPU tasks are complete

# Create single dataloader
num_workers = 0
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=False  # Set pin_memory to False if the data is already on GPU
)

# Compute scaling factors
input_stats = compute_nonzero_stats(X_train)
output_stats = compute_nonzero_stats(y_train)

# Create model with scaling layers
model = MultiTaskMLP(
    input_dim=8,
    output_dim=y_train.shape[1],
    input_stats=input_stats,
    output_stats=output_stats
)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training")
    model = nn.DataParallel(model)
model = model.to(device)


epochs = 50

# Setup optimizer and scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,                   # Base LR; scheduler controls the peak
    weight_decay=1e-2,
    betas=(0.9, 0.95),         # Faster adaptation
    eps=1e-8
)


scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,                 # Lower peak to avoid destabilizing regressor
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    div_factor=10.0,            # Start lr = 3e-3 / 10 = 3e-4
    final_div_factor=1e4,       # Final lr = 3e-3 / 1e4 = 3e-7 — more fine-tuning
    pct_start=0.2,              # Shorter warmup phase (20%) to reduce early forgetting
    anneal_strategy='cos',      # Smooth cosine decay
    cycle_momentum=False        # Good with AdamW
)

start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint["model_state_dict"]
    if isinstance(model, nn.DataParallel):
        new_state_dict = {"module." + k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Restore optimizer and scheduler states if available
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Set starting epoch
    start_epoch = checkpoint.get("epoch", 0)
    print(f"Resuming from epoch {start_epoch+1}")

for epoch in range(start_epoch, epochs):
    total_loss, class_losses, reg_losses, neg_mass_losses, avg_acc = train_epoch(
        model, train_loader, optimizer, scheduler, phase='combined'
    )
    
    # Compute R² scores
    r2_scores = compute_r2_scores(model, train_loader, device)
    r2_mean = np.mean(r2_scores)
    
    # Only print epoch summary
    print(f"Epoch {epoch+1}/{epochs}, Acc: {avg_acc:.4f}, "
          f"Loss: {total_loss:.4f}, Class: {class_losses:.4f}, "
          f"Reg: {reg_losses:.4f}, NegMass: {neg_mass_losses:.4f}, "
          f"Mean R²: {r2_mean:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, checkpoint_path)


# Create validation dataset and loader (add after train_loader definition)
X_val_tensor = torch.FloatTensor(X_val)
y_val_class_tensor = torch.LongTensor((y_val != 0).astype(int))
y_val_raw_tensor = torch.FloatTensor(y_val)

val_dataset = TensorDataset(X_val_tensor, y_val_class_tensor, y_val_raw_tensor)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle validation data
    pin_memory=True
)

# Evaluate model
X_test = torch.FloatTensor(X_test).to(device)
evaluate_model(model, X_test, y_test, y_test_class, output_stats)

# Create and save TorchScript version
class WrappedModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model  # Store the original model
    
    def forward(self, x):
        with torch.no_grad():
            _, reg_out = self.model(x, scale_input=True, unscale_output=True)  # Extract only the regression output
        del x  # Free memory
        return reg_out  # Return only the final predicted values

# Load the original model
original_model = model

# Convert to TorchScript
checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(checkpoint_dir, exist_ok=True)
# CPU version
model_cpu = model.to('cpu')
wrapped_model = WrappedModel(original_model)
scripted_model_cpu = torch.jit.script(wrapped_model)
torch.jit.save(
    scripted_model_cpu,
    os.path.join(checkpoint_dir, "constrained_regression_model_final_cpu_scripted.pt")
)
model = model.to(device)

print("Saved model successfully.")


training_time = time.time() - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)

print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}")


