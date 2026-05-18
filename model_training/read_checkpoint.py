######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: read_checkpoint.py                                                                       #
######################################################################################################
# This script reads and displays the contents of a PyTorch checkpoint file.                          #
######################################################################################################

import torch
import os

checkpoint_path = "checkpoint_fix.pth"

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("\n Checkpoint Keys ")
    for k in checkpoint.keys():
        print(f"- {k}")

    # Print epoch if stored
    if "epoch" in checkpoint:
        print(f"\nEpoch stored in checkpoint: {checkpoint['epoch']}")

    # Print model state_dict info
    if "model_state_dict" in checkpoint:
        print("\n Model State Dict ")
        for name, tensor in checkpoint["model_state_dict"].items():
            print(f"{name:40s} | shape: {tuple(tensor.shape)} | dtype: {tensor.dtype}")

    # Print optimizer info
    if "optimizer_state_dict" in checkpoint:
        print("\nOptimizer state_dict found.")
        print(f"Number of parameter groups: {len(checkpoint['optimizer_state_dict'].get('param_groups', []))}")

    # Print scheduler info
    if "scheduler_state_dict" in checkpoint:
        print("\nScheduler state_dict found.")

else:
    print("Checkpoint file not found.")

