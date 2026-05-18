######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: submit_training.sh                                                                       #
######################################################################################################
# This script submits a training job to the Slurm scheduler on GPU nodes.                            #
######################################################################################################

#!/bin/bash
#SBATCH --partition=gpu             # Use the GPU partition instead of the compute partition
#SBATCH --account=bdXXXX            # Account name
#SBATCH --nodes=1                   # Use one node
#SBATCH --time=08:00:00             
#SBATCH --gres=gpu:1                # Request one GPU (adjust if you need more)
#SBATCH --cpus-per-task=8           # You can specify the number of CPUs to use per task if necessary


srun python .../fix_loss_mask_model.py