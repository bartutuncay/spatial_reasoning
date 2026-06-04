#!/bin/bash
#SBATCH --job-name=autoencoder_train
#SBATCH --output=logs/training_output_%A.out
#SBATCH --error=logs/training_error_%A.err
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=120:00:00                     
#SBATCH --gpus=rtx_4090:1

# Load stack
module load stack/.2024-06-silent gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source .venv/bin/activate

# Run Python script
srun python3 spatial_reasoning/training_scripts/1_autoencoder.py
