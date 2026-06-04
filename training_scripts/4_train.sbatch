#!/bin/bash
#SBATCH --job-name=large_locator_train
#SBATCH --output=logs/training_output_locator_%A.out
#SBATCH --error=logs/training_error_locator_%A.err
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=120:00:00
#SBATCH --gpus=rtx_4090:1

module load stack/.2024-06-silent gcc/12.2.0 python_cuda/3.11.6 eth_proxy

source .venv/bin/activate

srun python3 spatial_reasoning/training_scripts/4_large_locator.py
