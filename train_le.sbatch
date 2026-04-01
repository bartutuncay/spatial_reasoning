#!/bin/bash
#SBATCH --job-name=large_autoencoder_train
#SBATCH -A es_dgess
#SBATCH --output=../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_locator/logs/training_output_le_%A.out
#SBATCH --error=../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_locator/logs/training_error_le_%A.err
#SBATCH --cpus-per-task=32                    # CPUs per task
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0:30:00                     
#SBATCH --gpus=rtx_4090:1

# Load stack
module load stack/.2024-06-silent gcc/12.2.0 python_cuda/3.11.6 eth_proxy

#source ../../../home/btuncay/gnn_312/bin/activate

# Run Python script
srun python3 gnn_spatial_reasoning/large_locator.py
