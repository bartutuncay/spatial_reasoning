#!/usr/bin/env bash
#SBATCH -J combined_dataset_write
#SBATCH --output=logs/dataset_out.out
#SBATCH --error=logs/dataset_error.err
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=2G

module purge
module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6

python3 spatial_reasoning/preprocessing_src/combine_data.py
