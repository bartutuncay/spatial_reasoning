#!/usr/bin/env bash
#SBATCH -J rw_latents
#SBATCH --output=logs/rw_latent.out
#SBATCH --error=logs/rw_latent.err
#SBATCH -t 01:00:00
#SBATCH -N 4
#SBATCH -c 16
#SBATCH --mem-per-cpu=1G

module purge
module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6

python3 -m spatial_reasoning.preprocessing_src.rw_to_png
