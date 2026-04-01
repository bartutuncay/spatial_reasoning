#!/usr/bin/env bash
#SBATCH -J rw_latents
#SBATCH -A es_dgess
#SBATCH --output=../../../scratch/btuncay/cog/gnn_spatial_reasoning/logs/rw_latent.out
#SBATCH --error=../../../scratch/btuncay/cog/gnn_spatial_reasoning/logs/rw_latent.err
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem-per-cpu=2G

module purge
module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6
source ../../../home/btuncay/gnn_312/bin/activate

cd ~/cog
python3 -m gnn_spatial_reasoning.preprocessing_src.rw_to_ae
