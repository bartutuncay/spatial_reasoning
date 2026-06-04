#!/usr/bin/env bash
#SBATCH -J pcd_extractor_combined
#SBATCH --output=logs/pcd_combined_reduced.out
#SBATCH --error=logs/pcd_combined_reduced.err
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem-per-cpu=2G

module purge
module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6
source .venv/bin/activate

python3 spatial_reasoning/preprocessing_src/large_graph_from_pcd.py
