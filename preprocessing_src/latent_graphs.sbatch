#!/usr/bin/env bash
#SBATCH -J rw_latent_graphs
#SBATCH --output=logs/rw_latent_graphs.out
#SBATCH --error=logs/rw_latent_graphs.err
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-5

module purge
module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6
source .venv/bin/activate

cd ..
python3 -m spatial_reasoning.preprocessing_src.rw_latent_graph --task-index ${SLURM_ARRAY_TASK_ID}