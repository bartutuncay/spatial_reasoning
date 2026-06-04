#!/usr/bin/env bash
#SBATCH -J agent_views
#SBATCH -A es_dgess
#SBATCH --output=logs/agent_view/output_%a.out
#SBATCH --error=logs/agent_view/error_%a.err
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-800

module purge
module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6

python3 spatial_reasoning/agent_src/vision.py --task-index ${SLURM_ARRAY_TASK_ID}