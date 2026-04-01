#!/usr/bin/env bash
#SBATCH -J pcd_extractor_anlieferung
#SBATCH -A es_dgess
#SBATCH --output=../../../scratch/btuncay/cog/gnn_spatial_reasoning/logs/pcd_slice/out_%a.out
#SBATCH --error=../../../scratch/btuncay/cog/gnn_spatial_reasoning/logs/pcd_slice/error_%a.err
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem-per-cpu=2G
#SBATCH --array=675-718

module purge
module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6

python3 gnn_spatial_reasoning/preprocessing_src/pcd_slice_extractor_camera.py --task-index ${SLURM_ARRAY_TASK_ID}
