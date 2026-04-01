#!/usr/bin/env bash
#SBATCH -J metrics_anlieferung
#SBATCH --output=../../../scratch/btuncay/cog/gnn_spatial_reasoning/logs/metrics_out.out
#SBATCH --error=../../../scratch/btuncay/cog/gnn_spatial_reasoning/logs/metrics_error.err
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=2G

module purge
module load stack/2025-06 gcc/8.5.0 python/3.11.9
source area_calc/cv_env/bin/activate

python3 gnn_spatial_reasoning/preprocessing_src/metrics.py
