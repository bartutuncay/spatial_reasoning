#!/bin/bash
# Render random walks from ScanNet meshes. Array task i -> scene%04d_00.
# CPU-only. Set NSEEDS via --export (default 26). Idempotent-ish (overwrites).
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=03:55:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-19
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/scannet_walks_%A_%a.txt
set -eo pipefail; set +u
source "$(conda info --base 2>/dev/null || echo $HOME/miniforge3)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
SID=$(printf "scene%04d_00" "$SLURM_ARRAY_TASK_ID")
python experiments/data/gen_scannet_walks.py --scene "$SID" --seeds "${NSEEDS:-26}" \
  --n-points "${NPOINTS:-1500000}" --out-root "${OUTROOT:-datasets_scannet/walks}"
