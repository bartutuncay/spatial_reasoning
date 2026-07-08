#!/bin/bash
# Branching-walk generation (Wave-3 A2). One scene per array task. CPU-only.
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=03:55:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/branchwalks_%A_%a.txt
set -eo pipefail; set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
SCENES=(office break_room pipes relief)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
python experiments/data/gen_branch_walks.py --scene "$SCENE" \
  --anchors "${ANCHORS:-60}" --k "${K:-4}"
