#!/bin/bash
# Localization job template. Config passed via --export (OBJ/MAP/HEAD/FRZ/ID/TIER/SCENE).
#SBATCH --partition=gpuhe.4h
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --account=ls_helbi
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/loc_%x_%j.txt
set -eo pipefail; set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
python -m experiments.exp_jepa.locate \
  --objective "${OBJ}" --map-cond "${MAP}" --head "${HEAD}" --freeze "${FRZ}" \
  --tier "${TIER:-shakedown}" --scene "${SCENE:-office}" \
  --out "results/locate_campaign/${ID}"
