#!/bin/bash
# Few-shot cross-scene job. Config via --export (OBJ/K/TGT/ID/RESDIR/TIER).
#SBATCH --partition=gpuhe.4h
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --account=ls_helbi
#SBATCH --time=00:40:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/fs_%x_%j.txt
set -eo pipefail; set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
export SJEPA_PROCESSED_ROOT="${SJEPA_PROCESSED_ROOT:-data_bartu}"
python -m experiments.exp_jepa.fewshot \
  --objective "${OBJ}" --target "${TGT:-office}" --few-shot "${K}" \
  --tier "${TIER:-shakedown}" --seed "${SEED:-0}" \
  --out "results/${RESDIR:-fewshot}/${ID}"
