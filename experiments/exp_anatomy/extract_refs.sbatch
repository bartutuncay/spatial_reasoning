#!/bin/bash
# Reference feature extraction. Config via --export (MODEL, optional LIMIT).
#SBATCH --partition=gpuhe.4h
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --account=ls_helbi
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/an_%x_%j.txt
set -eo pipefail; set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
export SJEPA_PROCESSED_ROOT="${SJEPA_PROCESSED_ROOT:-data_bartu}"
export HF_HOME=/cluster/scratch/aleonel/spatial_jepa/hf_cache
export HF_HUB_OFFLINE=1   # extraction uses the prefetched cache only
LIM=""
[ -n "${LIMIT}" ] && LIM="--limit ${LIMIT}"
python -m experiments.exp_anatomy.extract_refs --model "${MODEL}" \
  --processed-root "${SJEPA_PROCESSED_ROOT}" \
  --out-root "${OUTROOT:-results/anatomy_refs}" ${LIM}
