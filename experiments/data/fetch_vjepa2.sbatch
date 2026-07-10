#!/bin/bash
# Prefetch V-JEPA 2 (ViT-L) weights into the scratch HF cache. CPU-only.
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/fetch_vjepa2_%j.txt
set -eo pipefail; set +u
module load eth_proxy   # compute nodes reach the internet only via the ETH proxy
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
export HF_HOME=/cluster/scratch/aleonel/spatial_jepa/hf_cache
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("facebook/vjepa2-vitl-fpc64-256")
print("cached at", p)
PY
