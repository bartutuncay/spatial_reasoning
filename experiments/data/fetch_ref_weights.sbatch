#!/bin/bash
# Install transformers into the scratch env and prefetch reference weights. CPU.
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/fetch_refs_%j.txt
set -eo pipefail; set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
export HF_HOME=/cluster/scratch/aleonel/spatial_jepa/hf_cache
pip install --no-input "transformers>=4.45,<5" accelerate
python - <<'PY'
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
for m in ["facebook/dinov2-small", "facebook/dinov2-base",
          "google/siglip-base-patch16-224"]:
    AutoImageProcessor.from_pretrained(m); AutoModel.from_pretrained(m)
    print("cached", m, flush=True)
AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("cached qwen2-vl-2b", flush=True)
PY
touch hf_cache/.fetch_refs.done
echo OK
