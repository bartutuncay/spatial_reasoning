#!/usr/bin/env bash
# Build the Spatial-JEPA conda env ON SCRATCH (torch cu121 + PyG stack).
# Idempotent: re-running upgrades in place. All downloads happen on-cluster.
#
#   bash setup_env.sh [PREFIX]
# default PREFIX = /cluster/scratch/aleonel/spatial_jepa/env
set -euo pipefail

PREFIX="${1:-/cluster/scratch/aleonel/spatial_jepa/env}"
TORCH="2.4.1"
CU="cu121"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "${PREFIX}" ]; then
    echo "== creating conda env at ${PREFIX} (python 3.11) =="
    conda create -y -p "${PREFIX}" python=3.11
fi
conda activate "${PREFIX}"
python -m pip install --upgrade pip wheel

echo "== torch ${TORCH}+${CU} (CUDA runtime bundled in wheel) =="
pip install "torch==${TORCH}" torchvision --index-url "https://download.pytorch.org/whl/${CU}"

echo "== PyG compiled extensions matching torch ${TORCH}+${CU} =="
pip install torch_scatter torch_sparse torch_cluster pyg_lib \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CU}.html"

echo "== remaining deps =="
pip install -r "${HERE}/requirements.txt"

echo "== import smoke (cuda availability only true inside a GPU job) =="
python - <<'PY'
import torch, torch_geometric
from torch_geometric.nn import GENConv
from torch_geometric.nn import knn_graph
import open3d, cv2, shapely, networkx, py7zr  # noqa: F401
print("torch", torch.__version__, "cuda_build", torch.version.cuda,
      "cuda_avail", torch.cuda.is_available())
print("pyg", torch_geometric.__version__, "GENConv+knn_graph+open3d+cv2 import OK")
PY
echo "== env ready at ${PREFIX} =="
