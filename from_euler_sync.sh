#!/usr/bin/env bash
# Pull DOWN only the lightweight results (ledger + result.json + metrics).
# Checkpoints and weights are excluded so nothing large hits the metered laptop.
set -euo pipefail

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = src/
REMOTE_HOST="euler"
REMOTE_PATH="/cluster/scratch/aleonel/spatial_jepa"

mkdir -p "${LOCAL}/results"
# Exclude ALL weight/checkpoint forms (dirs and bare files) so nothing large
# reaches the metered laptop; only small json/txt/csv results come down.
rsync -av --progress \
  --exclude='*ckpt*/' --exclude='*checkpoint*/' \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.safetensors' \
  "${REMOTE_HOST}:${REMOTE_PATH}/results/" "${LOCAL}/results/"
echo "pulled results <- ${REMOTE_HOST}:${REMOTE_PATH}/results"
