#!/usr/bin/env bash
# Sync code (src/) UP to Euler scratch. Never uploads datasets, results, weights,
# or the venv (metered-connection rule: heavy data stays on-cluster).
set -euo pipefail

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = src/
REMOTE_HOST="euler"
REMOTE_PATH="/cluster/scratch/aleonel/spatial_jepa"

# NB: --delete must never touch scratch-generated trees: the conda env, the
# downloaded/processed data, results and logs all live under excluded dirs.
# Leading '/' anchors an exclude to the transfer ROOT only, so scratch-generated
# top-level dirs (conda env, data, results, logs) are protected from --delete
# WITHOUT also excluding nested code dirs like experiments/env/.
rsync -av --checksum --delete \
  --exclude='/env/' --exclude='/data/' --exclude='/data_bartu/' \
  --exclude='/results/' --exclude='/logs/' \
  --exclude='/datasets/' --exclude='/datasets_processed/' \
  --exclude='/datasets_replica/' --exclude='/hf_cache/' --exclude='/torch_hub/' \
  --exclude='/data_branch/' --exclude='/datasets_scannet/' \
  --exclude='/1_model/' --exclude='/4_model_locator/' \
  --exclude='.venv/' \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.pytest_cache/' --exclude='*.ipynb' --exclude='.ipynb_checkpoints/' \
  "${LOCAL}/" "${REMOTE_HOST}:${REMOTE_PATH}/"

ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_PATH}/logs ${REMOTE_PATH}/results"
echo "synced code -> ${REMOTE_HOST}:${REMOTE_PATH}"
