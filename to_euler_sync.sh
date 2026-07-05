#!/usr/bin/env bash
# Sync code (src/) UP to Euler scratch. Never uploads datasets, results, weights,
# or the venv (metered-connection rule: heavy data stays on-cluster).
set -euo pipefail

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = src/
REMOTE_HOST="euler"
REMOTE_PATH="/cluster/scratch/aleonel/spatial_jepa"

rsync -av --checksum --delete \
  --exclude='results/' --exclude='logs/' \
  --exclude='datasets_processed/' --exclude='datasets/*.7z' \
  --exclude='1_model/' --exclude='4_model_locator/' \
  --exclude='.venv/' \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.pytest_cache/' --exclude='*.ipynb' --exclude='.ipynb_checkpoints/' \
  "${LOCAL}/" "${REMOTE_HOST}:${REMOTE_PATH}/"

ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_PATH}/logs ${REMOTE_PATH}/results"
echo "synced code -> ${REMOTE_HOST}:${REMOTE_PATH}"
