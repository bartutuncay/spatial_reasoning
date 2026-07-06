#!/usr/bin/env bash
# THE thesis test: few-shot cross-scene localization.
# Pretrain on 4 rooms, few-shot localize the held-out target from K labels.
#   objective{scratch,jepa,symalign,contrastive,recon} x K{20,50,100} x seed{0,1}
# Default target = office (held out; pretrain on pipes+break_room+relief+hospital).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SB="experiments/exp_jepa/fewshot.sbatch"
TGT="${1:-office}"
mkdir -p logs results
n=0
for OBJ in scratch jepa symalign contrastive recon; do
  for K in 20 50 100; do
    for SEED in 0 1; do
      ID="fs_${TGT}_${OBJ}_K${K}_s${SEED}"
      jid=$(sbatch --parsable --job-name="$ID" \
        --export=ALL,OBJ=$OBJ,K=$K,TGT=$TGT,SEED=$SEED,ID=$ID,RESDIR=fewshot "$SB")
      echo "submitted $ID -> $jid"; n=$((n + 1))
    done
  done
done
echo "== submitted $n few-shot jobs (target=$TGT) =="
