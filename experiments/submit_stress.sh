#!/usr/bin/env bash
# HARD-REGIME degradation sweep (the real money plot): does a JEPA-pretrained
# representation degrade more gracefully than scratch as the map thins and the
# query blurs? objective{scratch,jepa,symalign} x map_frac{1.0,0.5,0.25} x
# blur{0,3} = 18 parallel jobs. Frozen encoder, stable barycenter head.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SB="experiments/exp_jepa/locate.sbatch"
mkdir -p logs results
n=0
for OBJ in scratch jepa symalign; do
  for MF in 1.0 0.5 0.25; do
    for BL in 0 3; do
      ID="stress_${OBJ}_mf${MF}_bl${BL}"
      jid=$(sbatch --parsable --job-name="$ID" \
        --export=ALL,OBJ=$OBJ,MAP=map,HEAD=barycenter,FRZ=1,MF=$MF,BL=$BL,RESDIR=stress_campaign,ID=$ID "$SB")
      echo "submitted $ID -> $jid"; n=$((n + 1))
    done
  done
done
echo "== submitted $n stress jobs =="
