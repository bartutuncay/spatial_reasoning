#!/usr/bin/env bash
# Fan out the localization ablation as PARALLEL SLURM jobs:
#   objective(5) x freeze(2) x { map: head(2) | no_map }  = 30 jobs.
# This is the money-plot matrix: rank the four objective arms on real pose ATE,
# and test the map-conditioning inversion (map vs no_map) per objective.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SB="experiments/exp_jepa/locate.sbatch"
mkdir -p logs results
n=0
for OBJ in scratch jepa symalign contrastive recon; do
  for FRZ in 1 0; do
    for HEAD in barycenter anchor_offset; do        # map-conditioned
      ID="loc_${OBJ}_map_${HEAD}_f${FRZ}"
      jid=$(sbatch --parsable --job-name="$ID" \
            --export=ALL,OBJ=$OBJ,MAP=map,HEAD=$HEAD,FRZ=$FRZ,ID=$ID "$SB")
      echo "submitted $ID -> $jid"; n=$((n + 1))
    done
    ID="loc_${OBJ}_nomap_f${FRZ}"                    # no-map APR (head N/A)
    jid=$(sbatch --parsable --job-name="$ID" \
          --export=ALL,OBJ=$OBJ,MAP=no_map,HEAD=barycenter,FRZ=$FRZ,ID=$ID "$SB")
    echo "submitted $ID -> $jid"; n=$((n + 1))
  done
done
echo "== submitted $n localization jobs =="
