#!/usr/bin/env bash
# Submit the wave-1 volley. Usage: submit_wave1.sh [FETCH_JOBID]
# If FETCH_JOBID is given, ref extraction waits on it (afterok) and the
# ref-probes wait on their extraction job.
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa
FETCH=${1:-}
DEP=""; [ -n "$FETCH" ] && DEP="--dependency=afterok:${FETCH}"

# trained-arm probes: 5 objectives x 2 seeds x 2 probes (shakedown)
for MOD in placerec depthprobe; do
  for OBJ in scratch recon symalign contrastive jepa; do
    for SEED in 0 1; do
      ID="an_${MOD}_${OBJ}_s${SEED}"
      sbatch --job-name "$ID" \
        --export=ALL,MODULE=$MOD,ID=$ID,RESDIR=anatomy,ARGS="--objective ${OBJ} --seed ${SEED} --tier shakedown" \
        experiments/exp_anatomy/probe.sbatch
    done
  done
done

# reference extraction (after weights land), then ref probes (after extraction)
for MODEL in dinov2s dinov2b siglip qwen2vl; do
  EX=$(sbatch --parsable $DEP --job-name "ex_${MODEL}" --export=ALL,MODEL=$MODEL,LIMIT= \
       experiments/exp_anatomy/extract_refs.sbatch)
  for MOD in placerec depthprobe; do
    ID="an_${MOD}_ref_${MODEL}"
    sbatch --dependency=afterok:${EX} --job-name "$ID" \
      --export=ALL,MODULE=$MOD,ID=$ID,RESDIR=anatomy,ARGS="--features-dir results/anatomy_refs/${MODEL} --tier shakedown" \
      experiments/exp_anatomy/probe.sbatch
  done
done
squeue -u aleonel -o "%.10i %.24j %.9P %.8T %.10M %R" | head -60
