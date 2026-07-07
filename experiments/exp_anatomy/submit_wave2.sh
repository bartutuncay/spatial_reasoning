#!/usr/bin/env bash
# Wave-2 volley: dynamical/geometric columns (C4 rollout, C5 navdist, C6 relpose)
# across 6 objective rows x 2 seeds, + rgb_only backfill on C3/C7, + reference
# rows (--features-dir) for the new columns. No new extraction (features cached).
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa
OBJS="scratch recon symalign contrastive jepa rgb_only"
REFS="dinov2s dinov2b siglip qwen2vl"

sub() {  # sub <jobname> <MODULE> "<ARGS>"
  sbatch --job-name "$1" --export=ALL,MODULE=$2,ID=$1,RESDIR=anatomy,ARGS="$3" \
    experiments/exp_anatomy/probe.sbatch
}

# --- trained-arm rows: C4 rollout, C5 navdist, C6 relpose (x2 seeds) ---
for OBJ in $OBJS; do
  for SEED in 0 1; do
    sub "an_rollout_${OBJ}_s${SEED}"  rollout      "--objective ${OBJ} --seed ${SEED} --horizon 2 --tier shakedown"
    sub "an_navdist_${OBJ}_s${SEED}"  spatialpairs "--objective ${OBJ} --seed ${SEED} --channel navdist --tier shakedown"
    sub "an_relpose_${OBJ}_s${SEED}"  spatialpairs "--objective ${OBJ} --seed ${SEED} --channel relpose --tier shakedown"
  done
done

# --- rgb_only backfill on the Wave-1 columns C3 (depth) + C7 (placerec) ---
for SEED in 0 1; do
  sub "an_placerec_rgb_only_s${SEED}"   placerec   "--objective rgb_only --seed ${SEED} --tier shakedown"
  sub "an_depthprobe_rgb_only_s${SEED}" depthprobe "--objective rgb_only --seed ${SEED} --tier shakedown"
done

# --- reference rows for the new columns (frozen features already cached) ---
for MODEL in $REFS; do
  FD="results/anatomy_refs/${MODEL}"
  sub "an_rollout_ref_${MODEL}" rollout      "--features-dir ${FD} --horizon 2 --tier shakedown"
  sub "an_navdist_ref_${MODEL}" spatialpairs "--features-dir ${FD} --channel navdist --tier shakedown"
  sub "an_relpose_ref_${MODEL}" spatialpairs "--features-dir ${FD} --channel relpose --tier shakedown"
done

squeue -u aleonel -o "%.10i %.30j %.9P %.8T %.10M" | head -70
