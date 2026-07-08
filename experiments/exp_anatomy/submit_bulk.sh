#!/usr/bin/env bash
# BULK-tier full matrix (1500-step pretrain) x 3 seeds — the credibility run.
# Usage: submit_bulk.sh <phase>
#   phase efficient : placerec, depthprobe, rollout (smooth)  [already I/O-fixed]
#   phase pairs     : navdist, relpose                        [after spatialpairs fix]
# Bulk pretrain is long -> 3:55h walltime override on the 4h partition.
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa
PHASE="${1:?phase: efficient|pairs}"
POOL="${2:-gpuhe}"          # gpuhe = 4090/ls_helbi ; gpupr = a100/es_dgess
TRAINED="scratch recon symalign contrastive jepa rgb_only fuse_cj_25 fuse_cj_50 fuse_cj_75"
REFS="dinov2s dinov2b siglip qwen2vl"
SEEDS="0 1 2"

if [ "$POOL" = "gpupr" ]; then
  POOLARGS="--partition=gpupr.4h --account=es_dgess --gpus=nvidia_a100_80gb_pcie:1"
else
  POOLARGS="--partition=gpuhe.4h --account=ls_helbi --gpus=nvidia_geforce_rtx_4090:1"
fi

sub() {  # sub <jobname> <MODULE> "<ARGS>"
  sbatch $POOLARGS --time=03:55:00 --job-name "$1" \
    --export=ALL,MODULE=$2,ID=$1,RESDIR=anatomy_bulk,ARGS="$3" \
    experiments/exp_anatomy/probe.sbatch >/dev/null
}

if [ "$PHASE" = "efficient" ]; then
  declare -A PMOD=( [placerec]=placerec [depthprobe]=depthprobe [rollout]=rollout )
  PARGS_rollout="--horizon 2"
  for P in placerec depthprobe rollout; do
    EXTRA=""; [ "$P" = "rollout" ] && EXTRA="--horizon 2"
    for OBJ in $TRAINED; do for S in $SEEDS; do
      sub "b_${P}_${OBJ}_s${S}" "${PMOD[$P]}" "--objective ${OBJ} --seed ${S} --tier bulk ${EXTRA}"
    done; done
    for M in $REFS; do for S in $SEEDS; do
      sub "b_${P}_ref_${M}_s${S}" "${PMOD[$P]}" "--features-dir results/anatomy_refs/${M} --seed ${S} --tier bulk ${EXTRA}"
    done; done
  done
elif [ "$PHASE" = "pairs" ]; then
  for CH in navdist relpose; do
    for OBJ in $TRAINED; do for S in $SEEDS; do
      sub "b_${CH}_${OBJ}_s${S}" spatialpairs "--objective ${OBJ} --seed ${S} --channel ${CH} --tier bulk"
    done; done
    for M in $REFS; do for S in $SEEDS; do
      sub "b_${CH}_ref_${M}_s${S}" spatialpairs "--features-dir results/anatomy_refs/${M} --seed ${S} --channel ${CH} --tier bulk"
    done; done
  done
fi
squeue -u aleonel -h -o "%j" | grep -c "^b_" | xargs echo "bulk jobs queued so far:"
