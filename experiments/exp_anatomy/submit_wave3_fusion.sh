#!/usr/bin/env bash
# Wave-3 A1: fusion-frontier battery. 3 lambdas x 2 seeds x 5 probes = 30 jobs.
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa

sub() {  # sub <jobname> <MODULE> "<ARGS>"
  sbatch --job-name "$1" --export=ALL,MODULE=$2,ID=$1,RESDIR=anatomy,ARGS="$3" \
    experiments/exp_anatomy/probe.sbatch
}

for LAM in 25 50 75; do
  OBJ="fuse_cj_${LAM}"
  for SEED in 0 1; do
    sub "an_placerec_${OBJ}_s${SEED}"   placerec     "--objective ${OBJ} --seed ${SEED} --tier shakedown"
    sub "an_depthprobe_${OBJ}_s${SEED}" depthprobe   "--objective ${OBJ} --seed ${SEED} --tier shakedown"
    sub "an_rollout_${OBJ}_s${SEED}"    rollout      "--objective ${OBJ} --seed ${SEED} --horizon 2 --tier shakedown"
    sub "an_navdist_${OBJ}_s${SEED}"    spatialpairs "--objective ${OBJ} --seed ${SEED} --channel navdist --tier shakedown"
    sub "an_relpose_${OBJ}_s${SEED}"    spatialpairs "--objective ${OBJ} --seed ${SEED} --channel relpose --tier shakedown"
  done
done
squeue -u aleonel -h -o "%j" | grep -c "fuse_cj" | xargs echo "fusion jobs queued:"
