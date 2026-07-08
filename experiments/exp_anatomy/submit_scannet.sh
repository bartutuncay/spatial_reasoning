#!/usr/bin/env bash
# ScanNet GENERALITY matrix (shakedown tier, 2 seeds). Extract ref features on
# ScanNet walks, then run the full probe battery with the ScanNet scene set.
# Split across both GPU pools. Results -> results/anatomy_scannet.
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa
SCENES_CSV=$(printf "scene%04d_00," $(seq 0 19) | sed 's/,$//')
COMMON="SJEPA_SCENES=${SCENES_CSV},SJEPA_PROCESSED_ROOT=datasets_scannet/walks"
A100="--partition=gpupr.4h --account=es_dgess --gpus=nvidia_a100_80gb_pcie:1"
HE="--partition=gpuhe.4h --account=ls_helbi --gpus=nvidia_geforce_rtx_4090:1"
TRAINED="scratch recon symalign contrastive jepa rgb_only fuse_cj_25 fuse_cj_50 fuse_cj_75"
REFS="dinov2s dinov2b siglip qwen2vl"

# 1) ScanNet ref feature extraction (a100), results/anatomy_refs_scannet/<model>
declare -A EXJ
for M in $REFS; do
  EXJ[$M]=$(sbatch --parsable $A100 --job-name sx_${M} \
    --export=ALL,${COMMON},MODEL=${M},OUTROOT=results/anatomy_refs_scannet \
    experiments/exp_anatomy/extract_refs.sbatch)
done

i=0
sub() {  # sub <pool> <name> <MODULE> "<ARGS>"
  sbatch $1 --time=03:55:00 --job-name "$2" \
    --export=ALL,${COMMON},MODULE=$3,ID=$2,RESDIR=anatomy_scannet,ARGS="$4" \
    experiments/exp_anatomy/probe.sbatch >/dev/null
}
alt() { i=$((i+1)); [ $((i%2)) -eq 0 ] && echo "$HE" || echo "$A100"; }

# 2) trained rows: 5 probes x 2 seeds, alternating pools
for OBJ in $TRAINED; do for S in 0 1; do
  sub "$(alt)" "sn_placerec_${OBJ}_s${S}"   placerec     "--objective ${OBJ} --seed ${S} --tier shakedown"
  sub "$(alt)" "sn_depthprobe_${OBJ}_s${S}" depthprobe   "--objective ${OBJ} --seed ${S} --tier shakedown"
  sub "$(alt)" "sn_rollout_${OBJ}_s${S}"    rollout      "--objective ${OBJ} --seed ${S} --horizon 2 --tier shakedown"
  sub "$(alt)" "sn_navdist_${OBJ}_s${S}"    spatialpairs "--objective ${OBJ} --seed ${S} --channel navdist --tier shakedown"
  sub "$(alt)" "sn_relpose_${OBJ}_s${S}"    spatialpairs "--objective ${OBJ} --seed ${S} --channel relpose --tier shakedown"
done; done

# 3) ref rows (afterok on their ScanNet extraction)
for M in $REFS; do
  FD="results/anatomy_refs_scannet/${M}"
  for spec in "placerec placerec" "depthprobe depthprobe" "rollout rollout --horizon 2" \
              "navdist spatialpairs --channel navdist" "relpose spatialpairs --channel relpose"; do
    set -- $spec; CH=$1; MOD=$2; shift 2; EXTRA="$*"
    sbatch $HE --time=01:30:00 --dependency=afterok:${EXJ[$M]} --job-name "sn_${CH}_ref_${M}" \
      --export=ALL,${COMMON},MODULE=${MOD},ID=sn_${CH}_ref_${M},RESDIR=anatomy_scannet,ARGS="--features-dir ${FD} ${EXTRA} --tier shakedown" \
      experiments/exp_anatomy/probe.sbatch >/dev/null
  done
done
squeue -u aleonel -h -o "%j" | grep -cE "^sn_|^sx_" | xargs echo "ScanNet jobs queued:"
