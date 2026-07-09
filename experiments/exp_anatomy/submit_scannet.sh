#!/usr/bin/env bash
# ScanNet GENERALITY matrix. Usage: submit_scannet.sh [shakedown|bulk]
#   shakedown : 2 seeds, extracts ref features -> results/anatomy_scannet
#   bulk      : 3 seeds, 1500-step conditioning, REUSES existing ref features
#               -> results/anatomy_scannet_bulk (firms up depth/pairwise columns)
# Split across both GPU pools; comma-valued SJEPA_SCENES exported in-shell (SLURM
# --export is comma-delimited).
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa
TIER="${1:-shakedown}"
export SJEPA_SCENES=$(printf "scene%04d_00," $(seq 0 19) | sed 's/,$//')
export SJEPA_PROCESSED_ROOT="datasets_scannet/walks"
A100="--partition=gpupr.4h --account=es_dgess --gpus=nvidia_a100_80gb_pcie:1"
HE="--partition=gpuhe.4h --account=ls_helbi --gpus=nvidia_geforce_rtx_4090:1"
TRAINED="scratch recon symalign contrastive jepa rgb_only fuse_cj_25 fuse_cj_50 fuse_cj_75"
REFS="dinov2s dinov2b siglip qwen2vl"
if [ "$TIER" = "bulk" ]; then SEEDS="0 1 2"; RESDIR=anatomy_scannet_bulk; else SEEDS="0 1"; RESDIR=anatomy_scannet; fi

# 1) ref feature extraction — only if missing (bulk reuses shakedown's features)
declare -A EXJ; NEED_EXTRACT=0
for M in $REFS; do [ -f "results/anatomy_refs_scannet/${M}/scene0000_00.pt" ] || NEED_EXTRACT=1; done
for M in $REFS; do
  if [ "$NEED_EXTRACT" = "1" ]; then
    EXJ[$M]=$(sbatch --parsable $A100 --job-name sx_${M} \
      --export=ALL,MODEL=${M},OUTROOT=results/anatomy_refs_scannet \
      experiments/exp_anatomy/extract_refs.sbatch)
  else EXJ[$M]=""; fi
done

i=0
sub() {  # sub <pool> <name> <MODULE> "<ARGS>"
  sbatch $1 --time=03:55:00 --job-name "$2" \
    --export=ALL,MODULE=$3,ID=$2,RESDIR=${RESDIR},ARGS="$4 --tier ${TIER}" \
    experiments/exp_anatomy/probe.sbatch >/dev/null
}
alt() { i=$((i+1)); [ $((i%2)) -eq 0 ] && echo "$HE" || echo "$A100"; }

# 2) trained rows
for OBJ in $TRAINED; do for S in $SEEDS; do
  sub "$(alt)" "sn_placerec_${OBJ}_s${S}"   placerec     "--objective ${OBJ} --seed ${S}"
  sub "$(alt)" "sn_depthprobe_${OBJ}_s${S}" depthprobe   "--objective ${OBJ} --seed ${S}"
  sub "$(alt)" "sn_rollout_${OBJ}_s${S}"    rollout      "--objective ${OBJ} --seed ${S} --horizon 2"
  sub "$(alt)" "sn_navdist_${OBJ}_s${S}"    spatialpairs "--objective ${OBJ} --seed ${S} --channel navdist"
  sub "$(alt)" "sn_relpose_${OBJ}_s${S}"    spatialpairs "--objective ${OBJ} --seed ${S} --channel relpose"
done; done

# 3) ref rows (3 seeds too, for head-init variance; afterok only if re-extracting)
for M in $REFS; do
  FD="results/anatomy_refs_scannet/${M}"; DEP=""; [ -n "${EXJ[$M]}" ] && DEP="--dependency=afterok:${EXJ[$M]}"
  for S in $SEEDS; do
    for spec in "placerec placerec" "depthprobe depthprobe" "rollout rollout --horizon 2" \
                "navdist spatialpairs --channel navdist" "relpose spatialpairs --channel relpose"; do
      set -- $spec; CH=$1; MOD=$2; shift 2; EXTRA="$*"
      sbatch $HE --time=01:30:00 $DEP --job-name "sn_${CH}_ref_${M}_s${S}" \
        --export=ALL,MODULE=${MOD},ID=sn_${CH}_ref_${M}_s${S},RESDIR=${RESDIR},ARGS="--features-dir ${FD} ${EXTRA} --seed ${S} --tier ${TIER}" \
        experiments/exp_anatomy/probe.sbatch >/dev/null
    done
  done
done
squeue -u aleonel -h -o "%j" | grep -cE "^sn_|^sx_" | xargs echo "ScanNet ${TIER} jobs queued:"
