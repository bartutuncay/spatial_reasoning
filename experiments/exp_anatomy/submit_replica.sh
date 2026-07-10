#!/usr/bin/env bash
# Replica MULTI-ROOM matrix: the geometric-evidence dataset (nav-dist/rel-pose
# get real pairwise variance that single-room ScanNet crops lack).
#   refs: extract 5 frozen models -> results/anatomy_refs_replica
#   matrix: 9 trained rows + 5 refs x 5 probes x 3 seeds, bulk tier
#   -> results/anatomy_replica_bulk
# Comma-valued SJEPA_SCENES exported in-shell (SLURM --export is comma-delimited).
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa
export SJEPA_SCENES="apartment_0,apartment_1,apartment_2,frl_apartment_0,frl_apartment_1,frl_apartment_2,frl_apartment_3,frl_apartment_4,frl_apartment_5,hotel_0,office_0,room_0"
export SJEPA_PROCESSED_ROOT="datasets_replica/walks"
A100="--partition=gpupr.4h --account=es_dgess --gpus=nvidia_a100_80gb_pcie:1"
HE="--partition=gpuhe.4h --account=ls_helbi --gpus=nvidia_geforce_rtx_4090:1"
TRAINED="scratch recon symalign contrastive jepa rgb_only fuse_cj_25 fuse_cj_50 fuse_cj_75"
REFS="dinov2s dinov2b siglip qwen2vl vjepa2"
SEEDS="0 1 2"
RESDIR=anatomy_replica_bulk
i=0
nextpool() { i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then POOLV="$HE"; else POOLV="$A100"; fi; }

# 1) ref feature extraction
declare -A EXJ
for M in $REFS; do
  EXJ[$M]=$(sbatch --parsable $A100 --time=02:30:00 --job-name rx_${M} \
    --export=ALL,MODEL=${M},OUTROOT=results/anatomy_refs_replica \
    experiments/exp_anatomy/extract_refs.sbatch)
done
echo "ref extraction queued"

sub() {  # sub <name> <MODULE> "<ARGS>" [DEP]
  nextpool
  sbatch $POOLV --time=03:55:00 ${4:-} --job-name "$1" \
    --export=ALL,MODULE=$2,ID=$1,RESDIR=${RESDIR},ARGS="$3 --tier bulk" \
    experiments/exp_anatomy/probe.sbatch >/dev/null
}

# 2) trained rows (no branch walks on Replica -> no rollout_act)
for OBJ in $TRAINED; do for S in $SEEDS; do
  sub "rp_placerec_${OBJ}_s${S}"   placerec     "--objective ${OBJ} --seed ${S}"
  sub "rp_depthprobe_${OBJ}_s${S}" depthprobe   "--objective ${OBJ} --seed ${S}"
  sub "rp_rollout_${OBJ}_s${S}"    rollout      "--objective ${OBJ} --seed ${S} --horizon 2"
  sub "rp_navdist_${OBJ}_s${S}"    spatialpairs "--objective ${OBJ} --seed ${S} --channel navdist"
  sub "rp_relpose_${OBJ}_s${S}"    spatialpairs "--objective ${OBJ} --seed ${S} --channel relpose"
done; done

# 3) ref rows (afterok on extraction)
for M in $REFS; do
  FD="results/anatomy_refs_replica/${M}"; DEP="--dependency=afterok:${EXJ[$M]}"
  for S in $SEEDS; do
    for spec in "placerec placerec" "depthprobe depthprobe" "rollout rollout --horizon 2" \
                "navdist spatialpairs --channel navdist" "relpose spatialpairs --channel relpose"; do
      set -- $spec; CH=$1; MOD=$2; shift 2; EXTRA="$*"
      sub "rp_${CH}_ref_${M}_s${S}" "$MOD" "--features-dir ${FD} ${EXTRA} --seed ${S}" "$DEP"
    done
  done
done
squeue -u aleonel -h -o "%j" | grep -cE "^rp_|^rx_" | xargs echo "Replica matrix jobs queued:"
