#!/usr/bin/env bash
# Review-driven robustness sweeps (src/PAPER_REVIEW.md, Tier 2). ETH3D defaults.
#   A: 1500-step encoder checkpoints (one per row x seed; reused by B/C)
#   B: rollout horizon sweep k in {1,4,8}            -> anatomy_horizon_k<K>
#   C: probe head-capacity flip (linear<->mlp)       -> anatomy_probecap
#   D: training-length curves c/j x {5000,15000}     -> anatomy_curves_st<S>
#   E: oracle-state positive control                 -> anatomy_controls
#   F: V-JEPA 2 reference row: fetch -> extract -> bulk probes + sweeps
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa

A100="--partition=gpupr.4h --account=es_dgess --gpus=nvidia_a100_80gb_pcie:1"
HE="--partition=gpuhe.4h --account=ls_helbi --gpus=nvidia_geforce_rtx_4090:1"
A100L="--partition=gpupr.24h --account=es_dgess --gpus=nvidia_a100_80gb_pcie:1"
HEL="--partition=gpuhe.24h --account=ls_helbi --gpus=nvidia_geforce_rtx_4090:1"
ROWS="recon symalign contrastive jepa rgb_only fuse_cj_25 fuse_cj_50 fuse_cj_75"
REFS="dinov2s dinov2b siglip qwen2vl"
SEEDS="0 1 2"
i=0
nextpool() { i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then POOLV="$HE"; else POOLV="$A100"; fi; }

probe() {  # probe <ID> <MODULE> <RESDIR> <ARGS> [DEP] [TIME]
  nextpool
  sbatch $POOLV --time="${6:-01:00:00}" ${5:-} --job-name "$1" \
    --export=ALL,MODULE=$2,ID=$1,RESDIR=$3,ARGS="$4" \
    experiments/exp_anatomy/probe.sbatch >/dev/null
}

# ---- Phase A: encoder checkpoints @1500 steps -------------------------------
declare -A CK
for OBJ in $ROWS; do for S in $SEEDS; do
  ID="enc_${OBJ}_st1500_s${S}"
  nextpool
  CK[${OBJ}_${S}]=$(sbatch --parsable $POOLV --time=01:50:00 --job-name "$ID" \
    --export=ALL,MODULE=pretrain_ckpt,ID=$ID,RESDIR=anatomy_encoders,ARGS="--objective ${OBJ} --seed ${S} --steps 1500" \
    experiments/exp_anatomy/probe.sbatch)
done; done
ckpt() { echo "results/anatomy_encoders/enc_${1}_st1500_s${2}/encoder.pt"; }
echo "phase A: 24 encoder ckpts queued"

# ---- Phase B: horizon sweep (k=2 already in anatomy_bulk) -------------------
for K in 1 4 8; do
  for OBJ in $ROWS; do for S in $SEEDS; do
    probe "h${K}_rollout_${OBJ}_s${S}" rollout "anatomy_horizon_k${K}" \
      "--objective ${OBJ} --encoder-ckpt $(ckpt $OBJ $S) --seed ${S} --tier bulk --horizon ${K}" \
      "--dependency=afterok:${CK[${OBJ}_${S}]}"
  done; done
  for S in $SEEDS; do
    probe "h${K}_rollout_scratch_s${S}" rollout "anatomy_horizon_k${K}" \
      "--objective scratch --seed ${S} --tier bulk --horizon ${K}"
  done
  for M in $REFS; do for S in $SEEDS; do
    probe "h${K}_rollout_ref_${M}_s${S}" rollout "anatomy_horizon_k${K}" \
      "--features-dir results/anatomy_refs/${M} --seed ${S} --tier bulk --horizon ${K}"
  done; done
done
echo "phase B: horizon sweep queued (k=1,4,8)"

# ---- Phase C: head-capacity flip --------------------------------------------
# placerec/depthprobe default linear -> mlp ; rollout/navdist/relpose default mlp -> linear
capset() {  # capset <rowtag> <base-args> [DEP]
  probe "pc_placerec_${1}_s${S}"   placerec     anatomy_probecap "${2} --head mlp --seed ${S} --tier bulk" "${3:-}"
  probe "pc_depthprobe_${1}_s${S}" depthprobe   anatomy_probecap "${2} --head mlp --seed ${S} --tier bulk" "${3:-}"
  probe "pc_rollout_${1}_s${S}"    rollout      anatomy_probecap "${2} --head linear --horizon 2 --seed ${S} --tier bulk" "${3:-}"
  probe "pc_navdist_${1}_s${S}"    spatialpairs anatomy_probecap "${2} --head linear --channel navdist --seed ${S} --tier bulk" "${3:-}"
  probe "pc_relpose_${1}_s${S}"    spatialpairs anatomy_probecap "${2} --head linear --channel relpose --seed ${S} --tier bulk" "${3:-}"
}
for OBJ in $ROWS; do for S in $SEEDS; do
  capset "$OBJ" "--objective ${OBJ} --encoder-ckpt $(ckpt $OBJ $S)" "--dependency=afterok:${CK[${OBJ}_${S}]}"
done; done
for S in $SEEDS; do capset scratch "--objective scratch"; done
for M in $REFS; do for S in $SEEDS; do
  capset "ref_${M}" "--features-dir results/anatomy_refs/${M}"
done; done
echo "phase C: capacity sweep queued"

# ---- Phase D: training-length curves ----------------------------------------
for OBJ in contrastive jepa; do for S in $SEEDS; do
  ID5="enc_${OBJ}_st5000_s${S}"; nextpool
  J5=$(sbatch --parsable $POOLV --time=03:55:00 --job-name "$ID5" \
    --export=ALL,MODULE=pretrain_ckpt,ID=$ID5,RESDIR=anatomy_encoders,ARGS="--objective ${OBJ} --seed ${S} --steps 5000" \
    experiments/exp_anatomy/probe.sbatch)
  ID15="enc_${OBJ}_st15000_s${S}"
  if [ $((i % 2)) -eq 0 ]; then P24="$HEL"; else P24="$A100L"; fi
  J15=$(sbatch --parsable $P24 --time=12:00:00 --job-name "$ID15" \
    --export=ALL,MODULE=pretrain_ckpt,ID=$ID15,RESDIR=anatomy_encoders,ARGS="--objective ${OBJ} --seed ${S} --steps 15000" \
    experiments/exp_anatomy/probe.sbatch)
  for ST in 5000 15000; do
    if [ "$ST" = "5000" ]; then DEP="--dependency=afterok:$J5"; CKP="results/anatomy_encoders/${ID5}/encoder.pt"
    else DEP="--dependency=afterok:$J15"; CKP="results/anatomy_encoders/${ID15}/encoder.pt"; fi
    for spec in "placerec placerec" "depthprobe depthprobe" "rollout rollout --horizon 2" \
                "navdist spatialpairs --channel navdist" "relpose spatialpairs --channel relpose"; do
      set -- $spec; CH=$1; MOD=$2; shift 2; EXTRA="$*"
      probe "cv${ST}_${CH}_${OBJ}_s${S}" "$MOD" "anatomy_curves_st${ST}" \
        "--objective ${OBJ} --encoder-ckpt ${CKP} ${EXTRA} --seed ${S} --tier bulk" "$DEP"
    done
  done
done; done
echo "phase D: training-length curves queued (5000, 15000 steps)"

# ---- Phase E: oracle-state positive control ---------------------------------
for S in $SEEDS; do
  probe "oc_rollout_smooth_s${S}" rollout anatomy_controls \
    "--oracle-state --seed ${S} --tier bulk --horizon 2" "" 00:45:00
  probe "oc_rollout_branch_s${S}" rollout anatomy_controls \
    "--oracle-state --walks branching --seed ${S} --tier bulk --horizon 2" "" 00:45:00
done
echo "phase E: oracle controls queued"

# ---- Phase F: V-JEPA 2 reference row ----------------------------------------
JF=$(sbatch --parsable experiments/data/fetch_vjepa2.sbatch)
JX=$(sbatch --parsable $A100 --dependency=afterok:$JF --job-name sx_vjepa2 \
  --export=ALL,MODEL=vjepa2,OUTROOT=results/anatomy_refs \
  experiments/exp_anatomy/extract_refs.sbatch)
FD="results/anatomy_refs/vjepa2"
for S in $SEEDS; do
  for spec in "placerec placerec" "depthprobe depthprobe" "rollout rollout --horizon 2" \
              "navdist spatialpairs --channel navdist" "relpose spatialpairs --channel relpose"; do
    set -- $spec; CH=$1; MOD=$2; shift 2; EXTRA="$*"
    probe "b_${CH}_ref_vjepa2_s${S}" "$MOD" anatomy_bulk \
      "--features-dir ${FD} ${EXTRA} --seed ${S} --tier bulk" "--dependency=afterok:$JX"
  done
  for K in 1 4 8; do
    probe "h${K}_rollout_ref_vjepa2_s${S}" rollout "anatomy_horizon_k${K}" \
      "--features-dir ${FD} --seed ${S} --tier bulk --horizon ${K}" "--dependency=afterok:$JX"
  done
done
echo "phase F: vjepa2 fetch/extract/probes queued"

# ---- Phase G: rollout re-measure under the FIXED within-scene split ---------
# The pooled-index split accidentally held out the last scene wholesale; all
# prior rollout numbers were cross-scene. Re-measure k=2 into anatomy_bulk
# (same IDs, overwrite; old tallies preserved in git) + corrected action-gap
# into anatomy_controls.
for OBJ in $ROWS; do for S in $SEEDS; do
  DEP="--dependency=afterok:${CK[${OBJ}_${S}]}"
  probe "b_rollout_${OBJ}_s${S}" rollout anatomy_bulk \
    "--objective ${OBJ} --encoder-ckpt $(ckpt $OBJ $S) --seed ${S} --tier bulk --horizon 2" "$DEP"
  probe "ra_rolloutact_${OBJ}_s${S}" rollout anatomy_controls \
    "--objective ${OBJ} --encoder-ckpt $(ckpt $OBJ $S) --walks branching --seed ${S} --tier bulk --horizon 2" "$DEP"
done; done
for S in $SEEDS; do
  probe "b_rollout_scratch_s${S}" rollout anatomy_bulk \
    "--objective scratch --seed ${S} --tier bulk --horizon 2"
  probe "ra_rolloutact_scratch_s${S}" rollout anatomy_controls \
    "--objective scratch --walks branching --seed ${S} --tier bulk --horizon 2"
done
for M in $REFS; do for S in $SEEDS; do
  probe "b_rollout_ref_${M}_s${S}" rollout anatomy_bulk \
    "--features-dir results/anatomy_refs/${M} --seed ${S} --tier bulk --horizon 2"
done; done
echo "phase G: ETH3D rollout re-measure (fixed split) queued"

# ---- Phase H: ScanNet rollout re-measure (LAST: comma-valued env export) ----
export SJEPA_SCENES=$(printf "scene%04d_00," $(seq 0 19) | sed 's/,$//')
export SJEPA_PROCESSED_ROOT="datasets_scannet/walks"
SN_ROWS="scratch recon symalign contrastive jepa rgb_only fuse_cj_25 fuse_cj_50 fuse_cj_75"
for OBJ in $SN_ROWS; do for S in $SEEDS; do
  probe "sn_rollout_${OBJ}_s${S}" rollout anatomy_scannet_bulk \
    "--objective ${OBJ} --seed ${S} --tier bulk --horizon 2" "" 03:55:00
done; done
for M in $REFS; do for S in $SEEDS; do
  probe "sn_rollout_ref_${M}_s${S}" rollout anatomy_scannet_bulk \
    "--features-dir results/anatomy_refs_scannet/${M} --seed ${S} --tier bulk --horizon 2"
done; done
echo "phase H: ScanNet rollout re-measure queued"

squeue -u aleonel -h | wc -l | xargs echo "TOTAL jobs now in queue:"
