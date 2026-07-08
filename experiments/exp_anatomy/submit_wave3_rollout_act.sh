#!/usr/bin/env bash
# Wave-3 A2: action-conditioned rollout on branching walks.
# 1) extract ref features for branch walks (4 GPU jobs), then
# 2) rollout --walks branching for 6 base + 3 fusion rows x 2 seeds + 4 refs.
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa

# ref feature extraction for branch walks (afterok chains the ref-row rollouts)
declare -A EXJ
for MODEL in dinov2s dinov2b siglip qwen2vl; do
  EXJ[$MODEL]=$(sbatch --parsable --job-name "exb_${MODEL}" \
    --export=ALL,MODEL=$MODEL \
    --wrap="module load eth_proxy 2>/dev/null; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate /cluster/scratch/aleonel/spatial_jepa/env; export HF_HOME=/cluster/scratch/aleonel/spatial_jepa/hf_cache HF_HUB_OFFLINE=1; python -m experiments.exp_anatomy.extract_refs --model $MODEL --processed-root data_branch --walks-subdir branch_walks --out-root results/anatomy_refs_branch" \
    --partition=gpuhe.4h --gpus=nvidia_geforce_rtx_4090:1 --account=ls_helbi \
    --time=00:40:00 --mem-per-cpu=8G --cpus-per-task=4 --chdir=/cluster/scratch/aleonel/spatial_jepa \
    --output=logs/exb_%x_%j.txt)
done

sub() { sbatch --job-name "$1" --export=ALL,MODULE=rollout,ID=$1,RESDIR=anatomy,ARGS="$2" \
  experiments/exp_anatomy/probe.sbatch >/dev/null; }

# trained-arm + fusion rows (self-contained pretrain; no dependency)
for OBJ in scratch recon symalign contrastive jepa rgb_only fuse_cj_25 fuse_cj_50 fuse_cj_75; do
  for SEED in 0 1; do
    sub "an_rolloutact_${OBJ}_s${SEED}" "--objective ${OBJ} --seed ${SEED} --walks branching --walks-root data_branch --horizon 2 --tier shakedown"
  done
done

# reference rows (afterok on their branch-feature extraction)
for MODEL in dinov2s dinov2b siglip qwen2vl; do
  sbatch --dependency=afterok:${EXJ[$MODEL]} --job-name "an_rolloutact_ref_${MODEL}" \
    --export=ALL,MODULE=rollout,ID=an_rolloutact_ref_${MODEL},RESDIR=anatomy,ARGS="--features-dir results/anatomy_refs_branch/${MODEL} --walks branching --walks-root data_branch --horizon 2 --tier shakedown" \
    experiments/exp_anatomy/probe.sbatch >/dev/null
done
squeue -u aleonel -h -o "%j" | grep -cE "rolloutact|exb_" | xargs echo "action-rollout + extract jobs queued:"
