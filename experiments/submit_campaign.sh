#!/usr/bin/env bash
# Regenerate specs + sbatch and submit the whole campaign as PARALLEL SLURM jobs.
# Run on the Euler login node (this only submits; jobs run on compute nodes).
#
#   bash experiments/submit_campaign.sh [WAVE_GLOB]
# WAVE_GLOB defaults to 'W*' (all); use 'W1_*' for Stage-A only, 'W0_*' for probes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GLOB="${1:-W*}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX:-/cluster/scratch/aleonel/spatial_jepa/env}"

# regenerate specs + fresh sbatch scripts (clears stale)
python -m experiments.make_specs
rm -rf experiments/exp_jepa/slurm/generated
python -m experiments.gen_sbatch

mkdir -p logs results
n=0
for f in experiments/exp_jepa/slurm/generated/${GLOB}.sh; do
  [ -e "$f" ] || continue
  jid=$(sbatch --parsable "$f")
  echo "submitted $(basename "$f" .sh) -> job $jid"
  n=$((n + 1))
done
echo "== submitted $n jobs (glob=${GLOB}) =="
echo "collect later with: python -m experiments.campaign"
