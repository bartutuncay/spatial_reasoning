#!/usr/bin/env bash
# Provision one ETH3D scene as a CPU batch job (align/graph are too heavy for a
# login node). Submit with:
#   sbatch --export=ALL,SCENE=office,NSEEDS=3 experiments/data/provision.sbatch
#SBATCH --job-name=sj_provision
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=03:55:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/provision_%x_%j.txt
set -eo pipefail; set +u

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
echo "node=$(hostname) python=$(which python3)"

bash experiments/data/provision.sh "${SCENE:-office}" "${NSEEDS:-3}"
