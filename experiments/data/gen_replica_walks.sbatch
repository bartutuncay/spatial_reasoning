#!/bin/bash
# Render random walks from Replica v1 meshes. Array task i -> REPLICA_SCENES[i].
# CPU-only. Overrides via --export: NSEEDS (26), NSTEPS (40), NPTS (3000000).
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=03:55:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/replicawalks_%A_%a.txt
set -eo pipefail; set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
SCENES=(apartment_0 apartment_1 apartment_2 frl_apartment_0 frl_apartment_1 \
        frl_apartment_2 frl_apartment_3 frl_apartment_4 frl_apartment_5 \
        hotel_0 office_0 room_0)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
python experiments/data/gen_replica_walks.py --scene "$SCENE" \
  --seeds "${NSEEDS:-26}" --steps "${NSTEPS:-40}" --n-points "${NPTS:-3000000}"
