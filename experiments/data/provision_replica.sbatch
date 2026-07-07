#!/bin/bash
# Download Replica v1 (~100GB, tar parts) to scratch. CPU-only, idempotent.
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=03:55:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/provision_replica_%j.txt
set -eo pipefail
DEST=datasets_replica
DONE="$DEST/.provision_replica.done"
[ -f "$DONE" ] && { echo "already provisioned"; exit 0; }
mkdir -p "$DEST"
cd "$DEST"
[ -d Replica-Dataset ] || git clone --depth 1 https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
# download.sh wgets the release tar parts, concatenates, and extracts into $1
bash download.sh ../replica_v1
cd ..
ls replica_v1 | head -20
N=$(ls -d replica_v1/*/ 2>/dev/null | wc -l)
echo "scenes extracted: $N"
[ "$N" -ge 18 ] || { echo "FAIL: expected >=18 scene dirs"; exit 1; }
rm -f replica_v1_0.tar.gz.part* replica_v1/*.tar* 2>/dev/null || true
touch ".provision_replica.done"
echo OK
