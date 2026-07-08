#!/bin/bash
# Download a SCOPED ScanNet v2 subset to scratch: cleaned meshes (_vh_clean_2.ply)
# + per-scan .txt (axis alignment) for a fixed 20-scene subset. NEVER the full
# 1.3TB release, NEVER the laptop. CPU-only. Idempotent (script skips existing).
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=03:55:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/provision_scannet_%j.txt
set -eo pipefail
module load eth_proxy
DEST=datasets_scannet
mkdir -p "$DEST"
cd "$DEST"
[ -f download-scannet.py ] || wget -q http://kaldir.vc.cit.tum.de/scannet/download-scannet.py
# First 20 spaces, first scan each: scene0000_00 .. scene0019_00
N=${NSCENES:-20}
for i in $(seq 0 $((N-1))); do
  SID=$(printf "scene%04d_00" "$i")
  for TYPE in _vh_clean_2.ply .txt; do
    # feed the two ToS 'press any key' prompts via yes ''
    yes '' | python download-scannet.py -o . --id "$SID" --type "$TYPE" || \
      echo "WARN: $SID $TYPE failed"
  done
done
echo "=== downloaded meshes ==="
find . -name "*_vh_clean_2.ply" | wc -l | xargs echo "scene meshes:"
du -sh . 2>/dev/null
touch .provision_scannet.done
echo OK
