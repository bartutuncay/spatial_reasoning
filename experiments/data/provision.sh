#!/usr/bin/env bash
# Provision ONE ETH3D scene end-to-end (CPU only): download -> extract -> align
# -> scene graph -> random-walk .pt samples. Reuses Bartu's preprocessing_src/
# and agent_src/ (parameterized via SJEPA_LOCATION / SJEPA_SCENE env vars).
#
# datasets/ and datasets_processed/ live directly under the repo ROOT on scratch
# and are excluded from to_euler_sync --delete, so downloads persist.
#
#   bash experiments/data/provision.sh <scene_key> [num_walk_seeds]
# clean scene keys: office (recommended pilot), anlieferung, break_room, pipes
set -euo pipefail

SCENE_KEY="${1:-office}"
N_SEEDS="${2:-5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
cd "$ROOT"

# scene_key -> eth3d download name | project folder | single-scan(1/0)
case "$SCENE_KEY" in
  office)      DL=office;        LOC=office;      SINGLE=0 ;;
  anlieferung) DL=delivery_area; LOC=anlieferung; SINGLE=0 ;;
  break_room)  DL=kicker;        LOC=break_room;  SINGLE=0 ;;
  pipes)       DL=pipes;         LOC=pipes;       SINGLE=1 ;;
  *) echo "unsupported scene_key '$SCENE_KEY' (use: office anlieferung break_room pipes)"; exit 1 ;;
esac
SCENE="${LOC}/${DL}"
PLY="datasets_processed/${SCENE}/scan_raw/combined_aligned.ply"

echo "== [$SCENE_KEY] scene=$SCENE single=$SINGLE seeds=$N_SEEDS =="

# 1. download the raw scan archive to scratch (skip if present)
ARCHIVE="datasets/${LOC}/${DL}_scan_raw.7z"
mkdir -p "datasets/${LOC}"
if [ ! -f "$ARCHIVE" ]; then
  echo "== download ${DL}_scan_raw.7z =="
  # download to a temp file and rename on success, so an interrupted download
  # never leaves a truncated archive that the [ -f ] guard would skip re-fetching
  wget -q -O "${ARCHIVE}.part" "https://www.eth3d.net/data/${DL}_scan_raw.7z"
  mv "${ARCHIVE}.part" "$ARCHIVE"
fi

# 2. extract
echo "== extract =="
SJEPA_LOCATION="$LOC" python3 preprocessing_src/extract_dataset_eth3d.py

# 3. combined_aligned.ply (align multi-scan, or copy the single scan)
if [ "$SINGLE" = "1" ]; then
  echo "== single scan -> combined_aligned.ply =="
  cp "datasets_processed/${SCENE}/scan_raw/scan1.ply" "$PLY"
else
  echo "== align =="
  SJEPA_SCENE="$SCENE" python3 preprocessing_src/align_pcds.py
fi
[ -f "$PLY" ] || { echo "ERROR: $PLY not produced"; exit 1; }

# 4. scene graph (scan_pcd_graph/combined_aligned.pt)
echo "== scene graph =="
SJEPA_SCENE="$SCENE" python3 preprocessing_src/large_graph_from_pcd.py

# 5. random-walk .pt training samples (one array index = one RNG seed ~= 40 .pt)
echo "== random walks =="
for s in $(seq 1 "$N_SEEDS"); do
  python3 agent_src/random_walk_json.py --scene "$SCENE_KEY" --task-index "$s"
done

NW=$(ls "datasets_processed/${LOC}/random_walks/"*.pt 2>/dev/null | wc -l)
GRAPH_OK=$([ -f "datasets_processed/${LOC}/scan_pcd_graph/combined_aligned.pt" ] && echo yes || echo NO)
echo "== DONE [$SCENE_KEY]: ${NW} walk .pt samples, scene_graph=${GRAPH_OK} =="
