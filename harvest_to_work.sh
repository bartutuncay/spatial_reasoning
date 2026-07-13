#!/bin/bash
# Selective backup of scratch -> /cluster/work/gess/cog (scratch purges after
# 15 idle days). Policy: champions + main results ONLY (~10 G) - the group
# work/project quotas are tight (project is over soft quota; do NOT write there).
# Idempotent rsync; re-run after every wave (login node or CPU job).
#
#   INCLUDED  results/            all result.json evidence + tallies + figures,
#                                 anatomy_encoders (champion ckpts), ref features
#             modality assets     datasets_replica/{layouts,objects}, data_bartu
#   EXCLUDED  generated walks     (195 G; regenerate: gen_replica_walks.sbatch,
#                                 gen_scannet_walks.sbatch, gen_branch_walks.sbatch)
#             raw datasets        (ScanNet: provision_scannet.sbatch; Replica:
#                                 public download), hf_cache, env, pilot_sweeps
set -euo pipefail
SRC=/cluster/scratch/aleonel/spatial_jepa
DST=/cluster/work/gess/cog/spatial_jepa
mkdir -p "$DST"

# Guard: a purged/empty scratch must NOT propagate --delete to the backup.
# (15-day purge or a re-provision can leave results/ present but empty; rsyncing
# that with --delete would wipe the only surviving copy of the evidence base.)
n_results=$(find "$SRC/results" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$n_results" -lt 5 ]; then
  echo "ABORT: only $n_results result dirs under $SRC/results (scratch purged?);" \
       "refusing to --delete the backup at $DST" >&2
  exit 1
fi

rsync -a --delete --exclude 'pilot_sweeps' "$SRC/results/" "$DST/results/"
rsync -a "$SRC/datasets_replica/layouts" "$SRC/datasets_replica/objects" \
      "$DST/replica_assets/" 2>/dev/null || true
rsync -a "$SRC/data_bartu" "$DST/" 2>/dev/null || true
[ -d "$SRC/data_bartu_assets" ] && rsync -a "$SRC/data_bartu_assets" "$DST/"

cat > "$DST/README.md" <<EOF
# spatial_jepa selective backup (from scratch; auto-written by harvest_to_work.sh)
Last harvest: $(date -Iseconds)

results/           every wave's result.json + matrix_tally.jsonl + figures;
                   anatomy_encoders = champion checkpoints (row x seed x steps);
                   anatomy_refs* = frozen-model feature caches.
replica_assets/    layout maps + object tables (modality axis).
EXCLUDED (regenerable): generated walk datasets (gen_*_walks.sbatch),
raw ScanNet/Replica downloads, hf_cache, conda env, pilot results.
Code + paper: git repos (laptop is source of truth).
EOF
du -sh "$DST"/* 2>/dev/null
echo "harvest complete -> $DST"
