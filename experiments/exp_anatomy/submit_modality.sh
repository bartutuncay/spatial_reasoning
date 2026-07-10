#!/usr/bin/env bash
# Conditioning-MODALITY matrix (rows cj<lam>@<modality>).
#   assets (CPU) -> modality encoder ckpts (afterok) -> 5 probes each (afterok)
#   Replica: {pcd, depth, layout, objgraph} x lam{0,50,100} -> anatomy_replica_mod
#   ETH3D:   {pcd, depth, layout}           x lam{0,50,100} -> anatomy_eth3d_mod
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa
A100="--partition=gpupr.4h --account=es_dgess --gpus=nvidia_a100_80gb_pcie:1"
HE="--partition=gpuhe.4h --account=ls_helbi --gpus=nvidia_geforce_rtx_4090:1"
SEEDS="0 1 2"
LAMS="0 50 100"
RP_SCENES="apartment_0 apartment_1 apartment_2 frl_apartment_0 frl_apartment_1 frl_apartment_2 frl_apartment_3 frl_apartment_4 frl_apartment_5 hotel_0 office_0 room_0"
ETH_SCENES="office pipes break_room relief hospital"
i=0
nextpool() { i=$((i+1)); if [ $((i % 2)) -eq 0 ]; then POOLV="$HE"; else POOLV="$A100"; fi; }

# ---- Phase 0: assets (layouts everywhere + Replica object tables), one CPU job
cat > /tmp/modality_assets.sbatch <<EOF
#!/bin/bash
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=03:55:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/modality_assets_%j.txt
set -eo pipefail; set +u
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
for SC in $RP_SCENES; do
  python -m experiments.data.gen_layouts --walks-root datasets_replica/walks --scene "\$SC" --out-root datasets_replica/layouts
  python -m experiments.data.gen_objgraphs --scene "\$SC"
done
for SC in $ETH_SCENES; do
  python -m experiments.data.gen_layouts --walks-root data_bartu --scene "\$SC" --out-root data_bartu_assets/layouts
done
echo ASSETS_DONE
EOF
J0=$(sbatch --parsable /tmp/modality_assets.sbatch)
echo "assets job $J0"

enc() {  # enc <ID> <objective> <extra-env>  -> job id
  nextpool
  sbatch --parsable $POOLV --time=03:55:00 --dependency=afterok:$J0 --job-name "$1" \
    --export=ALL,MODULE=pretrain_ckpt,ID=$1,RESDIR=anatomy_encoders_mod,ARGS="--objective $2 --seed $3 --steps 1500" \
    experiments/exp_anatomy/probe.sbatch
}
probe5() {  # probe5 <ID> <objective> <seed> <ckpt-job> <RESDIR>
  local ID=$1 OBJ=$2 S=$3 DEP=$4 RD=$5
  local CKP="results/anatomy_encoders_mod/${ID}/encoder.pt"
  local spec CH rest MOD EXTRA
  for spec in "placerec placerec" "depthprobe depthprobe" "rollout rollout --horizon 2" \
              "navdist spatialpairs --channel navdist" "relpose spatialpairs --channel relpose"; do
    CH=${spec%% *}; rest=${spec#* }; MOD=${rest%% *}; EXTRA=${rest#"$MOD"}
    nextpool
    sbatch $POOLV --time=01:30:00 --dependency=afterok:${DEP} --job-name "${ID}_${CH}" \
      --export=ALL,MODULE=${MOD},ID=${ID}_${CH},RESDIR=${RD},ARGS="--objective ${OBJ} --encoder-ckpt ${CKP}${EXTRA} --seed ${S} --tier bulk" \
      experiments/exp_anatomy/probe.sbatch >/dev/null
  done
}

# ---- ETH3D block (default scenes/root; assets under data_bartu_assets) ------
export SJEPA_ASSETS_ROOT="data_bartu_assets"
for MOD in pcd depth layout; do for L in $LAMS; do for S in $SEEDS; do
  ID="me_cj${L}_${MOD}_s${S}"
  J=$(enc "$ID" "cj${L}@${MOD}" "$S")
  probe5 "$ID" "cj${L}@${MOD}" "$S" "$J" anatomy_eth3d_mod
done; done; done
echo "ETH3D modality block queued"

# ---- Replica block (LAST: comma-valued env export) ---------------------------
export SJEPA_SCENES=$(echo $RP_SCENES | tr " " ",")
export SJEPA_PROCESSED_ROOT="datasets_replica/walks"
export SJEPA_ASSETS_ROOT="datasets_replica"
for MOD in pcd depth layout objgraph; do for L in $LAMS; do for S in $SEEDS; do
  ID="mr_cj${L}_${MOD}_s${S}"
  J=$(enc "$ID" "cj${L}@${MOD}" "$S")
  probe5 "$ID" "cj${L}@${MOD}" "$S" "$J" anatomy_replica_mod
done; done; done
echo "Replica modality block queued"
squeue -u aleonel -h -o "%j" | grep -cE "^me_|^mr_" | xargs echo "modality matrix jobs queued:"
