#!/usr/bin/env bash
#SBATCH -J random_walk_habitat
#SBATCH --output=logs/random_walk_habitat/output_%a.out
#SBATCH --error=logs/random_walk_habitat/error_%a.err
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=6G
#SBATCH --array=0-10
#SBATCH --gpus=rtx_3090:1

module purge
module load stack/.2024-06-silent gcc/12.2.0 cuda/11.8.0 eth_proxy

source ~/.bashrc
conda activate habitat
python -c "import habitat_sim; print('habitat_sim import ok:', habitat_sim.__file__)"

srun python spatial_reasoning/agent_src/random_walk_habitat.py --task-index ${SLURM_ARRAY_TASK_ID}
