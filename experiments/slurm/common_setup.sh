#!/usr/bin/env bash
# Sourced by every generated sbatch. Activates the environment, sets paths,
# and defines run_with_usr1_forwarding (checkpoint-signal forwarding).
#
# Env selection via exported ENV_KIND (default venv):
#   venv  -> source $ENV_PATH/bin/activate   (reuse Bartu's shared venv in-place)
#   conda -> conda activate $ENV_NAME
# Module stack matches what Bartu's venv was built against.

set +u

: "${ENV_KIND:=venv}"
: "${ENV_PATH:=/cluster/scratch/aleonel/spatial_jepa/.venv}"
: "${ENV_NAME:=spatial_jepa}"
: "${SJEPA_MODULES:=stack/.2024-06-silent gcc/12.2.0 python_cuda/3.11.6 eth_proxy}"

module purge 2>/dev/null || true
# shellcheck disable=SC2086
module load ${SJEPA_MODULES}

if [[ "${ENV_KIND}" == "conda" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
else
    # venvs are not relocatable: activate at its original absolute path.
    source "${ENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p logs results

# Forward SIGUSR1 (raised by --signal=B:USR1@120) to the training process and
# its children so a checkpoint-then-resubmit handler can run before the job is
# killed. Robustly re-waits when the wait is interrupted by the trap.
run_with_usr1_forwarding() {
    "$@" &
    local pid=$!
    trap "
        echo \"[\$(date -Is)] forwarding SIGUSR1 to pid=${pid}\"
        kill -USR1 ${pid} 2>/dev/null || true
        for c in \$(pgrep -P ${pid} 2>/dev/null); do
            kill -USR1 \"\$c\" 2>/dev/null || true
        done
    " USR1
    local rc=0 wait_rc
    while true; do
        wait_rc=0
        wait "${pid}" || wait_rc=$?
        if (( wait_rc == 0 )); then rc=0; break; fi
        if (( wait_rc > 128 )); then continue; fi
        rc=${wait_rc}; break
    done
    trap - USR1
    return ${rc}
}
