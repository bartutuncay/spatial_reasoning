"""Render one sbatch script per spec.

Mirrors the sibling header conventions: ``--gpus=<gres>:1``, ``--mem-per-cpu``,
combined ``--output`` (no ``--error``), ``--chdir`` = on-cluster ROOT, and
``python -m <module> <args> --out results/<id>`` after sourcing common_setup.sh
(which activates the shared venv / conda env based on ENV_KIND).
"""
from __future__ import annotations

from pathlib import Path

from experiments.config import (
    ENV_KIND,
    ENV_NAME,
    ENV_PATH,
    ENV_PREFIX,
    GPU_TABLE,
    RESULTS_DIR,
    ROOT,
)

TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name=sj_{id}
#SBATCH --partition={partition}
#SBATCH --gpus={gres}:1
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --mem-per-cpu={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --chdir={root}
#SBATCH --output=logs/sj_{id}_%j.txt
#SBATCH --signal=B:USR1@120
set -eo pipefail; set +u
cd {root}
export ENV_KIND={env_kind} ENV_PREFIX={env_prefix} ENV_PATH={env_path} ENV_NAME={env_name}
source experiments/slurm/common_setup.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
run_with_usr1_forwarding python -m {module} {args} --seed {seed} --out {results_dir}/{id}
"""


def _flatten_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if v is None:
            parts.append(k)               # bare flag
        else:
            parts.append(f"{k} {v}")
    return " ".join(parts)


def render(spec: dict, root: str = ROOT) -> str:
    slurm = spec["slurm"]
    partition, gres = GPU_TABLE[slurm["gpu"]]
    if partition is None:
        raise ValueError(f"spec {spec['id']} uses cpu gpu-key; cpu partition not wired yet")
    return TEMPLATE.format(
        id=spec["id"],
        partition=partition,
        gres=gres,
        account=slurm["account"],
        time=slurm["time"],
        mem=slurm["mem_per_cpu"],
        cpus=slurm["cpus"],
        root=root,
        env_kind=ENV_KIND,
        env_prefix=ENV_PREFIX,
        env_path=ENV_PATH,
        env_name=ENV_NAME,
        module=spec["module"],
        args=_flatten_args(spec["args"]),
        seed=spec["seed"],
        results_dir=RESULTS_DIR,
    )


def emit(spec: dict, gen_dir, root: str = ROOT) -> Path:
    gen_dir = Path(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    path = gen_dir / f"{spec['id']}.sh"
    path.write_text(render(spec, root=root))
    return path


def main():
    import json

    here = Path(__file__).resolve().parent / "exp_jepa"
    specs_dir = here / "specs"
    gen_dir = here / "slurm" / "generated"
    n = 0
    for p in sorted(specs_dir.glob("*.json")):
        emit(json.loads(p.read_text()), gen_dir)
        n += 1
    print(f"emitted {n} sbatch scripts to {gen_dir}")


if __name__ == "__main__":
    main()
