"""Campaign configuration for the Spatial-JEPA EULER campaign.

Mirrors the sibling CognitivePrimitives scaffolding conventions (verbatim spec
ids, JSONL + Markdown LEDGER, sbatch via ``conda run -n <env> python -m <module>``),
adapted for this project and the ``aleonel`` account.
"""
from __future__ import annotations

# On-cluster project root: code is rsynced here, results/ live beneath it.
ROOT = "/cluster/scratch/aleonel/spatial_jepa"

# Environment to run workers in. Reusing Bartu's shared venv is preferred; if
# ENV_KIND == "venv" the sbatch sources ENV_PATH/bin/activate, else conda.
ENV_KIND = "venv"                    # "venv" | "conda"
ENV_PATH = "/cluster/scratch/aleonel/spatial_jepa/.venv"  # updated once Bartu shares his
ENV_NAME = "spatial_jepa"            # used when ENV_KIND == "conda"

# (partition, gres-string) by GPU key. Mirrors the sibling table, which is
# proven to schedule under the aleonel account.
GPU_TABLE = {
    "rtx_4090": ("gpuhe.4h", "nvidia_geforce_rtx_4090"),
    "rtx_3090": ("gpuhe.4h", "nvidia_geforce_rtx_3090"),
    "a100_40":  ("gpupr.4h", "a100-pcie-40gb"),
    "a100_80":  ("gpupr.4h", "nvidia_a100_80gb_pcie"),
    "cpu":      (None, None),
}

ACCOUNT_SHORT = "ls_helbi"   # short batched jobs (politically cheap)
ACCOUNT_LONG = "es_dgess"    # long / defensible jobs

DEFAULT_SLURM = {
    "gpu": "rtx_4090",
    "account": ACCOUNT_SHORT,
    "time": "03:55:00",
    "mem_per_cpu": "8G",
    "cpus": 4,
}
