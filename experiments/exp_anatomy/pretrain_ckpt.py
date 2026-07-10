"""Pretrain one conditioning row's encoder and save it as a checkpoint, so the
probe sweeps (horizon, head capacity, interventions) and the training-length
curves reuse a single encoder instead of re-pretraining inside every probe job.

Reproduces the probes' in-job pretrain exactly: same seeding call order
(manual_seed -> model build -> pretrain_encoder), same data roots, so a probe
run with --encoder-ckpt matches an in-job pretrain at equal (objective, seed,
steps).

Runs under the generic probe.sbatch (MODULE=pretrain_ckpt): --out is a result
directory; the checkpoint lands at <out>/encoder.pt and a result.json records
the run for the tally (channel encoder:<row>, verdict UNSCORED).
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_anatomy.common import (  # noqa: E402
    SCENES, finish, pretrain_encoder,
)
from experiments.exp_jepa.locate import _load_module  # noqa: E402


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)
    autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
    vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
    if args.objective != "scratch":
        pretrain_encoder(vae, autoenc, [root / s / "random_walks" for s in SCENES],
                         args.objective, args.steps, dev)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ck_path = out_dir / "encoder.pt"
    torch.save({"state_dict": {k: v.cpu() for k, v in vae.state_dict().items()},
                "objective": args.objective, "steps": args.steps,
                "seed": args.seed, "latent_dim": args.latent_dim,
                "scenes": list(SCENES)}, ck_path)
    return {"status": "ok", "verdict": "UNSCORED",
            "channel": f"encoder:{args.objective}", "primary": None,
            "metrics": {"row": args.objective, "seed": args.seed,
                        "steps": args.steps, "latent_dim": args.latent_dim,
                        "ckpt": str(ck_path)},
            "notes": f"encoder {args.objective} s{args.seed} @{args.steps} steps -> {ck_path}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--processed-root",
                    default=os.environ.get("SJEPA_PROCESSED_ROOT", "data_bartu"))
    ap.add_argument("--tier", default="bulk")           # accepted for sbatch symmetry
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    try:
        result = run(args)
    except Exception as e:
        result = {"status": "failed", "verdict": "DEAD",
                  "notes": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-2500:]}
    finish(args.out, result, t0)


if __name__ == "__main__":
    main()
