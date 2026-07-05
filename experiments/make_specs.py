"""Enumerate campaign specs (one JSON file per spec, verbatim id).

Wave 0 = Day-1 de-risking probes (P1-P5).
Wave 1 = Stage-A single-axis shakedown: vary one axis at a time around a
default so a dead setting is caught cheaply before the bulk cross-product.

A spec is a plain dict:
    {id, wave, hypothesis, channel, module, args, seed, slurm, est_gpu_h}
`args` maps CLI flags to values; a value of None emits a bare flag.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.config import ACCOUNT_SHORT, DEFAULT_SLURM

SPECS_DIRNAME = "specs"

# --------------------------------------------------------------------------- #
# Stage-A axes. The default point (one value per axis) is the anchor; each axis
# is then swept alone. `rotation=dir` predicts the view_dir forward vector
# (the only orientation the data supports; full 6-DoF needs richer walks).
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "objective": "jepa",
    "direction": "rgb2geo",
    "predictor": "deep",
    "mask_ratio": 0.5,
    "ema": 0.996,
    "collapse": "ema_vicreg",
    "rotation": "dir",
    "latent_dim": 128,
    "node_feats": "full",
}
AXES = {
    "objective": ["jepa", "symalign", "recon", "contrastive"],  # the killer ablation
    "direction": ["rgb2geo", "geo2rgb", "bidir"],
    "predictor": ["shallow", "deep"],
    "mask_ratio": [0.0, 0.25, 0.5, 0.75],
    "ema": [0.99, 0.996, 0.999],
    "collapse": ["ema", "vicreg", "ema_vicreg", "whiten"],
    "rotation": ["dir", "none"],
    "latent_dim": [64, 128, 256],
    "node_feats": ["full", "geom_only", "no_rgb"],
}

_HYPO = {
    "objective": "predict-in-latent (jepa) removes the map-conditioning inversion vs recon/symalign/contrastive",
    "direction": "RGB->geometry prediction is the informative direction at inference",
    "predictor": "an asymmetric predictor is load-bearing (vs symalign)",
    "mask_ratio": "intra-graph masking earns the world-model over plain alignment",
    "ema": "an EMA target prevents collapse and beats same-encoder re-encode",
    "collapse": "EMA+VICReg keeps embedding rank high on 13 scenes",
    "rotation": "a direction head makes pose-recall computable and beats NN-rotation",
    "latent_dim": "representation quality vs latent width",
    "node_feats": "geometry-only vs +rgb (appearance-retrieval rebuttal)",
}


def _slurm(gpu="rtx_4090", account=ACCOUNT_SHORT, time="03:55:00", est_gpu_h=1.0):
    s = dict(DEFAULT_SLURM)
    s.update(gpu=gpu, account=account, time=time)
    return s


def _spec(sid, wave, channel, module, args, hypothesis, seed=0, slurm=None, est_gpu_h=1.0):
    return {
        "id": sid,
        "wave": wave,
        "hypothesis": hypothesis,
        "channel": channel,
        "module": module,
        "args": args,
        "seed": seed,
        "slurm": slurm or _slurm(est_gpu_h=est_gpu_h),
        "est_gpu_h": est_gpu_h,
    }


def _probe_specs():
    """Wave 0 — cheap diagnostics, each writes a real LEDGER row."""
    mod = "experiments.exp_jepa.probes"
    probes = [
        ("P1", "probe", "pose-recall computable with oracle rotation on ETH3D GT"),
        ("P2", "probe", "map-conditioning inversion: is it global or a few pathological scenes"),
        ("P3", "probe", "trivial baselines the method must beat: predict-centroid, retrieval-NN, NN-rotation"),
        ("P4", "probe", "partial-map crossover: does the map help more as it sparsens"),
        ("P5", "probe", "wip/ rotation-prototype triage ranked by 1-epoch loss"),
    ]
    out = []
    for pid, channel, hypo in probes:
        out.append(_spec(
            sid=f"W0_{pid}", wave=0, channel=channel, module=mod,
            args={"--probe": pid}, hypothesis=hypo, est_gpu_h=0.3,
        ))
    return out


def _stage_a_specs():
    """Wave 1 — single-axis shakedown around DEFAULTS."""
    mod = "experiments.exp_jepa.train"
    out = []

    def _args(cfg):
        a = {"--tier": "shakedown"}
        for k, v in cfg.items():
            a[f"--{k.replace('_', '-')}"] = v
        return a

    # anchor (all defaults)
    out.append(_spec(
        sid="W1_anchor", wave=1, channel="anchor", module=mod,
        args=_args(DEFAULTS), hypothesis="default configuration reference", est_gpu_h=1.0,
    ))
    # sweep each axis alone
    for axis, values in AXES.items():
        for v in values:
            if v == DEFAULTS[axis]:
                continue  # already the anchor
            cfg = dict(DEFAULTS)
            cfg[axis] = v
            vtag = str(v).replace(".", "p")
            out.append(_spec(
                sid=f"W1_{axis}_{vtag}", wave=1, channel=axis, module=mod,
                args=_args(cfg), hypothesis=_HYPO[axis], est_gpu_h=1.0,
            ))
    return out


def build_specs():
    """Return the full list of spec dicts. Ids are unique by construction."""
    specs = _probe_specs() + _stage_a_specs()
    ids = [s["id"] for s in specs]
    assert len(ids) == len(set(ids)), "duplicate spec ids"
    return specs


def write_specs(specs, specs_dir):
    specs_dir = Path(specs_dir)
    specs_dir.mkdir(parents=True, exist_ok=True)
    for s in specs:
        (specs_dir / f"{s['id']}.json").write_text(json.dumps(s, indent=2))
    return len(specs)


def main():
    n = write_specs(build_specs(), Path(__file__).resolve().parent / "exp_jepa" / SPECS_DIRNAME)
    print(f"wrote {n} specs")


if __name__ == "__main__":
    main()
