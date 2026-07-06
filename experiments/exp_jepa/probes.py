"""Day-1 de-risking probes (P1-P5). Each writes results/<id>/result.json so it
flows through the LEDGER (see experiments/campaign.py).

P3 (trivial baselines) is model-free and runnable as soon as one scene is
provisioned: it establishes the predict-scene-centroid floor the learned
localizer MUST beat (vuln 3). The model-dependent probes (P1/P2/P4/P5) are
stubbed until the JEPA trainer + heads land, and for now emit an honest PENDING
result so the pipeline stays green.

Run:  python -m experiments.exp_jepa.probes --probe P3 --scene office --out results/jepa_campaign/W0_P3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# make the repo's metrics importable whether run from ROOT or elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.pose import ate_rmse, translation_errors  # noqa: E402


def _scene_node_positions(processed_root: str, scene_folder: str) -> np.ndarray:
    """Load the scene graph and return its Nx3 node positions (world frame)."""
    gpath = Path(processed_root) / scene_folder / "scan_pcd_graph" / "combined_aligned.pt"
    data = torch.load(gpath, map_location="cpu", weights_only=False)
    pos = data.pos if hasattr(data, "pos") else data["pos"]
    return np.asarray(pos, dtype=float)


def _walk_gt_positions(processed_root: str, scene_folder: str) -> np.ndarray:
    """Stack the GT camera positions (`loc`) from every random-walk .pt sample."""
    wdir = Path(processed_root) / scene_folder / "random_walks"
    locs = []
    for p in sorted(glob.glob(str(wdir / "*.pt"))):
        s = torch.load(p, map_location="cpu", weights_only=False)
        loc = s["loc"] if isinstance(s, dict) else s.loc
        locs.append(np.asarray(loc, dtype=float).reshape(3))
    if not locs:
        raise FileNotFoundError(f"no random-walk .pt under {wdir}")
    return np.stack(locs)


def _robust_bounds(node_pos, lo=1.0, hi=99.0):
    """Percentile bounds per axis to reject far-flung ETH3D scan outliers."""
    p_lo = np.percentile(node_pos, lo, axis=0)
    p_hi = np.percentile(node_pos, hi, axis=0)
    keep = np.all((node_pos >= p_lo) & (node_pos <= p_hi), axis=1)
    return node_pos[keep], keep.mean()


def probe_p3(args) -> dict:
    """Predict-scene-centroid baseline: the floor the locator must beat.

    Raw ETH3D combined_aligned.ply carries far-away outlier points, so we report
    BOTH the naive centroid and an outlier-robust (1-99 percentile trimmed)
    centroid; the trimmed one is the meaningful floor.
    """
    node_pos = _scene_node_positions(args.processed_root, args.scene_folder)
    gt = _walk_gt_positions(args.processed_root, args.scene_folder)

    naive_c = node_pos.mean(axis=0)
    inl_pos, inl_frac = _robust_bounds(node_pos)
    robust_c = inl_pos.mean(axis=0)

    naive_pred = np.tile(naive_c, (gt.shape[0], 1))
    robust_pred = np.tile(robust_c, (gt.shape[0], 1))
    r_errs = translation_errors(robust_pred, gt)

    full_extent = float(np.linalg.norm(node_pos.max(0) - node_pos.min(0)))
    inl_extent = float(np.linalg.norm(inl_pos.max(0) - inl_pos.min(0)))
    return {
        "status": "ok",
        "verdict": "EXISTS",
        "channel": "probe",
        "primary": round(float(ate_rmse(robust_pred, gt)), 4),
        "metrics": {
            "centroid_ate_rmse_robust_m": float(ate_rmse(robust_pred, gt)),
            "centroid_ate_rmse_naive_m": float(ate_rmse(naive_pred, gt)),
            "robust_err_median_m": float(np.median(r_errs)),
            "robust_err_p90_m": float(np.percentile(r_errs, 90)),
            "n_queries": int(gt.shape[0]),
            "n_scene_nodes": int(node_pos.shape[0]),
            "inlier_frac": float(inl_frac),
            "scene_extent_full_m": full_extent,
            "scene_extent_inlier_m": inl_extent,
        },
        "notes": ("predict-scene-centroid floor (outlier-trimmed); learned locator "
                  "must beat this per-sample. Large full extent => raw .ply has outliers."),
    }


def _stub(probe: str) -> dict:
    return {
        "status": "ok",
        "verdict": "PENDING",
        "channel": "probe",
        "notes": f"{probe} needs the JEPA encoder/locator; not yet implemented",
    }


PROBES = {
    "P1": lambda a: _stub("P1"),   # pose-recall computability (needs rotation/model)
    "P2": lambda a: _stub("P2"),   # map-conditioning inversion (needs model)
    "P3": probe_p3,                # trivial baselines (model-free) -- implemented
    "P4": lambda a: _stub("P4"),   # partial-map crossover (needs model)
    "P5": lambda a: _stub("P5"),   # wip rotation triage (needs training)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=list(PROBES))
    ap.add_argument("--scene", default="office", help="random-walk config key")
    ap.add_argument("--scene-folder", default=None,
                    help="processed folder (defaults to --scene)")
    ap.add_argument("--processed-root",
                    default=os.environ.get("SJEPA_PROCESSED_ROOT", "datasets_processed"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.scene_folder = args.scene_folder or args.scene

    t0 = time.time()
    result = PROBES[args.probe](args)
    result.setdefault("id", Path(args.out).name)
    result["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result.setdefault("gpu_h", round((time.time() - t0) / 3600.0, 5))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{args.probe}] {result.get('verdict')} -> {out/'result.json'}")
    print(json.dumps(result.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
