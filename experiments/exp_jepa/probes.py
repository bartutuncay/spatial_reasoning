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


def probe_p3(args) -> dict:
    """Predict-scene-centroid baseline: the floor the locator must beat."""
    node_pos = _scene_node_positions(args.processed_root, args.scene_folder)
    gt = _walk_gt_positions(args.processed_root, args.scene_folder)
    centroid = node_pos.mean(axis=0)
    pred = np.tile(centroid, (gt.shape[0], 1))
    errs = translation_errors(pred, gt)
    scene_extent = float(np.linalg.norm(node_pos.max(0) - node_pos.min(0)))
    return {
        "status": "ok",
        "verdict": "EXISTS",
        "channel": "probe",
        "primary": round(float(ate_rmse(pred, gt)), 4),
        "metrics": {
            "centroid_ate_rmse_m": float(ate_rmse(pred, gt)),
            "centroid_err_median_m": float(np.median(errs)),
            "centroid_err_p90_m": float(np.percentile(errs, 90)),
            "n_queries": int(gt.shape[0]),
            "n_scene_nodes": int(node_pos.shape[0]),
            "scene_diag_extent_m": scene_extent,
        },
        "notes": "predict-scene-centroid floor; learned locator must beat this per-sample",
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

    result = PROBES[args.probe](args)
    result.setdefault("id", Path(args.out).name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{args.probe}] {result.get('verdict')} -> {out/'result.json'}")
    print(json.dumps(result.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
