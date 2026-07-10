"""Per-scene top-down occupancy maps ("architectural 2D layouts") for the
conditioning-modality axis.

Dataset-agnostic: world points are reconstructed from the walk frames
themselves (graph.pos is camera-aligned: world = pos[:, :3] @ R + loc with
R = rotation_a_to_b(view_dir, +x)), so the map lives in exactly the frame the
probes' loc/view_dir use - no re-derivation of dataset-specific transforms.

Output: <out-root>/<scene>.pt with {occ [H,W] float, hmax [H,W] float,
origin (x0,y0), res} consumed by exp_anatomy.modality.layout_crop.

Usage: python -m experiments.data.gen_layouts --walks-root datasets_replica/walks \
           --scene frl_apartment_0 --out-root datasets_replica/layouts
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent_src" / "rw_small_step"))
from pcd_slice_methods import rotation_a_to_b  # noqa: E402

RES = 0.1
FRAME_STRIDE = 3         # every 3rd frame is plenty for a static map
PTS_PER_FRAME = 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    files = sorted((Path(args.walks_root) / args.scene / "random_walks").glob("*.pt"))
    files = files[::FRAME_STRIDE]
    if not files:
        raise FileNotFoundError(f"no walks for {args.scene}")
    rng = np.random.default_rng(0)
    W = []
    for f in files:
        s = torch.load(f, map_location="cpu", weights_only=False)
        pos = np.asarray(s["graph"].pos, np.float32)[:, :3]
        if len(pos) > PTS_PER_FRAME:
            pos = pos[rng.choice(len(pos), PTS_PER_FRAME, replace=False)]
        vd = np.asarray(s["view_dir"], np.float32).reshape(3)
        loc = np.asarray(s["loc"], np.float32).reshape(3)
        R = rotation_a_to_b(vd, np.array([1.0, 0.0, 0.0]))
        W.append(pos @ R + loc)                    # aligned rows = (world-loc)@R.T
    W = np.concatenate(W)
    x0, y0 = W[:, 0].min() - 0.5, W[:, 1].min() - 0.5
    nx = int(np.ceil((W[:, 0].max() + 0.5 - x0) / RES))
    ny = int(np.ceil((W[:, 1].max() + 0.5 - y0) / RES))
    occ = np.zeros((nx, ny), np.float32)
    hmax = np.zeros((nx, ny), np.float32)
    ix = np.clip(((W[:, 0] - x0) / RES).astype(int), 0, nx - 1)
    iy = np.clip(((W[:, 1] - y0) / RES).astype(int), 0, ny - 1)
    occ[ix, iy] = 1.0
    z = W[:, 2] - W[:, 2].min()
    np.maximum.at(hmax, (ix, iy), z)
    hmax /= max(hmax.max(), 1e-6)
    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    torch.save({"occ": occ, "hmax": hmax, "origin": (float(x0), float(y0)),
                "res": RES}, out / f"{args.scene}.pt")
    print(f"{args.scene}: layout {nx}x{ny} cells from {len(W)} pts "
          f"({occ.mean():.2%} occupied) -> {out / (args.scene + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
