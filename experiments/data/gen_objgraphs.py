"""Per-scene object tables (the "3D model"/FBX-equivalent modality) from
Replica's semantic annotations, transformed into the SAME gravity-aligned
frame the walks use (recomputes gen_replica_walks.gravity_align on the vertex
cloud - deterministic, so the transform matches the rendered walks exactly).

Output: <out-root>/<scene>.pt with {centroid [N,3], extent [N,3], cls [N]}
consumed by exp_anatomy.modality.obj_tokens.

Usage: python -m experiments.data.gen_objgraphs --scene frl_apartment_0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def gravity_transform(pts):
    """Replicate gen_replica_walks.gravity_align decisions; return (order, flip)."""
    ext = pts.max(0) - pts.min(0)
    up_ax = int(np.argmin(ext))
    order = list(range(3)) if up_ax == 2 else [i for i in range(3) if i != up_ax] + [up_ax]
    p = pts[:, order]
    z = p[:, 2]
    lo = (z <= z.min() + 0.5).sum()
    hi = (z >= z.max() - 0.5).sum()
    return order, bool(hi > lo)


def apply(points, order, flip):
    p = points[:, order].copy()
    if flip:
        p[:, 2] = -p[:, 2]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--replica-root", default="datasets_replica/replica_v1")
    ap.add_argument("--out-root", default="datasets_replica/objects")
    args = ap.parse_args()

    sroot = Path(args.replica_root) / args.scene
    info = json.load(open(sroot / "habitat" / "info_semantic.json"))
    from plyfile import PlyData
    v = PlyData.read(str(sroot / "mesh.ply"))["vertex"].data
    verts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    order, flip = gravity_transform(verts)

    cen, ext, cls = [], [], []
    for o in info.get("objects", []):
        bb = o.get("oriented_bbox", {}).get("abb")
        if not bb:
            continue
        cen.append(bb["center"]); ext.append(np.abs(bb["sizes"]))
        cls.append(int(o.get("class_id", 0)))
    cen = apply(np.asarray(cen, np.float32), order, flip)
    ext = np.asarray(ext, np.float32)[:, order]
    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    torch.save({"centroid": cen, "extent": ext,
                "cls": np.asarray(cls, np.int64)}, out / f"{args.scene}.pt")
    print(f"{args.scene}: {len(cen)} objects (order={order} flip={flip}) "
          f"-> {out / (args.scene + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
