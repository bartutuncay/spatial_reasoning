"""C5 navigation-distance + C6 relative-pose (correspondence) probes.

From a PAIR of frozen image latents (z_i, z_j) of the same scene, read off a
geometric relationship:
  --channel navdist : regress ||loc_i - loc_j||  (metric spatial separation);
                      floor = train-mean distance; report R^2.
  --channel relpose : regress relative translation DIRECTION (unit loc_j-loc_i)
                      + relative heading cos<vd_i,vd_j>; floor = mean direction;
                      report direction cosine (higher = better).
Pairs are sampled WITHIN a scene (cross-scene coords are incomparable), seed-split.
Rows: --objective (self-contained pretrain) or --features-dir (frozen refs)."""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_anatomy.common import (  # noqa: E402
    SCENES, finish, load_walk, pretrain_encoder, scene_files, split_seeds, walk_sequences,
)
from experiments.exp_jepa.fewshot import PRETRAIN_STEPS  # noqa: E402
from experiments.exp_jepa.locate import _load_module, _mlp  # noqa: E402


def _scene_split_files(root, sc, rng):
    seqs = walk_sequences(root, sc)
    tr_seqs, ev_seqs = split_seeds(seqs, n_eval=max(1, len(seqs) // 5))
    tr = [f for fs in tr_seqs.values() for f in fs]
    ev = [f for fs in ev_seqs.values() for f in fs]
    return tr, ev


def _sample_pairs(files, n, rng):
    if len(files) < 2:
        return []
    i = rng.integers(0, len(files), size=n)
    j = rng.integers(0, len(files), size=n)
    keep = i != j
    return list(zip(np.asarray(files)[i[keep]], np.asarray(files)[j[keep]]))


def run(args):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)

    if args.features_dir:
        feat, loc = {}, {}
        for sc in SCENES:
            d = load_walk(Path(args.features_dir) / f"{sc}.pt")
            for f, z, l in zip(d["files"], np.asarray(d["features"]), np.asarray(d["loc"])):
                feat[f] = z; loc[f] = l
        dim = int(next(iter(feat.values())).shape[0])

        def encode(f):
            return feat[f]
    else:
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        if args.objective != "scratch":
            pretrain_encoder(vae, autoenc, [root / s / "random_walks" for s in SCENES],
                             args.objective, PRETRAIN_STEPS.get(args.tier, 200), dev)
        vae.img_enc.eval()
        dim = args.latent_dim
        _cache = {}

        def encode(f):
            if f not in _cache:
                s = load_walk(f)
                img = torch.as_tensor(np.asarray(s["img"]), dtype=torch.float32
                                      ).permute(2, 0, 1)[None].to(dev)
                with torch.no_grad():
                    _, mu, _ = vae.img_enc(img)
                _cache[f] = mu[0].cpu().numpy()
            return _cache[f]

    n_pairs = {"pilot": 200, "shakedown": 4000, "bulk": 20000}.get(args.tier, 4000)
    n_ev = max(50, n_pairs // 3)

    def targets(fi, fj):
        si, sj = load_walk(fi), load_walk(fj)
        li = np.asarray(si["loc"], dtype=np.float32).reshape(3)
        lj = np.asarray(sj["loc"], dtype=np.float32).reshape(3)
        vi = np.asarray(si["view_dir"], dtype=np.float32).reshape(3)
        vj = np.asarray(sj["view_dir"], dtype=np.float32).reshape(3)
        d = lj - li
        dist = float(np.linalg.norm(d))
        direction = d / (dist + 1e-8)
        heading = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8))
        return dist, direction, heading

    def build(split):
        Z, DIST, DIR, HEAD = [], [], [], []
        per_scene = (n_pairs if split == "tr" else n_ev) // len(SCENES) + 1
        for sc in SCENES:
            tr_f, ev_f = _scene_split_files(root, sc, rng)
            files = tr_f if split == "tr" else ev_f
            for fi, fj in _sample_pairs(files, per_scene, rng):
                dist, direction, heading = targets(fi, fj)
                Z.append(np.concatenate([encode(fi), encode(fj)]))
                DIST.append(dist); DIR.append(direction); HEAD.append(heading)
        return (np.stack(Z), np.asarray(DIST, np.float32),
                np.stack(DIR), np.asarray(HEAD, np.float32))

    Ztr, Dtr, DIRtr, Htr = build("tr")
    Zev, Dev, DIRev, Hev = build("ev")
    Xtr = torch.as_tensor(Ztr, dtype=torch.float32, device=dev)
    Xev = torch.as_tensor(Zev, dtype=torch.float32, device=dev)

    if args.channel == "navdist":
        head = _mlp(2 * dim, dim, 1).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        y = torch.as_tensor(Dtr, dtype=torch.float32, device=dev)[:, None]
        for _ in range(args.head_steps):
            perm = torch.randperm(len(Xtr), device=dev)[:256]
            loss = F.smooth_l1_loss(head(Xtr[perm]), y[perm])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = head(Xev).cpu().numpy().reshape(-1)
        ss_res = float(((pred - Dev) ** 2).sum())
        ss_tot = float(((Dev - Dtr.mean()) ** 2).sum()) + 1e-8
        r2 = 1.0 - ss_res / ss_tot
        mae = float(np.abs(pred - Dev).mean())
        floor_mae = float(np.abs(Dev - Dtr.mean()).mean())
        row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
        return {"status": "ok",
                "verdict": "EXISTS" if r2 > 0.02 else "WEAK",
                "channel": f"navdist:{row}", "primary": round(r2, 4),
                "metrics": {"r2": r2, "mae_m": mae, "floor_mae_m": floor_mae,
                            "n_train": int(len(Dtr)), "n_eval": int(len(Dev)),
                            "dim": dim, "row": row, "seed": args.seed, "tier": args.tier},
                "notes": f"navdist {row}: R2 {r2:.3f}, MAE {mae:.2f}m (floor {floor_mae:.2f}m)"}

    # relpose: predict [unit-direction(3), heading(1)]
    head = _mlp(2 * dim, dim, 4).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    tgt = torch.as_tensor(np.concatenate([DIRtr, Htr[:, None]], 1), dtype=torch.float32, device=dev)
    for _ in range(args.head_steps):
        perm = torch.randperm(len(Xtr), device=dev)[:256]
        out = head(Xtr[perm])
        d = F.normalize(out[:, :3], dim=1)
        loss = (1 - (d * F.normalize(tgt[perm, :3], dim=1)).sum(1)).mean() \
            + F.smooth_l1_loss(out[:, 3], tgt[perm, 3])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = head(Xev).cpu().numpy()
    pdir = out[:, :3] / (np.linalg.norm(out[:, :3], axis=1, keepdims=True) + 1e-8)
    dir_cos = float((pdir * DIRev).sum(1).mean())
    mean_dir = DIRtr.mean(0); mean_dir /= (np.linalg.norm(mean_dir) + 1e-8)
    floor_cos = float((DIRev @ mean_dir).mean())
    head_mae = float(np.abs(out[:, 3] - Hev).mean())
    floor_head_mae = float(np.abs(Hev - Htr.mean()).mean())
    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    return {"status": "ok",
            "verdict": "EXISTS" if dir_cos > floor_cos + 0.03 else "WEAK",
            "channel": f"relpose:{row}", "primary": round(dir_cos, 4),
            "metrics": {"dir_cos": dir_cos, "floor_dir_cos": floor_cos,
                        "heading_mae": head_mae, "floor_heading_mae": floor_head_mae,
                        "n_train": int(len(Htr)), "n_eval": int(len(Hev)),
                        "dim": dim, "row": row, "seed": args.seed, "tier": args.tier},
            "notes": f"relpose {row}: dir-cos {dir_cos:.3f} (floor {floor_cos:.3f}), "
                     f"heading-MAE {head_mae:.3f} (floor {floor_head_mae:.3f})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon", "rgb_only"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--channel", default="navdist", choices=["navdist", "relpose"])
    ap.add_argument("--tier", default="shakedown")
    ap.add_argument("--head-steps", type=int, default=400)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--processed-root",
                    default=os.environ.get("SJEPA_PROCESSED_ROOT", "data_bartu"))
    ap.add_argument("--seed", type=int, default=0)
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
