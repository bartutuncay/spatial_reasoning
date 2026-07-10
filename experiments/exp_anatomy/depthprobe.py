"""C3 metric-depth probe: linear head from the frozen global latent to a 16x16
log-depth grid. Seed-split within every scene; floor = train-mean grid.
Rows: --objective (self-contained pretrain) or --features-dir (frozen refs)."""
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
    SCENES, build_encoder, depth_grid, encode_arm_latents, finish, scene_files, seed_split,
)
from experiments.exp_jepa.locate import _load_module  # noqa: E402

G = 16  # grid side


def _load_features(features_dir):
    Z, files = [], []
    for sc in SCENES:
        d = torch.load(Path(features_dir) / f"{sc}.pt", weights_only=False)
        Z.append(np.asarray(d["features"])); files += list(d["files"])
    return np.concatenate(Z), files


def _depth_targets(files):
    Y, M = [], []
    for f in files:
        s = torch.load(f, map_location="cpu", weights_only=False)
        g, m = depth_grid(torch.as_tensor(np.asarray(s["depth"], dtype=np.float32)), G)
        Y.append(g); M.append(m)
    return torch.stack(Y), torch.stack(M)


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)

    tr_files, ev_files = [], []
    for sc in SCENES:
        tr, ev = seed_split(scene_files(root, sc))
        if args.tier == "pilot":
            tr, ev = tr[:8], ev[:8]
        tr_files += tr; ev_files += ev

    if args.features_dir:
        Z, files = _load_features(args.features_dir)
        pos = {f: i for i, f in enumerate(files)}
        Ztr = Z[[pos[f] for f in tr_files]]; Zev = Z[[pos[f] for f in ev_files]]
    else:
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        args._pre_steps = build_encoder(vae, autoenc, root, args, dev)
        vae.img_enc.eval()
        Ztr, _, _ = encode_arm_latents(vae, tr_files, dev)
        Zev, _, _ = encode_arm_latents(vae, ev_files, dev)

    Ytr, Mtr = _depth_targets(tr_files)
    Yev, Mev = _depth_targets(ev_files)

    # Scale-invariant probe protocol: z-score features by TRAIN stats before the
    # linear head. Un-normalized rows (e.g. rgb_only InfoNCE latents with huge
    # norms) otherwise diverge the head during training -> inf/NaN AbsRel.
    fmu = Ztr.mean(0); fsd = Ztr.std(0) + 1e-6
    Ztr = (Ztr - fmu) / fsd
    Zev = (Zev - fmu) / fsd

    head = (torch.nn.Sequential(torch.nn.Linear(Ztr.shape[1], 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, G * G))
            if args.head == "mlp" else torch.nn.Linear(Ztr.shape[1], G * G)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    X = torch.as_tensor(Ztr, dtype=torch.float32, device=dev)
    Y = Ytr.to(dev); M = Mtr.float().to(dev)
    for _ in range(args.head_steps):
        perm = torch.randperm(len(X), device=dev)[:128]
        err = (head(X[perm]) - Y[perm]) ** 2
        loss = (err * M[perm]).sum() / M[perm].sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()

    def absrel_d125(pred_log, y_log, m):
        # Clamp to a physical depth range [1mm, 1km] before exp: a divergent linear
        # head (seen with high-norm rgb_only latents) can emit huge log-depths that
        # overflow to inf. Clamping bounds the metric without touching in-range rows.
        p = torch.exp(pred_log[m].clamp(-6.9, 6.9))
        y = torch.exp(y_log[m])
        absrel = float(((p - y).abs() / y).mean())
        d125 = float((torch.maximum(p / y, y / p) < 1.25).float().mean())
        return absrel, d125

    with torch.no_grad():
        Pev = head(torch.as_tensor(Zev, dtype=torch.float32, device=dev)).cpu()
    absrel, d125 = absrel_d125(Pev, Yev, Mev)
    mean_grid = (Ytr * Mtr.float()).sum(0) / Mtr.float().sum(0).clamp(min=1)  # train-mean floor
    fl_absrel, fl_d125 = absrel_d125(mean_grid.expand_as(Yev), Yev, Mev)

    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    return {"status": "ok",
            "verdict": "EXISTS" if absrel < fl_absrel - 0.01 else "WEAK",
            "channel": f"depthprobe:{row}", "primary": round(absrel, 4),
            "metrics": {"absrel": absrel, "delta125": d125,
                        "floor_absrel": fl_absrel, "floor_delta125": fl_d125,
                        "n_train": len(tr_files), "n_eval": len(ev_files),
                        "head": args.head,
                        "pretrain_steps": getattr(args, "_pre_steps", None),
                        "row": row, "seed": args.seed, "tier": args.tier},
            "notes": f"depth {row}: AbsRel {absrel:.3f} (floor {fl_absrel:.3f}), "
                     f"d1.25 {d125:.3f} (floor {fl_d125:.3f})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon", "rgb_only",
                             "fuse_cj_25", "fuse_cj_50", "fuse_cj_75"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--encoder-ckpt", default=None)   # reuse a pretrain_ckpt encoder
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
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
