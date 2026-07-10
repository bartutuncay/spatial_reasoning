"""Mechanism INTERVENTION: remove the linearly-decodable appearance subspace
from a row's latents and measure the place-recognition drop, against a
random-subspace-removal control of equal dimension.

Converts the correlational mechanism claim (I(z;appearance) tracks the
discriminative axis, Fig. 3) into a causal one: if appearance information is
what discrimination uses, deleting it should collapse place-rec far more than
deleting a random subspace of the same size.

Protocol per (row, seed):
  1. encode all frames (encoder ckpt / in-job pretrain / features-dir),
     per-scene seed-split like placerec;
  2. ridge-fit W: z -> 48-d color histogram on train; appearance subspace =
     top-k right singular vectors of W (k = 95% spectral mass, capped);
  3. probe place-rec on z, on z with the appearance subspace projected out,
     and on z with a random k-dim subspace projected out.
Channel intervene:<row>; primary = acc drop (appearance-removed vs base).
"""
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
    SCENES, build_encoder, encode_arm_latents, finish, load_walk, scene_files,
    seed_split,
)
from experiments.exp_anatomy.diagnose import _hist48  # noqa: E402
from experiments.exp_jepa.locate import _load_module  # noqa: E402


def _probe_acc(Ztr, ytr, Zev, yev, dev, steps=300, seed=0):
    torch.manual_seed(seed)
    D, C = Ztr.shape[1], len(SCENES)
    head = torch.nn.Linear(D, C).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    X = torch.as_tensor(Ztr, dtype=torch.float32, device=dev)
    T = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    for _ in range(steps):
        perm = torch.randperm(len(X), device=dev)[:256]
        loss = F.cross_entropy(head(X[perm]), T[perm])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(torch.as_tensor(Zev, dtype=torch.float32, device=dev)).argmax(1).cpu().numpy()
    return float((pred == yev).mean())


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)

    # ---- gather features + scene labels + appearance targets, seed-split ----
    Ztr, ytr, Atr = [], [], []
    Zev, yev = [], []
    if args.features_dir:
        for si, sc in enumerate(SCENES):
            d = torch.load(Path(args.features_dir) / f"{sc}.pt", weights_only=False)
            tr_idx, ev_idx = seed_split(d["files"])
            pos = {f: i for i, f in enumerate(d["files"])}
            feats = np.asarray(d["features"])
            Ztr.append(feats[[pos[f] for f in tr_idx]]); ytr.append(np.full(len(tr_idx), si))
            Zev.append(feats[[pos[f] for f in ev_idx]]); yev.append(np.full(len(ev_idx), si))
            Atr.append(np.stack([_hist48(load_walk(f)["img"]) for f in tr_idx]))
    else:
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        build_encoder(vae, autoenc, root, args, dev)
        vae.img_enc.eval()
        for si, sc in enumerate(SCENES):
            tr_f, ev_f = seed_split(scene_files(root, sc))
            if args.tier == "pilot":
                tr_f, ev_f = tr_f[:8], ev_f[:8]
            ztr, _, _ = encode_arm_latents(vae, tr_f, dev)
            zev, _, _ = encode_arm_latents(vae, ev_f, dev)
            Ztr.append(ztr); ytr.append(np.full(len(tr_f), si))
            Zev.append(zev); yev.append(np.full(len(ev_f), si))
            Atr.append(np.stack([_hist48(load_walk(f)["img"]) for f in tr_f]))
    Ztr = np.concatenate(Ztr); ytr = np.concatenate(ytr); Atr = np.concatenate(Atr)
    Zev = np.concatenate(Zev); yev = np.concatenate(yev)
    D = Ztr.shape[1]

    # ---- standardize (train stats), ridge-fit z -> appearance ---------------
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    Zs_tr = (Ztr - mu) / sd; Zs_ev = (Zev - mu) / sd
    Ac = Atr - Atr.mean(0)
    lam = 1e-2 * len(Zs_tr)
    W = np.linalg.solve(Zs_tr.T @ Zs_tr + lam * np.eye(D), Zs_tr.T @ Ac)  # (D,48)
    # appearance subspace = top-k left singular vectors of W in z-space
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    k_int = int(np.searchsorted(np.cumsum(S**2) / (S**2).sum(), 0.95) + 1)
    k = int(min(max(k_int, 1), args.max_k, D - 1))
    V = U[:, :k]                                       # (D,k), orthonormal

    def remove(Z, B):
        return Z - (Z @ B) @ B.T

    rng = np.random.default_rng(args.seed)
    R, _ = np.linalg.qr(rng.standard_normal((D, k)))   # random k-dim control

    acc_base = _probe_acc(Zs_tr, ytr, Zs_ev, yev, dev, seed=args.seed)
    acc_noapp = _probe_acc(remove(Zs_tr, V), ytr, remove(Zs_ev, V), yev, dev, seed=args.seed)
    acc_norand = _probe_acc(remove(Zs_tr, R), ytr, remove(Zs_ev, R), yev, dev, seed=args.seed)

    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    drop_app = acc_base - acc_noapp
    drop_rand = acc_base - acc_norand
    return {"status": "ok",
            "verdict": "EXISTS" if drop_app > drop_rand + 0.05 else "WEAK",
            "channel": f"intervene:{row}", "primary": round(drop_app, 4),
            "metrics": {"acc_base": acc_base, "acc_no_appearance": acc_noapp,
                        "acc_no_random": acc_norand, "drop_appearance": drop_app,
                        "drop_random": drop_rand, "k": k, "dim": int(D),
                        "row": row, "seed": args.seed, "tier": args.tier},
            "notes": f"intervene {row}: base {acc_base:.3f}, -appearance({k}d) "
                     f"{acc_noapp:.3f} (drop {drop_app:+.3f}), -random({k}d) "
                     f"{acc_norand:.3f} (drop {drop_rand:+.3f})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--max-k", type=int, default=32)
    ap.add_argument("--tier", default="bulk")
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
