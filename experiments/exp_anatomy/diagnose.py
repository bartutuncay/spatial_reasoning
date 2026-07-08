"""A3 mechanism diagnostics: per-row information measures that should EXPLAIN the
capability dissociation.

Per row (arm via --objective, or frozen ref via --features-dir):
  i_pose_nats    InfoNCE lower bound on I(z; pose)  [pose = (loc, view_dir)]
  i_appear_nats  InfoNCE lower bound on I(z; appearance) [48-d color histogram]
  alignment      E||z1-z2||^2 over augmentation pairs (arms only; NaN for refs)
  uniformity     log E exp(-2||zi-zj||^2) on normalized z (Wang & Isola)
  eff_rank       effective rank of the eval embedding matrix
  smoothness     mean consecutive-step delta / mean pairwise distance (scale-free)

Hypothesis: discriminative-axis capability tracks uniformity + i_appear;
rollout capability tracks smoothness (low) + low eff_rank.
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
    SCENES, finish, load_walk, pretrain_encoder, split_seeds, to_uint8, walk_sequences,
)
from experiments.exp_jepa.fewshot import PRETRAIN_STEPS  # noqa: E402
from experiments.exp_jepa.jepa import effective_rank  # noqa: E402
from experiments.exp_jepa.locate import _load_module, _mlp  # noqa: E402


def _hist48(img):
    u = to_uint8(np.asarray(img))
    h = [np.histogram(u[..., c], bins=16, range=(0, 255))[0] for c in range(3)]
    h = np.concatenate(h).astype(np.float32)
    return h / (h.sum() + 1e-8)


def _infonce_bound(Ztr, Ttr, Zev, Tev, dev, steps=300, emb=64, seed=0):
    """Train critics f(z), g(t); return eval InfoNCE bound on I in nats
    (bound = log B - CE, capped by log B)."""
    torch.manual_seed(seed)
    if len(Ztr) < 4 or len(Zev) < 4:
        return float("nan")
    f = _mlp(Ztr.shape[1], 128, emb).to(dev)
    g = _mlp(Ttr.shape[1], 128, emb).to(dev)
    opt = torch.optim.AdamW(list(f.parameters()) + list(g.parameters()), lr=1e-3)
    X = torch.as_tensor(Ztr, dtype=torch.float32, device=dev)
    T = torch.as_tensor(Ttr, dtype=torch.float32, device=dev)
    B = min(256, len(X))
    for _ in range(steps):
        perm = torch.randperm(len(X), device=dev)[:B]
        a = F.normalize(f(X[perm]), dim=1)
        b = F.normalize(g(T[perm]), dim=1)
        logits = a @ b.T / 0.1
        lab = torch.arange(len(perm), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Xe = torch.as_tensor(Zev, dtype=torch.float32, device=dev)
        Te = torch.as_tensor(Tev, dtype=torch.float32, device=dev)
        Be = min(256, len(Xe))                       # eval batch (handle small sets)
        bounds = []
        for i in range(0, len(Xe) - Be + 1, Be):
            a = F.normalize(f(Xe[i:i + Be]), dim=1)
            b = F.normalize(g(Te[i:i + Be]), dim=1)
            logits = a @ b.T / 0.1
            lab = torch.arange(Be, device=dev)
            ce = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            bounds.append(float(np.log(Be) - ce.item()))
    return float(np.mean(bounds)) if bounds else float("nan")


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)

    # ---- gather per-frame Z, pose, appearance over ALL scenes, seed-split ----
    tr_files, ev_files, seq_ev = [], [], []
    for sc in SCENES:
        seqs = walk_sequences(root, sc)
        tr_s, ev_s = split_seeds(seqs, n_eval=max(1, len(seqs) // 5))
        tr = [f for fs in tr_s.values() for f in fs]
        ev = [f for fs in ev_s.values() for f in fs]
        if args.tier == "pilot":
            tr, ev = tr[:40], ev[:40]
            ev_s = {k: v[:10] for k, v in list(ev_s.items())[:2]}
        tr_files += tr; ev_files += ev
        seq_ev += list(ev_s.values())          # eval sequences for smoothness

    vae = None
    if args.features_dir:
        feat = {}
        for sc in SCENES:
            d = load_walk(Path(args.features_dir) / f"{sc}.pt")
            for f, z in zip(d["files"], np.asarray(d["features"])):
                feat[f] = z

        def encode(f):
            return feat[f]
    else:
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        if args.objective != "scratch":
            pretrain_encoder(vae, autoenc, [root / s / "random_walks" for s in SCENES],
                             args.objective, PRETRAIN_STEPS.get(args.tier, 200), dev)
        vae.img_enc.eval()
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

    def gather(files):
        Z, P, A = [], [], []
        for f in files:
            s = load_walk(f)
            Z.append(encode(f))
            P.append(np.concatenate([np.asarray(s["loc"], np.float32).reshape(3),
                                     np.asarray(s["view_dir"], np.float32).reshape(3)]))
            A.append(_hist48(s["img"]))
        return np.stack(Z), np.stack(P), np.stack(A)

    Ztr, Ptr, Atr = gather(tr_files)
    Zev, Pev, Aev = gather(ev_files)

    # standardize z for the critics (scale-invariant protocol, as depthprobe)
    fmu = Ztr.mean(0); fsd = Ztr.std(0) + 1e-6
    Ztr_s = (Ztr - fmu) / fsd; Zev_s = (Zev - fmu) / fsd

    i_pose = _infonce_bound(Ztr_s, Ptr, Zev_s, Pev, dev, seed=args.seed)
    i_appear = _infonce_bound(Ztr_s, Atr, Zev_s, Aev, dev, seed=args.seed + 1)

    # uniformity (normalized eval z)
    Zn = torch.as_tensor(Zev_s, dtype=torch.float32)
    Zn = F.normalize(Zn, dim=1)
    idx = torch.randperm(len(Zn))[:512]
    D2 = torch.cdist(Zn[idx], Zn[idx]) ** 2
    off = D2[~torch.eye(len(idx), dtype=torch.bool)]
    uniformity = float(torch.log(torch.exp(-2 * off).mean()))

    # alignment over augmentation pairs (arms only)
    alignment = float("nan")
    if vae is not None:
        import torchvision.transforms as TT
        aug = TT.Compose([TT.RandomResizedCrop((192, 256), scale=(0.6, 1.0), antialias=True),
                          TT.ColorJitter(0.4, 0.4, 0.4, 0.1)])
        vals = []
        for f in ev_files[:200]:
            s = load_walk(f)
            img = torch.as_tensor(np.asarray(s["img"]), dtype=torch.float32
                                  ).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                _, z1, _ = vae.img_enc(aug(img))
                _, z2, _ = vae.img_enc(aug(img))
            z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
            vals.append(float(((z1 - z2) ** 2).sum()))
        alignment = float(np.mean(vals))

    eff = effective_rank(torch.as_tensor(Zev_s, dtype=torch.float32))

    # smoothness: consecutive delta / mean pairwise, on eval sequences
    pos = {f: i for i, f in enumerate(ev_files)}
    deltas = []
    for fs in seq_ev:
        fs = [f for f in fs if f in pos]
        for a, b in zip(fs[:-1], fs[1:]):
            deltas.append(np.linalg.norm(Zev_s[pos[a]] - Zev_s[pos[b]]))
    pair_idx = np.random.default_rng(0).integers(0, len(Zev_s), size=(2000, 2))
    pair_d = np.linalg.norm(Zev_s[pair_idx[:, 0]] - Zev_s[pair_idx[:, 1]], axis=1)
    smoothness = float(np.mean(deltas) / (np.mean(pair_d) + 1e-8)) if deltas else float("nan")

    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    m = {"i_pose_nats": i_pose, "i_appear_nats": i_appear, "alignment": alignment,
         "uniformity": uniformity, "eff_rank": eff, "smoothness": smoothness,
         "log_batch_cap_nats": float(np.log(256)),
         "n_train": len(tr_files), "n_eval": len(ev_files),
         "row": row, "seed": args.seed, "tier": args.tier}
    return {"status": "ok", "verdict": "UNSCORED",       # diagnostics, not a capability
            "channel": f"diagnose:{row}", "primary": round(i_pose, 4),
            "metrics": m,
            "notes": (f"diagnose {row}: I(z;pose) {i_pose:.2f}, I(z;appear) {i_appear:.2f} nats; "
                      f"unif {uniformity:.2f}, align {alignment:.3f}, rank {eff:.1f}, "
                      f"smooth {smoothness:.3f}")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon", "rgb_only",
                             "fuse_cj_25", "fuse_cj_50", "fuse_cj_75"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--tier", default="shakedown")
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
