"""C7 place recognition: which scene is this view from? Frozen encoder + linear
probe over 5 ETH3D scenes; floor = majority class; also NN retrieval recall@1.
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
    SCENES, encode_arm_latents, finish, pretrain_encoder, scene_files, seed_split,
)
from experiments.exp_jepa.fewshot import PRETRAIN_STEPS  # noqa: E402
from experiments.exp_jepa.locate import _load_module  # noqa: E402


def _gather(args, dev):
    """(Ztr, ytr, Zev, yev): features + scene labels per seed-split."""
    Ztr, ytr, Zev, yev = [], [], [], []
    if args.features_dir:                       # reference row: precomputed
        for si, sc in enumerate(SCENES):
            d = torch.load(Path(args.features_dir) / f"{sc}.pt", weights_only=False)
            tr_idx, ev_idx = seed_split(d["files"])
            pos = {f: i for i, f in enumerate(d["files"])}
            feats = np.asarray(d["features"])
            Ztr.append(feats[[pos[f] for f in tr_idx]])
            ytr.append(np.full(len(tr_idx), si))
            Zev.append(feats[[pos[f] for f in ev_idx]])
            yev.append(np.full(len(ev_idx), si))
    else:                                       # trained arm: pretrain then encode
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        root = Path(args.processed_root)
        if args.objective != "scratch":
            pretrain_encoder(vae, autoenc, [root / s / "random_walks" for s in SCENES],
                             args.objective, PRETRAIN_STEPS.get(args.tier, 200), dev)
        vae.img_enc.eval()
        for si, sc in enumerate(SCENES):
            tr_f, ev_f = seed_split(scene_files(root, sc))
            if args.tier == "pilot":
                tr_f, ev_f = tr_f[:8], ev_f[:8]
            ztr, _, _ = encode_arm_latents(vae, tr_f, dev)
            zev, _, _ = encode_arm_latents(vae, ev_f, dev)
            Ztr.append(ztr); ytr.append(np.full(len(tr_f), si))
            Zev.append(zev); yev.append(np.full(len(ev_f), si))
    return (np.concatenate(Ztr), np.concatenate(ytr),
            np.concatenate(Zev), np.concatenate(yev))


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ztr, ytr, Zev, yev = _gather(args, dev)
    D = Ztr.shape[1]

    head = torch.nn.Linear(D, len(SCENES)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    Xtr = torch.as_tensor(Ztr, dtype=torch.float32, device=dev)
    Ttr = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    for _ in range(args.head_steps):
        perm = torch.randperm(len(Xtr), device=dev)[:256]
        loss = F.cross_entropy(head(Xtr[perm]), Ttr[perm])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        logits = head(torch.as_tensor(Zev, dtype=torch.float32, device=dev))
    pred = logits.argmax(1).cpu().numpy()
    acc = float((pred == yev).mean())
    floor = float(np.bincount(yev, minlength=len(SCENES)).max() / len(yev))

    # NN retrieval recall@1 (cosine, train bank) — probe-free discriminability
    A = Ztr / (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-8)
    B = Zev / (np.linalg.norm(Zev, axis=1, keepdims=True) + 1e-8)
    nn_acc = float((ytr[np.argmax(B @ A.T, axis=1)] == yev).mean())

    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    return {"status": "ok",
            "verdict": "EXISTS" if acc > floor + 0.05 else "WEAK",
            "channel": f"placerec:{row}", "primary": round(acc, 4),
            "metrics": {"top1_acc": acc, "majority_floor": floor,
                        "nn_recall1": nn_acc, "n_train": int(len(ytr)),
                        "n_eval": int(len(yev)), "dim": int(D), "row": row,
                        "seed": args.seed, "tier": args.tier},
            "notes": f"placerec {row}: acc {acc:.3f} (floor {floor:.3f}, NN@1 {nn_acc:.3f})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon", "rgb_only",
                             "fuse_cj_25", "fuse_cj_50", "fuse_cj_75"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--tier", default="shakedown")
    ap.add_argument("--head-steps", type=int, default=300)
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
