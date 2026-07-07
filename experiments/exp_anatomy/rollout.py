"""C4 latent rollout: predict a future frame's latent from the current latent +
relative camera motion, along random-walk trajectories. Tests whether the space
supports forward dynamics (world-model capability).
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
    SCENES, finish, load_walk, pretrain_encoder, relative_action,
    split_seeds, walk_sequences,
)
from experiments.exp_jepa.fewshot import PRETRAIN_STEPS  # noqa: E402
from experiments.exp_jepa.locate import _load_module, _mlp  # noqa: E402


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)

    if args.features_dir:
        feat = {}
        for sc in SCENES:
            d = load_walk(Path(args.features_dir) / f"{sc}.pt")
            for f, z in zip(d["files"], np.asarray(d["features"])):
                feat[f] = z
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

    # collect all walks across scenes, split by (global) seed index
    all_seqs = {}
    for sc in SCENES:
        for seed, fs in walk_sequences(root, sc).items():
            all_seqs[f"{sc}:{seed}"] = fs
    idx_seqs = {i: v for i, v in enumerate(all_seqs.values())}
    tr_seqs, ev_seqs = split_seeds(idx_seqs, n_eval=max(2, len(idx_seqs) // 6))
    if args.tier == "pilot":
        tr_seqs = dict(list(tr_seqs.items())[:3]); ev_seqs = dict(list(ev_seqs.items())[:2])

    K = args.horizon

    def build(seqs):
        Z, A, Y = [], [], []
        for fs in seqs.values():
            for t in range(len(fs) - K):
                si, sj = load_walk(fs[t]), load_walk(fs[t + K])
                Z.append(encode(fs[t])); Y.append(encode(fs[t + K]))
                A.append(relative_action(si, sj))
        if not Z:
            return None, None, None
        return np.stack(Z), np.stack(A), np.stack(Y)

    Ztr, Atr, Ytr = build(tr_seqs)
    Zev, Aev, Yev = build(ev_seqs)
    if Ztr is None or Zev is None:
        raise RuntimeError(f"no rollout pairs at horizon {K}")

    g = _mlp(dim + 6, 2 * dim, dim).to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3)
    X = torch.as_tensor(np.concatenate([Ztr, Atr], 1), dtype=torch.float32, device=dev)
    Tt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    for _ in range(args.head_steps):
        perm = torch.randperm(len(X), device=dev)[:256]
        loss = F.mse_loss(g(X[perm]), Tt[perm])
        opt.zero_grad(); loss.backward(); opt.step()

    def cosd(a, b):
        a = F.normalize(torch.as_tensor(a, dtype=torch.float32), dim=1)
        b = F.normalize(torch.as_tensor(b, dtype=torch.float32), dim=1)
        return float((1 - (a * b).sum(1)).mean())

    with torch.no_grad():
        Xe = torch.as_tensor(np.concatenate([Zev, Aev], 1), dtype=torch.float32, device=dev)
        pred = g(Xe).cpu().numpy()
        Ash = Aev[np.random.permutation(len(Aev))]
        Xsh = torch.as_tensor(np.concatenate([Zev, Ash], 1), dtype=torch.float32, device=dev)
        pred_sh = g(Xsh).cpu().numpy()
    err = cosd(pred, Yev)               # model prediction
    floor = cosd(Zev, Yev)              # copy-last (predict current frame)
    shuf = cosd(pred_sh, Yev)           # shuffled-action control
    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    return {"status": "ok",
            "verdict": "EXISTS" if err < floor - 0.005 else "WEAK",
            "channel": f"rollout:{row}", "primary": round(err, 5),
            "metrics": {"cosdist": err, "copylast_floor": floor,
                        "shuffled_action": shuf, "action_gap": shuf - err,
                        "horizon": K, "n_train": int(len(Ztr)), "n_eval": int(len(Zev)),
                        "dim": dim, "row": row, "seed": args.seed, "tier": args.tier},
            "notes": f"rollout {row} k={K}: cosd {err:.4f} (copy-last {floor:.4f}, "
                     f"shuffled {shuf:.4f}, action-gap {shuf - err:+.4f})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon", "rgb_only"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--horizon", type=int, default=2)
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
