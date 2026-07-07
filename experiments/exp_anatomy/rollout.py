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

    # Probe in a STANDARDIZED latent space (per-dim z-score from train stats):
    # raw latents have a huge static/DC component and tiny frame-to-frame motion,
    # so a raw-space delta-R^2 is dominated by a near-zero denominator and the head
    # cannot fit the tiny target. Standardizing makes every dim O(1) and the motion
    # signal well-conditioned; deltas below are all in standardized space.
    mu = Ztr.mean(0); sd = Ztr.std(0) + 1e-6
    amu = Atr.mean(0); asd = Atr.std(0) + 1e-6
    Zs_tr = (Ztr - mu) / sd; Ys_tr = (Ytr - mu) / sd
    Zs_ev = (Zev - mu) / sd; Ys_ev = (Yev - mu) / sd
    As_tr = (Atr - amu) / asd; As_ev = (Aev - amu) / asd

    # Predictor outputs the standardized DELTA z'_{t+k}-z'_t (residual prediction):
    # an untrained head outputs ~0 -> R^2~0 (copy-last floor), so the metric is
    # bounded and only genuine motion prediction from [z', action] earns positive R^2.
    Dtr = (Ys_tr - Zs_tr).astype(np.float32)
    g = _mlp(dim + 6, 2 * dim, dim).to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3)
    X = torch.as_tensor(np.concatenate([Zs_tr, As_tr], 1), dtype=torch.float32, device=dev)
    Tt = torch.as_tensor(Dtr, dtype=torch.float32, device=dev)
    for _ in range(args.head_steps):
        perm = torch.randperm(len(X), device=dev)[:256]
        loss = F.mse_loss(g(X[perm]), Tt[perm])
        opt.zero_grad(); loss.backward(); opt.step()

    # Score the RESIDUAL (delta), not the absolute latent: consecutive frames are
    # near-identical so a DC-dominated absolute-cosine makes copy-last unbeatable.
    # delta-R^2 = 1 - ||pred_delta - true_delta||^2 / ||true_delta||^2; copy-last
    # (pred=z_t -> pred_delta=0) gives R^2=0 by construction, so any positive R^2
    # means the model predicts the motion from [z_t, action].
    def delta_r2(pred_delta, true_delta):
        ss_res = float(((pred_delta - true_delta) ** 2).sum())
        ss_tot = float((true_delta ** 2).sum()) + 1e-8
        return 1.0 - ss_res / ss_tot

    def delta_cos(pred_delta, true_delta):
        pd = F.normalize(torch.as_tensor(pred_delta, dtype=torch.float32), dim=1)
        td = F.normalize(torch.as_tensor(true_delta, dtype=torch.float32), dim=1)
        return float((pd * td).sum(1).mean())

    true_delta = (Ys_ev - Zs_ev).astype(np.float32)
    with torch.no_grad():
        Xe = torch.as_tensor(np.concatenate([Zs_ev, As_ev], 1), dtype=torch.float32, device=dev)
        pred = g(Xe).cpu().numpy()          # predicted standardized DELTA
        Ash = As_ev[np.random.permutation(len(As_ev))]
        Xsh = torch.as_tensor(np.concatenate([Zs_ev, Ash], 1), dtype=torch.float32, device=dev)
        pred_sh = g(Xsh).cpu().numpy()
    r2 = delta_r2(pred, true_delta)         # model (copy-last floor = 0 by construction)
    r2_shuf = delta_r2(pred_sh, true_delta)  # shuffled-action control
    dcos = delta_cos(pred, true_delta)      # direction of predicted motion
    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    return {"status": "ok",
            "verdict": "EXISTS" if r2 > 0.02 else "WEAK",
            "channel": f"rollout:{row}", "primary": round(r2, 4),
            "metrics": {"delta_r2": r2, "copylast_floor_r2": 0.0,
                        "shuffled_action_r2": r2_shuf, "action_gap": r2 - r2_shuf,
                        "delta_cos": dcos, "horizon": K, "n_train": int(len(Ztr)),
                        "n_eval": int(len(Zev)), "dim": dim, "row": row,
                        "seed": args.seed, "tier": args.tier},
            "notes": f"rollout {row} k={K}: delta-R2 {r2:.3f} (floor 0), "
                     f"shuffled-R2 {r2_shuf:.3f}, action-gap {r2 - r2_shuf:+.3f}, "
                     f"delta-cos {dcos:.3f}"}


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
