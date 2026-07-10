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
    SCENES, build_encoder, finish, load_walk, relative_action,
    split_seeds, walk_sequences,
)
from experiments.exp_jepa.locate import _load_module, _mlp  # noqa: E402


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)

    # Per-file caches populated in ONE pass (see below): z latent, loc, view_dir.
    # Loading each file once (not per-pair) and batching the encode is essential —
    # branch-walk .pt files carry large scene graphs, so per-pair load_walk is the
    # dominant cost and blew the walltime.
    zc, locc, vdc = {}, {}, {}
    vae = None
    pre_steps = None
    if args.oracle_state:
        # Positive control (harness validation): the "representation" is the
        # ground-truth camera state [loc, view_dir]. Next-state is then exactly
        # determined by the action, so the shuffle gap MUST be large if the
        # harness can detect action-conditioning at all.
        dim = 6
    elif args.features_dir:
        feat = {}
        for sc in SCENES:
            p = Path(args.features_dir) / f"{sc}.pt"
            if not p.exists():
                continue                      # e.g. hospital has no branch walks
            d = load_walk(p)
            for f, z, l, v in zip(d["files"], np.asarray(d["features"]),
                                  np.asarray(d["loc"]), np.asarray(d["view_dir"])):
                zc[f] = z; locc[f] = l; vdc[f] = v
        if not zc:
            raise FileNotFoundError(f"no feature files under {args.features_dir}")
        dim = int(next(iter(zc.values())).shape[0])
    else:
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        pre_steps = build_encoder(vae, autoenc, root, args, dev)
        vae.img_enc.eval()
        dim = args.latent_dim
        print("encoder ready, building trajectories", flush=True)

    # collect trajectories across scenes.
    #   smooth   : Bartu's random walks, split by walk seed.
    #   branching: same-anchor multi-action walks (gen_branch_walks); split by
    #              ANCHOR so all branches of an anchor stay in one split — the
    #              eval set then contains same-state different-action futures,
    #              which is what makes the action-gap meaningful.
    all_seqs = {}
    if args.walks == "branching":
        from experiments.exp_anatomy.common import branch_sequences
        wroot = Path(args.walks_root)
        scenes_present = [sc for sc in SCENES if branch_sequences(wroot, sc)]
        if not scenes_present:
            raise FileNotFoundError(f"no branch walks under {wroot}")
        for sc in scenes_present:
            for (anchor, branch), fs in branch_sequences(wroot, sc).items():
                all_seqs[(sc, anchor, branch)] = fs
        anchors = sorted({(sc, a) for (sc, a, _) in all_seqs})
        n_ev = max(2, len(anchors) // 5)
        ev_anchors = set(anchors[-n_ev:])
        tr_seqs = {k: v for k, v in all_seqs.items() if (k[0], k[1]) not in ev_anchors}
        ev_seqs = {k: v for k, v in all_seqs.items() if (k[0], k[1]) in ev_anchors}
    else:
        for sc in SCENES:
            for seed, fs in walk_sequences(root, sc).items():
                all_seqs[f"{sc}:{seed}"] = fs
        idx_seqs = {i: v for i, v in enumerate(all_seqs.values())}
        tr_seqs, ev_seqs = split_seeds(idx_seqs, n_eval=max(2, len(idx_seqs) // 6))
    if args.tier == "pilot":
        tr_seqs = dict(list(tr_seqs.items())[:3]); ev_seqs = dict(list(ev_seqs.items())[:2])

    K = args.horizon

    # ONE pass over every unique file: cache latent + loc + view_dir, batch the
    # encode. For ref rows the caches are already populated from the feature files.
    if args.oracle_state:
        need = sorted({f for fs in list(tr_seqs.values()) + list(ev_seqs.values()) for f in fs})
        for f in need:
            s = load_walk(f)
            locc[f] = np.asarray(s["loc"], np.float32).reshape(3)
            vdc[f] = np.asarray(s["view_dir"], np.float32).reshape(3)
            zc[f] = np.concatenate([locc[f], vdc[f]])
        print(f"oracle state for {len(need)} frames", flush=True)
    elif vae is not None:
        need = sorted({f for fs in list(tr_seqs.values()) + list(ev_seqs.values()) for f in fs})
        BS = 16                               # matches proven-safe encode_arm_latents batch
        for i in range(0, len(need), BS):
            chunk = need[i:i + BS]
            imgs, samples = [], []
            for f in chunk:
                s = load_walk(f)
                samples.append(s)
                imgs.append(torch.as_tensor(np.asarray(s["img"]), dtype=torch.float32).permute(2, 0, 1))
            batch = torch.stack(imgs).to(dev)
            with torch.no_grad():
                _, mu, _ = vae.img_enc(batch)
            mu = mu.cpu().numpy()
            for j, f in enumerate(chunk):
                zc[f] = mu[j]
                locc[f] = np.asarray(samples[j]["loc"], np.float32).reshape(3)
                vdc[f] = np.asarray(samples[j]["view_dir"], np.float32).reshape(3)
        print(f"encoded {len(need)} frames", flush=True)

    def _action(fi, fj):
        return np.concatenate([locc[fj] - locc[fi], vdc[fj] - vdc[fi]]).astype(np.float32)

    def build(seqs):
        Z, A, Y = [], [], []
        for fs in seqs.values():
            for t in range(len(fs) - K):
                if fs[t] not in zc or fs[t + K] not in zc:
                    continue
                Z.append(zc[fs[t]]); Y.append(zc[fs[t + K]])
                A.append(_action(fs[t], fs[t + K]))
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
    g = (torch.nn.Linear(dim + 6, dim) if args.head == "linear"
         else _mlp(dim + 6, 2 * dim, dim)).to(dev)
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
    if args.oracle_state:
        row = "oracle_state"
    else:
        row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    chan = "rollout_act" if args.walks == "branching" else "rollout"
    return {"status": "ok",
            "verdict": "EXISTS" if r2 > 0.02 else "WEAK",
            "channel": f"{chan}:{row}", "primary": round(r2, 4),
            "metrics": {"delta_r2": r2, "copylast_floor_r2": 0.0,
                        "shuffled_action_r2": r2_shuf, "action_gap": r2 - r2_shuf,
                        "delta_cos": dcos, "horizon": K, "n_train": int(len(Ztr)),
                        "n_eval": int(len(Zev)), "dim": dim, "row": row,
                        "head": args.head, "pretrain_steps": pre_steps,
                        "seed": args.seed, "tier": args.tier},
            "notes": f"rollout {row} k={K}: delta-R2 {r2:.3f} (floor 0), "
                     f"shuffled-R2 {r2_shuf:.3f}, action-gap {r2 - r2_shuf:+.3f}, "
                     f"delta-cos {dcos:.3f}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon", "rgb_only",
                             "fuse_cj_25", "fuse_cj_50", "fuse_cj_75"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--encoder-ckpt", default=None)   # reuse a pretrain_ckpt encoder
    ap.add_argument("--oracle-state", action="store_true")  # positive control
    ap.add_argument("--head", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--walks", default="smooth", choices=["smooth", "branching"])
    ap.add_argument("--walks-root", default="data_branch")
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
