"""Few-shot cross-scene localization — THE test of the thesis.

Pretrain the encoder (--objective) on a POOL of scenes (SSL, no pose labels),
FREEZE it, then train a small pose head on a HELD-OUT target scene from only K
labels and eval on the rest. If a JEPA-pretrained representation localizes a NEW
room from few labels better than scratch, the thesis holds. Single-scene tests
can't show this: dense per-scene data makes localization pure interpolation, so
representation quality is irrelevant (confirmed: scratch==jepa at 1040 walks).

--objective scratch = random (unpretrained) frozen encoder baseline.
No map graph needed (APR head on the frozen latent), so it runs on walks alone.
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_jepa.jepa import (  # noqa: E402
    Predictor, clone_as_target, ema_update, jepa_objective,
)
from experiments.exp_jepa.locate import _load_module, _mlp, _QueryDS  # noqa: E402
from metrics.pose import ate_rmse  # noqa: E402

PRETRAIN_STEPS = {"pilot": 4, "shakedown": 200, "bulk": 1500}


def _pretrain_pool(vae, autoenc, walk_dirs, objective, steps, dev):
    ds = ConcatDataset([autoenc.RandomWalkAutoencoderDataset(str(d)) for d in walk_dirs])
    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=autoenc.collate_random_walk_autoencoder)
    predictor = Predictor(vae.pcd_enc.mu_head.out_features).to(dev)
    ema_pcd = clone_as_target(vae.pcd_enc).to(dev)
    opt = torch.optim.AdamW(list(vae.parameters()) + list(predictor.parameters()), lr=1e-5)
    it = iter(loader)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        img = b.img.permute(0, 3, 1, 2).float().to(dev)
        pcd = b.pcd.float().to(dev); bvec = b.batch.to(dev); ei = b.edge_index.to(dev)
        ew = autoenc.normalize_edge_weights(b.edge_weights.float()).to(dev)
        if objective == "recon":
            loss, _ = autoenc.compute_loss(
                vae(img, pcd, bvec, ei, ew, b.ei_camera.to(dev), b.ea_camera.to(dev)),
                img, b.depth.float().to(dev))
        else:
            _, mu_i, _ = vae.img_enc(img)
            _, mu_p, _ = vae.pcd_enc(pcd, bvec, ei, ew)
            if objective == "jepa":
                with torch.no_grad():
                    _, tgt, _ = ema_pcd(pcd, bvec, ei, ew)
                loss, _ = jepa_objective(mu_i, tgt, predictor=predictor, objective="jepa")
            else:
                loss, _ = jepa_objective(mu_i, mu_p, predictor=predictor, objective=objective)
        opt.zero_grad(); loss.backward(); opt.step()
        if objective == "jepa":
            ema_update(ema_pcd, vae.pcd_enc, 0.996)


def _apr(vae, head, files, dev):
    """Predict positions for a set of query files with the frozen encoder + head."""
    P, G = [], []
    with torch.no_grad():
        for img, loc, _ in DataLoader(_QueryDS(files), batch_size=16):
            _, q, _ = vae.img_enc(img.to(dev))
            P.append(head(q).cpu().numpy()); G.append(loc.numpy())
    return np.concatenate(P), np.concatenate(G)


def run(args):
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
    root = Path(args.processed_root)
    scenes = [s for s in args.scenes.split(",") if s]
    pool = [s for s in scenes if s != args.target]
    if not pool:
        raise ValueError("no pretrain scenes (pool empty)")

    vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
    if args.objective != "scratch":
        _pretrain_pool(vae, autoenc, [root / s / "random_walks" for s in pool],
                       args.objective, PRETRAIN_STEPS.get(args.tier, 200), dev)
    for p in vae.img_enc.parameters():
        p.requires_grad_(False)          # freeze encoder: test the representation
    vae.img_enc.eval()

    # target scene: K labeled train, disjoint held-out eval
    files = sorted(glob.glob(str(root / args.target / "random_walks" / "*.pt")))
    if len(files) < args.few_shot + 50:
        raise FileNotFoundError(f"target {args.target}: only {len(files)} walks")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)
    K = args.few_shot
    n_eval = min(250, len(files) - K)
    train_files, eval_files = files[:K], files[K:K + n_eval]

    head = _mlp(args.latent_dim, args.latent_dim, 3).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    trl = DataLoader(_QueryDS(train_files), batch_size=min(8, K), shuffle=True)
    head.train()
    for _ in range(args.head_steps):
        for img, loc, _ in trl:
            with torch.no_grad():
                _, q, _ = vae.img_enc(img.to(dev))
            loss = F.smooth_l1_loss(head(q), loc.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    pred, gt = _apr(vae, head, eval_files, dev)
    # floor: predict the mean of the FEW-SHOT TRAIN positions (no held-out peeking)
    _, tr_gt = _apr(vae, head, train_files, dev)
    floor = ate_rmse(np.tile(tr_gt.mean(0), (gt.shape[0], 1)), gt)
    ate = ate_rmse(pred, gt)
    return {
        "status": "ok",
        "verdict": "EXISTS" if ate < floor else "WEAK",
        "channel": f"fewshot:{args.objective}",
        "primary": round(float(ate), 4),
        "metrics": {
            "ate_m": float(ate), "train_centroid_floor_m": float(floor),
            "objective": args.objective, "target": args.target,
            "pool": ",".join(pool), "few_shot_K": K,
            "n_eval": len(eval_files), "device": str(dev),
        },
        "notes": (f"{args.objective}: pretrain[{'+'.join(pool)}] -> few-shot(K={K}) "
                  f"{args.target}: ATE {ate:.2f}m (train-centroid floor {floor:.2f}m)"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="jepa",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon"])
    ap.add_argument("--target", default="office")
    ap.add_argument("--scenes", default="office,pipes,break_room,relief,hospital")
    ap.add_argument("--few-shot", type=int, default=50)
    ap.add_argument("--head-steps", type=int, default=200)
    ap.add_argument("--tier", default="shakedown")
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--processed-root",
                    default=os.environ.get("SJEPA_PROCESSED_ROOT", "data_bartu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        result = run(args)
    except Exception as e:
        result = {"status": "failed", "verdict": "DEAD",
                  "notes": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2500:]}
    result["id"] = out.name
    result["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result.setdefault("gpu_h", round((time.time() - t0) / 3600.0, 5))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ("status", "verdict", "primary", "notes")}))


if __name__ == "__main__":
    main()
