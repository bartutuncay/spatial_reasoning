"""JEPA trainer: the four-arm objective over Bartu's RGB / point-cloud encoders.

Reuses the existing RandomWalkAutoencoderDataset + ImageGraphVAE (loaded via
importlib because their module name starts with a digit). Context = the RGB
latent (mu_img); target = the point-cloud-graph latent from an EMA/stop-grad
copy of the PCD encoder. --objective selects the four-arm killer ablation:

  jepa        : predictor(mu_img) -> EMA target mu_pcd, + VICReg
  symalign    : MSE(mu_img, mu_pcd), no predictor/EMA  (anti-reskin control)
  contrastive : InfoNCE(mu_img, mu_pcd)
  recon       : the base VAE reconstruction loss (pixels+depth), no JEPA

Writes results/<id>/result.json for the LEDGER, with an honest verdict
(UNSCORED unless the representation collapsed -> DEAD). Localization/rotation
read-off heads land in the next step; some sweep axes (mask-ratio, node-feats,
direction, rotation) are accepted but not yet wired and are flagged in notes.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_jepa.jepa import (  # noqa: E402
    Predictor,
    clone_as_target,
    effective_rank,
    ema_update,
    jepa_objective,
)

TIER_STEPS = {"pilot": 4, "shakedown": 60, "bulk": 4000}
NOT_WIRED = ("direction", "mask_ratio", "collapse", "rotation", "node_feats")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def train(args):
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")

    walks = Path(args.processed_root) / args.scene / "random_walks"
    if not any(walks.glob("*.pt")):
        raise FileNotFoundError(f"no random-walk .pt under {walks}")
    ds = autoenc.RandomWalkAutoencoderDataset(str(walks))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=autoenc.collate_random_walk_autoencoder)

    vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
    predictor = Predictor(args.latent_dim,
                          depth=1 if args.predictor == "shallow" else 2).to(dev)
    ema_pcd = clone_as_target(vae.pcd_enc).to(dev)

    params = (list(vae.parameters()) + list(predictor.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    steps = TIER_STEPS.get(args.tier, 60)
    # --collapse ablation: which anti-collapse mechanism the jepa arm uses.
    use_ema = args.objective == "jepa" and args.collapse in ("ema", "ema_vicreg")
    vicreg_w = (1.0, 0.04) if "vicreg" in args.collapse else (0.0, 0.0)
    losses, emb_buf = [], []   # accumulate context embeddings for a meaningful rank
    it = iter(loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        img = batch.img.permute(0, 3, 1, 2).to(torch.float32).to(dev)
        pcd = batch.pcd.to(torch.float32).to(dev)
        if args.node_feats in ("geom_only", "no_rgb"):
            pcd = pcd.clone()
            pcd[:, 4:7] = 0.0   # drop RGB -> geometry-only (appearance-retrieval rebuttal)
        bvec = batch.batch.to(dev)
        ei = batch.edge_index.to(dev)
        ew = autoenc.normalize_edge_weights(batch.edge_weights.to(torch.float32)).to(dev)

        if args.objective == "recon":
            depth = batch.depth.to(torch.float32).to(dev)
            out = vae(img, pcd, bvec, ei, ew,
                      batch.ei_camera.to(dev), batch.ea_camera.to(dev))
            loss, _ = autoenc.compute_loss(out, img, depth)
        else:
            _, mu_img, _ = vae.img_enc(img)                 # context (RGB)
            _, mu_pcd, _ = vae.pcd_enc(pcd, bvec, ei, ew)   # online target latent
            if args.objective == "jepa":
                if use_ema:
                    with torch.no_grad():
                        _, tgt_mu, _ = ema_pcd(pcd, bvec, ei, ew)
                else:
                    tgt_mu = mu_pcd.detach()
                loss, _ = jepa_objective(mu_img, tgt_mu, predictor=predictor,
                                         objective="jepa", vicreg_w=vicreg_w)
            else:  # symalign / contrastive
                loss, _ = jepa_objective(mu_img, mu_pcd, predictor=predictor,
                                         objective=args.objective)
            emb_buf.append(mu_img.detach().float().cpu())

        opt.zero_grad()
        loss.backward()
        opt.step()
        if use_ema:
            ema_update(ema_pcd, vae.pcd_enc, args.ema)

        losses.append(float(loss.detach()))

    # effective rank over ALL accumulated context embeddings (not one small
    # batch), and only judge collapse once there are enough samples to trust it.
    n_emb = sum(e.shape[0] for e in emb_buf)
    final_rank = effective_rank(torch.cat(emb_buf, dim=0)) if n_emb >= 8 else None
    collapsed = final_rank is not None and n_emb >= 16 and final_rank < 2.0
    unwired = [a for a in NOT_WIRED if getattr(args, a, None) not in (None, "full", "ema_vicreg")]
    return {
        "status": "ok",
        "verdict": "DEAD" if collapsed else "UNSCORED",
        "channel": "objective",
        "primary": round(losses[-1], 5) if losses else None,
        "metrics": {
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "rank_ctx_accum": final_rank,
            "n_ctx_emb": n_emb,
            "steps": len(losses),
            "device": str(dev),
            "objective": args.objective,
        },
        "notes": (f"{args.objective} {args.tier}; collapse={collapsed}. "
                  f"axes not yet wired: {unwired}" if unwired else
                  f"{args.objective} {args.tier}; collapse={collapsed}"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="jepa",
                    choices=["jepa", "symalign", "contrastive", "recon"])
    ap.add_argument("--scene", default="office")
    ap.add_argument("--tier", default="shakedown")
    ap.add_argument("--direction", default="rgb2geo")
    ap.add_argument("--predictor", default="deep")
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--collapse", default="ema_vicreg")
    ap.add_argument("--rotation", default="dir")
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--node-feats", default="full")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)  # lower: JEPA diverged at 1e-4
    ap.add_argument("--processed-root",
                    default=os.environ.get("SJEPA_PROCESSED_ROOT", "datasets_processed"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        result = train(args)
    except Exception as e:
        result = {"status": "failed", "verdict": "DEAD",
                  "notes": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-2500:]}
    result["id"] = out.name
    result["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result.setdefault("gpu_h", round((time.time() - t0) / 3600.0, 5))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ("status", "verdict", "primary", "notes")}))


if __name__ == "__main__":
    main()
