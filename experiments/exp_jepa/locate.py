"""Localization worker: RGB image -> camera position (+ heading), scored on the
REAL pose metric so the four-arm objective ablation can finally be RANKED
(SSL loss is not comparable across objectives; pose-recall / ATE is).

Pipeline (map-conditioned):
  query : ImgEnc(img) -> q [B,D]   (optionally pretrained with --objective, frozen or fine-tuned)
  scene : outlier-trimmed, subsampled graph nodes -> positions P[M,3], feats F[M,6];
          node_encoder(F) -> S[M,D]
  match : softmax(q @ S.T / sqrt(D)) -> w[B,M]
  head  : barycenter (w@P)  OR  anchor+offset (P[argmax] + MLP([q, w@S]))
  dir   : MLP(q) -> unit forward vector (the honest "rotation" for this data)
  --map-cond no_map : ignore the scene, regress position from q alone (APR).
          map vs no_map across arms = the MAP-CONDITIONING INVERSION test.

Self-contained per job (short SSL pretrain -> localizer train -> eval), so it
fans out trivially over objective x map-cond x head x freeze on EULER.
Train/eval are split by random-walk SEED (no pose-label leakage).
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_jepa.jepa import jepa_objective  # noqa: E402
from metrics.pose import ate_rmse, pose_recall, translation_errors  # noqa: E402

TIER = {"pilot": (4, 20), "shakedown": (40, 300), "bulk": (500, 4000)}  # (pretrain, locate) steps
RECALL_THRESH = [(0.5, None), (1.0, None), (2.0, None)]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


def _maybe_blur(img, sigma):
    """Gaussian blur the query image (degraded-query stress axis)."""
    if sigma and sigma > 0:
        import torchvision.transforms.functional as TF
        k = int(2 * round(2 * sigma) + 1)
        img = TF.gaussian_blur(img, kernel_size=k, sigma=float(sigma))
    return img


class Localizer(nn.Module):
    def __init__(self, img_enc, latent_dim, head, map_cond, direction):
        super().__init__()
        self.img_enc = img_enc
        self.head, self.map_cond, self.direction = head, map_cond, direction
        self.node_enc = _mlp(6, latent_dim, latent_dim)
        self.offset_mlp = _mlp(2 * latent_dim, latent_dim, 3)
        self.offset_scale = nn.Parameter(torch.ones(3))    # learnable bound on the offset
        self.apr_mlp = _mlp(latent_dim, latent_dim, 3)     # no-map regressor
        self.dir_mlp = _mlp(latent_dim, latent_dim, 3)
        self.d = latent_dim

    def forward(self, img, P, F_feat):
        _, q, _ = self.img_enc(img)                        # [B, D]
        if self.map_cond == "no_map":
            pos = self.apr_mlp(q)
        else:
            S = self.node_enc(F_feat)                      # [M, D]
            scores = q @ S.t() / (self.d ** 0.5)           # [B, M]
            w = torch.softmax(scores, dim=-1)
            if self.head == "anchor_offset":
                anchor = P[scores.argmax(dim=-1)]          # [B, 3]
                ctx = w @ S                                # [B, D]
                # BOUNDED local correction: tanh*scale so the offset can't blow
                # up in a large / outlier-laden world frame (was 144m unbounded)
                off = torch.tanh(self.offset_mlp(torch.cat([q, ctx], dim=-1)))
                pos = anchor + off * self.offset_scale
            else:  # barycenter
                pos = w @ P                                # [B, 3]
        d = F.normalize(self.dir_mlp(q), dim=-1) if self.direction else None
        return pos, d


def _robust_scene(g, m_sub, dev, map_frac=1.0):
    P = np.asarray(g.pos, dtype=np.float64)
    C = np.asarray(g.rgb, dtype=np.float64) if hasattr(g, "rgb") else np.zeros_like(P)
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    keep = np.all((P >= lo) & (P <= hi), axis=1)
    P, C = P[keep], C[keep]
    target = max(32, int(m_sub * map_frac))   # map_frac<1 => partial/sparse map
    if P.shape[0] > target:
        idx = np.random.default_rng(0).choice(P.shape[0], target, replace=False)
        P, C = P[idx], C[idx]
    Pn = (P - P.mean(0)) / (P.std(0) + 1e-6)
    F_feat = np.concatenate([Pn, C], axis=1)               # [M, 6]
    return (torch.tensor(P, dtype=torch.float32, device=dev),
            torch.tensor(F_feat, dtype=torch.float32, device=dev))


class _QueryDS(torch.utils.data.Dataset):
    """Loads (img, loc, view_dir) from a list of .pt files (localizer phase)."""
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        s = torch.load(self.files[i], map_location="cpu", weights_only=False)
        return (torch.as_tensor(s["img"], dtype=torch.float32).permute(2, 0, 1),
                torch.as_tensor(s["loc"], dtype=torch.float32).reshape(3),
                torch.as_tensor(s["view_dir"], dtype=torch.float32).reshape(3))


def _seed_split(walks):
    files = sorted(glob.glob(str(walks / "*.pt")))
    seeds = sorted({int(re.search(r"rw_(\d+)_", Path(f).name).group(1)) for f in files})
    eval_seed = seeds[-1]
    tr = [f for f in files if not Path(f).name.startswith(f"rw_{eval_seed}_")]
    ev = [f for f in files if Path(f).name.startswith(f"rw_{eval_seed}_")]
    return tr, ev


def _pretrain(vae, autoenc, walks, objective, steps, dev):
    from experiments.exp_jepa.jepa import Predictor, clone_as_target, ema_update
    ds = autoenc.RandomWalkAutoencoderDataset(str(walks))
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


def run(args):
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
    walks = Path(args.processed_root) / args.scene / "random_walks"
    graph = torch.load(Path(args.processed_root) / args.scene / "scan_pcd_graph" /
                       "combined_aligned.pt", map_location="cpu", weights_only=False)
    P, Ffeat = _robust_scene(graph, args.scene_nodes, dev, map_frac=args.map_frac)

    pre_steps, loc_steps = TIER.get(args.tier, (40, 300))
    vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
    if args.objective != "scratch":
        _pretrain(vae, autoenc, walks, args.objective, pre_steps, dev)
    if args.freeze:
        for p in vae.img_enc.parameters():
            p.requires_grad_(False)

    model = Localizer(vae.img_enc, args.latent_dim, args.head, args.map_cond,
                      bool(args.direction)).to(dev)
    tr_files, ev_files = _seed_split(walks)
    tr = DataLoader(_QueryDS(tr_files), batch_size=8, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    model.train()
    it = iter(tr)
    for _ in range(loc_steps):
        try:
            img, loc, vd = next(it)
        except StopIteration:
            it = iter(tr); img, loc, vd = next(it)
        img, loc, vd = img.to(dev), loc.to(dev), vd.to(dev)
        img = _maybe_blur(img, args.blur)
        pos, d = model(img, P, Ffeat)
        loss = F.smooth_l1_loss(pos, loc)
        if d is not None:
            loss = loss + (1.0 - F.cosine_similarity(d, F.normalize(vd, dim=-1), dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- eval on held-out seed ----
    model.eval()
    preds, gts, dirs, gtdirs = [], [], [], []
    with torch.no_grad():
        for img, loc, vd in DataLoader(_QueryDS(ev_files), batch_size=16):
            pos, d = model(_maybe_blur(img.to(dev), args.blur), P, Ffeat)
            preds.append(pos.cpu().numpy()); gts.append(loc.numpy())
            if d is not None:
                dirs.append(d.cpu().numpy()); gtdirs.append(F.normalize(vd, dim=-1).numpy())
    pred = np.concatenate(preds); gt = np.concatenate(gts)
    ate = ate_rmse(pred, gt)
    recall = pose_recall(pred, gt, RECALL_THRESH)
    floor = ate_rmse(np.tile(P.cpu().numpy().mean(0), (gt.shape[0], 1)), gt)  # centroid floor
    dir_err = None
    if dirs:
        cos = np.clip((np.concatenate(dirs) * np.concatenate(gtdirs)).sum(-1), -1, 1)
        dir_err = float(np.degrees(np.arccos(cos)).mean())

    beats = ate < floor
    return {
        "status": "ok",
        "verdict": "EXISTS" if beats else "WEAK",
        "channel": f"locate:{args.objective}",
        "primary": round(float(ate), 4),
        "metrics": {
            "ate_m": float(ate), "centroid_floor_m": float(floor),
            "beats_floor": bool(beats), "dir_err_deg": dir_err,
            **{f"recall@{k}": v for k, v in recall.items()},
            "objective": args.objective, "map_cond": args.map_cond,
            "head": args.head, "freeze": bool(args.freeze),
            "map_frac": args.map_frac, "blur": args.blur,
            "n_eval": int(gt.shape[0]), "device": str(dev),
        },
        "notes": (f"{args.objective}/{args.map_cond}/{args.head}/"
                  f"{'frozen' if args.freeze else 'ft'}: ATE {ate:.2f}m vs floor {floor:.2f}m"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="jepa",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon"])
    ap.add_argument("--map-cond", default="map", choices=["map", "no_map"])
    ap.add_argument("--head", default="anchor_offset", choices=["barycenter", "anchor_offset"])
    ap.add_argument("--freeze", type=int, default=1)
    ap.add_argument("--direction", type=int, default=1)
    ap.add_argument("--scene", default="office")
    ap.add_argument("--tier", default="shakedown")
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--scene-nodes", type=int, default=2048)
    ap.add_argument("--map-frac", type=float, default=1.0)   # partial-map stress
    ap.add_argument("--blur", type=float, default=0.0)       # degraded-query stress
    ap.add_argument("--processed-root",
                    default=os.environ.get("SJEPA_PROCESSED_ROOT", "datasets_processed"))
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
