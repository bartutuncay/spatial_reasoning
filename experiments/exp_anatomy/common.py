"""Shared utilities for capability-anatomy probes (wave 1)."""
import glob
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_jepa.locate import _QueryDS  # noqa: E402

SCENES = ["office", "pipes", "break_room", "relief", "hospital"]


def seed_split(files):
    """Train/eval split by random-walk seed (rw_<seed>_*); last seed = eval."""
    seeds = sorted({int(re.search(r"rw_(\d+)_", Path(f).name).group(1)) for f in files})
    ev_seed = seeds[-1]
    tr = [f for f in files if not Path(f).name.startswith(f"rw_{ev_seed}_")]
    ev = [f for f in files if Path(f).name.startswith(f"rw_{ev_seed}_")]
    return tr, ev


def scene_files(root, scene):
    return sorted(glob.glob(str(Path(root) / scene / "random_walks" / "*.pt")))


def depth_grid(depth, g=16):
    """Downsample a [H,W] depth map to a g*g log-depth grid, masking cells with
    no valid (>0) pixels. Pools SUM(valid depth)/COUNT(valid) per cell so
    invalid pixels never bias a cell."""
    d = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)  # NaN/inf = invalid
    valid = (d > 0).float()
    s = F.adaptive_avg_pool2d(d * valid, g)
    c = F.adaptive_avg_pool2d(valid, g)
    mask = (c > 0).reshape(-1)
    grid = torch.where(c > 0, s / c.clamp(min=1e-8), torch.ones_like(s))
    return torch.log(grid.clamp(min=1e-6)).reshape(-1), mask


def to_uint8(img):
    a = np.asarray(img, dtype=np.float32)
    if a.max() <= 1.5:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def encode_arm_latents(vae, files, dev, bs=16):
    """Frozen img_enc mu for a list of walk files -> (Z[N,D], loc[N,3], vd[N,3])."""
    Z, L, V = [], [], []
    with torch.no_grad():
        for img, loc, vd in DataLoader(_QueryDS(files), batch_size=bs):
            _, mu, _ = vae.img_enc(img.to(dev))
            Z.append(mu.cpu().numpy()); L.append(loc.numpy()); V.append(vd.numpy())
    return np.concatenate(Z), np.concatenate(L), np.concatenate(V)


def load_walk(f):
    return torch.load(f, map_location="cpu", weights_only=False)


def _step_of(f):
    return int(Path(f).name.rsplit("_", 1)[1].split(".")[0])


def walk_sequences(root, scene):
    """seed -> [files] ordered by step NUMERICALLY (sorted(glob) is lexicographic)."""
    seqs = {}
    for f in scene_files(root, scene):
        seed = int(re.search(r"rw_(\d+)_", Path(f).name).group(1))
        seqs.setdefault(seed, []).append(f)
    return {s: sorted(fs, key=_step_of) for s, fs in seqs.items()}


def split_seeds(seqs, n_eval=5):
    """Hold out the highest n_eval seed keys for eval."""
    ev_seeds = sorted(seqs)[-n_eval:]
    tr = {s: v for s, v in seqs.items() if s not in ev_seeds}
    ev = {s: v for s, v in seqs.items() if s in ev_seeds}
    return tr, ev


def relative_action(si, sj):
    """[Δloc(3, world), Δview_dir(3, world)] from sample i -> j."""
    li = np.asarray(si["loc"], dtype=np.float32).reshape(3)
    lj = np.asarray(sj["loc"], dtype=np.float32).reshape(3)
    vi = np.asarray(si["view_dir"], dtype=np.float32).reshape(3)
    vj = np.asarray(sj["view_dir"], dtype=np.float32).reshape(3)
    return np.concatenate([lj - li, vj - vi]).astype(np.float32)


def pretrain_encoder(vae, autoenc, walk_dirs, objective, steps, dev):
    """Dispatch: rgb_only -> augmentation-InfoNCE on img_enc only; else the
    cross-modal four-arm pretrainer (fewshot._pretrain_pool)."""
    if objective != "rgb_only":
        from experiments.exp_jepa.fewshot import _pretrain_pool
        return _pretrain_pool(vae, autoenc, walk_dirs, objective, steps, dev)
    import torchvision.transforms as TT
    from torch.utils.data import ConcatDataset, DataLoader
    from experiments.exp_jepa.jepa import info_nce
    ds = ConcatDataset([autoenc.RandomWalkAutoencoderDataset(str(d)) for d in walk_dirs])
    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=autoenc.collate_random_walk_autoencoder)
    aug = TT.Compose([TT.RandomResizedCrop((192, 256), scale=(0.6, 1.0), antialias=True),
                      TT.RandomHorizontalFlip(),
                      TT.ColorJitter(0.4, 0.4, 0.4, 0.1)])
    opt = torch.optim.AdamW(vae.img_enc.parameters(), lr=1e-4)
    it = iter(loader)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        img = b.img.permute(0, 3, 1, 2).float().to(dev) / 255.0
        _, z1, _ = vae.img_enc(aug(img))
        _, z2, _ = vae.img_enc(aug(img))
        loss = info_nce(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()


def finish(out_dir, result, t0):
    """Stamp id/date/gpu_h, write result.json, print the one-line summary."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    result["id"] = out.name
    result["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result.setdefault("gpu_h", round((time.time() - t0) / 3600.0, 5))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ("status", "verdict", "primary", "notes")}))
