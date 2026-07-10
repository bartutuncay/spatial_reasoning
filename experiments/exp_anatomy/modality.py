"""Conditioning-MODALITY axis: what must the cross-modal target carry?

Rows cj<pct>@<modality>: lam*InfoNCE(mu_img, mu_target) + (1-lam)*JEPA
(predictor -> EMA(target_enc)), one implementation for every modality so the
target is the only variable:

  depth    : the frame's own depth map (egocentric, view-local 2.5D);
             already stored in every walk .pt - no new data.
  layout   : oriented top-down occupancy crop around the camera (allocentric,
             architectural 2D); scene map from gen_layouts.py.
  objgraph : nearby object instances as a set (class, centroid, extent) -
             the information content of a CAD/FBX scene model (allocentric,
             symbolic); scene table from gen_objgraphs.py (Replica semantics).
  pcd      : the existing view-frustum point-cloud graph - the within-axis
             anchor tying this matrix to the objective-axis rows.

Hypothesis: the representation inherits the reference frame of its target -
allocentric targets (layout/objgraph) should lift the relational columns.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

MODALITIES = ("depth", "layout", "objgraph", "pcd")
LAYOUT_RES = 0.1          # m per cell
LAYOUT_SIZE = 64          # 64x64 crop = 6.4 m window
OBJ_RADIUS = 5.0          # objects within 5 m condition the frame
OBJ_MAX = 40
N_CLASSES = 128           # class-id embedding table (Replica has ~90)


def parse_row(objective):
    """'cj50@layout' -> (0.5, 'layout'); None if not a modality row."""
    if "@" not in objective:
        return None
    head, mod = objective.split("@", 1)
    if not head.startswith("cj") or mod not in MODALITIES:
        return None
    return int(head[2:]) / 100.0, mod


class ConvEnc(nn.Module):
    """Small CNN -> latent mean; input-size agnostic (adaptive pool)."""

    def __init__(self, in_ch, latent_dim):
        super().__init__()
        chs = [in_ch, 32, 64, 128, 128]
        blocks = []
        for a, b in zip(chs[:-1], chs[1:]):
            blocks += [nn.Conv2d(a, b, 3, stride=2, padding=1),
                       nn.GroupNorm(8, b), nn.SiLU()]
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.mu_head = nn.Linear(128 * 16, latent_dim)

    def forward(self, x):
        h = self.pool(self.conv(x)).flatten(1)
        mu = self.mu_head(h)
        return None, mu, None                     # (_, mu, _) like the VAE encs


class SetEnc(nn.Module):
    """Object-set encoder: per-object MLP -> mean+max pool -> latent mean."""

    def __init__(self, latent_dim, feat_dim=6, emb_dim=32, hid=128):
        super().__init__()
        self.emb = nn.Embedding(N_CLASSES, emb_dim)
        self.phi = nn.Sequential(nn.Linear(feat_dim + emb_dim, hid), nn.SiLU(),
                                 nn.Linear(hid, hid), nn.SiLU())
        self.mu_head = nn.Linear(2 * hid, latent_dim)

    def forward(self, feats, cls, mask):
        # feats [B,N,6], cls [B,N] long, mask [B,N] bool (True = real object)
        h = self.phi(torch.cat([feats, self.emb(cls)], -1))
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        hmax = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        mu = self.mu_head(torch.cat([mean, hmax], -1))
        return None, mu, None


def _yaw_rot(view_dir):
    """2D rotation matrix aligning the view heading with +x."""
    yaw = math.atan2(float(view_dir[1]), float(view_dir[0]))
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def layout_crop(scene_map, loc, view_dir):
    """Oriented LAYOUT_SIZE^2 occupancy+height crop around (loc, heading)."""
    occ, hmax = scene_map["occ"], scene_map["hmax"]
    x0, y0 = scene_map["origin"]
    R = _yaw_rot(view_dir)
    half = LAYOUT_SIZE // 2
    ii, jj = np.meshgrid(np.arange(LAYOUT_SIZE), np.arange(LAYOUT_SIZE), indexing="ij")
    # crop-frame metric offsets (heading = +x = i axis)
    dx = (ii - half + 0.5) * LAYOUT_RES
    dy = (jj - half + 0.5) * LAYOUT_RES
    world = np.stack([dx, dy], -1) @ R + np.asarray(loc[:2], np.float32)
    ci = ((world[..., 0] - x0) / LAYOUT_RES).astype(int)
    cj = ((world[..., 1] - y0) / LAYOUT_RES).astype(int)
    ok = (ci >= 0) & (ci < occ.shape[0]) & (cj >= 0) & (cj < occ.shape[1])
    out = np.zeros((2, LAYOUT_SIZE, LAYOUT_SIZE), np.float32)
    out[0][ok] = occ[ci[ok], cj[ok]]
    out[1][ok] = hmax[ci[ok], cj[ok]]
    return out


def obj_tokens(scene_objs, loc, view_dir):
    """Objects within OBJ_RADIUS as camera-frame tokens: [dxyz(3), extent(3)]."""
    cen, ext, cls = scene_objs["centroid"], scene_objs["extent"], scene_objs["cls"]
    d = cen - np.asarray(loc, np.float32)
    near = np.linalg.norm(d[:, :2], axis=1) <= OBJ_RADIUS
    d, e, c = d[near][:OBJ_MAX], ext[near][:OBJ_MAX], cls[near][:OBJ_MAX]
    R = _yaw_rot(view_dir)
    d2 = d.copy(); d2[:, :2] = d[:, :2] @ R.T           # rotate into heading frame
    feats = np.zeros((OBJ_MAX, 6), np.float32)
    clsv = np.zeros(OBJ_MAX, np.int64)
    mask = np.zeros(OBJ_MAX, bool)
    n = len(d2)
    feats[:n] = np.concatenate([d2, e], 1)
    clsv[:n] = np.clip(c, 0, N_CLASSES - 1)
    mask[:n] = True
    return feats, clsv, mask


class ModalityDataset(Dataset):
    """(img, target payload) pairs for depth/layout/objgraph conditioning."""

    def __init__(self, walk_dir, modality, assets_dir=None):
        self.files = sorted(Path(walk_dir).glob("*.pt"))
        self.modality = modality
        self.scene = Path(walk_dir).parent.name
        self.assets = None
        if modality == "layout":
            self.assets = torch.load(Path(assets_dir) / f"{self.scene}.pt",
                                     map_location="cpu", weights_only=False)
        elif modality == "objgraph":
            self.assets = torch.load(Path(assets_dir) / f"{self.scene}.pt",
                                     map_location="cpu", weights_only=False)
            self.assets = {k: np.asarray(v) for k, v in self.assets.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        s = torch.load(self.files[i], map_location="cpu", weights_only=False)
        img = torch.as_tensor(np.asarray(s["img"]), dtype=torch.float32)
        loc = np.asarray(s["loc"], np.float32).reshape(3)
        vd = np.asarray(s["view_dir"], np.float32).reshape(3)
        if self.modality == "depth":
            d = np.nan_to_num(np.asarray(s["depth"], np.float32), nan=0.0)
            d = np.log1p(np.clip(d, 0, 100))[None]        # [1,H,W], bounded
            return img, torch.from_numpy(d)
        if self.modality == "layout":
            return img, torch.from_numpy(layout_crop(self.assets, loc, vd))
        feats, cls, mask = obj_tokens(self.assets, loc, vd)
        return img, (torch.from_numpy(feats), torch.from_numpy(cls),
                     torch.from_numpy(mask))


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    p = [b[1] for b in batch]
    if isinstance(p[0], tuple):
        return imgs, tuple(torch.stack([x[k] for x in p]) for k in range(3))
    return imgs, torch.stack(p)


def pretrain_modality(vae, autoenc, walk_dirs, objective, steps, dev,
                      assets_root=None):
    """cj<pct>@<modality> pretrain: identical loop for every modality."""
    from experiments.exp_jepa.jepa import (
        Predictor, clone_as_target, ema_update, info_nce, jepa_objective,
    )
    lam, mod = parse_row(objective)
    latent = vae.img_enc.mu_head.out_features if hasattr(vae.img_enc, "mu_head") \
        else vae.pcd_enc.mu_head.out_features

    if mod == "pcd":
        tgt_enc = vae.pcd_enc
        ds = ConcatDataset([autoenc.RandomWalkAutoencoderDataset(str(d)) for d in walk_dirs])
        loader = DataLoader(ds, batch_size=4, shuffle=True,
                            collate_fn=autoenc.collate_random_walk_autoencoder)
    else:
        tgt_enc = (ConvEnc(1, latent) if mod == "depth"
                   else ConvEnc(2, latent) if mod == "layout"
                   else SetEnc(latent)).to(dev)
        sub = {"depth": None, "layout": "layouts", "objgraph": "objects"}[mod]
        adir = None if sub is None else Path(assets_root) / sub
        ds = ConcatDataset([ModalityDataset(d, mod, adir) for d in walk_dirs])
        loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=_collate)

    predictor = Predictor(latent).to(dev)
    ema_tgt = clone_as_target(tgt_enc).to(dev)
    opt = torch.optim.AdamW(list(vae.img_enc.parameters())
                            + list(tgt_enc.parameters())
                            + list(predictor.parameters()), lr=1e-5)

    def encode_target(enc, b):
        if mod == "pcd":
            pcd = b.pcd.float().to(dev); bvec = b.batch.to(dev)
            ei = b.edge_index.to(dev)
            ew = autoenc.normalize_edge_weights(b.edge_weights.float()).to(dev)
            return enc(pcd, bvec, ei, ew)[1]
        if mod == "objgraph":
            feats, cls, mask = (t.to(dev) for t in b)
            return enc(feats, cls, mask)[1]
        return enc(b.to(dev))[1]

    it = iter(loader)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        if mod == "pcd":
            img = b.img.permute(0, 3, 1, 2).float().to(dev)
            payload = b
        else:
            img, payload = b
            img = img.permute(0, 3, 1, 2).float().to(dev)
        _, mu_i, _ = vae.img_enc(img)
        mu_t = encode_target(tgt_enc, payload)
        with torch.no_grad():
            tgt = encode_target(ema_tgt, payload)
        j_loss, _ = jepa_objective(mu_i, tgt, predictor=predictor, objective="jepa")
        loss = lam * info_nce(mu_i, mu_t) + (1.0 - lam) * j_loss
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(ema_tgt, tgt_enc, 0.996)
