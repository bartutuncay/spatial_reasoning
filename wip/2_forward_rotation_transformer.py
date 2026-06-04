from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gnn_spatial_reasoning.preprocessing_src.dataloader_autoencoder import make_loader


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


img_enc_mod = load_module("nips_img_enc_action", ROOT / "gnn_spatial_reasoning_nips/models/1_img_enc.py")
Bottleneck = img_enc_mod.Bottleneck
ImgEnc = img_enc_mod.ImgEnc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

LATENT_DIM = 128
MODEL_DIM = 128
DROPOUT = 0.1
EPOCHS = 80000
ENCODE_BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 32
LR = 2e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD = 1.0
TRAJ_WEIGHT = 5
DIR_WEIGHT = 2.0
VAL_FRACTION = 0.2
SPLIT_SEED = 5

ALIAS = "0505_pair_action_stratified"
RW_ROOTS = [
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_translation",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/break_room/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/relief/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/terrains/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/rw_translation",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/random_walks",
]
VAE_WEIGHTS = Path("../../../scratch/btuncay/cog/gnn_spatial_reasoning/1_model/0420_nips/model_weights_7800.pt")
OUT_ROOT = Path("../../../scratch/btuncay/cog/gnn_spatial_reasoning/2_model_movement")
RW_RE = re.compile(r"rw_(\d+)_(\d+)\.pt$")
WORLD_UP = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
WORLD_FORWARD = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)


class ImgOnlyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = ImgEnc(Bottleneck, [3, 4, 6, 3], LATENT_DIM)


class ActionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(LATENT_DIM * 3),
            nn.Linear(LATENT_DIM * 3, MODEL_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.GELU(),
            nn.Linear(MODEL_DIM, 5),
        )

    def forward(self, latents: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        n = latents.size(1)
        out = latents.new_zeros(latents.size(0), n, 5)
        if n < 2:
            return out
        prev = latents[:, :-1]
        curr = latents[:, 1:]
        out[:, 1:] = self.net(torch.cat([prev, curr, curr - prev], dim=-1))
        return out


class Walks(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def strip_module(state):
    return {k.removeprefix("module."): v for k, v in state.items()}


def camera_basis(view_dir: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    view_dir = view_dir.float()
    norm = view_dir.norm(dim=-1, keepdim=True)
    ok = torch.isfinite(view_dir).all(dim=-1, keepdim=True) & (norm > eps)
    z = torch.where(norm > eps, view_dir / norm.clamp_min(eps), WORLD_FORWARD.to(view_dir.device).expand_as(view_dir))
    z = torch.where(ok, z, WORLD_FORWARD.to(view_dir.device).expand_as(view_dir))
    up = WORLD_UP.to(view_dir.device).expand_as(z)
    x = torch.cross(up, z, dim=-1)
    bad = x.norm(dim=-1, keepdim=True) < eps
    if bad.any():
        candidates = torch.eye(3, device=view_dir.device, dtype=view_dir.dtype)
        fallback = candidates[torch.argmin(torch.abs(z @ candidates.T), dim=-1)]
        x = torch.where(bad, torch.cross(fallback, z, dim=-1), x)
    x = F.normalize(x, dim=-1, eps=eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def ground_forward(view_dir: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    fwd = view_dir.float().clone()
    fwd[..., 2] = 0.0
    norm = fwd.norm(dim=-1, keepdim=True)
    fallback = WORLD_FORWARD.to(view_dir.device).expand_as(fwd)
    return torch.where(norm > eps, fwd / norm.clamp_min(eps), fallback)


def ground_right(view_dir: torch.Tensor) -> torch.Tensor:
    up = WORLD_UP.to(view_dir.device).expand_as(view_dir)
    return F.normalize(torch.cross(up, ground_forward(view_dir), dim=-1), dim=-1, eps=1e-9)


def make_action(src_loc, dst_loc, src_dir, dst_dir):
    dp = dst_loc - src_loc
    move = torch.cat(
        [
            (dp * ground_right(src_dir)).sum(dim=-1, keepdim=True),
            (dp * ground_forward(src_dir)).sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    rot = torch.bmm(camera_basis(src_dir).transpose(1, 2), F.normalize(dst_dir, dim=-1).unsqueeze(-1)).squeeze(-1)
    return torch.cat([move, rot], dim=-1)


def rollout(start_loc, start_dir, actions):
    locs = [start_loc]
    dirs = [F.normalize(start_dir, dim=-1)]
    for step in actions.unbind(dim=1):
        move = step[:, :2]
        local_dir = F.normalize(step[:, 2:], dim=-1, eps=1e-8)
        locs.append(locs[-1] + move[:, :1] * ground_right(dirs[-1]) + move[:, 1:2] * ground_forward(dirs[-1]))
        dirs.append(F.normalize(torch.bmm(camera_basis(dirs[-1]), local_dir.unsqueeze(-1)).squeeze(-1), dim=-1))
    return torch.stack(locs, dim=1), torch.stack(dirs, dim=1)


def parse_name(path):
    match = RW_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"bad random walk name: {path}")
    return int(match.group(1)), int(match.group(2))


def scene_name(root):
    root = Path(root)
    return root.parent.name if root.name in {"random_walks", "rw_translation"} else root.name


def load_encoder():
    state = torch.load(VAE_WEIGHTS, map_location="cpu", weights_only=False)
    state = strip_module(state.get("state_dict", state) if isinstance(state, dict) else state)
    model = ImgOnlyEncoder().to(device)
    model.load_state_dict(state, strict=False)
    return model.eval()


def encode_all(encoder):
    obs = []
    for root in [Path(p) for p in RW_ROOTS]:
        loader = make_loader(root, batch_size=ENCODE_BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)
        scene = scene_name(root)
        count = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                img = batch.img.permute(0, 3, 1, 2).contiguous().float()
                _, mu, _ = encoder.img_enc(img)
                for i, path in enumerate(batch.path):
                    walk, step = parse_name(path)
                    obs.append((scene, walk, step, path, mu[i].cpu(), batch.loc[i].cpu().float(), batch.view_dir[i].cpu().float()))
                    count += 1
        print(f"encoded {count} from {scene}")
    return sorted(obs, key=lambda x: (x[0], x[1], x[2]))


def build_sequences(obs):
    groups = {}
    for item in obs:
        groups.setdefault((item[0], item[1]), []).append(item)
    sequences = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda x: x[2])
        if len(items) < 2:
            continue
        lat = torch.stack([x[4] for x in items])
        loc = torch.stack([x[5] for x in items])
        dirs = F.normalize(torch.stack([x[6] for x in items]), dim=-1)
        actions = torch.zeros(len(items), 5)
        valid = torch.zeros(len(items), dtype=torch.bool)
        actions[1:] = make_action(loc[:-1], loc[1:], dirs[:-1], dirs[1:])
        valid[1:] = True
        sequences.append(dict(scene=key[0], walk=key[1], latents=lat, locs=loc, dirs=dirs, actions=actions, valid=valid))
    return sequences


def collate(batch):
    b = len(batch)
    n = max(x["latents"].size(0) for x in batch)
    lat = torch.zeros(b, n, LATENT_DIM)
    loc = torch.zeros(b, n, 3)
    dirs = torch.zeros(b, n, 3)
    act = torch.zeros(b, n, 5)
    valid = torch.zeros(b, n, dtype=torch.bool)
    pad = torch.ones(b, n, dtype=torch.bool)
    for i, x in enumerate(batch):
        m = x["latents"].size(0)
        lat[i, :m] = x["latents"]
        loc[i, :m] = x["locs"]
        dirs[i, :m] = x["dirs"]
        act[i, :m] = x["actions"]
        valid[i, :m] = x["valid"]
        pad[i, :m] = False
    return lat, loc, dirs, act, valid, pad


def split_sequences(sequences):
    if len(sequences) < 2:
        return sequences, []
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    by_scene = {}
    for seq in sequences:
        by_scene.setdefault(seq["scene"], []).append(seq)
    train, val = [], []
    for scene in sorted(by_scene):
        scene_sequences = by_scene[scene]
        if len(scene_sequences) < 2:
            train.extend(scene_sequences)
            continue
        order = torch.randperm(len(scene_sequences), generator=generator).tolist()
        n_val = min(len(scene_sequences) - 1, max(1, round(len(scene_sequences) * VAL_FRACTION)))
        val_ids = set(order[:n_val])
        train.extend(seq for i, seq in enumerate(scene_sequences) if i not in val_ids)
        val.extend(seq for i, seq in enumerate(scene_sequences) if i in val_ids)
    return train, val


def write_split(path, train_sequences, val_sequences):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "scene", "walk", "steps", "valid_steps"])
        writer.writeheader()
        for split, sequences in [("train", train_sequences), ("val", val_sequences)]:
            for seq in sequences:
                writer.writerow(
                    {
                        "split": split,
                        "scene": seq["scene"],
                        "walk": seq["walk"],
                        "steps": int(seq["latents"].size(0)),
                        "valid_steps": int(seq["valid"].sum().item()),
                    }
                )


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def loss_terms(model, lat, loc, dirs, act, valid, pad):
    pred = model(lat, pad)
    pred_action = torch.cat([pred[..., :2], F.normalize(pred[..., 2:], dim=-1, eps=1e-8)], dim=-1)
    pred_loc, pred_dir = rollout(loc[:, 0], dirs[:, 0], pred_action[:, 1:])
    trans_loss = F.smooth_l1_loss(pred_action[..., :2][valid], act[..., :2][valid])
    rot_loss = F.smooth_l1_loss(pred_action[..., 2:][valid], act[..., 2:][valid])
    traj_loss = F.smooth_l1_loss(pred_loc[valid], loc[valid])
    dir_loss = F.smooth_l1_loss(pred_dir[valid], dirs[valid])
    loss = trans_loss + rot_loss + TRAJ_WEIGHT * traj_loss + DIR_WEIGHT * dir_loss
    return dict(loss=loss, translation=trans_loss, rotation=rot_loss, trajectory=traj_loss, direction=dir_loss)


def run_epoch(model, loader, optim=None):
    training = optim is not None
    model.train(training)
    sums = dict(loss=0.0, translation=0.0, rotation=0.0, trajectory=0.0, direction=0.0)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        if len(loader) == 0:
            out = {k: float("nan") for k in sums}
            out["valid_steps"] = 0
            return out
        count = 0
        valid_count = 0
        for lat, loc, dirs, act, valid, pad in loader:
            lat, loc, dirs, act, valid, pad = lat.to(device), loc.to(device), dirs.to(device), act.to(device), valid.to(device), pad.to(device)
            if training:
                optim.zero_grad(set_to_none=True)
            losses = loss_terms(model, lat, loc, dirs, act, valid, pad)
            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                optim.step()
            batch_valid = int(valid.sum().item())
            count += 1
            valid_count += batch_valid
            for key, val in losses.items():
                sums[key] += float(val.detach())
    out = {k: v / max(1, count) for k, v in sums.items()}
    out["valid_steps"] = valid_count
    return out


def prefix_keys(prefix, losses):
    return {f"{prefix}_{key}": val for key, val in losses.items()}


def train_epoch(model, loader, optim):
    return run_epoch(model, loader, optim)


def val_epoch(model, loader):
    return run_epoch(model, loader)


def train():
    out = OUT_ROOT / ALIAS
    encoder = load_encoder()
    sequences = build_sequences(encode_all(encoder))
    train_sequences, val_sequences = split_sequences(sequences)
    train_loader = DataLoader(Walks(train_sequences), batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(Walks(val_sequences), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)
    model = ActionTransformer().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS)
    write_split(out / "split.csv", train_sequences, val_sequences)
    print(f"training {ALIAS}: {len(train_sequences)} train, {len(val_sequences)} val sequences")
    best_val = float("inf")
    best_epoch = -1
    for epoch in range(EPOCHS):
        train_losses = train_epoch(model, train_loader, optim)
        val_losses = val_epoch(model, val_loader)
        sched.step()
        improved = val_losses["loss"] == val_losses["loss"] and val_losses["loss"] < best_val
        if improved:
            best_val = val_losses["loss"]
            best_epoch = epoch
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "model_weights_best.pt")
        row = {
            "epoch": epoch,
            **prefix_keys("train", train_losses),
            **prefix_keys("val", val_losses),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "is_best": int(improved),
            "num_sequences": len(sequences),
            "num_train_sequences": len(train_sequences),
            "num_val_sequences": len(val_sequences),
        }
        append_csv(out / "losses.csv", row)
        if epoch % 10 == 0:
            print(row)
        if epoch % 1000 == 0:
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / f"model_weights_{epoch}.pt")
    torch.save(model.state_dict(), out / "model_weights.pt")


if __name__ == "__main__":
    train()
