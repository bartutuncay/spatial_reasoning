from __future__ import annotations

import csv
import torch
import importlib.util
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Tuple, Dict, Sequence
import torch.nn.functional as F

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

base = load_module("nips_translation_transformer_base", Path(__file__).with_name("2_translation_transformer.py"))
device = base.device
torch.set_default_dtype(torch.float32)
#TODO: calculate vector-vector distribution transforms --> predict physical transforms
#TODO: attention mechanism within vectors: calculate transform of significant sections
LATENT_DIM = 128
TRAIN_BATCH_SIZE = 30
LEARNING_RATE = 2e-6
WEIGHT_DECAY = 1e-6
CLIP_GRAD_NORM = 1.0
MODEL_ROOT = base.MODEL_ROOT
VAE_WEIGHTS = base.VAE_WEIGHTS
LATENT_BRANCH = base.LATENT_BRANCH
RANDOM_WALK_ROOTS = [
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_translation",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/break_room/random_walks",
    #"../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/hospital/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/relief/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/terrains/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/rw_translation",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/random_walks",
]

ALIAS = "0511_lstm_l2"
EPOCHS = 20000
HIDDEN_DIM = 16
OUT_DIM = 5
LSTM_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.0
TRANSLATION_WEIGHT = 1.0
ANGLE_WEIGHT = 2.0
TRAJECTORY_WEIGHT = 0.1
DIRECTION_WEIGHT = 0.5 #0.5
VAL_FRACTION = 0.15
SPLIT_SEED = 1
CHECKPOINT_EVERY = 200
    

class TransformLSTM(nn.Module):
    def __init__(
        self,
        in_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        out_dim: int = OUT_DIM,
        layers: int = LSTM_LAYERS,
        bidirectional: bool = BIDIRECTIONAL,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        self.node_dim = hidden_dim * (2 if self.bidirectional else 1)

        self.lift = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.Linear(self.node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        latents: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze = latents.dim() == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        if padding_mask is None:
            padding_mask = torch.zeros(latents.shape[:2], device=latents.device, dtype=torch.bool)

        x = self.lift(latents.float())
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        h, _ = self.rnn(x)
        transforms = self.head(h)
        transforms = transforms.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        transforms[:, 0] = 0.0
        return transforms.squeeze(0) if squeeze else transforms

def normalize(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def camera_frame_axes(
    view_dir: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward = normalize(view_dir.float(), eps=eps)
    world_up = forward.new_tensor([0.0, 0.0, 1.0]).expand_as(forward)

    right = torch.cross(world_up, forward, dim=-1)
    right_norm = right.norm(dim=-1, keepdim=True)
    bad_right = right_norm.squeeze(-1) < eps
    if bool(bad_right.any()):
        candidates = torch.eye(3, device=forward.device, dtype=forward.dtype)
        fallback_up = candidates[torch.argmin(torch.abs(forward @ candidates.T), dim=-1)]
        fallback_right = torch.cross(fallback_up, forward, dim=-1)
        right = torch.where(bad_right.unsqueeze(-1), fallback_right, right)

    right = normalize(right, eps=eps)
    local_up = normalize(torch.cross(forward, right, dim=-1), eps=eps)
    return forward, right, local_up


def local_delta_to_world(delta: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
    forward, right, local_up = camera_frame_axes(view_dir)
    return (
        delta[..., :1] * forward
        + delta[..., 1:2] * right
        + delta[..., 2:3] * local_up
    )


def apply_local_yaw(view_dir: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    if yaw.dim() == view_dir.dim() - 1:
        yaw = yaw.unsqueeze(-1)
    forward, right, _ = camera_frame_axes(view_dir)
    return normalize(torch.cos(yaw) * forward + torch.sin(yaw) * right)


def camera_transform(
    dir1: torch.Tensor,
    loc1: torch.Tensor,
    dir2: torch.Tensor,
    loc2: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    squeeze = dir1.dim() == 1
    if squeeze:
        dir1 = dir1.unsqueeze(0)
        loc1 = loc1.unsqueeze(0)
        dir2 = dir2.unsqueeze(0)
        loc2 = loc2.unsqueeze(0)

    forward, right, local_up = camera_frame_axes(dir1, eps=eps)
    target_forward = normalize(dir2.float(), eps=eps)
    world_to_local = torch.stack([forward, right, local_up], dim=-2)

    delta = torch.bmm(world_to_local, (loc2.float() - loc1.float()).unsqueeze(-1)).squeeze(-1)
    target_dir = torch.bmm(world_to_local, target_forward.unsqueeze(-1)).squeeze(-1)
    out = torch.stack(
        [
            delta[:, 0],
            delta[:, 1],
            delta[:, 2],
            torch.atan2(delta[:, 1], delta[:, 0]),
            torch.atan2(target_dir[:, 1], target_dir[:, 0]),
        ],
        dim=-1,
    )
    return out.squeeze(0) if squeeze else out


def split_sequences(sequences: Sequence) -> Tuple[list, list]:
    if len(sequences) < 2:
        return list(sequences), []

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    by_scene: Dict[str, list] = {}
    for sequence in sequences:
        by_scene.setdefault(sequence.scene_name, []).append(sequence)

    train, val = [], []
    for scene_name in sorted(by_scene):
        scene_sequences = by_scene[scene_name]
        if len(scene_sequences) < 2:
            train.extend(scene_sequences)
            continue
        order = torch.randperm(len(scene_sequences), generator=generator).tolist()
        num_val = min(len(scene_sequences) - 1, max(1, round(len(scene_sequences) * VAL_FRACTION)))
        val_ids = set(order[:num_val])
        train.extend(sequence for idx, sequence in enumerate(scene_sequences) if idx not in val_ids)
        val.extend(sequence for idx, sequence in enumerate(scene_sequences) if idx in val_ids)
    return train, val


def write_split(path: Path, train_sequences: Sequence, val_sequences: Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "scene", "walk", "steps"])
        writer.writeheader()
        for split_name, sequences in [("train", train_sequences), ("val", val_sequences)]:
            for sequence in sequences:
                writer.writerow(
                    {
                        "split": split_name,
                        "scene": sequence.scene_name,
                        "walk": sequence.walk_id,
                        "steps": int(sequence.latents.size(0)),
                    }
                )


def prefix_keys(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def unit_direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = normalize(pred)
    target = normalize(target)
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean()


def immediate_transform_targets(
    locs: torch.Tensor,
    view_dirs: torch.Tensor,
    valid_steps: torch.Tensor,
) -> torch.Tensor:
    targets = locs.new_zeros((*locs.shape[:2], OUT_DIM))
    if locs.size(1) < 2:
        return targets

    flat_targets = camera_transform(
        view_dirs[:, :-1].reshape(-1, 3),
        locs[:, :-1].reshape(-1, 3),
        view_dirs[:, 1:].reshape(-1, 3),
        locs[:, 1:].reshape(-1, 3),
    ).reshape(locs.size(0), locs.size(1) - 1, OUT_DIM)
    targets[:, 1:] = flat_targets
    return targets.masked_fill(~valid_steps.unsqueeze(-1), 0.0)


def rollout_transforms(
    transforms: torch.Tensor,
    start_locs: torch.Tensor,
    start_dirs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loc = start_locs
    view_dir = normalize(start_dirs)
    locs = [loc]
    dirs = [view_dir]

    for step in range(1, transforms.size(1)):
        transform = transforms[:, step]
        loc = loc + local_delta_to_world(transform[:, :3], view_dir)
        view_dir = apply_local_yaw(view_dir, transform[:, 4:5])
        locs.append(loc)
        dirs.append(view_dir)

    return torch.stack(locs, dim=1), torch.stack(dirs, dim=1)


def loss_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    locs: torch.Tensor,
    view_dirs: torch.Tensor,
    valid_steps: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    valid = (
        valid_steps
        & torch.isfinite(pred).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(locs).all(dim=-1)
        & torch.isfinite(view_dirs).all(dim=-1)
    )
    if not valid.any():
        zero = pred.sum() * 0.0
        return {
            "loss": zero,
            "translation": zero,
            "angle": zero,
            "trajectory": zero,
            "direction": zero,
            "trajectory_steps": zero,
        }

    translation_loss = F.smooth_l1_loss(pred[..., :3][valid], target[..., :3][valid])
    angle_error = wrap_angle(pred[..., 3:][valid] - target[..., 3:][valid])
    angle_loss = F.smooth_l1_loss(angle_error, torch.zeros_like(angle_error))
    pred_locs, pred_dirs = rollout_transforms(pred, locs[:, 0], view_dirs[:, 0])
    trajectory_loss = F.smooth_l1_loss(pred_locs[valid], locs[valid])
    direction_loss = unit_direction_loss(pred_dirs[valid], view_dirs[valid])
    loss = (
        TRANSLATION_WEIGHT * translation_loss
        + ANGLE_WEIGHT * angle_loss
        + TRAJECTORY_WEIGHT * trajectory_loss
        + DIRECTION_WEIGHT * direction_loss
    )
    return {
        "loss": loss,
        "translation": translation_loss,
        "angle": angle_loss,
        "trajectory": trajectory_loss,
        "direction": direction_loss,
        "trajectory_steps": valid.sum().to(dtype=pred.dtype),
    }


def run_epoch(
    model: TransformLSTM,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    if loader is None:
        return {
            "loss": float("nan"),
            "translation": float("nan"),
            "angle": float("nan"),
            "trajectory": float("nan"),
            "direction": float("nan"),
            "trajectory_steps": 0,
            "sequences": 0,
        }

    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "translation": 0.0, "angle": 0.0, "trajectory": 0.0, "direction": 0.0}
    count = 0
    sequence_count = 0
    trajectory_steps = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            latents = batch["latents"].to(device)
            locs = batch["locs"].to(device)
            view_dirs = batch["view_dirs"].to(device)
            valid_steps = batch["valid_steps"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred = model(latents, padding_mask=padding_mask)
            target = immediate_transform_targets(locs.float(), view_dirs.float(), valid_steps)
            losses = loss_terms(
                pred,
                target,
                locs.float(),
                view_dirs.float(),
                valid_steps,
            )

            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()

            count += 1
            sequence_count += int(latents.size(0))
            trajectory_steps += int(losses["trajectory_steps"].detach().item())
            for key, value in losses.items():
                if key == "trajectory_steps":
                    continue
                sums[key] += float(value.detach())

    if count == 0:
        out = {key: float("nan") for key in sums}
    else:
        out = {key: value / count for key, value in sums.items()}
    out["trajectory_steps"] = trajectory_steps
    out["sequences"] = sequence_count
    return out


def train() -> None:
    output_dir = MODEL_ROOT / ALIAS
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "losses.csv"
    write_header = not csv_path.exists()

    encoder = base.load_encoder(VAE_WEIGHTS, latent_dim=LATENT_DIM)
    observations = base.encode_all_observations(
        encoder,
        RANDOM_WALK_ROOTS,
        latent_branch=LATENT_BRANCH,
    )
    sequences = base.build_walk_sequences(observations)
    train_sequences, val_sequences = split_sequences(sequences)
    if not train_sequences:
        raise ValueError("No train sequences were built from the encoded walk sequences.")

    train_loader = DataLoader(
        base.WalkSequenceDataset(train_sequences),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=base.collate_walk_sequences,
    )
    val_loader = (
        DataLoader(
            base.WalkSequenceDataset(val_sequences),
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=base.collate_walk_sequences,
        )
        if val_sequences
        else None
    )

    model = TransformLSTM().to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    write_split(output_dir / "split.csv", train_sequences, val_sequences)
    print(
        f"training {ALIAS}: {len(train_sequences)} train, {len(val_sequences)} val sequences, "
        f"lstm_layers={LSTM_LAYERS}, bidirectional={BIDIRECTIONAL}, branch={LATENT_BRANCH}"
    )

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(EPOCHS):
        train_losses = run_epoch(model, train_loader, optimizer)
        val_losses = run_epoch(model, val_loader)
        scheduler.step()

        monitor_loss = val_losses["loss"] if val_sequences else train_losses["loss"]
        improved = monitor_loss == monitor_loss and monitor_loss < best_val
        if improved:
            best_val = monitor_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "model_weights_best.pt")

        row = {
            "epoch": epoch,
            **prefix_keys("train", train_losses),
            **prefix_keys("val", val_losses),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "is_best": int(improved),
            "num_observations": len(observations),
            "num_sequences": len(sequences),
            "num_train_sequences": len(train_sequences),
            "num_val_sequences": len(val_sequences),
            "lstm_layers": LSTM_LAYERS,
            "bidirectional": int(BIDIRECTIONAL),
            "translation_weight": TRANSLATION_WEIGHT,
            "angle_weight": ANGLE_WEIGHT,
            "trajectory_weight": TRAJECTORY_WEIGHT,
            "direction_weight": DIRECTION_WEIGHT,
        }
        base.append_loss_row(csv_path, row, write_header=write_header)
        write_header = False

        if epoch % 10 == 0:
            print(row)
        if epoch % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), output_dir / f"model_weights_{epoch}.pt")

    torch.save(model.state_dict(), output_dir / "model_weights.pt")
    print("completed")


if __name__ == "__main__":
    train()
