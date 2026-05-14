from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GENConv


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module(
    "nips_forward_rotation_transformer_base",
    Path(__file__).with_name("2_forward_rotation_transformer.py"),
)

device = base.device
torch.set_default_dtype(torch.float32)

LATENT_DIM = base.LATENT_DIM
MODEL_DIM = 32
DROPOUT = 0.25
EPOCHS = 40000
TRAIN_BATCH_SIZE = base.TRAIN_BATCH_SIZE
LR = base.LR
WEIGHT_DECAY = 5e-4
VAE_WEIGHTS = base.VAE_WEIGHTS
OUT_ROOT = base.OUT_ROOT
TEMPORAL_K = 4
TRANSLATION_WEIGHT = 1.0 #2
ROTATION_WEIGHT = 1.0
TRAJECTORY_WEIGHT = 3 #0.5
DIRECTION_WEIGHT = 3.0 #1
GNN_LAYERS = 1
EARLY_STOP_PATIENCE = 2000
EARLY_STOP_MIN_DELTA = 1e-4
ALIAS = "0507_gnn_zero_edge_relpose"

ImgOnlyEncoder = base.ImgOnlyEncoder
make_action = base.make_action
strip_module = base.strip_module


class ActionGNN(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = MODEL_DIM,
        temporal_k: int = TEMPORAL_K,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.temporal_k = temporal_k
        self.hidden_dim = hidden_dim
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(latent_dim + 1),
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList(
            GENConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, norm="layer")
            for _ in range(GNN_LAYERS)
        )
        self.pose_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        latents: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = torch.zeros(latents.shape[:2], device=latents.device, dtype=torch.bool)

        batch_size, num_steps, _ = latents.shape
        h = latents.new_zeros(batch_size * num_steps, self.hidden_dim)
        edge_batch, edge_src_step, edge_dst_step = build_temporal_edges(padding_mask, self.temporal_k)

        if edge_batch.numel() > 0:
            edge_index = temporal_edge_index(edge_batch, edge_src_step, edge_dst_step, num_steps)
            edge_attr = self.edge_encoder(
                temporal_edge_features(latents, edge_batch, edge_src_step, edge_dst_step, self.temporal_k)
            )
            for conv in self.convs:
                h = h + F.gelu(conv(h, edge_index, edge_attr=edge_attr))

        decoded = self.pose_decoder(h).view(batch_size, num_steps, 3)
        coords = decoded[..., :2].masked_fill(padding_mask.unsqueeze(-1), 0.0)
        angles = decoded[..., 2:].masked_fill(padding_mask.unsqueeze(-1), 0.0)

        frame_dir = fixed_frame_dir(latents.new_zeros(batch_size, num_steps, 3))
        locs = planar_coords_to_world(coords, frame_dir)
        dirs = apply_planar_rotation(frame_dir, angles)
        locs = locs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        dirs = dirs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return dict(
            loc=locs,
            direction=dirs,
            move=coords,
            rotation=angles,
            edge_batch=edge_batch,
            edge_src_step=edge_src_step,
            edge_dst_step=edge_dst_step,
        )


def encode_all(encoder):
    return base.encode_all(encoder)


def build_sequences(obs):
    return base.build_sequences(obs)


def collate(batch):
    return base.collate(batch)


def build_temporal_edges(
    padding_mask: torch.Tensor,
    temporal_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_batch, edge_src_step, edge_dst_step = [], [], []
    batch_size = padding_mask.size(0)
    for batch_id in range(batch_size):
        valid_steps = torch.nonzero(~padding_mask[batch_id], as_tuple=False).flatten()
        count = int(valid_steps.numel())
        if count < 2:
            continue

        for delta in range(1, min(temporal_k, count - 1) + 1):
            src_steps = valid_steps[:-delta]
            dst_steps = valid_steps[delta:]
            batch_ids = torch.full((src_steps.numel(),), batch_id, device=padding_mask.device, dtype=torch.long)
            edge_batch.extend([batch_ids, batch_ids])
            edge_src_step.extend([src_steps, dst_steps])
            edge_dst_step.extend([dst_steps, src_steps])

    if not edge_batch:
        empty = torch.empty(0, device=padding_mask.device, dtype=torch.long)
        return empty, empty, empty
    return torch.cat(edge_batch), torch.cat(edge_src_step), torch.cat(edge_dst_step)


def temporal_edge_index(
    edge_batch: torch.Tensor,
    edge_src_step: torch.Tensor,
    edge_dst_step: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    return torch.stack(
        [
            edge_batch * num_steps + edge_src_step,
            edge_batch * num_steps + edge_dst_step,
        ],
        dim=0,
    )


def temporal_edge_features(
    latents: torch.Tensor,
    edge_batch: torch.Tensor,
    edge_src_step: torch.Tensor,
    edge_dst_step: torch.Tensor,
    temporal_k: int,
) -> torch.Tensor:
    src_latents = latents[edge_batch, edge_src_step]
    dst_latents = latents[edge_batch, edge_dst_step]
    edge_delta = (edge_dst_step - edge_src_step).to(dtype=latents.dtype).unsqueeze(-1)
    edge_delta = edge_delta / max(1, temporal_k)
    return torch.cat([dst_latents - src_latents, edge_delta], dim=-1)


def identity_planar_rotation(shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros((*shape, 1))


def fixed_frame_dir(reference: torch.Tensor) -> torch.Tensor:
    return base.WORLD_FORWARD.to(device=reference.device, dtype=reference.dtype).expand_as(reference)


def apply_planar_rotation(src_dir: torch.Tensor, local_rot: torch.Tensor) -> torch.Tensor:
    angle = local_rot[..., :1]
    return F.normalize(
        torch.sin(angle) * base.ground_right(src_dir)
        + torch.cos(angle) * base.ground_forward(src_dir),
        dim=-1,
        eps=1e-8,
    )


def planar_coords_to_world(coords: torch.Tensor, frame_dir: torch.Tensor) -> torch.Tensor:
    return (
        coords[..., :1] * base.ground_right(frame_dir)
        + coords[..., 1:2] * base.ground_forward(frame_dir)
    )


def relative_planar_coords(
    src_loc: torch.Tensor,
    dst_loc: torch.Tensor,
    src_dir: torch.Tensor,
) -> torch.Tensor:
    delta = dst_loc - src_loc
    return torch.cat(
        [
            (delta * base.ground_right(src_dir)).sum(dim=-1, keepdim=True),
            (delta * base.ground_forward(src_dir)).sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )


def relative_planar_direction(src_dir: torch.Tensor, dst_dir: torch.Tensor) -> torch.Tensor:
    return apply_planar_rotation(
        fixed_frame_dir(src_dir),
        planar_rotation_target(src_dir, dst_dir),
    )


def immediate_planar_actions(
    locs: torch.Tensor,
    dirs: torch.Tensor,
    padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    moves = locs.new_zeros((*locs.shape[:2], 2))
    rotations = locs.new_zeros((*locs.shape[:2], 1))
    if locs.size(1) < 2:
        return moves, rotations

    pair_valid = ~(padding_mask[:, :-1] | padding_mask[:, 1:])
    moves[:, 1:] = relative_planar_coords(
        locs[:, :-1].reshape(-1, 3),
        locs[:, 1:].reshape(-1, 3),
        dirs[:, :-1].reshape(-1, 3),
    ).view(locs.size(0), locs.size(1) - 1, 2)
    rotations[:, 1:] = planar_rotation_target(
        dirs[:, :-1].reshape(-1, 3),
        dirs[:, 1:].reshape(-1, 3),
    ).view(locs.size(0), locs.size(1) - 1, 1)
    moves[:, 1:] = moves[:, 1:].masked_fill(~pair_valid.unsqueeze(-1), 0.0)
    rotations[:, 1:] = rotations[:, 1:].masked_fill(~pair_valid.unsqueeze(-1), 0.0)
    return moves, rotations


def rollout_planar(
    start_loc: torch.Tensor,
    start_dir: torch.Tensor,
    moves: torch.Tensor,
    rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out motion in the start node's canonical planar frame."""
    if moves.size(1) != rotations.size(1):
        raise ValueError(
            "moves and rotations must have the same sequence length, "
            f"got {moves.size(1)} and {rotations.size(1)}"
        )

    locs = [torch.zeros_like(start_loc)]
    dirs = [fixed_frame_dir(start_dir)]
    for move, rotation in zip(moves.unbind(dim=1), rotations.unbind(dim=1)):
        step_dir = dirs[-1]
        locs.append(
            locs[-1]
            + move[:, :1] * base.ground_right(step_dir)
            + move[:, 1:2] * base.ground_forward(step_dir)
        )
        dirs.append(apply_planar_rotation(step_dir, rotation))
    return torch.stack(locs, dim=1), torch.stack(dirs, dim=1)


def planar_rotation_target(src_dir: torch.Tensor, dst_dir: torch.Tensor) -> torch.Tensor:
    dst_forward = base.ground_forward(dst_dir)
    right = (dst_forward * base.ground_right(src_dir)).sum(dim=-1, keepdim=True)
    forward = (dst_forward * base.ground_forward(src_dir)).sum(dim=-1, keepdim=True)
    return torch.atan2(right, forward)


def unit_direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1, eps=1e-8)
    target = F.normalize(target, dim=-1, eps=1e-8)
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean()


def wrap_planar_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def planar_rotation_loss(raw_rot: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = wrap_planar_angle(raw_rot - target)
    return (1.0 - torch.cos(err)).mean() + F.smooth_l1_loss(err, torch.zeros_like(err), beta=0.25)


def immediate_planar_rotation_targets(dirs: torch.Tensor) -> torch.Tensor:
    out = identity_planar_rotation(dirs.shape[:2], dirs)
    if dirs.size(1) > 1:
        out[:, 1:] = planar_rotation_target(
            dirs[:, :-1].reshape(-1, 3),
            dirs[:, 1:].reshape(-1, 3),
        ).view(dirs.size(0), dirs.size(1) - 1, 1)
    return out


def edge_relative_pose(
    locs: torch.Tensor,
    dirs: torch.Tensor,
    edge_batch: torch.Tensor,
    edge_src_step: torch.Tensor,
    edge_dst_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_loc = locs[edge_batch, edge_src_step]
    dst_loc = locs[edge_batch, edge_dst_step]
    src_dir = dirs[edge_batch, edge_src_step]
    dst_dir = dirs[edge_batch, edge_dst_step]
    coords = relative_planar_coords(src_loc, dst_loc, src_dir)
    rel_dirs = relative_planar_direction(src_dir, dst_dir)
    return coords, rel_dirs


def loss_terms(model, lat, loc, dirs, act, valid, pad):
    out = model(lat, pad)
    pred_coords = out["move"]
    pred_rel_dirs = out["direction"]
    edge_batch = out["edge_batch"]
    edge_src_step = out["edge_src_step"]
    edge_dst_step = out["edge_dst_step"]

    if edge_batch.numel() > 0:
        target_coords, target_rel_dirs = edge_relative_pose(
            loc,
            dirs,
            edge_batch,
            edge_src_step,
            edge_dst_step,
        )
        pred_edge_coords = pred_coords[edge_batch, edge_dst_step]
        pred_edge_dirs = pred_rel_dirs[edge_batch, edge_dst_step]
        finite = (
            torch.isfinite(target_coords).all(dim=-1)
            & torch.isfinite(target_rel_dirs).all(dim=-1)
        )
        if finite.any():
            coord_loss = F.smooth_l1_loss(pred_edge_coords[finite], target_coords[finite], beta=0.10)
            dir_loss = unit_direction_loss(pred_edge_dirs[finite], target_rel_dirs[finite])
        else:
            coord_loss = pred_coords.new_tensor(0.0)
            dir_loss = pred_coords.new_tensor(0.0)
    else:
        coord_loss = pred_coords.new_tensor(0.0)
        dir_loss = pred_coords.new_tensor(0.0)

    loss = (
        TRANSLATION_WEIGHT * coord_loss
        + DIRECTION_WEIGHT * dir_loss
    )
    return dict(
        loss=loss,
        coordinates=coord_loss,
        direction=dir_loss,
    )


def run_epoch(model, loader, optim=None):
    training = optim is not None
    model.train(training)
    sums = dict(
        loss=0.0,
        coordinates=0.0,
        direction=0.0,
    )
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        if len(loader) == 0:
            out = {k: float("nan") for k in sums}
            out["valid_steps"] = 0
            return out
        count = 0
        valid_count = 0
        for lat, loc, dirs, act, valid, pad in loader:
            lat = lat.to(device)
            loc = loc.to(device)
            dirs = dirs.to(device)
            act = act.to(device)
            valid = valid.to(device)
            pad = pad.to(device)
            if training:
                optim.zero_grad(set_to_none=True)
            losses = loss_terms(model, lat, loc, dirs, act, valid, pad)
            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP_GRAD)
                optim.step()
            count += 1
            valid_count += int(valid.sum().item())
            for key, val in losses.items():
                sums[key] += float(val.detach())
    out = {key: val / max(1, count) for key, val in sums.items()}
    out["valid_steps"] = valid_count
    return out


def train():
    out = OUT_ROOT / ALIAS
    encoder = base.load_encoder()
    sequences = build_sequences(encode_all(encoder))
    train_sequences, val_sequences = base.split_sequences(sequences)
    train_loader = DataLoader(base.Walks(train_sequences), batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(base.Walks(val_sequences), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)
    model = ActionGNN().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS)

    base.write_split(out / "split.csv", train_sequences, val_sequences)
    print(
        f"training {ALIAS}: {len(train_sequences)} train, {len(val_sequences)} val sequences, "
        f"temporal_k={TEMPORAL_K}, layers={GNN_LAYERS}"
    )

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(EPOCHS):
        train_losses = run_epoch(model, train_loader, optim)
        val_losses = run_epoch(model, val_loader)
        sched.step()

        improved = val_losses["loss"] == val_losses["loss"] and val_losses["loss"] < best_val - EARLY_STOP_MIN_DELTA
        if improved:
            best_val = val_losses["loss"]
            best_epoch = epoch
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "model_weights_best.pt")

        row = {
            "epoch": epoch,
            **base.prefix_keys("train", train_losses),
            **base.prefix_keys("val", val_losses),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "is_best": int(improved),
            "num_sequences": len(sequences),
            "num_train_sequences": len(train_sequences),
            "num_val_sequences": len(val_sequences),
            "temporal_k": TEMPORAL_K,
            "gnn_layers": GNN_LAYERS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        }
        base.append_csv(out / "losses.csv", row)
        if epoch % 10 == 0:
            print(row)
        if epoch % 1000 == 0:
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / f"model_weights_{epoch}.pt")
        if best_epoch >= 0 and epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(
                f"early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch}, best_val_loss={best_val}"
            )
            break

    torch.save(model.state_dict(), out / "model_weights.pt")


if __name__ == "__main__":
    train()
