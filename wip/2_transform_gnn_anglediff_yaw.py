from __future__ import annotations

import csv
import torch
import importlib.util
import sys
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GENConv
from torch_geometric.loader import DataLoader
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

LATENT_DIM = 128
TRAIN_BATCH_SIZE = 20
LEARNING_RATE = 4e-5
WEIGHT_DECAY = 1e-6
CLIP_GRAD_NORM = 1.0
MODEL_ROOT = base.MODEL_ROOT
VAE_WEIGHTS = base.VAE_WEIGHTS
LATENT_BRANCH = base.LATENT_BRANCH
RANDOM_WALK_ROOTS = [
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_translation_anlieferung",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/break_room/random_walks",
    #"../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/hospital/random_walks",
    #"../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/relief/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/relief/rw_translation_relief",
    #"../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/terrains/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/rw_translation_office",
    #"../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00000-kfPV7w3FaU5/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00001-UVdNNRcVyV1/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00002-FxCkHAfgh7A/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00003-NtVbfPCkBFy/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00004-VqCaAuuoeWk/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00005-yPKGKBCyYx8/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00006-HkseAnWCgqk/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00007-UQuchpekHRJ/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00008-VYnUX657cVo/random_walks",
    "../../../scratch/btuncay/cog/baselines/datasets/habitat/train/00009-vLpv2VX547B/random_walks",
]

ALIAS = "0520_gnn_k1_l0_d64_anglediff_yawonly"
EPOCHS = 40000
HIDDEN_DIM = 64
TARGET_DIM = 4
OUT_DIM = 1
GNN_LAYERS = 0 #2
K_NEIGHBORS = 1 #4
TRANSLATION_WEIGHT = 0
ANGLE_WEIGHT = 1.5
TRAJECTORY_WEIGHT = 0.1 #0.2
DIRECTION_WEIGHT = 0.05 #0.5
MOMENTUM_WEIGHT = 0.01
VAL_FRACTION = 0.15
SPLIT_SEED = 1
CHECKPOINT_EVERY = 200
#TODO: probabilistic predictions

# generate knn edges
# relative node-node knn transforms

# nodes initialized as zero
# node attrs: latents (translation), zeros
# edges: latent differences (translation)

# genconv (node i->j): --> relative diff
# This yaw-only variant predicts only the frame-to-frame yaw delta. During loss
# and rollout it reuses the GT frame-centric translation from the edge target.

class TransformGNN(nn.Module):
    def __init__(self, in_dim: int, k: int, hidden_dim: int, out_dim: int, layers: int):
        super().__init__()
        self.k = k
        self.out_dim = out_dim

        self.lift = MLP(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=3,
        )
        self.lift_edges = MLP(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=3,
        )
        self.convs = nn.ModuleList(
            GENConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=hidden_dim,
                norm="layer",
            )
            for _ in range(layers)
        )
        self.edge_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            #nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, nodes_in: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_delta = nodes_in[dst] - nodes_in[src]

        x = self.lift(nodes_in)
        edge_attr = self.lift_edges(edge_delta)

        if edge_index.numel() > 0:
            for conv in self.convs:
                x = x + F.gelu(conv(x, edge_index, edge_attr=edge_attr))
                #edge_attr = edge_attr + F.gelu(edge_update(edge_hidden))
        
        edge_hidden = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.edge_head(edge_hidden)

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
    world_to_local = torch.stack([forward, right, local_up], dim=-2)

    delta = torch.bmm(world_to_local, (loc2.float() - loc1.float()).unsqueeze(-1)).squeeze(-1)
    d1_xy = normalize(dir1[:,:2]) # project on xy plane
    d2_xy = normalize(dir2[:,:2])
    d1 = torch.atan2(d1_xy[:,0],d1_xy[:,1])
    d2 = torch.atan2(d2_xy[:,0],d2_xy[:,1])
    angle_delta = torch.atan2(torch.sin(d1 - d2), torch.cos(d1 - d2))
    out = torch.stack(
        [
            delta[:, 0],
            delta[:, 1],
            delta[:, 2],
            angle_delta
        ],
        dim=-1,
    )
    return out.squeeze(0) if squeeze else out


def make_edge_index(num_nodes: int, k: int) -> torch.Tensor:
    edges = [[i, j] for j in range(num_nodes)
             for i in range(max(0, j-k), min(j+k+1, num_nodes)) if i != j]
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def pairwise_camera_transforms(
    view_dirs: torch.Tensor,
    locs: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return locs.new_empty((0, TARGET_DIM), dtype=torch.float32)
    src, dst = edge_index
    return camera_transform(view_dirs[src], locs[src], view_dirs[dst], locs[dst]).float()


def sequence_to_graph(sequence, k: int) -> Data:
    edge_index = make_edge_index(sequence.latents.size(0), k)
    return Data(
        x=sequence.latents.float(),
        edge_index=edge_index,
        y=pairwise_camera_transforms(sequence.view_dirs, sequence.locs, edge_index),
        loc=sequence.locs.float(),
        view_dir=sequence.view_dirs.float(),
    )


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
        writer = csv.DictWriter(handle, fieldnames=["split", "scene", "walk", "steps", "edges"])
        writer.writeheader()
        for split_name, sequences in [("train", train_sequences), ("val", val_sequences)]:
            for sequence in sequences:
                writer.writerow(
                    {
                        "split": split_name,
                        "scene": sequence.scene_name,
                        "walk": sequence.walk_id,
                        "steps": int(sequence.latents.size(0)),
                        "edges": int(make_edge_index(sequence.latents.size(0), K_NEIGHBORS).size(1)),
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


def compose_transform_with_gt_translation(
    pred_yaw: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([target[:, :3], pred_yaw[:, :1]], dim=-1)


def rollout_immediate_edges(
    pred: torch.Tensor,
    edge_index: torch.Tensor,
    locs: torch.Tensor,
    view_dirs: torch.Tensor,
    ptr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_locs, target_locs = [], []
    pred_dirs, target_dirs = [], []
    immediate = edge_index[1] == edge_index[0] + 1

    for graph_idx in range(ptr.numel() - 1):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        if end - start < 2:
            continue

        edge_mask = immediate & (edge_index[0] >= start) & (edge_index[1] < end)
        edge_ids = torch.nonzero(edge_mask, as_tuple=False).flatten()
        if edge_ids.numel() == 0:
            continue

        edge_ids = edge_ids[torch.argsort(edge_index[0, edge_ids])]
        expected_steps = end - start - 1
        if edge_ids.numel() != expected_steps:
            continue

        loc = locs[start]
        view_dir = view_dirs[start]
        graph_pred_locs = [loc]
        graph_pred_dirs = [view_dir]

        for edge_id in edge_ids:
            transform = pred[edge_id]
            loc = loc + local_delta_to_world(
                transform[:3].unsqueeze(0),
                view_dir.unsqueeze(0),
            ).squeeze(0)
            view_dir = apply_local_yaw(
                view_dir.unsqueeze(0),
                transform[3:4].unsqueeze(0),
            ).squeeze(0)
            graph_pred_locs.append(loc)
            graph_pred_dirs.append(view_dir)

        pred_locs.append(torch.stack(graph_pred_locs[1:], dim=0))
        target_locs.append(locs[start + 1:end])
        pred_dirs.append(torch.stack(graph_pred_dirs[1:], dim=0))
        target_dirs.append(view_dirs[start + 1:end])

    if not pred_locs:
        empty = locs.new_empty((0, 3))
        return empty, empty, empty, empty

    return (
        torch.cat(pred_locs, dim=0),
        torch.cat(target_locs, dim=0),
        torch.cat(pred_dirs, dim=0),
        torch.cat(target_dirs, dim=0),
    )


def trajectory_loss_terms(
    pred: torch.Tensor,
    edge_index: torch.Tensor,
    locs: torch.Tensor,
    view_dirs: torch.Tensor,
    ptr: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_locs, target_locs, pred_dirs, target_dirs = rollout_immediate_edges(
        pred,
        edge_index,
        locs,
        view_dirs,
        ptr,
    )
    if pred_locs.numel() == 0:
        zero = pred.sum() * 0.0
        return {"trajectory": zero, "direction": zero, "trajectory_steps": zero}

    finite = (
        torch.isfinite(pred_locs).all(dim=-1)
        & torch.isfinite(target_locs).all(dim=-1)
        & torch.isfinite(pred_dirs).all(dim=-1)
        & torch.isfinite(target_dirs).all(dim=-1)
    )
    if not finite.any():
        zero = pred.sum() * 0.0
        return {"trajectory": zero, "direction": zero, "trajectory_steps": zero}

    trajectory_loss = F.smooth_l1_loss(pred_locs[finite], target_locs[finite])
    direction_loss = unit_direction_loss(pred_dirs[finite], target_dirs[finite])
    return {
        "trajectory": trajectory_loss,
        "direction": direction_loss,
        "trajectory_steps": finite.sum().to(dtype=pred.dtype),
    }

def momentum_losses(pred,edge_index,locs,view_dirs,ptr) -> torch.Tensor:
    pred_locs, target_locs, pred_dirs, target_dirs = rollout_immediate_edges(
        pred,edge_index,locs,view_dirs,ptr)
    vel_losses = []
    acc_losses = []
    dir_acc_losses = []
    rot_losses = []
    rot_acc_losses = []

    offset = 0
    for graph_idx in range(ptr.numel() - 1):
        start = int(ptr[graph_idx].item())
        end = int(ptr[graph_idx + 1].item())
        steps = end - start - 1
        graph_pred_locs = pred_locs[offset:offset + steps]
        graph_target_locs = target_locs[offset:offset + steps]
        graph_pred_dirs = pred_dirs[offset:offset + steps]
        graph_target_dirs = target_dirs[offset:offset + steps]
        offset += steps

        target_disps = graph_target_locs[1:]-graph_target_locs[:-1]
        target_rots = graph_target_dirs[1:]-graph_target_dirs[:-1]
        pred_disps = graph_pred_locs[1:]-graph_pred_locs[:-1]
        pred_rots = graph_pred_dirs[1:]-graph_pred_dirs[:-1]

        vel_target = torch.norm(target_disps, dim=1)
        vel_pred = torch.norm(pred_disps, dim=1)
        rot_vel_target = torch.norm(target_rots, dim=1)
        rot_vel_pred = torch.norm(pred_rots, dim=1)

        vel_losses.append(F.smooth_l1_loss(vel_target,vel_pred))
        acc_losses.append(F.smooth_l1_loss(torch.diff(vel_target),torch.diff(vel_pred)))
        dir_acc_losses.append(F.mse_loss(torch.diff(target_disps,dim=0),torch.diff(pred_disps,dim=0)))
        rot_losses.append(F.smooth_l1_loss(rot_vel_target,rot_vel_pred))
        rot_acc_losses.append(F.mse_loss(torch.diff(rot_vel_target,dim=0),torch.diff(rot_vel_pred,dim=0)))

    if not vel_losses:
        return pred.sum() * 0.0

    vel_loss = torch.stack(vel_losses).mean()
    acc_loss = torch.stack(acc_losses).mean()
    dir_acc_loss = torch.stack(dir_acc_losses).mean()
    rot_loss = torch.stack(rot_losses).mean()
    rot_acc_loss = torch.stack(rot_acc_losses).mean()
    #return {"velocity":vel_loss,"acceleration":acc_loss,
    #        "dir_accel":dir_acc_loss,"rotation":rot_loss}
    return 0.4*vel_loss + 0.2*acc_loss + 0.5*dir_acc_loss + rot_loss + rot_acc_loss

def loss_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
    locs: torch.Tensor,
    view_dirs: torch.Tensor,
    ptr: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    valid = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    if not valid.any():
        zero = pred.sum() * 0.0
        return {
            "loss": zero,
            "translation": zero,
            "angle": zero,
            "trajectory": zero,
            "direction": zero,
            "trajectory_steps": zero,
            "momentum": zero,
        }

    edge_pred = pred[valid]
    edge_target = target[valid]
    translation_loss = pred.sum() * 0.0
    #angle_error = wrap_angle(edge_pred[:, 3:] - edge_target[:, 3:])
    angle_error = wrap_angle(edge_pred[:, :1] - edge_target[:, 3:4])
    angle_loss = F.mse_loss(angle_error, torch.zeros_like(angle_error))
    pred_transform = compose_transform_with_gt_translation(pred, target)
    trajectory_terms = trajectory_loss_terms(pred_transform, edge_index, locs, view_dirs, ptr)
    momentum_loss = momentum_losses(pred_transform, edge_index, locs, view_dirs, ptr)
    loss = (
        TRANSLATION_WEIGHT * translation_loss.mean()
        + ANGLE_WEIGHT * angle_loss.mean()
        + TRAJECTORY_WEIGHT * trajectory_terms["trajectory"].mean()
        + DIRECTION_WEIGHT * trajectory_terms["direction"].mean()
        + MOMENTUM_WEIGHT * momentum_loss.mean()
    )
    return {
        "loss": loss,
        "translation": translation_loss.mean(),
        "angle": angle_loss.mean(),
        "trajectory": trajectory_terms["trajectory"].mean(),
        "direction": trajectory_terms["direction"].mean(),
        "trajectory_steps": trajectory_terms["trajectory_steps"].mean(),
        "momentum": momentum_loss.mean(),
    }


def run_epoch(
    model: TransformGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "translation": 0.0, "angle": 0.0, "trajectory": 0.0, "direction": 0.0, "momentum": 0.0}
    count = 0
    edge_count = 0
    trajectory_steps = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            batch = batch.to(device)
            if batch.edge_index.numel() == 0:
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred = model(batch.x.float(), batch.edge_index)
            target = batch.y.float()
            losses = loss_terms(
                pred,
                target,
                batch.edge_index,
                batch.loc.float(),
                batch.view_dir.float(),
                batch.ptr,
            )

            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()

            count += 1
            edge_count += int(target.size(0))
            trajectory_steps += int(losses["trajectory_steps"].detach().item())
            for key, value in losses.items():
                if key == "trajectory_steps":
                    continue
                sums[key] += float(value.detach())

    if count == 0:
        out = {key: float("nan") for key in sums}
    else:
        out = {key: value / count for key, value in sums.items()}
    out["edges"] = edge_count
    out["trajectory_steps"] = trajectory_steps
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
    train_graphs = [sequence_to_graph(sequence, K_NEIGHBORS) for sequence in train_sequences]
    val_graphs = [sequence_to_graph(sequence, K_NEIGHBORS) for sequence in val_sequences]
    if not train_graphs:
        raise ValueError("No train graphs were built from the encoded walk sequences.")

    train_loader = DataLoader(
        train_graphs,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = TransformGNN(
        in_dim=LATENT_DIM,
        k=K_NEIGHBORS,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        layers=GNN_LAYERS,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    write_split(output_dir / "split.csv", train_sequences, val_sequences)
    num_edges = sum(graph.edge_index.size(1) for graph in train_graphs + val_graphs)
    print(
        f"training {ALIAS}: {len(train_sequences)} train, {len(val_sequences)} val sequences, "
        f"{num_edges} pairwise edges, k={K_NEIGHBORS}, layers={GNN_LAYERS}, branch={LATENT_BRANCH}"
    )

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(EPOCHS):
        train_losses = run_epoch(model, train_loader, optimizer)
        val_losses = run_epoch(model, val_loader)
        scheduler.step()

        monitor_loss = val_losses["loss"] if val_graphs else train_losses["loss"]
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
            "k_neighbors": K_NEIGHBORS,
            "gnn_layers": GNN_LAYERS,
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
