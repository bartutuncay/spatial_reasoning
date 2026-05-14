from __future__ import annotations

import csv
import torch
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from typing import Tuple, Dict, List, Sequence
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
TRAIN_BATCH_SIZE = 30
LEARNING_RATE = 2e-5
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
    #"../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/random_walks",
]

ALIAS = "0512_gru_stable"
EPOCHS = 40000
HIDDEN_DIM = 16 #16
OUT_DIM = 5
GRU_LAYERS = 2
DROPOUT = 0.0
TRANSLATION_WEIGHT = 1.0
ANGLE_WEIGHT = 1.5
TRAJECTORY_WEIGHT = 0.2
DIRECTION_WEIGHT = 0.01 #0.5
COVARIANCE_WEIGHT = 0.01
POSE_LOGVAR_MIN = -6.0
POSE_LOGVAR_MAX = 4.0
COVARIANCE_TARGET_EPS = 1e-4
VAL_FRACTION = 0.15
SPLIT_SEED = 1
CHECKPOINT_EVERY = 200

@dataclass(frozen=True)
class ProbEncodedObservation:
    path: str
    scene_name: str
    walk_id: int
    step_id: int
    latent_mu: torch.Tensor
    latent_logvar: torch.Tensor
    loc: torch.Tensor
    view_dir: torch.Tensor


@dataclass(frozen=True)
class ProbWalkSequence:
    scene_name: str
    walk_id: int
    paths: Tuple[str, ...]
    latent_mu: torch.Tensor
    latent_logvar: torch.Tensor
    locs: torch.Tensor
    view_dirs: torch.Tensor
    translations: torch.Tensor
    valid_steps: torch.Tensor

    @property
    def latents(self) -> torch.Tensor:
        return self.latent_mu


class ProbWalkSequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[ProbWalkSequence]):
        if not sequences:
            raise ValueError("ProbWalkSequenceDataset requires at least one walk sequence.")
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sequence = self.sequences[index]
        return {
            "scene_name": sequence.scene_name,
            "walk_id": sequence.walk_id,
            "paths": sequence.paths,
            "latent_mu": sequence.latent_mu,
            "latent_logvar": sequence.latent_logvar,
            "locs": sequence.locs,
            "view_dirs": sequence.view_dirs,
            "translations": sequence.translations,
            "valid_steps": sequence.valid_steps,
        }


def collate_prob_walk_sequences(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not batch:
        raise ValueError("collate_prob_walk_sequences received an empty batch.")

    batch_size = len(batch)
    max_steps = max(item["latent_mu"].size(0) for item in batch)
    latent_dim = batch[0]["latent_mu"].size(-1)

    latent_mu = torch.zeros(batch_size, max_steps, latent_dim, dtype=torch.float32)
    latent_logvar = torch.zeros(batch_size, max_steps, latent_dim, dtype=torch.float32)
    locs = torch.zeros(batch_size, max_steps, 3, dtype=torch.float32)
    view_dirs = torch.zeros(batch_size, max_steps, 3, dtype=torch.float32)
    translations = torch.zeros(batch_size, max_steps, 3, dtype=torch.float32)
    valid_steps = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    padding_mask = torch.ones(batch_size, max_steps, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    scene_names: List[str] = []
    walk_ids: List[int] = []
    paths: List[Tuple[str, ...]] = []

    for idx, item in enumerate(batch):
        length = item["latent_mu"].size(0)
        latent_mu[idx, :length] = item["latent_mu"]
        latent_logvar[idx, :length] = item["latent_logvar"]
        locs[idx, :length] = item["locs"]
        view_dirs[idx, :length] = item["view_dirs"]
        translations[idx, :length] = item["translations"]
        valid_steps[idx, :length] = item["valid_steps"]
        padding_mask[idx, :length] = False
        lengths[idx] = length

        scene_names.append(item["scene_name"])
        walk_ids.append(item["walk_id"])
        paths.append(item["paths"])

    return {
        "scene_name": scene_names,
        "walk_id": walk_ids,
        "paths": paths,
        "latent_mu": latent_mu,
        "latent_logvar": latent_logvar,
        "locs": locs,
        "view_dirs": view_dirs,
        "translations": translations,
        "valid_steps": valid_steps,
        "padding_mask": padding_mask,
        "lengths": lengths,
    }


def encode_prob_observations(
    model: base.ImageGraphVAE,
    loader,
    latent_branch: str,
    scene_name: str,
) -> List[ProbEncodedObservation]:
    if latent_branch not in {"img", "pcd"}:
        raise ValueError(f"Unsupported latent branch '{latent_branch}'.")

    observations: List[ProbEncodedObservation] = []
    model.eval()
    mu_key = f"mu_{latent_branch}"
    logvar_key = f"logvar_{latent_branch}"

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            img = batch.img.permute(0, 3, 1, 2).contiguous().float()
            ew_points = base.normalize_edge_weights(batch.edge_weights.to(torch.float32))
            outputs = model(
                img,
                batch.pcd.float(),
                batch.batch,
                batch.edge_index,
                ew_points,
                batch.ei_camera,
                batch.ea_camera.float(),
            )

            latent_mu = outputs[mu_key].detach().cpu()
            latent_logvar = outputs[logvar_key].detach().cpu()
            locs = batch.loc.detach().cpu().float()
            view_dirs = F.normalize(batch.view_dir.detach().cpu().float(), dim=-1, eps=1e-8)

            for idx, path in enumerate(batch.path):
                walk_id, step_id = base.parse_random_walk_name(path)
                observations.append(
                    ProbEncodedObservation(
                        path=path,
                        scene_name=scene_name,
                        walk_id=walk_id,
                        step_id=step_id,
                        latent_mu=latent_mu[idx],
                        latent_logvar=latent_logvar[idx],
                        loc=locs[idx],
                        view_dir=view_dirs[idx],
                    )
                )

    observations.sort(key=lambda obs: (obs.scene_name, obs.walk_id, obs.step_id))
    return observations


def encode_all_prob_observations(
    model: base.ImageGraphVAE,
    root_dirs: Sequence[str | Path],
    latent_branch: str,
) -> List[ProbEncodedObservation]:
    observations: List[ProbEncodedObservation] = []
    for root_dir in base.resolve_random_walk_roots(root_dirs):
        scene_name = base.scene_name_from_root(root_dir)
        loader = base.make_encode_loader(root_dir)
        scene_observations = encode_prob_observations(
            model,
            loader,
            latent_branch=latent_branch,
            scene_name=scene_name,
        )
        observations.extend(scene_observations)
        print(f"encoded {len(scene_observations)} probabilistic observations from scene={scene_name}")

    observations.sort(key=lambda obs: (obs.scene_name, obs.walk_id, obs.step_id))
    return observations


def build_prob_walk_sequences(
    observations: Sequence[ProbEncodedObservation],
) -> List[ProbWalkSequence]:
    grouped: Dict[Tuple[str, int], List[ProbEncodedObservation]] = {}
    for obs in observations:
        grouped.setdefault((obs.scene_name, obs.walk_id), []).append(obs)
    for walk_observations in grouped.values():
        walk_observations.sort(key=lambda obs: obs.step_id)

    sequences: List[ProbWalkSequence] = []
    for (scene_name, walk_id), walk_observations in grouped.items():
        if len(walk_observations) < 2:
            continue

        num_steps = len(walk_observations)
        latent_mu = torch.stack([obs.latent_mu for obs in walk_observations], dim=0)
        latent_logvar = torch.stack([obs.latent_logvar for obs in walk_observations], dim=0)
        locs = torch.stack([obs.loc for obs in walk_observations], dim=0)
        view_dirs = torch.stack([obs.view_dir for obs in walk_observations], dim=0)
        translations = torch.zeros(num_steps, 3, dtype=torch.float32)
        valid_steps = torch.zeros(num_steps, dtype=torch.bool)

        for step_idx in range(1, num_steps):
            source = walk_observations[step_idx - 1]
            target = walk_observations[step_idx]
            translations[step_idx] = base.batch_equivariant_translation(
                source.loc.unsqueeze(0),
                target.loc.unsqueeze(0),
                source.view_dir.unsqueeze(0),
            ).squeeze(0).cpu()
            valid_steps[step_idx] = True

        sequences.append(
            ProbWalkSequence(
                scene_name=scene_name,
                walk_id=walk_id,
                paths=tuple(obs.path for obs in walk_observations),
                latent_mu=latent_mu,
                latent_logvar=latent_logvar,
                locs=locs,
                view_dirs=view_dirs,
                translations=translations,
                valid_steps=valid_steps,
            )
        )

    sequences.sort(key=lambda sequence: (sequence.scene_name, sequence.walk_id))
    return sequences


class LatentNormalizer(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.register_buffer("global_mean", torch.zeros(D))
        self.register_buffer("global_std", torch.ones(D))
        self.alpha = 3.0
        self.beta = 0.99
        self.ln = nn.LayerNorm(4 * D)

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        squeeze = mu.dim() == 2
        if squeeze:
            mu = mu.unsqueeze(0)
            logvar = logvar.unsqueeze(0)

        logvar = logvar.float().clamp(min=-10.0, max=10.0)
        mu = (mu.float() - self.global_mean) / (self.global_std + 1e-5)
        conf = torch.exp(-0.5 * logvar).clamp(max=10.0)
        mu = mu * conf

        ema = []
        m = mu[:, 0]
        for t in range(mu.shape[1]):
            m = self.beta * m + (1 - self.beta) * mu[:, t]
            ema.append(mu[:, t] - m)
        mu_hp = torch.stack(ema, dim=1)

        delta = mu_hp[:, 1:] - mu_hp[:, :-1]
        delta = torch.tanh(self.alpha * delta)
        accel = delta[:, 1:] - delta[:, :-1]
        delta = F.pad(delta, (0, 0, 1, 0))
        accel = F.pad(accel, (0, 0, 2, 0))

        x = torch.cat([mu_hp, delta, accel, logvar], dim=-1)
        x = self.ln(x)
        return x.squeeze(0) if squeeze else x


class TransformGRU(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        out_dim: int = OUT_DIM,
        layers: int = GRU_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.normalizer = LatentNormalizer(latent_dim)
        token_dim = 16 * latent_dim

        self.token_lift = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.pose_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

    def motion_tokens(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(mu, logvar)
        prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        return torch.cat([prev, x, x - prev, x * prev], dim=-1)

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        start_locs: torch.Tensor | None = None,
        start_dirs: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        squeeze = mu.dim() == 2
        if squeeze:
            mu = mu.unsqueeze(0)
            logvar = logvar.unsqueeze(0)
        if padding_mask is None:
            padding_mask = torch.zeros(mu.shape[:2], device=mu.device, dtype=torch.bool)

        tokens = self.motion_tokens(mu, logvar)
        x = self.token_lift(tokens)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden, _ = self.rnn(x)
        pose_cov = self.pose_head(hidden)
        pose, pose_logvar = pose_cov.split(self.out_dim, dim=-1)
        pose_logvar = pose_logvar.clamp(min=POSE_LOGVAR_MIN, max=POSE_LOGVAR_MAX)
        pose = pose.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        pose_logvar = pose_logvar.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        pose[:, 0] = 0.0
        pose_logvar[:, 0] = 0.0

        out = {"pose": pose, "logvar": pose_logvar, "covariance": torch.exp(pose_logvar)}
        if start_locs is not None and start_dirs is not None:
            if squeeze:
                start_locs = start_locs.unsqueeze(0)
                start_dirs = start_dirs.unsqueeze(0)
            locs, dirs = rollout_transforms(pose, start_locs, start_dirs)
            out["locs"] = locs
            out["view_dirs"] = dirs

        if squeeze:
            out = {key: value.squeeze(0) for key, value in out.items()}
        return out


def fit_latent_normalizer(model: TransformGRU, sequences: Sequence[ProbWalkSequence]) -> None:
    latent_mu = torch.cat([sequence.latent_mu.float() for sequence in sequences], dim=0)
    model.normalizer.global_mean.copy_(latent_mu.mean(dim=0).to(model.normalizer.global_mean.device))
    model.normalizer.global_std.copy_(
        latent_mu.std(dim=0, unbiased=False).clamp_min(1e-4).to(model.normalizer.global_std.device)
    )


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


def split_sequences(sequences: Sequence[ProbWalkSequence]) -> Tuple[list, list]:
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


def write_split(
    path: Path,
    train_sequences: Sequence[ProbWalkSequence],
    val_sequences: Sequence[ProbWalkSequence],
) -> None:
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
                        "steps": int(sequence.latent_mu.size(0)),
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


def covariance_calibration_loss(error: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    target_logvar = torch.log(error.detach().pow(2) + COVARIANCE_TARGET_EPS)
    target_logvar = target_logvar.clamp(min=POSE_LOGVAR_MIN, max=POSE_LOGVAR_MAX)
    return F.smooth_l1_loss(logvar, target_logvar)


def loss_terms(
    pred: Dict[str, torch.Tensor],
    target: torch.Tensor,
    locs: torch.Tensor,
    view_dirs: torch.Tensor,
    valid_steps: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pose = pred["pose"]
    pose_logvar = pred["logvar"]
    valid = (
        valid_steps
        & torch.isfinite(pose).all(dim=-1)
        & torch.isfinite(pose_logvar).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(locs).all(dim=-1)
        & torch.isfinite(view_dirs).all(dim=-1)
    )
    if not valid.any():
        zero = pose.sum() * 0.0
        return {
            "loss": zero,
            "translation": zero,
            "angle": zero,
            "covariance": zero,
            "trajectory": zero,
            "direction": zero,
            "trajectory_steps": zero,
        }

    translation_error = pose[..., :3][valid] - target[..., :3][valid]
    angle_error = wrap_angle(pose[..., 3:][valid] - target[..., 3:][valid])
    translation_loss = F.smooth_l1_loss(translation_error, torch.zeros_like(translation_error))
    angle_loss = F.smooth_l1_loss(angle_error, torch.zeros_like(angle_error))
    covariance_loss = covariance_calibration_loss(
        torch.cat([translation_error, angle_error], dim=-1),
        pose_logvar[valid],
    )

    pred_locs, pred_dirs = rollout_transforms(pose, locs[:, 0], view_dirs[:, 0])
    trajectory_loss = F.smooth_l1_loss(pred_locs[valid], locs[valid])
    direction_loss = unit_direction_loss(pred_dirs[valid], view_dirs[valid])
    loss = (
        TRANSLATION_WEIGHT * translation_loss
        + ANGLE_WEIGHT * angle_loss
        + COVARIANCE_WEIGHT * covariance_loss
        + TRAJECTORY_WEIGHT * trajectory_loss
        + DIRECTION_WEIGHT * direction_loss
    )
    return {
        "loss": loss,
        "translation": translation_loss,
        "angle": angle_loss,
        "covariance": covariance_loss,
        "trajectory": trajectory_loss,
        "direction": direction_loss,
        "trajectory_steps": valid.sum().to(dtype=pose.dtype),
    }


def run_epoch(
    model: TransformGRU,
    loader: DataLoader | None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    if loader is None:
        return {
            "loss": float("nan"),
            "translation": float("nan"),
            "angle": float("nan"),
            "covariance": float("nan"),
            "trajectory": float("nan"),
            "direction": float("nan"),
            "trajectory_steps": 0,
            "sequences": 0,
        }

    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "translation": 0.0, "angle": 0.0, "covariance": 0.0, "trajectory": 0.0, "direction": 0.0}
    count = 0
    sequence_count = 0
    trajectory_steps = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            latent_mu = batch["latent_mu"].to(device)
            latent_logvar = batch["latent_logvar"].to(device)
            locs = batch["locs"].to(device)
            view_dirs = batch["view_dirs"].to(device)
            valid_steps = batch["valid_steps"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred = model(latent_mu, latent_logvar, padding_mask=padding_mask)
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
            sequence_count += int(latent_mu.size(0))
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
    observations = encode_all_prob_observations(
        encoder,
        RANDOM_WALK_ROOTS,
        latent_branch=LATENT_BRANCH,
    )
    sequences = build_prob_walk_sequences(observations)
    train_sequences, val_sequences = split_sequences(sequences)

    if not train_sequences:
        raise ValueError("No train sequences were built from the encoded walk sequences.")

    train_loader = DataLoader(
        ProbWalkSequenceDataset(train_sequences),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_prob_walk_sequences,
    )
    val_loader = (
        DataLoader(
            ProbWalkSequenceDataset(val_sequences),
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_prob_walk_sequences,
        )
        if val_sequences
        else None
    )

    model = TransformGRU().to(device)
    fit_latent_normalizer(model, train_sequences)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    write_split(output_dir / "split.csv", train_sequences, val_sequences)
    print(
        f"training {ALIAS}: {len(train_sequences)} train, {len(val_sequences)} val sequences, "
        f"gru_layers={GRU_LAYERS}, branch={LATENT_BRANCH}"
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
            "gru_layers": GRU_LAYERS,
            "translation_weight": TRANSLATION_WEIGHT,
            "angle_weight": ANGLE_WEIGHT,
            "covariance_weight": COVARIANCE_WEIGHT,
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
