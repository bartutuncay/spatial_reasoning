from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

img_enc_mod = load_module("nips_img_enc", ROOT / "gnn_spatial_reasoning_nips/models/1_img_enc.py")
img_dec_mod = load_module("nips_img_dec", ROOT / "gnn_spatial_reasoning_nips/models/1_img_dec.py")
pcd_enc_mod = load_module("nips_pcd_enc", ROOT / "gnn_spatial_reasoning_nips/models/1_pcd_enc.py")
pcd_dec_mod = load_module("nips_pcd_dec", ROOT / "gnn_spatial_reasoning_nips/models/1_pcd_dec.py")

Bottleneck = img_enc_mod.Bottleneck
ImgEnc = img_enc_mod.ImgEnc
ImgDec = img_dec_mod.ImgDec
PCDEnc = pcd_enc_mod.PCDEnc
PCDDecoder = pcd_dec_mod.PCDDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

LATENT_DIM = 128
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 512
HEAD_HIDDEN_DIM = 128
DROPOUT = 0.1
EPOCHS = 80000
ENCODE_BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 1.0
TRAJECTORY_LOSS_WEIGHT = 1.0

LATENT_BRANCH = "img"

alias = "0505"

rw_list = [
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_translation",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/break_room/random_walks",
    #"../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/hospital/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/relief/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/terrains/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/rw_translation",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/random_walks",
]

VAE_WEIGHTS = Path(
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/1_model/0420_nips/model_weights_7800.pt"
)
MODEL_ROOT = Path("../../../scratch/btuncay/cog/gnn_spatial_reasoning/2_model_movement")

RW_NAME_RE = re.compile(r"rw_(\d+)_(\d+)\.pt$")
WORLD_UP = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
DEFAULT_FORWARD = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module.") :]] = value
        else:
            stripped[key] = value
    return stripped


def parse_random_walk_name(path: str | Path) -> Tuple[int, int]:
    name = Path(path).name
    match = RW_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Expected random-walk file names like 'rw_<walk_id>_<step_id>.pt', got '{name}'."
        )
    return int(match.group(1)), int(match.group(2))


def batch_equivariant_translation(
    source_loc: torch.Tensor,
    target_loc: torch.Tensor,
    source_view_dir: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    source_loc = source_loc.float()
    target_loc = target_loc.float()
    dp = target_loc - source_loc
    rot_camera_to_world = camera_to_world_rotation(source_view_dir, eps=eps)
    translation_camera = torch.bmm(
        rot_camera_to_world.transpose(1, 2),
        dp.unsqueeze(-1),
    ).squeeze(-1)
    return translation_camera


def camera_to_world_rotation(
    view_dir: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    view_dir = view_dir.float()
    view_norm = view_dir.norm(dim=-1, keepdim=True)
    default_forward = DEFAULT_FORWARD.to(view_dir.device).expand_as(view_dir)
    valid_view = torch.isfinite(view_dir).all(dim=-1, keepdim=True) & (view_norm > eps)
    z_axis = torch.where(valid_view, view_dir / view_norm.clamp_min(eps), default_forward)

    up = WORLD_UP.to(view_dir.device).expand_as(z_axis)
    x_axis = torch.cross(up, z_axis, dim=-1)
    x_norm = x_axis.norm(dim=-1, keepdim=True)
    bad = x_norm.squeeze(-1) < eps

    if bad.any():
        candidates = torch.eye(3, device=view_dir.device, dtype=view_dir.dtype)
        alignment = torch.abs(z_axis @ candidates.T)
        fallback_up = candidates[torch.argmin(alignment, dim=-1)]
        fallback_x = torch.cross(fallback_up, z_axis, dim=-1)
        fallback_norm = fallback_x.norm(dim=-1, keepdim=True)
        x_axis = torch.where(bad.unsqueeze(-1), fallback_x, x_axis)
        x_norm = torch.where(bad.unsqueeze(-1), fallback_norm, x_norm)

    x_axis = x_axis / x_norm.clamp_min(eps)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


class ImageGraphVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.img_enc = ImgEnc(Bottleneck, [3, 4, 6, 3], latent_dim)
        self.img_dec = ImgDec(latent_dim=latent_dim, out_channels=3, base_channels=32, out_hw=(192, 256), expansion=Bottleneck.expansion)
        self.pcd_enc = PCDEnc(latent_dim=latent_dim, layers_points=2, layers_camera=4, layers_mlp=2, z_dim=latent_dim)
        self.pcd_dec = PCDDecoder(nodes_dim=7, layers_mlp=2, latent_dim=latent_dim, nodes_k=3, layers_points=2, layers_camera=4)

    def forward(self, img, pcd, batch, ei_points, ew_points, ei_camera, ea_camera):
        z_img, mu_img, logvar_img = self.img_enc(img)
        z_pcd, mu_pcd, logvar_pcd = self.pcd_enc(pcd, batch, ei_points, ew_points, ei_camera, ea_camera)

        img_img = self.img_dec(z_img)
        pcd_img = self.img_dec(z_pcd)
        pcd_pred, pcd_batch, pcd_ei, pcd_ew, pcd_ei_c, pcd_ea_c = self.pcd_dec(z_img, img)
        _, mu_pcd_p, _ = self.pcd_enc(pcd_pred, pcd_batch, pcd_ei, pcd_ew, pcd_ei_c, pcd_ea_c)

        return {
            "img_img": img_img,
            "pcd_img": pcd_img,
            "pcd_pcd_pred": pcd_pred,
            "pcd_pcd_ei": pcd_ei,
            "mu_img": mu_img,
            "logvar_img": logvar_img,
            "mu_pcd": mu_pcd,
            "logvar_pcd": logvar_pcd,
            "mu_pcd_p": mu_pcd_p,
        }


@dataclass(frozen=True)
class EncodedObservation:
    path: str
    scene_name: str
    walk_id: int
    step_id: int
    latent: torch.Tensor
    loc: torch.Tensor
    view_dir: torch.Tensor


@dataclass(frozen=True)
class WalkSequence:
    scene_name: str
    walk_id: int
    paths: Tuple[str, ...]
    latents: torch.Tensor
    locs: torch.Tensor
    view_dirs: torch.Tensor
    translations: torch.Tensor
    valid_steps: torch.Tensor


class WalkSequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[WalkSequence]):
        if not sequences:
            raise ValueError("WalkSequenceDataset requires at least one walk sequence.")
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sequence = self.sequences[index]
        return {
            "scene_name": sequence.scene_name,
            "walk_id": sequence.walk_id,
            "paths": sequence.paths,
            "latents": sequence.latents,
            "locs": sequence.locs,
            "view_dirs": sequence.view_dirs,
            "translations": sequence.translations,
            "valid_steps": sequence.valid_steps,
        }


def collate_walk_sequences(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not batch:
        raise ValueError("collate_walk_sequences received an empty batch.")

    batch_size = len(batch)
    max_steps = max(item["latents"].size(0) for item in batch)
    latent_dim = batch[0]["latents"].size(-1)

    latents = torch.zeros(batch_size, max_steps, latent_dim, dtype=torch.float32)
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
        walk_latents = item["latents"]
        walk_locs = item["locs"]
        walk_view_dirs = item["view_dirs"]
        walk_translations = item["translations"]
        walk_valid_steps = item["valid_steps"]
        length = walk_latents.size(0)

        latents[idx, :length] = walk_latents
        locs[idx, :length] = walk_locs
        view_dirs[idx, :length] = walk_view_dirs
        translations[idx, :length] = walk_translations
        valid_steps[idx, :length] = walk_valid_steps
        padding_mask[idx, :length] = False
        lengths[idx] = length

        scene_names.append(item["scene_name"])
        walk_ids.append(item["walk_id"])
        paths.append(item["paths"])

    return {
        "scene_name": scene_names,
        "walk_id": walk_ids,
        "paths": paths,
        "latents": latents,
        "locs": locs,
        "view_dirs": view_dirs,
        "translations": translations,
        "valid_steps": valid_steps,
        "padding_mask": padding_mask,
        "lengths": lengths,
    }


class CausalWalkTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        head_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(latent_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 3),
        )

    def forward(
        self,
        latents: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_steps = latents.size(1)
        x = self.input_proj(latents) + sinusoidal_position_encoding(
            length=num_steps,
            dim=self.model_dim,
            device=latents.device,
            dtype=latents.dtype,
        )
        causal_mask = causal_attention_mask(num_steps, latents.device)
        hidden = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.output_head(hidden)


def sinusoidal_position_encoding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding.unsqueeze(0)


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


def camera_delta_to_world(
    delta_camera: torch.Tensor,
    source_view_dir: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    rot_camera_to_world = camera_to_world_rotation(source_view_dir, eps=eps)
    return torch.matmul(rot_camera_to_world, delta_camera.unsqueeze(-1)).squeeze(-1)


def integrate_camera_deltas(
    start_locs: torch.Tensor,
    deltas_camera: torch.Tensor,
    source_view_dirs: torch.Tensor,
) -> torch.Tensor:
    if deltas_camera.size(1) == 0:
        return start_locs.unsqueeze(1)

    world_steps = camera_delta_to_world(deltas_camera, source_view_dirs)
    relative_locs = torch.cumsum(world_steps, dim=1)
    pred_locs = torch.cat(
        [start_locs.unsqueeze(1), start_locs.unsqueeze(1) + relative_locs],
        dim=1,
    )
    return pred_locs


def group_observations_by_walk(
    observations: Sequence[EncodedObservation],
) -> Dict[Tuple[str, int], List[EncodedObservation]]:
    grouped: Dict[Tuple[str, int], List[EncodedObservation]] = {}
    for obs in observations:
        grouped.setdefault((obs.scene_name, obs.walk_id), []).append(obs)
    for walk_observations in grouped.values():
        walk_observations.sort(key=lambda obs: obs.step_id)
    return grouped


def build_walk_sequences(observations: Sequence[EncodedObservation]) -> List[WalkSequence]:
    grouped = group_observations_by_walk(observations)
    sequences: List[WalkSequence] = []

    for (scene_name, walk_id), walk_observations in grouped.items():
        if len(walk_observations) < 2:
            continue

        num_steps = len(walk_observations)
        latents = torch.stack([obs.latent for obs in walk_observations], dim=0)
        locs = torch.stack([obs.loc for obs in walk_observations], dim=0)
        view_dirs = torch.stack([obs.view_dir for obs in walk_observations], dim=0)
        translations = torch.zeros(num_steps, 3, dtype=torch.float32)
        valid_steps = torch.zeros(num_steps, dtype=torch.bool)

        for step_idx in range(1, num_steps):
            source = walk_observations[step_idx - 1]
            target = walk_observations[step_idx]
            translations[step_idx] = batch_equivariant_translation(
                source.loc.unsqueeze(0),
                target.loc.unsqueeze(0),
                source.view_dir.unsqueeze(0),
            ).squeeze(0).cpu()
            valid_steps[step_idx] = True

        sequences.append(
            WalkSequence(
                scene_name=scene_name,
                walk_id=walk_id,
                paths=tuple(obs.path for obs in walk_observations),
                latents=latents,
                locs=locs,
                view_dirs=view_dirs,
                translations=translations,
                valid_steps=valid_steps,
            )
        )

    sequences.sort(key=lambda sequence: (sequence.scene_name, sequence.walk_id))
    return sequences

def normalize_edge_weights(edge_weights: torch.Tensor) -> torch.Tensor:
    scale = edge_weights.mean().clamp_min(1e-6)
    return torch.exp(-(edge_weights ** 2) / (2 * scale ** 2))

def encode_observations(
    model: ImageGraphVAE,
    loader,
    latent_branch: str,
    scene_name: str,
) -> List[EncodedObservation]:
    observations: List[EncodedObservation] = []
    model.eval()

    if latent_branch not in {"img", "pcd"}:
        raise ValueError(f"Unsupported latent branch '{latent_branch}'.")

    mu_key = f"mu_{latent_branch}"

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            img = batch.img.permute(0, 3, 1, 2).contiguous().float()
            ew_points = normalize_edge_weights(batch.edge_weights.to(torch.float32))

            outputs = model(img,batch.pcd.float(),batch.batch,batch.edge_index,ew_points,
                batch.ei_camera,batch.ea_camera.float(),
            )

            latents = outputs[mu_key].detach().cpu()
            locs = batch.loc.detach().cpu().float()
            view_dirs = F.normalize(batch.view_dir.detach().cpu().float(), dim=-1, eps=1e-8)

            for idx, path in enumerate(batch.path):
                walk_id, step_id = parse_random_walk_name(path)
                observations.append(
                    EncodedObservation(
                        path=path,
                        scene_name=scene_name,
                        walk_id=walk_id,
                        step_id=step_id,
                        latent=latents[idx],
                        loc=locs[idx],
                        view_dir=view_dirs[idx],
                    )
                )

    observations.sort(key=lambda obs: (obs.scene_name, obs.walk_id, obs.step_id))
    return observations

def encode_observations_var(
    model: ImageGraphVAE,
    loader,
    latent_branch: str,
    scene_name: str,
) -> List[EncodedObservation]:
    observations: List[EncodedObservation] = []
    model.eval()

    if latent_branch not in {"img", "pcd"}:
        raise ValueError(f"Unsupported latent branch '{latent_branch}'.")

    mu_key = f"mu_{latent_branch}"
    var_key = f"logvar_{latent_branch}"

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            img = batch.img.permute(0, 3, 1, 2).contiguous().float()
            ew_points = normalize_edge_weights(batch.edge_weights.to(torch.float32))

            outputs = model(img,batch.pcd.float(),batch.batch,batch.edge_index,ew_points,
                batch.ei_camera,batch.ea_camera.float(),
            )

            latents_mu = outputs[mu_key].detach().cpu()
            latents_var = outputs[var_key].detach().cpu()
            locs = batch.loc.detach().cpu().float()
            view_dirs = F.normalize(batch.view_dir.detach().cpu().float(), dim=-1, eps=1e-8)

            for idx, path in enumerate(batch.path):
                walk_id, step_id = parse_random_walk_name(path)
                observations.append(
                    EncodedObservation(
                        path=path,
                        scene_name=scene_name,
                        walk_id=walk_id,
                        step_id=step_id,
                        latent_mu=latents_mu[idx],
                        latent_var=latents_var[idx],
                        loc=locs[idx],
                        view_dir=view_dirs[idx],
                    )
                )

    observations.sort(key=lambda obs: (obs.scene_name, obs.walk_id, obs.step_id))
    return observations


def load_encoder(weights_path: Path, latent_dim: int) -> ImageGraphVAE:
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing ImageGraphVAE checkpoint: {weights_path}")

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = strip_module_prefix(state)

    model = ImageGraphVAE(latent_dim=latent_dim).to(device)
    incompatible = model.load_state_dict(state, strict=False)

    loaded_encoder_keys = [key for key in state.keys() if key.startswith(("img_enc.", "pcd_enc."))]
    if not loaded_encoder_keys:
        raise ValueError(
            f"Checkpoint {weights_path} did not contain ImageGraphVAE encoder weights."
        )

    if incompatible.missing_keys:
        print(f"load_state_dict missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"load_state_dict unexpected keys: {len(incompatible.unexpected_keys)}")

    model.eval()
    return model


def make_encode_loader(root_dir: Path):
    return make_loader(
        root_dir=root_dir,
        batch_size=ENCODE_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def resolve_random_walk_roots(root_dirs: Sequence[str | Path]) -> List[Path]:
    roots = [Path(root_dir) for root_dir in root_dirs]
    missing = [root for root in roots if not root.is_dir()]
    if missing:
        missing_str = ", ".join(str(root) for root in missing)
        raise FileNotFoundError(f"Missing random-walk directories: {missing_str}")
    return roots


def scene_name_from_root(root_dir: Path) -> str:
    if root_dir.name == "random_walks" and root_dir.parent.name:
        return root_dir.parent.name
    return root_dir.name


def encode_all_observations(
    model: ImageGraphVAE,
    root_dirs: Sequence[str | Path],
    latent_branch: str,
) -> List[EncodedObservation]:
    observations: List[EncodedObservation] = []

    for root_dir in resolve_random_walk_roots(root_dirs):
        scene_name = scene_name_from_root(root_dir)
        loader = make_encode_loader(root_dir)
        scene_observations = encode_observations(
            model,
            loader,
            latent_branch=latent_branch,
            scene_name=scene_name,
        )
        observations.extend(scene_observations)
        print(f"encoded {len(scene_observations)} observations from scene={scene_name}")

    observations.sort(key=lambda obs: (obs.scene_name, obs.walk_id, obs.step_id))
    return observations

def encode_all_observations_var(
    model: ImageGraphVAE,
    root_dirs: Sequence[str | Path],
    latent_branch: str,
) -> List[EncodedObservation]:
    observations: List[EncodedObservation] = []

    for root_dir in resolve_random_walk_roots(root_dirs):
        scene_name = scene_name_from_root(root_dir)
        loader = make_encode_loader(root_dir)
        scene_observations = encode_observations_var(
            model,
            loader,
            latent_branch=latent_branch,
            scene_name=scene_name,
        )
        observations.extend(scene_observations)
        print(f"encoded {len(scene_observations)} observations from scene={scene_name}")

    observations.sort(key=lambda obs: (obs.scene_name, obs.walk_id, obs.step_id))
    return observations


def append_loss_row(csv_path: Path, row: Dict[str, float], write_header: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(
    model: CausalWalkTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    model.train(True)
    loss_sums = {
        "train_loss": 0.0,
        "train_delta_loss": 0.0,
        "train_trajectory_loss": 0.0,
    }
    num_batches = 0

    for batch in loader:
        latents = batch["latents"].to(device)
        locs = batch["locs"].to(device)
        view_dirs = batch["view_dirs"].to(device)
        translations = batch["translations"].to(device)
        valid_steps = batch["valid_steps"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_translations = model(latents, padding_mask=padding_mask)
        delta_loss = F.smooth_l1_loss(pred_translations[valid_steps], translations[valid_steps])

        pred_locs = integrate_camera_deltas(
            locs[:, 0],
            pred_translations[:, 1:],
            view_dirs[:, :-1],
        )
        trajectory_loss = F.smooth_l1_loss(pred_locs[valid_steps], locs[valid_steps])
        loss = delta_loss + TRAJECTORY_LOSS_WEIGHT * trajectory_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        loss_sums["train_loss"] += float(loss.detach())
        loss_sums["train_delta_loss"] += float(delta_loss.detach())
        loss_sums["train_trajectory_loss"] += float(trajectory_loss.detach())
        num_batches += 1

    if num_batches == 0:
        return {key: 0.0 for key in loss_sums}
    return {key: value / num_batches for key, value in loss_sums.items()}


@torch.no_grad()
def predict_walk_trajectory(
    model: CausalWalkTransformer,
    latents: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    padding_mask = torch.zeros(latents.size(0), latents.size(1), dtype=torch.bool, device=latents.device)
    pred = model(latents, padding_mask=padding_mask)
    return pred.squeeze(0)


def train() -> None:
    output_dir = MODEL_ROOT / alias
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "losses.csv"
    write_header = not csv_path.exists()

    encoder = load_encoder(VAE_WEIGHTS, latent_dim=LATENT_DIM)
    observations = encode_all_observations(
        encoder,
        rw_list,
        latent_branch=LATENT_BRANCH,
    )
    sequences = build_walk_sequences(observations)
    dataset = WalkSequenceDataset(sequences)
    train_loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_walk_sequences,
    )

    model = CausalWalkTransformer(
        latent_dim=LATENT_DIM,
        model_dim=TRANSFORMER_DIM,
        num_heads=TRANSFORMER_HEADS,
        num_layers=TRANSFORMER_LAYERS,
        ff_dim=TRANSFORMER_FF_DIM,
        head_hidden_dim=HEAD_HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    num_supervised_steps = int(sum(sequence.valid_steps.sum().item() for sequence in sequences))

    print("starting training")
    print(alias)
    print(
        f"encoded {len(observations)} observations into {len(sequences)} walk sequences "
        f"with {num_supervised_steps} supervised steps using branch={LATENT_BRANCH}"
    )

    for epoch in range(EPOCHS):
        loss_dict = run_epoch(model, train_loader, optimizer)
        scheduler.step()

        row = {
            "epoch": epoch,
            **loss_dict,
            "trajectory_loss_weight": TRAJECTORY_LOSS_WEIGHT,
            "num_observations": float(len(observations)),
            "num_sequences": float(len(sequences)),
            "num_supervised_steps": float(num_supervised_steps),
        }
        append_loss_row(csv_path, row, write_header=write_header)
        write_header = False

        if epoch % 10 == 0:
            print(f"epoch {epoch}: {row}")
        if epoch % 1000 == 0:
            torch.save(model.state_dict(), output_dir / f"model_weights_{epoch}.pt")

    torch.save(model.state_dict(), output_dir / "model_weights.pt")
    print("completed")


if __name__ == "__main__":
    train()
