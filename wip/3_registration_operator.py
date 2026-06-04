from __future__ import annotations

import math
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GENConv, global_mean_pool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
from gnn_spatial_reasoning.preprocessing_src.dataloader_autoencoder import make_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

EPOCHS = 20000
BATCH_SIZE = 2
ENCODE_BATCH_SIZE = 8
LATENT_DIM = 128
HIDDEN_DIM = 128
DESCRIPTOR_DIM = 128
GRAPH_LAYERS = 2
PAIR_MIN_DISTANCE = 0.75
PAIR_MAX_DISTANCE = 4.0
PAIR_TOP_K = 4
MAX_MATCH_POINTS = 768
MAX_LOSS_POINTS = 2048
ALIGN_KEEP_RATIO = 0.90
CLIP_GRAD_NORM = 1.0
PRIOR_TRANSLATION_WEIGHT = 0.05
PRIOR_DROPOUT = 0.20
PRIOR_NOISE_STD = 0.15
ALIAS = "0426"
LATENT_BRANCH = "img"
VAE_WEIGHTS = Path("../../../scratch/btuncay/cog/gnn_spatial_reasoning/1_model/0420_nips/model_weights_7800.pt")
MODEL_ROOT = Path("../../../scratch/btuncay/cog/gnn_spatial_reasoning/3_model_recon")
LOCAL_FORWARD = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
RW_NAME_RE = re.compile(r"rw_(\d+)_(\d+)\.pt$")

rw_list = [
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/break_room/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/hospital/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/relief/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/terrains/random_walks",
    "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/office/random_walks",
]

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


def _ensure_graph_tensors(graph: Data) -> Data:
    edge_index = getattr(graph, "edge_index", None)
    clean_graph = Data(
        pos=graph.pos.float(),
        edge_index=edge_index.long() if edge_index is not None else torch.empty((2, 0), dtype=torch.long),
    )
    clean_graph.num_nodes = int(graph.pos.size(0))

    if hasattr(graph, "rgb") and graph.rgb is not None:
        clean_graph.rgb = graph.rgb.float()
    elif hasattr(graph, "x") and graph.x is not None:
        clean_graph.x = graph.x.float()

    return clean_graph


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
class RegistrationObservation:
    path: str
    name: str
    scene_name: str
    walk_id: int
    step_id: int
    graph: Data
    latent: torch.Tensor
    loc: torch.Tensor
    view_dir: torch.Tensor


def make_encode_loader(root_dir: Path) -> DataLoader:
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

def normalize_edge_weights(edge_weights: torch.Tensor) -> torch.Tensor:
    scale = edge_weights.mean().clamp_min(1e-6)
    return torch.exp(-(edge_weights ** 2) / (2 * scale ** 2))

def encode_observations(
    model: ImageGraphVAE,
    loader: DataLoader,
    latent_branch: str,
    scene_name: str,
) -> List[RegistrationObservation]:
    observations: List[RegistrationObservation] = []
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
                    RegistrationObservation(
                        path=path,
                        name=Path(path).name,
                        scene_name=scene_name,
                        walk_id=walk_id,
                        step_id=step_id,
                        graph=_ensure_graph_tensors(batch.graph[idx]),
                        latent=latents[idx],
                        loc=locs[idx],
                        view_dir=view_dirs[idx],
                    )
                )

    observations.sort(key=lambda obs: (obs.scene_name, obs.walk_id, obs.step_id))
    return observations


def encode_all_observations(
    model: ImageGraphVAE,
    root_dirs: Sequence[str | Path],
    latent_branch: str,
) -> List[RegistrationObservation]:
    observations: List[RegistrationObservation] = []

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


class LatentRegistrationDataset(Dataset):
    def __init__(
        self,
        observations: Sequence[RegistrationObservation],
        pair_min_distance: float = PAIR_MIN_DISTANCE,
        pair_max_distance: float = PAIR_MAX_DISTANCE,
        pair_top_k: int = PAIR_TOP_K,
    ):
        self.pair_min_distance = pair_min_distance
        self.pair_max_distance = pair_max_distance
        self.pair_top_k = pair_top_k

        self.entries = list(observations)
        if len(self.entries) < 2:
            raise ValueError("Need at least two encoded random-walk observations for registration training.")

        self.pair_candidates = self._build_pairs()

    def __len__(self) -> int:
        return len(self.entries)

    def _build_pairs(self) -> List[torch.Tensor]:
        pair_candidates: List[torch.Tensor | None] = [None] * len(self.entries)
        scene_to_indices: Dict[str, List[int]] = {}

        for idx, entry in enumerate(self.entries):
            scene_to_indices.setdefault(entry.scene_name, []).append(idx)

        for scene_indices in scene_to_indices.values():
            if len(scene_indices) == 1:
                only_idx = scene_indices[0]
                pair_candidates[only_idx] = torch.tensor([only_idx], dtype=torch.long)
                continue

            scene_locs = torch.stack([self.entries[idx].loc for idx in scene_indices], dim=0)
            scene_latents = torch.stack([self.entries[idx].latent for idx in scene_indices], dim=0)
            scene_latents = F.normalize(scene_latents, dim=-1)

            similarities = scene_latents @ scene_latents.T
            distances = torch.cdist(scene_locs, scene_locs)
            scene_index_tensor = torch.tensor(scene_indices, dtype=torch.long)

            for local_idx, global_idx in enumerate(scene_indices):
                mask = (
                    (distances[local_idx] >= self.pair_min_distance)
                    & (distances[local_idx] <= self.pair_max_distance)
                )
                mask[local_idx] = False

                if not mask.any():
                    mask = torch.ones_like(mask, dtype=torch.bool)
                    mask[local_idx] = False

                candidate_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if candidate_ids.numel() == 0:
                    pair_candidates[global_idx] = torch.tensor([global_idx], dtype=torch.long)
                    continue

                candidate_scores = similarities[local_idx, candidate_ids]
                ranked_ids = candidate_ids[torch.argsort(candidate_scores, descending=True)]
                pair_candidates[global_idx] = scene_index_tensor[ranked_ids[: self.pair_top_k]].clone()

        return [
            candidates if candidates is not None else torch.tensor([idx], dtype=torch.long)
            for idx, candidates in enumerate(pair_candidates)
        ]

    def _load_entry(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index]
        return {
            "graph": entry.graph,
            "latent": entry.latent,
            "loc": entry.loc,
            "view_dir": entry.view_dir,
            "name": entry.name,
        }

    def __getitem__(self, index: int) -> Dict[str, Dict[str, torch.Tensor]]:
        candidates = self.pair_candidates[index]
        if candidates.numel() == 1:
            pair_index = int(candidates.item())
        else:
            pair_choice = torch.randint(candidates.numel(), (1,)).item()
            pair_index = int(candidates[pair_choice].item())

        return {
            "source": self._load_entry(index),
            "target": self._load_entry(pair_index),
        }


def collate_registration_pairs(
    samples: Sequence[Dict[str, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
    source_graphs = Batch.from_data_list(
        [sample["source"]["graph"] for sample in samples]
    )
    target_graphs = Batch.from_data_list(
        [sample["target"]["graph"] for sample in samples]
    )

    return {
        "source_graphs": source_graphs,
        "target_graphs": target_graphs,
        "source_latent": torch.stack(
            [sample["source"]["latent"].float() for sample in samples],
            dim=0,
        ),
        "target_latent": torch.stack(
            [sample["target"]["latent"].float() for sample in samples],
            dim=0,
        ),
        "source_loc": torch.stack(
            [sample["source"]["loc"].float() for sample in samples],
            dim=0,
        ),
        "target_loc": torch.stack(
            [sample["target"]["loc"].float() for sample in samples],
            dim=0,
        ),
        "source_view_dir": torch.stack(
            [sample["source"]["view_dir"].float() for sample in samples],
            dim=0,
        ),
        "target_view_dir": torch.stack(
            [sample["target"]["view_dir"].float() for sample in samples],
            dim=0,
        ),
        "source_name": [sample["source"]["name"] for sample in samples],
        "target_name": [sample["target"]["name"] for sample in samples],
    }


def point_mask_from_pos(pos: torch.Tensor) -> torch.Tensor:
    if pos.size(-1) < 4:
        raise ValueError(
            "Registration expects camera-centered graph nodes stored as [distance, unit_vector]."
        )
    return (pos[:, 0].abs() > 1e-8) | (pos[:, 1:4].abs().sum(dim=-1) > 1e-8)


def graph_rgb_tensor(graph_or_batch: Data) -> torch.Tensor:
    if hasattr(graph_or_batch, "rgb") and graph_or_batch.rgb is not None:
        return graph_or_batch.rgb.float()

    if hasattr(graph_or_batch, "x") and graph_or_batch.x is not None:
        if graph_or_batch.x.size(-1) >= 7:
            return graph_or_batch.x[:, 4:7].float()
        if graph_or_batch.x.size(-1) >= 6:
            return graph_or_batch.x[:, -3:].float()

    pos = graph_or_batch.pos.float()
    return pos.new_zeros((pos.size(0), 3))


def graph_dirs_from_pos(pos: torch.Tensor) -> torch.Tensor:
    dirs = pos.new_zeros((pos.size(0), 3))
    mask = point_mask_from_pos(pos)
    dirs[mask] = F.normalize(pos[mask, 1:4], dim=-1, eps=1e-8)
    return dirs


def rotation_a_to_b(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1, eps=eps)
    b = F.normalize(b.float(), dim=-1, eps=eps)

    v = torch.cross(a, b, dim=-1)
    c = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    s = torch.linalg.norm(v)
    eye = torch.eye(3, device=a.device, dtype=a.dtype)

    if float(s) < eps:
        if float(c) > 0.0:
            return eye

        axis = torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype)
        if abs(float(a[0])) > 0.9:
            axis = torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype)

        ortho = F.normalize(torch.cross(a, axis, dim=-1), dim=-1, eps=eps)
        return -eye + 2.0 * torch.outer(ortho, ortho)

    k = v / s
    kx, ky, kz = k.unbind(dim=0)
    skew = torch.stack(
        [
            torch.stack([torch.zeros_like(kx), -kz, ky]),
            torch.stack([kz, torch.zeros_like(kx), -kx]),
            torch.stack([-ky, kx, torch.zeros_like(kx)]),
        ],
        dim=0,
    )
    return eye + skew * s + (skew @ skew) * (1.0 - c)


def local_to_world_rotation(view_dir: torch.Tensor) -> torch.Tensor:
    return rotation_a_to_b(view_dir, LOCAL_FORWARD.to(device=view_dir.device, dtype=view_dir.dtype))


def apply_transform(
    xyz: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return xyz @ rotation + translation


def relative_transform_batch(
    source_loc: torch.Tensor,
    source_view_dir: torch.Tensor,
    target_loc: torch.Tensor,
    target_view_dir: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotations = []
    translations = []

    for loc_s, vd_s, loc_t, vd_t in zip(source_loc, source_view_dir, target_loc, target_view_dir):
        rot_s = local_to_world_rotation(vd_s)
        rot_t = local_to_world_rotation(vd_t)
        rotations.append(rot_s @ rot_t.T)
        translations.append((loc_s - loc_t) @ rot_t.T)

    return torch.stack(rotations, dim=0), torch.stack(translations, dim=0)


def weighted_kabsch(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = weights.clamp_min(1e-6)
    weights = weights / weights.sum().clamp_min(1e-6)

    source_mean = (weights[:, None] * source_xyz).sum(dim=0, keepdim=True)
    target_mean = (weights[:, None] * target_xyz).sum(dim=0, keepdim=True)

    source_centered = source_xyz - source_mean
    target_centered = target_xyz - target_mean
    covariance = source_centered.T @ (weights[:, None] * target_centered)

    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    rotation = u @ vh

    if torch.det(rotation) < 0:
        vh = vh.clone()
        vh[-1, :] *= -1
        rotation = u @ vh

    translation = target_mean.squeeze(0) - source_mean.squeeze(0) @ rotation
    return rotation, translation


def rotation_geodesic_loss(pred_rotation: torch.Tensor, gt_rotation: torch.Tensor) -> torch.Tensor:
    relative_rotation = pred_rotation @ gt_rotation.transpose(1, 2)
    trace = relative_rotation[:, 0, 0] + relative_rotation[:, 1, 1] + relative_rotation[:, 2, 2]
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine).mean()


def trimmed_mean(values: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    keep_count = max(1, int(math.ceil(values.numel() * keep_ratio)))
    return torch.topk(values, k=keep_count, largest=False).values.mean()


def trimmed_chamfer_distance(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    keep_ratio: float = ALIGN_KEEP_RATIO,
) -> torch.Tensor:
    if source_xyz.numel() == 0 or target_xyz.numel() == 0:
        return source_xyz.new_zeros(())

    dist_sq = torch.cdist(source_xyz, target_xyz).square()
    source_min = dist_sq.min(dim=1).values
    target_min = dist_sq.min(dim=0).values
    return trimmed_mean(source_min, keep_ratio) + trimmed_mean(target_min, keep_ratio)


def subsample_indices(num_points: int, max_points: int, tensor_device: torch.device) -> torch.Tensor:
    if num_points <= max_points:
        return torch.arange(num_points, device=tensor_device, dtype=torch.long)
    step = float(num_points) / float(max_points)
    idx = torch.floor(torch.arange(max_points, device=tensor_device) * step).long()
    return idx.clamp_max(num_points - 1)


def subsample_xyz_rgb(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    max_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = subsample_indices(xyz.size(0), max_points, xyz.device)
    return xyz[idx], rgb[idx]


class LatentGraphDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.context_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs = nn.ModuleList(
            [GENConv(in_channels=hidden_dim, out_channels=hidden_dim, norm="layer") for _ in range(layers)]
        )
        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graphs: Batch, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        pos = graphs.pos.float()
        rgb = graph_rgb_tensor(graphs)
        batch = graphs.batch
        point_mask = point_mask_from_pos(pos)
        dirs = graph_dirs_from_pos(pos)

        node_inputs = torch.cat([dirs, rgb], dim=-1)
        node_hidden = self.node_encoder(node_inputs)
        point_summary = global_mean_pool(node_hidden[point_mask], batch[point_mask], size=latents.size(0))
        latent_context = self.latent_proj(latents)[batch]
        summary_context = point_summary[batch]
        hidden = self.context_fuse(torch.cat([node_hidden, latent_context, summary_context], dim=-1))

        if graphs.edge_index.numel() > 0:
            for conv in self.convs:
                hidden = hidden + F.gelu(conv(hidden, graphs.edge_index))

        pred_dist = pos.new_zeros(pos.size(0))
        pred_dist[point_mask] = F.softplus(self.distance_head(hidden[point_mask]).squeeze(-1)) + 1e-3
        pred_xyz = dirs * pred_dist.unsqueeze(-1)

        return {
            "dirs": dirs,
            "rgb": rgb,
            "hidden": hidden,
            "point_mask": point_mask,
            "pred_dist": pred_dist,
            "pred_xyz": pred_xyz,
        }


class LatentTranslationPrior(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.translation_head = nn.Linear(hidden_dim, 3)
        self.strength_head = nn.Linear(hidden_dim, 1)

    def forward(self, source_latent: torch.Tensor, target_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_delta = target_latent - source_latent
        hidden = self.backbone(latent_delta)
        translation = self.translation_head(hidden)
        strength = torch.sigmoid(self.strength_head(hidden)).squeeze(-1)
        return translation, strength


class CrossAttentionPointMatcher(nn.Module):
    def __init__(self, descriptor_dim: int, knn: int = 8, max_points: int = MAX_MATCH_POINTS):
        super().__init__()
        self.knn = knn
        self.max_points = max_points
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(1 + knn + 3, descriptor_dim),
            nn.LayerNorm(descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.LayerNorm(descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )
        self.query_proj = nn.Linear(descriptor_dim, descriptor_dim)
        self.key_proj = nn.Linear(descriptor_dim, descriptor_dim)
        self.value_proj = nn.Linear(descriptor_dim, descriptor_dim)
        self.match_confidence = nn.Sequential(
            nn.Linear(descriptor_dim * 2, descriptor_dim),
            nn.LayerNorm(descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.prior_log_sigma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _knn_distances(self, xyz: torch.Tensor) -> torch.Tensor:
        num_points = xyz.size(0)
        if num_points <= 1:
            return xyz.new_zeros((num_points, self.knn))

        centered = xyz - xyz.mean(dim=0, keepdim=True)
        distances = torch.cdist(centered, centered)
        diag_mask = torch.eye(num_points, device=xyz.device, dtype=torch.bool)
        distances = distances.masked_fill(diag_mask, float("inf"))

        k = min(self.knn, num_points - 1)
        knn_distances = torch.topk(distances, k=k, largest=False).values
        if k < self.knn:
            padding = xyz.new_zeros((num_points, self.knn - k))
            knn_distances = torch.cat([knn_distances, padding], dim=1)
        return knn_distances

    def encode_points(self, xyz: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        centered = xyz - xyz.mean(dim=0, keepdim=True)
        radius = centered.norm(dim=-1, keepdim=True)
        local_distances = self._knn_distances(xyz)
        descriptors = self.descriptor_mlp(torch.cat([radius, local_distances, rgb], dim=-1))
        return F.normalize(descriptors, dim=-1)

    def register_pair(
        self,
        source_xyz: torch.Tensor,
        source_rgb: torch.Tensor,
        target_xyz: torch.Tensor,
        target_rgb: torch.Tensor,
        translation_prior: torch.Tensor | None = None,
        prior_strength: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if source_xyz.numel() == 0 or target_xyz.numel() == 0:
            rotation = torch.eye(3, device=source_xyz.device, dtype=source_xyz.dtype)
            translation = source_xyz.new_zeros(3)
            return {
                "rotation": rotation,
                "translation": translation,
                "aligned_xyz": source_xyz,
                "mean_confidence": source_xyz.new_zeros(()),
            }

        source_match_xyz, source_match_rgb = subsample_xyz_rgb(source_xyz, source_rgb, self.max_points)
        target_match_xyz, target_match_rgb = subsample_xyz_rgb(target_xyz, target_rgb, self.max_points)

        if source_match_xyz.size(0) < 3 or target_match_xyz.size(0) < 3:
            rotation = torch.eye(3, device=source_xyz.device, dtype=source_xyz.dtype)
            translation = target_match_xyz.mean(dim=0) - source_match_xyz.mean(dim=0)
            return {
                "rotation": rotation,
                "translation": translation,
                "aligned_xyz": apply_transform(source_xyz, rotation, translation),
                "mean_confidence": source_xyz.new_ones(()),
            }

        source_desc = self.encode_points(source_match_xyz, source_match_rgb)
        target_desc = self.encode_points(target_match_xyz, target_match_rgb)
        queries = self.query_proj(source_desc)
        keys = self.key_proj(target_desc)
        values = self.value_proj(target_desc)

        logits = (queries @ keys.T) / math.sqrt(queries.size(-1))

        if translation_prior is not None:
            if prior_strength is None:
                prior_strength = logits.new_ones(())
            prior_sigma = self.prior_log_sigma.exp().clamp_min(0.1)
            prior_shifted_source = source_match_xyz + translation_prior.reshape(1, 3)
            prior_dist_sq = torch.cdist(prior_shifted_source, target_match_xyz).square()
            prior_bias = -prior_dist_sq / (2.0 * prior_sigma.square())
            logits = logits + prior_strength.reshape(1, 1) * prior_bias

        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = scale * logits
        attention = torch.softmax(logits, dim=-1)
        matched_xyz = attention @ target_match_xyz
        attended_target = attention @ values
        confidence = torch.sigmoid(
            self.match_confidence(torch.cat([queries, attended_target], dim=-1)).squeeze(-1)
        )
        weights = (attention.max(dim=-1).values * confidence).clamp_min(1e-4)

        rotation, translation = weighted_kabsch(source_match_xyz, matched_xyz, weights)
        aligned_xyz = apply_transform(source_xyz, rotation, translation)

        return {
            "rotation": rotation,
            "translation": translation,
            "aligned_xyz": aligned_xyz,
            "mean_confidence": weights.mean(),
        }


def split_decoded_graphs(graphs: Batch, decoded: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    outputs = []
    batch = graphs.batch
    gt_dist = graphs.pos[:, 0].float()
    point_mask = decoded["point_mask"]

    for graph_idx in range(graphs.num_graphs):
        graph_mask = point_mask & (batch == graph_idx)
        dirs = decoded["dirs"][graph_mask]
        gt_dist_graph = gt_dist[graph_mask]
        outputs.append(
            {
                "xyz_pred": decoded["pred_xyz"][graph_mask],
                "xyz_gt": dirs * gt_dist_graph.unsqueeze(-1),
                "rgb": decoded["rgb"][graph_mask],
                "dist_pred": decoded["pred_dist"][graph_mask],
                "dist_gt": gt_dist_graph,
            }
        )

    return outputs


class RegistrationVAE(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, descriptor_dim: int):
        super().__init__()
        self.decoder = LatentGraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, layers=GRAPH_LAYERS)
        self.translation_prior = LatentTranslationPrior(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.matcher = CrossAttentionPointMatcher(descriptor_dim=descriptor_dim, knn=8, max_points=MAX_MATCH_POINTS)

    def forward(
        self,
        source_graphs: Batch,
        target_graphs: Batch,
        source_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        source_decoded = self.decoder(source_graphs, source_latents)
        target_decoded = self.decoder(target_graphs, target_latents)
        prior_translation, prior_strength = self.translation_prior(source_latents, target_latents)

        matcher_translation_prior = prior_translation
        matcher_prior_strength = prior_strength
        if self.training:
            noise = torch.randn_like(prior_translation) * PRIOR_NOISE_STD
            keep_mask = (torch.rand_like(prior_strength) > PRIOR_DROPOUT).float()
            matcher_translation_prior = prior_translation + noise
            matcher_prior_strength = prior_strength * keep_mask

        source_split = split_decoded_graphs(source_graphs, source_decoded)
        target_split = split_decoded_graphs(target_graphs, target_decoded)

        rotations = []
        translations = []
        aligned_source_xyz = []
        confidences = []

        for source_graph, target_graph in zip(source_split, target_split):
            pair_idx = len(rotations)
            pair_out = self.matcher.register_pair(
                source_graph["xyz_pred"],
                source_graph["rgb"],
                target_graph["xyz_pred"],
                target_graph["rgb"],
                translation_prior=matcher_translation_prior[pair_idx],
                prior_strength=matcher_prior_strength[pair_idx],
            )
            rotations.append(pair_out["rotation"])
            translations.append(pair_out["translation"])
            aligned_source_xyz.append(pair_out["aligned_xyz"])
            confidences.append(pair_out["mean_confidence"])

        return {
            "source_graphs": source_split,
            "target_graphs": target_split,
            "rotation": torch.stack(rotations, dim=0),
            "translation": torch.stack(translations, dim=0),
            "aligned_source_xyz": aligned_source_xyz,
            "prior_translation": prior_translation,
            "prior_strength": prior_strength,
            "mean_confidence": torch.stack(confidences, dim=0).mean(),
        }


def _decoded_graphs_device(decoded_graphs: Sequence[Dict[str, torch.Tensor]]) -> torch.device:
    for graph in decoded_graphs:
        if "dist_pred" in graph and graph["dist_pred"].numel() >= 0:
            return graph["dist_pred"].device
    return device


def reconstruction_loss(decoded_graphs: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
    losses = []
    for graph in decoded_graphs:
        if graph["dist_gt"].numel() == 0:
            continue
        losses.append(F.smooth_l1_loss(graph["dist_pred"], graph["dist_gt"]))
    if not losses:
        return torch.zeros((), device=_decoded_graphs_device(decoded_graphs))
    return torch.stack(losses, dim=0).mean()


def alignment_loss(
    aligned_source_xyz: Sequence[torch.Tensor],
    target_graphs: Sequence[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    losses = []
    for source_xyz, target_graph in zip(aligned_source_xyz, target_graphs):
        source_xyz_sub = source_xyz[subsample_indices(source_xyz.size(0), MAX_LOSS_POINTS, source_xyz.device)]
        target_xyz = target_graph["xyz_gt"]
        target_xyz_sub = target_xyz[subsample_indices(target_xyz.size(0), MAX_LOSS_POINTS, target_xyz.device)]
        losses.append(trimmed_chamfer_distance(source_xyz_sub, target_xyz_sub))
    if not losses:
        return torch.zeros((), device=_decoded_graphs_device(target_graphs))
    return torch.stack(losses, dim=0).mean()


def registration_loss(
    outputs: Dict[str, torch.Tensor],
    gt_rotation: torch.Tensor,
    gt_translation: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    source_recon = reconstruction_loss(outputs["source_graphs"])
    target_recon = reconstruction_loss(outputs["target_graphs"])
    recon_loss = 0.5 * (source_recon + target_recon)
    align_loss = alignment_loss(outputs["aligned_source_xyz"], outputs["target_graphs"])
    translation_loss = F.smooth_l1_loss(outputs["translation"], gt_translation)
    prior_translation_loss = F.smooth_l1_loss(outputs["prior_translation"], gt_translation)
    rotation_loss = rotation_geodesic_loss(outputs["rotation"], gt_rotation)
    world_loss = gt_translation.new_zeros(())

    loss = (
        0.35 * recon_loss
        + 0.30 * align_loss
        + 0.20 * rotation_loss
        + 0.10 * translation_loss
        + PRIOR_TRANSLATION_WEIGHT * prior_translation_loss
    )

    loss_dict = {
        "loss": float(loss.detach()),
        "recon_loss": float(recon_loss.detach()),
        "align_loss": float(align_loss.detach()),
        "rotation_loss": float(rotation_loss.detach()),
        "translation_loss": float(translation_loss.detach()),
        "prior_translation_loss": float(prior_translation_loss.detach()),
        "world_loss": float(world_loss.detach()),
        "prior_strength": float(outputs["prior_strength"].mean().detach()),
        "mean_confidence": float(outputs["mean_confidence"].detach()),
    }
    return loss, loss_dict


def move_batch_to_device(batch: Dict[str, torch.Tensor], tensor_device: torch.device) -> Dict[str, torch.Tensor]:
    batch["source_graphs"] = batch["source_graphs"].to(tensor_device)
    batch["target_graphs"] = batch["target_graphs"].to(tensor_device)
    batch["source_latent"] = batch["source_latent"].to(tensor_device)
    batch["target_latent"] = batch["target_latent"].to(tensor_device)
    batch["source_loc"] = batch["source_loc"].to(tensor_device)
    batch["target_loc"] = batch["target_loc"].to(tensor_device)
    batch["source_view_dir"] = batch["source_view_dir"].to(tensor_device)
    batch["target_view_dir"] = batch["target_view_dir"].to(tensor_device)
    return batch


def run_epoch(
    model: RegistrationVAE,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scheduler,
) -> Tuple[float, Dict[str, float]]:
    model.train(True)
    running_loss = 0.0
    running_metrics = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "align_loss": 0.0,
        "rotation_loss": 0.0,
        "translation_loss": 0.0,
        "prior_translation_loss": 0.0,
        "world_loss": 0.0,
        "prior_strength": 0.0,
        "mean_confidence": 0.0,
    }
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        gt_rotation, gt_translation = relative_transform_batch(
            batch["source_loc"],
            batch["source_view_dir"],
            batch["target_loc"],
            batch["target_view_dir"],
        )

        opt.zero_grad(set_to_none=True)
        outputs = model(
            batch["source_graphs"],
            batch["target_graphs"],
            batch["source_latent"],
            batch["target_latent"],
        )
        loss, loss_dict = registration_loss(outputs, gt_rotation, gt_translation)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        opt.step()

        running_loss += float(loss.detach())
        for key, value in loss_dict.items():
            running_metrics[key] += value
        num_batches += 1

    if num_batches == 0:
        return 0.0, running_metrics

    scheduler.step()
    averaged_metrics = {key: value / num_batches for key, value in running_metrics.items()}
    return running_loss / num_batches, averaged_metrics


def make_registration_loader(
    observations: Sequence[RegistrationObservation],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = LatentRegistrationDataset(
        observations=observations,
        pair_min_distance=PAIR_MIN_DISTANCE,
        pair_max_distance=PAIR_MAX_DISTANCE,
        pair_top_k=PAIR_TOP_K,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,#torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_registration_pairs,
    )


def train() -> None:
    output_dir = MODEL_ROOT / ALIAS
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "losses.csv"
    write_header = not csv_path.exists()

    encoder = load_encoder(VAE_WEIGHTS, latent_dim=LATENT_DIM)
    observations = encode_all_observations(
        encoder,
        rw_list,
        latent_branch=LATENT_BRANCH,
    )
    loader = make_registration_loader(observations=observations)
    num_candidate_pairs = int(sum(candidates.numel() for candidates in loader.dataset.pair_candidates))

    model = RegistrationVAE(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        descriptor_dim=DESCRIPTOR_DIM,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, 0.0, -1)

    print("starting training")
    print(ALIAS)
    print(
        f"encoded {len(observations)} observations into {num_candidate_pairs} candidate registration pairs "
        f"using branch={LATENT_BRANCH}"
    )

    for epoch in range(EPOCHS):
        _, loss_dict = run_epoch(model, loader, opt, scheduler)
        row = {
            "epoch": epoch,
            "num_observations": float(len(observations)),
            "num_candidate_pairs": float(num_candidate_pairs),
            **loss_dict,
        }
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=write_header, index=False)
        write_header = False

        if epoch % 10 == 0:
            print(f"epoch {epoch}", loss_dict)
        if epoch % 200 == 0:
            torch.save(model.state_dict(), output_dir / f"model_weights_{epoch}.pt")

    torch.save(model.state_dict(), output_dir / "model_weights.pt")
    print("completed")


if __name__ == "__main__":
    train()
