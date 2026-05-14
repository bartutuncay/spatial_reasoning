from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gnn_spatial_reasoning.models.locator_vae_2 import Locator
from gnn_spatial_reasoning.preprocessing_src.dataloader_autoencoder import make_loader


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


img_enc_mod = load_module("nips_img_enc", ROOT / "gnn_spatial_reasoning_nips/models/1_img_enc.py")
pcd_enc_mod = load_module("nips_pcd_enc", ROOT / "gnn_spatial_reasoning_nips/models/1_pcd_enc.py")

Bottleneck = img_enc_mod.Bottleneck
ImgEnc = img_enc_mod.ImgEnc
PCDEnc = pcd_enc_mod.PCDEnc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

EPOCHS = int(os.environ.get("LOCATOR_EPOCHS", "20000"))
WALK_BATCH_SIZE = int(os.environ.get("LOCATOR_WALK_BATCH_SIZE", "2"))
WALK_LOADER_WORKERS = int(os.environ.get("LOCATOR_WALK_WORKERS", "4"))
LATENT_DIM = int(os.environ.get("LOCATOR_LATENT_DIM", "64"))
HIDDEN_DIM = int(os.environ.get("LOCATOR_HIDDEN_DIM", "64"))
VAE_LATENT_DIM = int(os.environ.get("LOCATOR_VAE_LATENT_DIM", "128"))
SCENE_POOL_NODES = int(os.environ.get("LOCATOR_SCENE_POOL_NODES", "4096"))
VIEW_POOL_NODES = int(os.environ.get("LOCATOR_VIEW_POOL_NODES", "512"))
SCENE_CACHE_EVERY = max(1, int(os.environ.get("LOCATOR_SCENE_CACHE_EVERY", "2")))
CLIP_GRAD_NORM = float(os.environ.get("LOCATOR_CLIP_GRAD_NORM", "1.0"))
LEARNING_RATE = float(os.environ.get("LOCATOR_LR", "2e-5"))
WEIGHT_DECAY = float(os.environ.get("LOCATOR_WEIGHT_DECAY", "1e-4"))
GUIDE_LOSS_WEIGHT = float(os.environ.get("LOCATOR_GUIDE_LOSS_WEIGHT", "0.10"))
GUIDE_COSINE_WEIGHT = float(os.environ.get("LOCATOR_GUIDE_COSINE_WEIGHT", "0.05"))
GUIDE_DROPOUT = float(os.environ.get("LOCATOR_GUIDE_DROPOUT", "0.0"))
LOC_LOSS_KIND = os.environ.get("LOCATOR_LOC_LOSS", "mse").strip().lower()
TARGET_MODE = os.environ.get("LOCATOR_TARGET_MODE", "cloud_center").strip().lower()
VAE_BRANCH = os.environ.get("LOCATOR_VAE_BRANCH", "pcd").strip().lower()

PROCESSED_DATA_ROOT = Path(
    os.environ.get(
        "LOCATOR_PROCESSED_ROOT",
        "../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed",
    )
)
MODEL_ROOT = Path(
    os.environ.get(
        "LOCATOR_MODEL_ROOT",
        "../../../scratch/btuncay/cog/gnn_spatial_reasoning/4_model_locator",
    )
)
VAE_WEIGHTS = Path(
    os.environ.get(
        "LOCATOR_VAE_WEIGHTS",
        "../../../scratch/btuncay/cog/gnn_spatial_reasoning/1_model/0420_nips/model_weights_7800.pt",
    )
)
DATASET_ENV_VAR = "LOCATOR_DATASETS"
ALIAS = os.environ.get("LOCATOR_ALIAS", "0505")
WORLD_UP = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
SCENE_CACHE_KEYS = (
    "node_mu",
    "node_logvar",
    "node_pos",
    "mu",
    "logvar",
    "pool_score_mean",
)


@dataclass
class SceneData:
    name: str
    walk_root: Path
    scene_graph_path: Path
    walk_loader: object


@dataclass
class LocatorBatch:
    pcd: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    loc: torch.Tensor
    cloud_center_local: torch.Tensor
    path: Optional[List[str]] = None


class VAEGuideEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.img_enc = ImgEnc(Bottleneck, [3, 4, 6, 3], latent_dim)
        self.pcd_enc = PCDEnc(
            latent_dim=latent_dim,
            layers_points=2,
            layers_camera=4,
            layers_mlp=2,
            z_dim=latent_dim,
        )

    def encode_img(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.img_enc(img)

    def encode_pcd(
        self,
        pcd: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        ei_camera: torch.Tensor,
        ea_camera: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pcd_enc(pcd, batch, edge_index, edge_weights, ei_camera, ea_camera)


class GuidedLatentLocationVAE(nn.Module):
    def __init__(
        self,
        locator_latent_dim: int,
        hidden_dim: int,
        guide_latent_dim: int,
    ):
        super().__init__()
        self.locator = Locator(
            nodes_dim=6,
            latent_dim=locator_latent_dim,
            hidden_dim=hidden_dim,
            scene_pool_nodes=SCENE_POOL_NODES,
            view_pool_nodes=VIEW_POOL_NODES,
        )
        self.guide_latent_dim = guide_latent_dim
        self.guide_to_view = nn.Sequential(
            nn.LayerNorm(guide_latent_dim),
            nn.Linear(guide_latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, locator_latent_dim),
            nn.LayerNorm(locator_latent_dim),
        )
        self.fuse_gate = nn.Sequential(
            nn.Linear(locator_latent_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, locator_latent_dim),
        )
        self.fused_norm = nn.LayerNorm(locator_latent_dim)
        self.view_to_guide = nn.Sequential(
            nn.Linear(locator_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, guide_latent_dim),
        )

    def encode_scene(self, scene_graph, sample: bool = False):
        return self.locator.encode_scene(scene_graph, sample=sample)

    def _normalized_guide(self, guide_mu: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(guide_mu.float(), (guide_mu.size(-1),))

    def _fused_view_encoding(
        self,
        view_encoding: Dict[str, torch.Tensor],
        guide_mu: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        view_z = view_encoding["z"].float()
        guide_context = self.guide_to_view(guide_mu.float())

        if self.training and GUIDE_DROPOUT > 0.0:
            keep = torch.rand((guide_context.size(0), 1), device=guide_context.device) >= GUIDE_DROPOUT
            guide_context = guide_context * keep.to(dtype=guide_context.dtype)

        fuse_in = torch.cat([view_z, guide_context, torch.abs(view_z - guide_context)], dim=-1)
        guide_gate = torch.sigmoid(self.fuse_gate(fuse_in))
        fused_encoding = dict(view_encoding)
        fused_encoding["z"] = self.fused_norm(view_z + guide_gate * guide_context)
        return fused_encoding, guide_gate

    def forward_with_scene_encoding(
        self,
        scene_encoding: Dict[str, torch.Tensor],
        walk_batch: LocatorBatch,
        guide_mu: torch.Tensor,
        guide_logvar: torch.Tensor,
        sample_view: bool = True,
    ) -> Dict[str, torch.Tensor]:
        view_encoding = self.locator.encode_view(walk_batch, sample=sample_view)
        fused_view_encoding, guide_gate = self._fused_view_encoding(view_encoding, guide_mu)
        match_outputs = self.locator._scene_match(scene_encoding, fused_view_encoding)

        return {
            **match_outputs,
            "target_loc": walk_batch.loc.float(),
            "scene_mu": scene_encoding["mu"],
            "scene_logvar": scene_encoding["logvar"],
            "view_mu": view_encoding["mu"],
            "view_logvar": view_encoding["logvar"],
            "guide_mu": guide_mu,
            "guide_logvar": guide_logvar,
            "view_guide_pred": self.view_to_guide(view_encoding["mu"]),
            "guide_gate": guide_gate,
            "scene_pool_score_mean": scene_encoding["pool_score_mean"],
            "view_pool_score_mean": view_encoding["pool_score_mean"],
        }

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()

    def _guide_alignment_loss(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        guide_target = self._normalized_guide(outputs["guide_mu"].detach())
        guide_pred = outputs["view_guide_pred"]
        per_graph = F.smooth_l1_loss(guide_pred, guide_target, reduction="none").mean(dim=-1)
        guide_conf = torch.exp(
            -outputs["guide_logvar"].detach().mean(dim=-1).clamp(min=-6.0, max=6.0)
        )
        guide_loss = (per_graph * guide_conf).sum() / guide_conf.sum().clamp_min(1e-6)
        guide_cosine = 1.0 - F.cosine_similarity(guide_pred, guide_target, dim=-1).mean()
        return guide_loss, guide_cosine

    def loss(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if outputs["target_loc"] is None:
            raise ValueError("GuidedLatentLocationVAE.loss expects target locations.")

        if LOC_LOSS_KIND == "smooth_l1":
            loc_loss = F.smooth_l1_loss(outputs["pred_loc"], outputs["target_loc"])
        elif LOC_LOSS_KIND == "mse":
            loc_loss = F.mse_loss(outputs["pred_loc"], outputs["target_loc"])
        else:
            raise ValueError(f"Unsupported LOCATOR_LOC_LOSS='{LOC_LOSS_KIND}'.")

        kl_loss = self._kl_divergence(outputs["view_mu"], outputs["view_logvar"])
        weights = outputs["match_weights"].clamp_min(1e-8)
        entropy = -(weights * weights.log()).sum(dim=-1).mean()
        guide_loss, guide_cosine = self._guide_alignment_loss(outputs)

        loss = (
            loc_loss
            + (1e-2 * kl_loss)
            + (1e-4 * entropy)
            + (GUIDE_LOSS_WEIGHT * guide_loss)
            + (GUIDE_COSINE_WEIGHT * guide_cosine)
        )
        mean_error = torch.linalg.norm(outputs["pred_loc"] - outputs["target_loc"], dim=-1).mean()
        metrics = {
            "loc_loss": float(loc_loss.detach().item()),
            "loc_error": float(mean_error.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "match_entropy": float(entropy.detach().item()),
            "guide_loss": float(guide_loss.detach().item()),
            "guide_cosine": float(guide_cosine.detach().item()),
            "guide_gate": float(outputs["guide_gate"].detach().mean().item()),
            "pool_score_mean": float(
                0.5
                * (
                    outputs["scene_pool_score_mean"].detach().item()
                    + outputs["view_pool_score_mean"].detach().item()
                )
            ),
        }
        return loss, metrics


class LatentLocationVAE(GuidedLatentLocationVAE):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        guide_latent_dim: int = VAE_LATENT_DIM,
    ):
        super().__init__(
            locator_latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            guide_latent_dim=guide_latent_dim,
        )


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        stripped[key[len("module.") :] if key.startswith("module.") else key] = value
    return stripped


def load_vae_guide(weights_path: Path, latent_dim: int) -> VAEGuideEncoder:
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing ImageGraphVAE checkpoint: {weights_path}")

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = strip_module_prefix(state)

    model = VAEGuideEncoder(latent_dim=latent_dim).to(device)
    incompatible = model.load_state_dict(state, strict=False)
    loaded_encoder_keys = [key for key in state if key.startswith(("img_enc.", "pcd_enc."))]
    if not loaded_encoder_keys:
        raise ValueError(f"Checkpoint {weights_path} did not contain VAE encoder weights.")

    if incompatible.missing_keys:
        print(f"vae load missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"vae load unexpected keys: {len(incompatible.unexpected_keys)}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def normalize_edge_weights(edge_weights: torch.Tensor) -> torch.Tensor:
    scale = edge_weights.mean().clamp_min(1e-6)
    return torch.exp(-(edge_weights.pow(2)) / (2 * scale.pow(2)))


@torch.no_grad()
def encode_vae_guide(
    model: VAEGuideEncoder,
    walk_batch,
    branch: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    if branch == "pcd":
        edge_weights = normalize_edge_weights(walk_batch.edge_weights.float())
        _, mu, logvar = model.encode_pcd(
            walk_batch.pcd.float(),
            walk_batch.batch,
            walk_batch.edge_index,
            edge_weights,
            walk_batch.ei_camera,
            walk_batch.ea_camera.float(),
        )
        return mu.detach(), logvar.detach()
    if branch == "img":
        img = walk_batch.img.permute(0, 3, 1, 2).contiguous().float()
        _, mu, logvar = model.encode_img(img)
        return mu.detach(), logvar.detach()
    raise ValueError(f"Unsupported LOCATOR_VAE_BRANCH='{branch}'. Use 'pcd' or 'img'.")


def dir_has_pt_files(root: Path) -> bool:
    return root.is_dir() and any(root.glob("*.pt"))


def resolve_scene_graph_path(dataset_root: Path) -> Path:
    preferred = dataset_root / "scan_pcd_graph" / "combined_aligned.pt"
    if preferred.is_file():
        return preferred

    scan_root = dataset_root / "scan_pcd_graph"
    candidates = sorted(path for path in scan_root.glob("*.pt") if path.is_file())
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No scene graph `.pt` file found under {scan_root}.")


def discover_dataset_names(processed_root: Path) -> List[str]:
    discovered: List[str] = []
    if not processed_root.is_dir():
        return discovered

    for dataset_root in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        walk_root = dataset_root / "random_walks"
        if not dir_has_pt_files(walk_root):
            continue
        try:
            resolve_scene_graph_path(dataset_root)
        except FileNotFoundError:
            continue
        discovered.append(dataset_root.name)
    return discovered


def resolve_dataset_names() -> List[str]:
    requested = [name.strip() for name in os.environ.get(DATASET_ENV_VAR, "").split(",") if name.strip()]
    if requested:
        return requested

    discovered = discover_dataset_names(PROCESSED_DATA_ROOT)
    if discovered:
        return discovered

    raise FileNotFoundError(
        f"No datasets with both `random_walks` and `scan_pcd_graph/*.pt` were found under {PROCESSED_DATA_ROOT}."
    )


def build_scene_data(dataset_name: str) -> SceneData:
    dataset_root = PROCESSED_DATA_ROOT / dataset_name
    walk_root = dataset_root / "random_walks"
    if not dir_has_pt_files(walk_root):
        raise FileNotFoundError(f"Dataset `{dataset_name}` is missing random walks under {walk_root}.")

    scene_graph_path = resolve_scene_graph_path(dataset_root)
    return SceneData(
        name=dataset_name,
        walk_root=walk_root,
        scene_graph_path=scene_graph_path,
        walk_loader=make_loader(
            str(walk_root),
            batch_size=WALK_BATCH_SIZE,
            shuffle=True,
            num_workers=WALK_LOADER_WORKERS,
            drop_last=False,
        ),
    )


def build_scene_datasets(dataset_names: Sequence[str]) -> List[SceneData]:
    datasets = [build_scene_data(name) for name in dataset_names]
    if not datasets:
        raise ValueError("At least one dataset is required for locator training.")
    return datasets


def detach_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    return value


def move_to_device(value: Any, target_device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(target_device)
    if hasattr(value, "to"):
        return value.to(target_device)
    if isinstance(value, dict):
        return {key: move_to_device(item, target_device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, target_device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, target_device) for item in value)
    return value


def compact_scene_encoding(scene_encoding: Dict[str, Any]) -> Dict[str, Any]:
    return {key: scene_encoding[key] for key in SCENE_CACHE_KEYS if key in scene_encoding}


def load_scene_graph(scene_path: Path):
    return torch.load(scene_path, map_location="cpu", weights_only=False)


def build_scene_cache(
    model: GuidedLatentLocationVAE,
    scene_datasets: Sequence[SceneData],
) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    was_training = model.training
    model.train(False)

    with torch.inference_mode():
        for scene in scene_datasets:
            scene_graph = move_to_device(load_scene_graph(scene.scene_graph_path), device)
            encoded = model.encode_scene(scene_graph, sample=False)
            cache[scene.name] = detach_to_cpu(compact_scene_encoding(encoded))

    if was_training:
        model.train(True)
    return cache


def scatter_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = values.new_zeros((dim_size, values.size(-1)))
    counts = values.new_zeros((dim_size, 1))
    out.index_add_(0, index, values)
    counts.index_add_(0, index, values.new_ones((values.size(0), 1)))
    return out / counts.clamp_min(1.0)


def local_view_offset_to_world(offset: torch.Tensor, view_dir: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    forward = F.normalize(view_dir.float(), dim=-1, eps=eps)
    up_seed = WORLD_UP.to(device=offset.device, dtype=offset.dtype).expand_as(forward)
    right = torch.cross(up_seed, forward, dim=-1)
    right_norm = right.norm(dim=-1, keepdim=True)

    bad = right_norm.squeeze(-1) < eps
    if bad.any():
        alt_seed = torch.tensor([0.0, 1.0, 0.0], device=offset.device, dtype=offset.dtype).expand_as(forward)
        alt_right = torch.cross(alt_seed, forward, dim=-1)
        right = torch.where(bad.unsqueeze(-1), alt_right, right)
        right_norm = right.norm(dim=-1, keepdim=True)

    right = right / right_norm.clamp_min(eps)
    up = torch.cross(forward, right, dim=-1)
    return offset[:, 0:1] * forward - offset[:, 1:2] * right + offset[:, 2:3] * up


def locator_target_loc(walk_batch, cloud_center_local: torch.Tensor) -> torch.Tensor:
    loc = walk_batch.loc.float()
    if TARGET_MODE == "camera":
        return loc
    if TARGET_MODE == "legacy_cloud_center":
        return loc + cloud_center_local
    if TARGET_MODE == "cloud_center":
        if walk_batch.view_dir is None:
            return loc + cloud_center_local
        return loc + local_view_offset_to_world(cloud_center_local, walk_batch.view_dir.float())
    raise ValueError(
        f"Unsupported LOCATOR_TARGET_MODE='{TARGET_MODE}'. "
        "Use 'cloud_center', 'legacy_cloud_center', or 'camera'."
    )


def camera_free_locator_batch(walk_batch) -> LocatorBatch:
    pcd = walk_batch.pcd.float()
    graph_batch = walk_batch.batch.long()
    num_graphs = int(graph_batch.max().item()) + 1 if graph_batch.numel() > 0 else 1
    origin_mask = pcd[:, :4].abs().amax(dim=-1) < 1e-6
    point_mask = ~origin_mask
    if not point_mask.any():
        raise ValueError("Locator received a batch with no non-camera point nodes.")

    old_to_new = torch.full((pcd.size(0),), -1, device=pcd.device, dtype=torch.long)
    old_to_new[point_mask] = torch.arange(point_mask.sum(), device=pcd.device, dtype=torch.long)

    edge_index = walk_batch.edge_index.long()
    if edge_index.numel() > 0:
        src, dst = edge_index
        keep_edges = point_mask[src] & point_mask[dst]
        locator_edge_index = old_to_new[edge_index[:, keep_edges]]
    else:
        locator_edge_index = torch.empty((2, 0), device=pcd.device, dtype=torch.long)

    point_pcd = pcd[point_mask]
    locator_batch = graph_batch[point_mask]
    local_xyz = point_pcd[:, 1:4] * point_pcd[:, 0:1]
    xyzrgb = torch.cat([local_xyz, point_pcd[:, 4:]], dim=-1)
    cloud_center_local = scatter_mean(local_xyz, locator_batch, num_graphs)

    return LocatorBatch(
        pcd=xyzrgb,
        edge_index=locator_edge_index,
        batch=locator_batch,
        loc=locator_target_loc(walk_batch, cloud_center_local),
        cloud_center_local=cloud_center_local,
        path=walk_batch.path,
    )


def average_metrics(metrics_history: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_history:
        return {}

    keys = metrics_history[0].keys()
    return {key: sum(item[key] for item in metrics_history) / len(metrics_history) for key in keys}


def run_epoch(
    model: GuidedLatentLocationVAE,
    guide_model: VAEGuideEncoder,
    scene_datasets: Sequence[SceneData],
    scene_cache: Dict[str, Dict[str, Any]],
    opt: torch.optim.Optimizer,
    scheduler,
) -> Tuple[float, Dict[str, float]]:
    model.train(True)
    guide_model.eval()
    losses: List[float] = []
    metrics_history: List[Dict[str, float]] = []

    for scene in scene_datasets:
        scene_encoding = move_to_device(scene_cache[scene.name], device)

        for walk_batch in scene.walk_loader:
            walk_batch = walk_batch.to(device)
            guide_mu, guide_logvar = encode_vae_guide(guide_model, walk_batch, VAE_BRANCH)
            locator_batch = camera_free_locator_batch(walk_batch)
            if guide_mu.size(0) != locator_batch.loc.size(0):
                raise ValueError(
                    f"VAE guide batch size {guide_mu.size(0)} does not match locator batch size "
                    f"{locator_batch.loc.size(0)}."
                )

            opt.zero_grad(set_to_none=True)
            outputs = model.forward_with_scene_encoding(
                scene_encoding,
                locator_batch,
                guide_mu,
                guide_logvar,
                sample_view=True,
            )
            loss, metrics = model.loss(outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
            opt.step()

            losses.append(float(loss.detach().item()))
            metrics_history.append(metrics)

    scheduler.step()
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    return mean_loss, average_metrics(metrics_history)


def main() -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    output_dir = MODEL_ROOT / ALIAS
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "losses.csv"
    write_header = not csv_path.exists()

    dataset_names = resolve_dataset_names()
    scene_datasets = build_scene_datasets(dataset_names)

    print("starting guided large-locator training")
    print(ALIAS)
    print(f"device: {device}")
    print(f"target_mode: {TARGET_MODE}")
    print(f"vae_branch: {VAE_BRANCH}")
    print(f"vae_weights: {VAE_WEIGHTS}")
    print(f"using datasets: {', '.join(scene.name for scene in scene_datasets)}")
    for scene in scene_datasets:
        num_walks = sum(1 for _ in scene.walk_root.glob("*.pt"))
        print(f"  {scene.name}: {num_walks} walks, scene graph {scene.scene_graph_path.name}")

    guide_model = load_vae_guide(VAE_WEIGHTS, latent_dim=VAE_LATENT_DIM)
    model = LatentLocationVAE(
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        guide_latent_dim=VAE_LATENT_DIM,
    ).to(device)
    opt = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-6)
    scene_cache = build_scene_cache(model, scene_datasets)

    for epoch in range(EPOCHS):
        if epoch > 0 and (epoch % SCENE_CACHE_EVERY == 0):
            scene_cache = build_scene_cache(model, scene_datasets)

        mean_loss, metrics = run_epoch(model, guide_model, scene_datasets, scene_cache, opt, scheduler)
        row = {"epoch": epoch, "train_loss": mean_loss, **metrics}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=write_header, index=False)
        write_header = False

        if epoch % 10 == 0:
            print(f"epoch {epoch}: {row}")
        if epoch % 200 == 0:
            torch.save(model.state_dict(), output_dir / f"model_weights_{epoch}.pt")

    torch.save(model.state_dict(), output_dir / "model_weights.pt")
    print("completed")


if __name__ == "__main__":
    main()
