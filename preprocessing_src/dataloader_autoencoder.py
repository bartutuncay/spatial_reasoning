## Dataloader for the autoencoder
# This file contains the PyTorch dataloader for
# use with the variational autoencoder. It takes
# saved random walk sequences in .pt format containing
# point cloud and image data. It can work with images
# only for inference.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data


def _to_tensor(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value


def _stack_optional(values: Sequence[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        missing = [idx for idx, value in enumerate(values) if value is None]
        raise KeyError(f"Some samples are missing a required field: {missing}")
    return torch.stack([v for v in values if v is not None], dim=0)


@dataclass
class MultiModalBatch:
    img: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    pcd: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    edge_weights: Optional[torch.Tensor] = None
    ei_camera: Optional[torch.Tensor] = None
    ea_camera: Optional[torch.Tensor] = None
    visible_point_indices: Optional[List[torch.Tensor]] = None
    loc: Optional[torch.Tensor] = None
    view_dir: Optional[torch.Tensor] = None
    batch: Optional[torch.Tensor] = None
    graph: Optional[List[Data]] = None
    path: Optional[List[str]] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to(self, device: Union[str, torch.device]) -> "MultiModalBatch":
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if torch.is_tensor(value):
                setattr(self, field_name, value.to(device))
            elif isinstance(value, list):
                moved = []
                for item in value:
                    if torch.is_tensor(item):
                        moved.append(item.to(device))
                    elif hasattr(item, 'to'):
                        moved.append(item.to(device))
                    else:
                        moved.append(item)
                setattr(self, field_name, moved)
        return self


class RandomWalkAutoencoderDataset(Dataset):
    """
    Loads the `.pt` files written by `agent_src/random_walk_camera.py`.

    Each sample is expected to contain at least:
    - `img`: [H, W, 3]
    - `graph`: PyG `Data` with `pos`, `rgb`, `edge_index`, `edge_attr`,
      `ei_camera`, `ea_camera`

    The dataset also normalizes numpy arrays to tensors so downstream collate
    logic can stay simple.
    """

    def __init__(self, root_dir: Union[str, Path], pattern: str = "*.pt"):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matched {pattern} under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        sample = torch.load(path, map_location="cpu", weights_only=False)

        if not isinstance(sample, dict):
            raise TypeError(f"{path.name} did not load a dict. Got: {type(sample)}")

        for key, value in list(sample.items()):
            sample[key] = _to_tensor(value)

        graph = sample.get("graph")
        if graph is None or not isinstance(graph, Data):
            raise KeyError(f"{path.name} is missing a valid 'graph' field.")

        if not hasattr(graph, "pos") or not hasattr(graph, "rgb"):
            raise KeyError(f"{path.name}: graph must contain 'pos' and 'rgb'.")

        graph.pos = _to_tensor(graph.pos).to(torch.float32)
        graph.rgb = _to_tensor(graph.rgb).to(torch.float32)
        graph.edge_index = _to_tensor(graph.edge_index).to(torch.long)
        graph.edge_attr = _to_tensor(graph.edge_attr).to(torch.float32)
        graph.ei_camera = _to_tensor(graph.ei_camera).to(torch.long)
        graph.ea_camera = _to_tensor(graph.ea_camera).to(torch.float32)

        if hasattr(graph, "visible_point_indices") and graph.visible_point_indices is not None:
            graph.visible_point_indices = _to_tensor(graph.visible_point_indices).to(torch.long)

        if "img" in sample and sample["img"] is not None:
            sample["img"] = _to_tensor(sample["img"]).to(torch.float32)
        if "depth" in sample and sample["depth"] is not None:
            sample["depth"] = _to_tensor(sample["depth"]).to(torch.float32)
        if "loc" in sample and sample["loc"] is not None:
            sample["loc"] = _to_tensor(sample["loc"]).to(torch.float32)
        if "view_dir" in sample and sample["view_dir"] is not None:
            sample["view_dir"] = _to_tensor(sample["view_dir"]).to(torch.float32)

        sample["_path"] = str(path)
        return sample


def collate_random_walk_autoencoder(samples: Sequence[Dict[str, Any]]) -> MultiModalBatch:
    if not samples:
        return MultiModalBatch()

    imgs = [sample.get("img") for sample in samples]
    depths = [sample.get("depth") for sample in samples]
    locs = [sample.get("loc") for sample in samples]
    view_dirs = [sample.get("view_dir") for sample in samples]

    pcds: List[torch.Tensor] = []
    batch_parts: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_weights: List[torch.Tensor] = []
    camera_edge_indices: List[torch.Tensor] = []
    camera_edge_attrs: List[torch.Tensor] = []
    visible_point_indices: List[torch.Tensor] = []
    graphs: List[Data] = []
    paths: List[str] = []

    node_offset = 0
    for sample_idx, sample in enumerate(samples):
        graph = sample["graph"]
        graphs.append(graph)
        paths.append(sample["_path"])

        node_features = torch.cat([graph.pos, graph.rgb], dim=1).to(torch.float32)
        num_nodes = node_features.shape[0]

        pcds.append(node_features)
        batch_parts.append(torch.full((num_nodes,), sample_idx, dtype=torch.long))

        edge_indices.append(graph.edge_index.to(torch.long) + node_offset)
        edge_weights.append(graph.edge_attr.to(torch.float32))
        camera_edge_indices.append(graph.ei_camera.to(torch.long) + node_offset)
        camera_edge_attrs.append(graph.ea_camera.to(torch.float32))

        if hasattr(graph, "visible_point_indices") and graph.visible_point_indices is not None:
            visible_point_indices.append(graph.visible_point_indices.to(torch.long))
        else:
            visible_point_indices.append(torch.empty(0, dtype=torch.long))

        node_offset += num_nodes

    return MultiModalBatch(
        img=_stack_optional(imgs),
        depth=_stack_optional(depths),
        pcd=torch.cat(pcds, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_weights=torch.cat(edge_weights, dim=0),
        ei_camera=torch.cat(camera_edge_indices, dim=1),
        ea_camera=torch.cat(camera_edge_attrs, dim=0),
        visible_point_indices=visible_point_indices,
        loc=_stack_optional(locs),
        view_dir=_stack_optional(view_dirs),
        batch=torch.cat(batch_parts, dim=0),
        graph=graphs,
        path=paths,
    )


def make_loader(
    root_dir: Union[str, Path],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    pattern: str = "*.pt",
    drop_last: bool = False,
) -> DataLoader:
    dataset = RandomWalkAutoencoderDataset(root_dir=root_dir, pattern=pattern)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_random_walk_autoencoder,
        persistent_workers=(num_workers > 0),
    )
