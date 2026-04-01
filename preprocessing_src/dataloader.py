from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class MultiModalBatch:
    img: Optional[torch.Tensor] = None           # [B, C, H, W]
    pcd: Optional[torch.Tensor] = None           # [sumN, F]
    edge_index: Optional[torch.Tensor] = None    # [2, sumE]
    edge_weights: Optional[torch.Tensor] = None  # [sumE] or [sumE, ...]
    vis: Optional[torch.Tensor] = None           # user-defined shape
    batch: Optional[torch.Tensor] = None         # [sumN], sample index per node

    # Allow dict-style access: batch['img'] etc.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    # Match your training loop: batch = batch.to(device)
    def to(self, device: Union[str, torch.device]) -> "MultiModalBatch":
        for field_name in self.__dataclass_fields__:
            v = getattr(self, field_name)
            if torch.is_tensor(v):
                setattr(self, field_name, v.to(device))
        return self

class PtDictFolderDataset(Dataset):
    """
    Expects each .pt file to contain a dict with (some of) these keys:
      - 'img': Tensor [C,H,W] or [H,W,C] (you can adapt)
      - 'pcd': Tensor [N,F]
      - 'edge_index': LongTensor [2,E]
      - 'edge_weights': Tensor [E] (optional)
      - 'vis': Tensor (optional)
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
        sample = torch.load(path, map_location="cpu")

        if not isinstance(sample, dict):
            raise TypeError(f"{path.name} did not load a dict. Got: {type(sample)}")

        # (Optional) minimal sanity checks:
        if "pcd" in sample and sample["pcd"] is not None:
            if sample["pcd"].dim() != 2:
                raise ValueError(f"{path.name}: pcd must be [N,F], got {tuple(sample['pcd'].shape)}")

        if "edge_index" in sample and sample["edge_index"] is not None:
            ei = sample["edge_index"]
            if ei.dtype != torch.long:
                sample["edge_index"] = ei.long()
            if sample["edge_index"].dim() != 2 or sample["edge_index"].shape[0] != 2:
                raise ValueError(f"{path.name}: edge_index must be [2,E], got {tuple(sample['edge_index'].shape)}")

        # Add filename if useful for debugging
        sample["_path"] = str(path)
        return sample
    
def collate_pt_dicts(samples: Sequence[Dict[str, Any]]) -> MultiModalBatch:
    # Images: stack along batch dim if present
    imgs = [s.get("img", None) for s in samples]
    img = None
    if all(x is None for x in imgs):
        img = None
    else:
        # require all samples to have img
        if any(x is None for x in imgs):
            missing = [i for i, x in enumerate(imgs) if x is None]
            raise KeyError(f"Some samples missing 'img': indices {missing}")
        img = torch.stack([x for x in imgs], dim=0)  # [B,C,H,W]

    # Point clouds/node features: concat; create batch vector
    pcds = [s.get("pcd", None) for s in samples]
    if any(x is None for x in pcds):
        missing = [i for i, x in enumerate(pcds) if x is None]
        raise KeyError(f"Some samples missing 'pcd': indices {missing}")
    num_nodes = [p.shape[0] for p in pcds]
    pcd = torch.cat(pcds, dim=0)  # [sumN, F]

    batch_vec = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes)],
        dim=0,
    )  # [sumN]

    # Edges: re-index per sample then concat
    edge_indices = [s.get("edge_index", None) for s in samples]
    if any(ei is None for ei in edge_indices):
        missing = [i for i, ei in enumerate(edge_indices) if ei is None]
        raise KeyError(f"Some samples missing 'edge_index': indices {missing}")

    edge_weights_list = [s.get("edge_weights", None) for s in samples]
    use_edge_weights = not all(w is None for w in edge_weights_list)

    edge_index_cat: List[torch.Tensor] = []
    edge_weights_cat: List[torch.Tensor] = []

    node_offset = 0
    for i, (ei, n) in enumerate(zip(edge_indices, num_nodes)):
        ei = ei.clone()
        ei = ei + node_offset  # shift node ids
        edge_index_cat.append(ei)

        if use_edge_weights:
            w = edge_weights_list[i]
            if w is None:
                raise KeyError(f"Sample {i} missing 'edge_weights' but others have it.")
            edge_weights_cat.append(w)

        node_offset += n

    edge_index = torch.cat(edge_index_cat, dim=1)  # [2, sumE]
    edge_weights = torch.cat(edge_weights_cat, dim=0) if use_edge_weights else None

    # vis: either stack (if same shape) or concat (if per-node). Heuristic:
    vis_list = [s.get("vis", None) for s in samples]
    vis = None
    if not all(v is None for v in vis_list):
        if any(v is None for v in vis_list):
            missing = [i for i, v in enumerate(vis_list) if v is None]
            raise KeyError(f"Some samples missing 'vis': indices {missing}")

        # Heuristic: if first dim matches num_nodes per sample, treat as per-node and concat
        per_node = all(v.dim() >= 1 and v.shape[0] == num_nodes[i] for i, v in enumerate(vis_list))
        if per_node:
            vis = torch.cat(vis_list, dim=0)  # [sumN, ...]
        else:
            # Otherwise assume per-sample and stack
            vis = torch.stack(vis_list, dim=0)  # [B, ...]

    return MultiModalBatch(
        img=img,
        pcd=pcd,
        edge_index=edge_index,
        edge_weights=edge_weights,
        vis=vis,
        batch=batch_vec,
    )

def make_loader(
    root_dir: Union[str, Path],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    pattern: str = "*.pt",
) -> DataLoader:
    ds = PtDictFolderDataset(root_dir=root_dir, pattern=pattern)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_pt_dicts,
        persistent_workers=(num_workers > 0),
    )