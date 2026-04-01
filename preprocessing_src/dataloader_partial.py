import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch


class GraphFileDataset(Dataset):
    def __init__(self, root, pattern="*.pt", k=5):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(root, pattern)))
        self.k = k

        # neighbors.csv is assumed to contain graph indices already sorted by distance:
        # row i = [nearest_idx_1, nearest_idx_2, ...]
        self.neighbors = torch.tensor(
            pd.read_csv(os.path.join(root, "neighbors.csv")).to_numpy(),
            dtype=torch.long
        )

        if len(self.files) != self.neighbors.shape[0]:
            raise ValueError(
                f"Mismatch: found {len(self.files)} graph files but "
                f"{self.neighbors.shape[0]} rows in neighbors.csv"
            )

    def __len__(self):
        return len(self.files)

    def _load_graph(self, idx: int):
        return torch.load(self.files[int(idx)], map_location="cpu")

    def _get_knn_indices(self, idx: int):
        # Row already sorted by nearest-first
        knn_idx = self.neighbors[idx]

        # Remove self if it appears in the row
        knn_idx = knn_idx[knn_idx != idx]

        # Keep first k neighbors
        knn_idx = knn_idx[:self.k]

        return knn_idx

    def __getitem__(self, idx):
        graph = self._load_graph(idx)
        knn_idx = self._get_knn_indices(idx)
        neighbor_graphs = [self._load_graph(n_idx) for n_idx in knn_idx]

        return {
            "graph": graph,
            "graph_idx": idx,
            "neighbor_idx": knn_idx,
            "neighbor_graphs": neighbor_graphs,
        }


def pyg_collate_graphs_with_neighbors(batch):
    graphs = [item["graph"] for item in batch]
    graph_batch = Batch.from_data_list(graphs)

    flat_neighbors = []
    for item in batch:
        flat_neighbors.extend(item["neighbor_graphs"])

    neighbor_batch = Batch.from_data_list(flat_neighbors)

    return {
        "graph": graph_batch,
        "graph_idx": torch.tensor([item["graph_idx"] for item in batch], dtype=torch.long),
        "neighbor_idx": torch.stack([item["neighbor_idx"] for item in batch], dim=0),
        "neighbor_graphs": neighbor_batch,
    }


def make_loader_partial(
    root,
    batch_size,
    shuffle=True,
    num_workers=0,
    pattern="*.pt",
    drop_last=True,
    k=5,
):
    ds = GraphFileDataset(root=root, pattern=pattern, k=k)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pyg_collate_graphs_with_neighbors,
        drop_last=drop_last,
    )