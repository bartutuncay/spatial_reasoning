import os, glob, torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch

def to_graph_level(x):
    # if x is [d], make it [1, d]
    if torch.is_tensor(x) and x.dim() == 1:
        return x.unsqueeze(0)
    return x

class GraphFileDataset(Dataset):
    def __init__(self, root, pattern="*.pt"):
        self.files = sorted(glob.glob(os.path.join(root, pattern)))
        if not self.files:
            raise FileNotFoundError(f"No files found in {root} matching {pattern}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        if not isinstance(d, dict):
            d = {"latent": d}

        # Example: force graph-level semantics
        if "loc" in d:
            d["loc"] = to_graph_level(d["loc"])
        if "viewdir" in d:
            d["viewdir"] = to_graph_level(d["viewdir"])

        return Data(**d)

def pyg_collate_graphs(batch):
    # batch: List[Data]
    return Batch.from_data_list(batch)

def make_loader_walk(root, batch_size, shuffle=True, num_workers=0, pattern="*.pt", drop_last=True):
    ds = GraphFileDataset(root, pattern=pattern)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pyg_collate_graphs,
        drop_last=drop_last,
    )