import os, glob, torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch

class GraphFileDataset(Dataset):
    def __init__(self, root, pattern="*.pt"):
        self.files = sorted(glob.glob(os.path.join(root, pattern)))
        if not self.files:
            raise FileNotFoundError(f"No files found in {root} matching {pattern}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj = torch.load(self.files[idx], map_location="cpu")

        # Case 1: already a PyG Data object
        if isinstance(obj, Data):
            return obj

        # Case 2: dict -> Data(**dict)
        if isinstance(obj, dict):
            # Expect keys like x, edge_index, edge_attr, y, ...
            return Data(**obj)

        raise TypeError(
            f"Unsupported graph object type in {self.files[idx]}: {type(obj)}. "
            "Store torch_geometric.data.Data or a dict of tensors."
        )

def pyg_collate_graphs(batch):
    # batch: List[Data]
    return Batch.from_data_list(batch)

def make_loader_subgraph(root, batch_size, shuffle=True, num_workers=0, pattern="*.pt", drop_last=True):
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