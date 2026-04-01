class SinglePointDataset(Dataset):
    def __init__(self, root: str, index_file: str = "index.pt"):
        self.root = Path(root)
        idx = torch.load(self.root / index_file, map_location="cpu")
        self.files = idx["files"]


    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # Keep everything on CPU; DataLoader workers do the read/transform
        sample = torch.load(self.root / self.files[i], map_location="cpu")

        # Image: uint8 HWC -> float32 CHW in [0,1]
        img = sample["img_u8"]  # [H,W,C], uint8
        img = img.permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)

        # Others already typed/contiguous ideally
        return {
            "img": img,  # [3,H,W]
            "pcd": sample["pcd"],  # [N,6], float32
            "edge_index": sample["edge_index"],  # [2,E], int64
            "edge_weights": sample["edge_weights"],  # [E], float32
            "vis": sample["vis"],  # [2], int/float
        }


    def collate_mm_graph(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        # Stack fixed-size
        imgs = torch.stack([b["img"] for b in batch], dim=0)  # [B,3,H,W]
        vis = torch.stack([b["vis"] for b in batch], dim=0)   # [B,2]

        # Concatenate variable-size node features
        pcd_list = [b["pcd"] for b in batch]                  # each [N,6]
        node_counts = torch.tensor([p.shape[0] for p in pcd_list], dtype=torch.long)
        pcd = torch.cat(pcd_list, dim=0)                      # [sumN,6]

        # Build node->sample mapping (useful for pooling)
        node_batch = torch.repeat_interleave(torch.arange(len(batch), dtype=torch.long), node_counts)

        # Concatenate edges with offsets
        edge_index_list = []
        edge_weight_list = []
        offset = 0
        for b in batch:
            ei = b["edge_index"]          # [2,E]
            ew = b["edge_weights"]        # [E]
            edge_index_list.append(ei + offset)
            edge_weight_list.append(ew)
            offset += b["pcd"].shape[0]

        edge_index = torch.cat(edge_index_list, dim=1)        # [2,sumE]
        edge_weights = torch.cat(edge_weight_list, dim=0)     # [sumE]

        return {
            "img": imgs,
            "vis": vis,
            "pcd": pcd,
            "edge_index": edge_index,
            "edge_weights": edge_weights,
            "node_batch": node_batch,
            "node_counts": node_counts,
        }

class SingleFileDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path, map_location="cpu")
        self.img_u8 = self.data["img_u8"]
        self.pcd = self.data["pcd"]
        self.pcd_ptr = self.data["pcd_ptr"]
        self.edge_index = self.data["edge_index"]
        self.edge_weights = self.data["edge_weights"]
        self.edge_ptr = self.data["edge_ptr"]
        self.vis = self.data["vis"]

        self.M = self.vis.shape[0]

    def __len__(self) -> int:
        return self.M

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # Image: uint8 HWC -> float32 CHW
        img = self.img_u8[i].permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)

        n0, n1 = self.pcd_ptr[i].item(), self.pcd_ptr[i + 1].item()
        pcd_i = self.pcd[n0:n1]  # [N,6]

        e0, e1 = self.edge_ptr[i].item(), self.edge_ptr[i + 1].item()
        ei_global = self.edge_index[:, e0:e1]       # [2,E] global node indices
        ew = self.edge_weights[e0:e1]               # [E]

        # Convert global indices back to local for this sample (optional but convenient)
        ei_local = ei_global - n0

        return {
            "img": img,
            "pcd": pcd_i,
            "edge_index": ei_local,
            "edge_weights": ew,
            "vis": self.vis[i],
        }