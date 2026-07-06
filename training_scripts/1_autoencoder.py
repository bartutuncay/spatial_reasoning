import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

def _find_repo_root(start: Path) -> Path:
    # Walk up to the ancestor that actually contains models/1_img_enc.py, so the
    # sub-model loader works whether the repo is flat (spatial_jepa/) or nested.
    for p in [start, *start.parents]:
        if (p / "models" / "1_img_enc.py").exists():
            return p
    return start.parents[1]


ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from preprocessing_src.dataloader_autoencoder import (
    RandomWalkAutoencoderDataset,
    collate_random_walk_autoencoder,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


img_enc_mod = load_module("img_enc", ROOT / "models/1_img_enc.py")
img_dec_mod = load_module("img_dec", ROOT / "models/1_img_dec.py")
pcd_enc_mod = load_module("pcd_enc", ROOT / "models/1_pcd_enc.py")
pcd_dec_mod = load_module("pcd_dec", ROOT / "models/1_pcd_dec.py")

Bottleneck = img_enc_mod.Bottleneck
ImgEnc = img_enc_mod.ImgEnc
ImgDec = img_dec_mod.ImgDec
PCDEnc = pcd_enc_mod.PCDEnc
PCDDecoder = pcd_dec_mod.PCDDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20000
MAX_STEPS_PER_EPOCH = 200
CLIP_GRAD_NORM = 1.0
LATENT_DIM = 128
ALIAS = "0522"
SAVE_DIR = ROOT/"1_model" /ALIAS
TAN_FOV_X = torch.tan(torch.tensor(0.5 * 1.4773256165109574)).item()
TAN_FOV_Y = torch.tan(torch.tensor(0.5 * 1.0903791454597398)).item()

RW_LIST = [
    "datasets_processed/anlieferung/rw_translation_anlieferung",
    "datasets_processed/break_room/rw_translation_breakroom",
    "datasets_processed/relief/rw_translation_relief",
    "datasets_processed/terrains/rw_translation_terrains",
    "datasets_processed/office/rw_translation_office",
    "datasets_processed/habitat/train/00000-kfPV7w3FaU5/random_walks",
    "datasets_processed/habitat/train/00001-UVdNNRcVyV1/random_walks",
    "datasets_processed/habitat/train/00002-FxCkHAfgh7A/random_walks",
    "datasets_processed/habitat/train/00003-NtVbfPCkBFy/random_walks",
    "datasets_processed/habitat/train/00004-VqCaAuuoeWk/random_walks",
    "datasets_processed/habitat/train/00005-yPKGKBCyYx8/random_walks",
    "datasets_processed/habitat/train/00006-HkseAnWCgqk/random_walks",
    "datasets_processed/habitat/train/00007-UQuchpekHRJ/random_walks",
    "datasets_processed/habitat/train/00008-VYnUX657cVo/random_walks",
    "datasets_processed/habitat/train/00009-vLpv2VX547B/random_walks",
]


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


def normalize_edge_weights(edge_weights: torch.Tensor) -> torch.Tensor:
    scale = edge_weights.mean().clamp_min(1e-6)
    return torch.exp(-(edge_weights ** 2) / (2 * scale ** 2))


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def depth_loss(pcd_pred: torch.Tensor, depth_img: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if depth_img.dim() == 2:
        depth_img = depth_img.unsqueeze(0)

    batch_size, height, width = depth_img.shape
    dirs = F.normalize(pcd_pred[:, 1:4], dim=-1, eps=eps)
    radial_depth = pcd_pred[:, 0]
    forward_depth = radial_depth * dirs[:, 0]

    valid = (torch.linalg.norm(pcd_pred[:, 1:4], dim=-1) > eps) & (forward_depth > eps)
    if batch_size == 0 or not valid.any():
        return pcd_pred.new_zeros(())

    points_per_graph = pcd_pred.size(0) // batch_size
    point_batch = torch.arange(batch_size, device=pcd_pred.device).repeat_interleave(points_per_graph)
    point_batch = point_batch[: pcd_pred.size(0)]

    xn = -dirs[:, 1] / dirs[:, 0].clamp_min(eps)
    yn = dirs[:, 2] / dirs[:, 0].clamp_min(eps)
    u = ((xn / TAN_FOV_X) * 0.5 + 0.5) * (width - 1)
    v = (0.5 - (yn / TAN_FOV_Y) * 0.5) * (height - 1)
    ui = torch.round(u).long()
    vi = torch.round(v).long()

    valid = valid & torch.isfinite(u) & torch.isfinite(v)
    valid = valid & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not valid.any():
        return pcd_pred.new_zeros(())

    gt_depth = depth_img[point_batch[valid], vi[valid], ui[valid]]
    gt_valid = torch.isfinite(gt_depth) & (gt_depth > 0)
    if not gt_valid.any():
        return pcd_pred.new_zeros(())

    return F.smooth_l1_loss(forward_depth[valid][gt_valid], gt_depth[gt_valid])


def depth_smoothness_loss(pcd_pred: torch.Tensor, edge_index: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return pcd_pred.new_zeros(())
    src, dst = edge_index
    log_depth = torch.log(pcd_pred[:, 0].clamp_min(eps))
    return F.smooth_l1_loss(log_depth[src], log_depth[dst])


def compute_loss(out: dict, img: torch.Tensor, depth_img: torch.Tensor):
    loss_terms = {
        "img_img": F.l1_loss(out["img_img"], img),
        "pcd_img": F.l1_loss(out["pcd_img"], img),
        "pcd_depth": depth_loss(out["pcd_pcd_pred"], depth_img),
        "smooth": depth_smoothness_loss(out["pcd_pcd_pred"], out["pcd_pcd_ei"]),
        "latent": F.mse_loss(out["mu_img"], out["mu_pcd"]),
        "align_pcd": F.mse_loss(out["mu_pcd_p"], out["mu_pcd"].detach()),
        "kl_img": kl_divergence(out["mu_img"], out["logvar_img"]),
        "kl_pcd": kl_divergence(out["mu_pcd"], out["logvar_pcd"]),
    }
    loss = (
        3.0 * (loss_terms["img_img"] + loss_terms["pcd_img"])
        + 0.5 * loss_terms["pcd_depth"]
        + loss_terms["align_pcd"]
        + 0.3 * loss_terms["latent"]
        + 0.05 * (loss_terms["kl_img"] + loss_terms["kl_pcd"])
        + 0.02 * loss_terms["smooth"]
    )
    return loss, {key: value.item() for key, value in loss_terms.items()}


def build_loader():
    dataset = ConcatDataset([RandomWalkAutoencoderDataset(path) for path in RW_LIST])
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_random_walk_autoencoder,
        persistent_workers=True,
    )


def run_epoch(model, loader, optimizer, scheduler):
    model.train(True)
    loss_sums = None
    steps = 0

    for step, batch in enumerate(loader, start=1):
        if step > MAX_STEPS_PER_EPOCH:
            break

        batch = batch.to(DEVICE)
        img = batch.img.permute(0, 3, 1, 2).to(torch.float32)
        pcd = batch.pcd.to(torch.float32)
        depth_img = batch.depth.to(torch.float32)
        ew_points = normalize_edge_weights(batch.edge_weights.to(torch.float32))

        optimizer.zero_grad(set_to_none=True)
        out = model(img, pcd, batch.batch, batch.edge_index, ew_points, batch.ei_camera, batch.ea_camera.to(torch.float32))
        loss, loss_dict = compute_loss(out, img, depth_img)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        if loss_sums is None:
            loss_sums = {key: 0.0 for key in loss_dict}
        for key, value in loss_dict.items():
            loss_sums[key] += value
        steps += 1

    if steps == 0:
        return {key: 0.0 for key in ("img_img", "pcd_img", "pcd_depth", "smooth", "latent", "align_pcd", "kl_img", "kl_pcd")}

    scheduler.step()
    return {key: value / steps for key, value in loss_sums.items()}


def main():
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    torch.set_default_dtype(torch.float32)
    model = ImageGraphVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 0)
    loader = build_loader()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SAVE_DIR / "losses.csv"

    print("starting training")
    print(ALIAS)

    for epoch in range(EPOCHS):
        loss_dict = run_epoch(model, loader, optimizer, scheduler)
        row = {"epoch": epoch, **loss_dict}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

        if epoch % 10 == 0:
            print(f"epoch {epoch}", loss_dict)
        if epoch % 200 == 0:
            torch.save(model.state_dict(), SAVE_DIR / f"model_weights_{epoch}.pt")

    torch.save(model.state_dict(), SAVE_DIR / "model_weights.pt")
    print("completed")


if __name__ == "__main__":
    main()
