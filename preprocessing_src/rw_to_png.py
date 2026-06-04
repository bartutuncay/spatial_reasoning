## Helper function to extract images from saved .pt random walk sequences

from pathlib import Path

import cv2
import torch

from spatial_reasoning.preprocessing_src.dataloader_autoencoder import make_loader


device = torch.device("cpu")
torch.set_default_dtype(torch.float32)

random_walk_dir = Path("datasets_processed/test_terrace/random_walks")
png_dir = random_walk_dir.parent / "rw_images"
png_dir.mkdir(parents=True, exist_ok=True)

loader = make_loader(random_walk_dir, batch_size=1, shuffle=False, num_workers=4)

for batch in loader:
    img = batch["img"][0].cpu().numpy()

    if img.max() <= 1.0:
        img = img * 255.0

    img = img.clip(0, 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    name = Path(batch.path[0]).with_suffix(".png").name
    cv2.imwrite(str(png_dir / name), img)
