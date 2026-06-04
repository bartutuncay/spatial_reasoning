import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_uint8_rgb(img):
    img = img.detach().cpu() if torch.is_tensor(img) else torch.as_tensor(img)

    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.permute(1, 2, 0)
    if img.ndim == 2:
        img = img.unsqueeze(-1)

    img = img.numpy()
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]

    if np.issubdtype(img.dtype, np.floating):
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir.with_name(input_dir.name + "_png")
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(input_dir.rglob("*.pt"))
    for pt_path in pt_files:
        sample = load_pt(pt_path)
        img = sample["img"]

        rel_path = pt_path.relative_to(input_dir).with_suffix(".png")
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_uint8_rgb(img)).save(out_path)
        print(out_path)

    print(f"converted {len(pt_files)} files to {output_dir}")


if __name__ == "__main__":
    main()
