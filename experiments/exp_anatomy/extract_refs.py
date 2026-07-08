"""Extract frozen reference-model features for every walk frame (R8 row).
One model per invocation: dinov2s | dinov2b | siglip | qwen2vl."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_anatomy.common import SCENES, scene_files, to_uint8  # noqa: E402

HF = {"dinov2s": "facebook/dinov2-small", "dinov2b": "facebook/dinov2-base",
      "siglip": "google/siglip-base-patch16-224",
      "qwen2vl": "Qwen/Qwen2-VL-2B-Instruct"}


def _encoder(key, dev):
    from transformers import AutoImageProcessor, AutoModel
    if key == "qwen2vl":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(HF[key])
        vis = Qwen2VLForConditionalGeneration.from_pretrained(
            HF[key], torch_dtype=torch.float16).visual.to(dev).eval()
        merge = proc.image_processor.merge_size ** 2

        def enc(imgs):  # list of uint8 HWC
            batch = proc.image_processor(images=imgs, return_tensors="pt")
            with torch.no_grad():
                out = vis(batch["pixel_values"].to(dev).half(),
                          grid_thw=batch["image_grid_thw"].to(dev))
            # out: [total_merged_tokens, D] across the batch -> per-image counts
            thw = batch["image_grid_thw"]
            counts = (thw[:, 0] * thw[:, 1] * thw[:, 2] // merge).tolist()
            feats = torch.split(out, counts)
            return torch.stack([f.mean(0) for f in feats]).float().cpu().numpy()
        return enc
    proc = AutoImageProcessor.from_pretrained(HF[key])
    model = AutoModel.from_pretrained(HF[key]).to(dev).eval()
    if key == "siglip":
        model = model.vision_model

    def enc(imgs):
        batch = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**batch).last_hidden_state       # [B,T,D]
        return h.mean(1).float().cpu().numpy()
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF))
    ap.add_argument("--processed-root", default="data_bartu")
    ap.add_argument("--walks-subdir", default="random_walks")   # branch_walks for A2
    ap.add_argument("--out-root", default="results/anatomy_refs")
    ap.add_argument("--limit", type=int, default=0)   # pilot: cap files/scene
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = _encoder(args.model, dev)
    out_dir = Path(args.out_root) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    for sc in SCENES:
        files = scene_files(args.processed_root, sc, subdir=args.walks_subdir)
        if args.limit:
            files = files[:args.limit]
        if not files:
            print(f"{args.model}/{sc}: no walks under {args.processed_root} "
                  f"({args.walks_subdir}), skipped", flush=True)
            continue
        F_, L, V = [], [], []
        for i in range(0, len(files), args.bs):
            chunk = files[i:i + args.bs]
            samples = [torch.load(f, map_location="cpu", weights_only=False) for f in chunk]
            F_.append(enc([to_uint8(s["img"]) for s in samples]))
            L += [np.asarray(s["loc"], dtype=np.float32).reshape(3) for s in samples]
            V += [np.asarray(s["view_dir"], dtype=np.float32).reshape(3) for s in samples]
        torch.save({"features": np.concatenate(F_), "loc": np.stack(L),
                    "view_dir": np.stack(V), "files": files,
                    "model": args.model, "scene": sc}, out_dir / f"{sc}.pt")
        print(f"{args.model}/{sc}: {len(files)} frames, dim {np.concatenate(F_).shape[1]}",
              flush=True)


if __name__ == "__main__":
    main()
