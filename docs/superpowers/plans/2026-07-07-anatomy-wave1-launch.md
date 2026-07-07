# Capability-Anatomy Wave-1 Launch (First Volley) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the first capability-anatomy jobs running on Euler today: Replica download, reference-model weights, reference feature extraction, and two new probes (C7 place recognition, C3 depth) swept over the five existing objectives.

**Architecture:** New `experiments/exp_anatomy/` package mirrors `exp_jepa` conventions: self-contained workers (pretrain→probe→eval in one job), result.json LEDGER contract, PILOT→SHAKEDOWN→BULK tiers, sbatch via exported env vars. Reference rows use precomputed frozen features (extract once, probe many times); trained arms re-pretrain per job via `fewshot._pretrain_pool` (established trade-off).

**Tech Stack:** torch 2.4.1+cu121 conda env on scratch, PyG, transformers (new), SLURM `gpuhe.4h`/`normal.4h`.

## Global Constraints

- Dataset/model downloads go to `/cluster/scratch/aleonel/spatial_jepa`, NEVER the laptop; downloads and heavy compute run as SLURM batch jobs, never on the login node.
- Result contract per job: `result.json` with `{status, verdict(EXISTS|WEAK|DEAD|UNSCORED), channel, primary, metrics, notes, id, date, gpu_h}` (see `fewshot.py:128-141`).
- Commit messages: plain, no Claude/Co-Authored-By/session trailers.
- `to_euler_sync.sh` excludes are anchored; new scratch-generated top-level dirs must be added to excludes before any `--delete` sync (add `/hf_cache/` and `/datasets_replica/`).
- Scenes: `office,pipes,break_room,relief,hospital`; walks at `$SJEPA_PROCESSED_ROOT/<scene>/random_walks/*.pt`, sample keys: `img [H,W,3] float 0-255`, `depth [H,W]`, `loc [3]`, `view_dir [3]`; seed-split by `rw_(\d+)_` prefix, last seed = eval (`locate._seed_split`).
- MASt3R reference row is deferred to Plan B (needs vendored repo, not pip-installable); today's refs: DINOv2-S/B, SigLIP-base, Qwen2-VL-2B vision tower — all via `transformers` with `HF_HOME` on scratch.

---

### Task 1: Replica download job (fire immediately)

**Files:**
- Create: `experiments/data/provision_replica.sbatch`
- Modify: `to_euler_sync.sh` (add excludes)

**Interfaces:**
- Produces: `/cluster/scratch/aleonel/spatial_jepa/datasets_replica/replica_v1/<scene>/mesh.ply` (18 scenes) + `.provision_replica.done` sentinel. Plan C consumes.

- [ ] **Step 1: Add sync excludes** — in `to_euler_sync.sh` after the `/datasets_processed/` exclude add:

```bash
  --exclude='/datasets_replica/' --exclude='/hf_cache/' --exclude='/torch_hub/' \
```

- [ ] **Step 2: Write `experiments/data/provision_replica.sbatch`**

```bash
#!/bin/bash
# Download Replica v1 (~100GB, 17 tar parts) to scratch. CPU-only, idempotent.
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=03:55:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/provision_replica_%j.txt
set -eo pipefail
DEST=datasets_replica
DONE="$DEST/.provision_replica.done"
[ -f "$DONE" ] && { echo "already provisioned"; exit 0; }
mkdir -p "$DEST"
cd "$DEST"
[ -d Replica-Dataset ] || git clone --depth 1 https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
# download.sh wgets the release tar parts, concatenates, and extracts into $1
bash download.sh ../replica_v1
cd ..
ls replica_v1 | head -20
N=$(ls -d replica_v1/*/ | wc -l)
echo "scenes extracted: $N"
[ "$N" -ge 18 ] || { echo "FAIL: expected >=18 scene dirs"; exit 1; }
rm -f replica_v1_0.tar.gz.part* 2>/dev/null || true
touch ".provision_replica.done"
echo OK
```

- [ ] **Step 3: Sync + submit**

Run: `./to_euler_sync.sh && ssh euler 'cd /cluster/scratch/aleonel/spatial_jepa && sbatch experiments/data/provision_replica.sbatch'`
Expected: `Submitted batch job <id>`

- [ ] **Step 4: Commit** — `git add -A && git commit -m "Add Replica provisioning job (wave 1)"`

### Task 2: Reference weights prefetch job (fire immediately)

**Files:**
- Create: `experiments/data/fetch_ref_weights.sbatch`
- Modify: `experiments/env/requirements.txt` (document new deps)

**Interfaces:**
- Produces: `transformers` installed in the scratch conda env; HF weights cached under `ROOT/hf_cache` for `facebook/dinov2-small`, `facebook/dinov2-base`, `google/siglip-base-patch16-224`, `Qwen/Qwen2-VL-2B-Instruct`. Task 6 consumes (afterok).

- [ ] **Step 1: Append to `experiments/env/requirements.txt`:**

```
# Reference-model extraction (wave 1, installed by fetch_ref_weights.sbatch)
transformers>=4.45,<5
accelerate
```

- [ ] **Step 2: Write `experiments/data/fetch_ref_weights.sbatch`**

```bash
#!/bin/bash
# Install transformers into the scratch env and prefetch reference weights. CPU.
#SBATCH --partition=normal.4h
#SBATCH --account=ls_helbi
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/fetch_refs_%j.txt
set -eo pipefail; set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
export HF_HOME=/cluster/scratch/aleonel/spatial_jepa/hf_cache
pip install --no-input "transformers>=4.45,<5" accelerate
python - <<'PY'
import os
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
for m in ["facebook/dinov2-small", "facebook/dinov2-base",
          "google/siglip-base-patch16-224"]:
    AutoImageProcessor.from_pretrained(m); AutoModel.from_pretrained(m)
    print("cached", m, flush=True)
AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("cached qwen2-vl-2b", flush=True)
PY
touch hf_cache/.fetch_refs.done
echo OK
```

- [ ] **Step 3: Sync + submit** (same command pattern as Task 1 Step 3). Record job id — Task 6's extraction jobs depend on it.
- [ ] **Step 4: Commit** — `git commit -m "Add reference-weights prefetch job (wave 1)"`

### Task 3: `exp_anatomy` shared utilities (TDD)

**Files:**
- Create: `experiments/exp_anatomy/__init__.py` (empty), `experiments/exp_anatomy/common.py`
- Test: `tests/test_anatomy_common.py`

**Interfaces:**
- Produces (Tasks 4/5/6 consume):
  - `seed_split(files: list[str]) -> tuple[list[str], list[str]]` — train/eval by `rw_<seed>_` prefix, last seed = eval.
  - `depth_grid(depth: Tensor[H,W], g: int = 16) -> tuple[Tensor[g*g], Tensor[g*g]]` — (log-depth grid, valid mask); pools only over valid (>0) pixels.
  - `to_uint8(img: ndarray|Tensor[H,W,3]) -> ndarray[H,W,3] uint8` — handles 0–1 and 0–255 float inputs.
  - `encode_arm_latents(vae, files, dev, bs=16) -> tuple[ndarray[N,D], ndarray[N,3], ndarray[N,3]]` — frozen `img_enc` mu + loc + view_dir via `locate._QueryDS`.
  - `finish(out_dir: str, result: dict, t0: float) -> None` — stamps id/date/gpu_h, writes result.json, prints summary line (verbatim behavior of `fewshot.main` tail).

- [ ] **Step 1: Write failing tests `tests/test_anatomy_common.py`**

```python
import numpy as np
import torch
from experiments.exp_anatomy.common import seed_split, depth_grid, to_uint8

def test_seed_split_last_seed_eval():
    files = [f"x/rw_{s}_{i:03d}.pt" for s in (0, 1, 7) for i in range(3)]
    tr, ev = seed_split(files)
    assert all("rw_7_" in f for f in ev) and len(ev) == 3
    assert len(tr) == 6 and not any("rw_7_" in f for f in tr)

def test_depth_grid_masks_invalid():
    d = torch.zeros(64, 64); d[:32] = 2.0          # bottom half invalid (0)
    g, m = depth_grid(d, g=4)
    assert g.shape == (16,) and m.shape == (16,)
    assert m[:8].all() and not m[8:].any()
    assert torch.allclose(g[:8], torch.log(torch.tensor(2.0)).expand(8))

def test_to_uint8_handles_both_ranges():
    a = to_uint8(np.ones((4, 4, 3), dtype=np.float32) * 0.5)   # 0-1 range
    b = to_uint8(np.ones((4, 4, 3), dtype=np.float32) * 200.0) # 0-255 range
    assert a.dtype == np.uint8 and 126 <= a[0, 0, 0] <= 129
    assert b.dtype == np.uint8 and b[0, 0, 0] == 200
```

- [ ] **Step 2: Run** `python -m pytest tests/test_anatomy_common.py -q` — expect FAIL (module missing).
- [ ] **Step 3: Implement `experiments/exp_anatomy/common.py`**

```python
"""Shared utilities for capability-anatomy probes (wave 1)."""
import glob
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_jepa.locate import _QueryDS  # noqa: E402

SCENES = ["office", "pipes", "break_room", "relief", "hospital"]


def seed_split(files):
    seeds = sorted({int(re.search(r"rw_(\d+)_", Path(f).name).group(1)) for f in files})
    ev_seed = seeds[-1]
    tr = [f for f in files if not Path(f).name.startswith(f"rw_{ev_seed}_")]
    ev = [f for f in files if Path(f).name.startswith(f"rw_{ev_seed}_")]
    return tr, ev


def scene_files(root, scene):
    return sorted(glob.glob(str(Path(root) / scene / "random_walks" / "*.pt")))


def depth_grid(depth, g=16):
    """Downsample a [H,W] depth map to a g*g log-depth grid, masking cells
    with no valid (>0) pixels. Pools the SUM of valid depths / COUNT of valid
    pixels so invalid pixels never bias a cell."""
    d = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    valid = (d > 0).float()
    s = F.adaptive_avg_pool2d(d * valid, g)
    c = F.adaptive_avg_pool2d(valid, g)
    mask = (c > 0).reshape(-1)
    grid = torch.where(c > 0, s / c.clamp(min=1e-8), torch.ones_like(s))
    return torch.log(grid.clamp(min=1e-6)).reshape(-1), mask


def to_uint8(img):
    a = np.asarray(img, dtype=np.float32)
    if a.max() <= 1.5:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def encode_arm_latents(vae, files, dev, bs=16):
    Z, L, V = [], [], []
    with torch.no_grad():
        for img, loc, vd in DataLoader(_QueryDS(files), batch_size=bs):
            _, mu, _ = vae.img_enc(img.to(dev))
            Z.append(mu.cpu().numpy()); L.append(loc.numpy()); V.append(vd.numpy())
    return np.concatenate(Z), np.concatenate(L), np.concatenate(V)


def finish(out_dir, result, t0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    result["id"] = out.name
    result["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result.setdefault("gpu_h", round((time.time() - t0) / 3600.0, 5))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ("status", "verdict", "primary", "notes")}))
```

- [ ] **Step 4: Run** `python -m pytest tests/test_anatomy_common.py -q` — expect 3 passed. Run full suite `python -m pytest tests/ -q` — no regressions.
- [ ] **Step 5: Commit** — `git commit -m "Add exp_anatomy shared utilities with tests"`

### Task 4: C7 place-recognition probe

**Files:**
- Create: `experiments/exp_anatomy/placerec.py`, `experiments/exp_anatomy/probe.sbatch`

**Interfaces:**
- Consumes: `common.seed_split/scene_files/encode_arm_latents/finish`, `fewshot._pretrain_pool`, `locate._load_module`.
- Produces: `results/anatomy/<id>/result.json` with `channel="placerec:<row>"`, `primary=top1 acc`; `--features-dir` mode consumed by Task 6's ref jobs.

- [ ] **Step 1: Write `experiments/exp_anatomy/placerec.py`**

```python
"""C7 place recognition: which scene is this view from? Frozen encoder + linear
probe over 5 ETH3D scenes; floor = majority class; also NN retrieval recall@1.
Rows: --objective (self-contained pretrain) or --features-dir (frozen refs)."""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_anatomy.common import (  # noqa: E402
    SCENES, encode_arm_latents, finish, scene_files, seed_split,
)
from experiments.exp_jepa.fewshot import PRETRAIN_STEPS, _pretrain_pool  # noqa: E402
from experiments.exp_jepa.locate import _load_module  # noqa: E402


def _gather(args, dev):
    """Per split: (features [N,D], scene labels [N]) for train and eval."""
    Ztr, ytr, Zev, yev = [], [], [], []
    if args.features_dir:                       # reference row: precomputed
        for si, sc in enumerate(SCENES):
            d = torch.load(Path(args.features_dir) / f"{sc}.pt", weights_only=False)
            tr_idx, ev_idx = seed_split(d["files"])
            pos = {f: i for i, f in enumerate(d["files"])}
            for bucket, idxs in ((Ztr, tr_idx), (Zev, ev_idx)):
                take = [pos[f] for f in idxs]
                bucket.append(np.asarray(d["features"])[take])
                (ytr if bucket is Ztr else yev).append(np.full(len(take), si))
    else:                                       # trained arm: pretrain then encode
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        root = Path(args.processed_root)
        if args.objective != "scratch":
            _pretrain_pool(vae, autoenc, [root / s / "random_walks" for s in SCENES],
                           args.objective, PRETRAIN_STEPS.get(args.tier, 200), dev)
        vae.img_enc.eval()
        for si, sc in enumerate(SCENES):
            tr_f, ev_f = seed_split(scene_files(root, sc))
            if args.tier == "pilot":
                tr_f, ev_f = tr_f[:8], ev_f[:8]
            for bucket, ybucket, files in ((Ztr, ytr, tr_f), (Zev, yev, ev_f)):
                z, _, _ = encode_arm_latents(vae, files, dev)
                bucket.append(z); ybucket.append(np.full(len(files), si))
    return (np.concatenate(Ztr), np.concatenate(ytr),
            np.concatenate(Zev), np.concatenate(yev))


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ztr, ytr, Zev, yev = _gather(args, dev)
    D = Ztr.shape[1]

    head = torch.nn.Linear(D, len(SCENES)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    Xtr = torch.as_tensor(Ztr, dtype=torch.float32, device=dev)
    Ttr = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    for _ in range(args.head_steps):
        perm = torch.randperm(len(Xtr), device=dev)[:256]
        loss = F.cross_entropy(head(Xtr[perm]), Ttr[perm])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = head(torch.as_tensor(Zev, dtype=torch.float32, device=dev)).argmax(1).cpu().numpy()
    acc = float((pred == yev).mean())
    floor = float(max(np.bincount(yev, minlength=len(SCENES))) / len(yev))

    # NN retrieval recall@1 (cosine, train bank) — probe-free discriminability
    A = Ztr / (np.linalg.norm(Ztr, axis=1, keepdims=True) + 1e-8)
    B = Zev / (np.linalg.norm(Zev, axis=1, keepdims=True) + 1e-8)
    nn_acc = float((ytr[np.argmax(B @ A.T, axis=1)] == yev).mean())

    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    return {"status": "ok",
            "verdict": "EXISTS" if acc > floor + 0.05 else "WEAK",
            "channel": f"placerec:{row}", "primary": round(acc, 4),
            "metrics": {"top1_acc": acc, "majority_floor": floor,
                        "nn_recall1": nn_acc, "n_train": int(len(ytr)),
                        "n_eval": int(len(yev)), "dim": int(D), "row": row},
            "notes": f"placerec {row}: acc {acc:.3f} (floor {floor:.3f}, NN@1 {nn_acc:.3f})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--tier", default="shakedown")
    ap.add_argument("--head-steps", type=int, default=300)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--processed-root",
                    default=os.environ.get("SJEPA_PROCESSED_ROOT", "data_bartu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    try:
        result = run(args)
    except Exception as e:
        result = {"status": "failed", "verdict": "DEAD",
                  "notes": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-2500:]}
    finish(args.out, result, t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write generic `experiments/exp_anatomy/probe.sbatch`**

```bash
#!/bin/bash
# Generic anatomy probe job. Config via --export (MODULE/ARGS/ID/RESDIR).
#SBATCH --partition=gpuhe.4h
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --account=ls_helbi
#SBATCH --time=00:50:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/cluster/scratch/aleonel/spatial_jepa
#SBATCH --output=logs/an_%x_%j.txt
set -eo pipefail; set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /cluster/scratch/aleonel/spatial_jepa/env
export SJEPA_PROCESSED_ROOT="${SJEPA_PROCESSED_ROOT:-data_bartu}"
export HF_HOME=/cluster/scratch/aleonel/spatial_jepa/hf_cache
python -m "experiments.exp_anatomy.${MODULE}" ${ARGS} \
  --out "results/${RESDIR:-anatomy}/${ID}"
```

- [ ] **Step 3: Local CPU pilot** — `SJEPA_PROCESSED_ROOT=<local walks root> python -m experiments.exp_anatomy.placerec --objective scratch --tier pilot --head-steps 20 --out /tmp/pilot_pr` (use the local 2-scene subset if present; otherwise run this pilot on Euler as a single job). Expected: prints one JSON line, `status: ok`, `primary` ∈ [0,1], result.json exists.
- [ ] **Step 4: Commit** — `git commit -m "Add C7 place-recognition probe + generic probe sbatch"`

### Task 5: C3 depth probe

**Files:**
- Create: `experiments/exp_anatomy/depthprobe.py`

**Interfaces:**
- Consumes: `common.depth_grid/seed_split/scene_files/finish`, `fewshot._pretrain_pool`.
- Produces: `results/anatomy/<id>/result.json`, `channel="depthprobe:<row>"`, `primary=AbsRel` (lower better), `metrics.delta125`.

- [ ] **Step 1: Write `experiments/exp_anatomy/depthprobe.py`**

```python
"""C3 metric-depth probe: linear head from the frozen global latent to a 16x16
log-depth grid. Seed-split within every scene; floor = train-mean grid.
Rows: --objective (self-contained pretrain) or --features-dir (frozen refs)."""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp_anatomy.common import (  # noqa: E402
    SCENES, depth_grid, finish, scene_files, seed_split,
)
from experiments.exp_jepa.fewshot import PRETRAIN_STEPS, _pretrain_pool  # noqa: E402
from experiments.exp_jepa.locate import _load_module  # noqa: E402

G = 16  # grid side


def _load_features(features_dir):
    Z, files = [], []
    for sc in SCENES:
        d = torch.load(Path(features_dir) / f"{sc}.pt", weights_only=False)
        Z.append(np.asarray(d["features"])); files += list(d["files"])
    return np.concatenate(Z), files


def _depth_targets(files):
    Y, M = [], []
    for f in files:
        s = torch.load(f, map_location="cpu", weights_only=False)
        g, m = depth_grid(torch.as_tensor(s["depth"]), G)
        Y.append(g); M.append(m)
    return torch.stack(Y), torch.stack(M)


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)

    tr_files, ev_files = [], []
    for sc in SCENES:
        tr, ev = seed_split(scene_files(root, sc))
        if args.tier == "pilot":
            tr, ev = tr[:8], ev[:8]
        tr_files += tr; ev_files += ev

    if args.features_dir:
        Z, files = _load_features(args.features_dir)
        pos = {f: i for i, f in enumerate(files)}
        Ztr = Z[[pos[f] for f in tr_files]]; Zev = Z[[pos[f] for f in ev_files]]
    else:
        from experiments.exp_anatomy.common import encode_arm_latents
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        if args.objective != "scratch":
            _pretrain_pool(vae, autoenc, [root / s / "random_walks" for s in SCENES],
                           args.objective, PRETRAIN_STEPS.get(args.tier, 200), dev)
        vae.img_enc.eval()
        Ztr, _, _ = encode_arm_latents(vae, tr_files, dev)
        Zev, _, _ = encode_arm_latents(vae, ev_files, dev)

    Ytr, Mtr = _depth_targets(tr_files)
    Yev, Mev = _depth_targets(ev_files)

    head = torch.nn.Linear(Ztr.shape[1], G * G).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    X = torch.as_tensor(Ztr, dtype=torch.float32, device=dev)
    Y = Ytr.to(dev); M = Mtr.to(dev)
    for _ in range(args.head_steps):
        perm = torch.randperm(len(X), device=dev)[:128]
        err = (head(X[perm]) - Y[perm]) ** 2
        loss = (err * M[perm]).sum() / M[perm].sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()

    def absrel_d125(pred_log, y_log, m):
        p, y = torch.exp(pred_log[m]), torch.exp(y_log[m])
        absrel = float(((p - y).abs() / y).mean())
        d125 = float((torch.maximum(p / y, y / p) < 1.25).float().mean())
        return absrel, d125

    with torch.no_grad():
        Pev = head(torch.as_tensor(Zev, dtype=torch.float32, device=dev)).cpu()
    absrel, d125 = absrel_d125(Pev, Yev, Mev)
    mean_grid = (Ytr * Mtr).sum(0) / Mtr.sum(0).clamp(min=1)     # train-mean floor
    fl_absrel, fl_d125 = absrel_d125(mean_grid.expand_as(Yev), Yev, Mev)

    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    return {"status": "ok",
            "verdict": "EXISTS" if absrel < fl_absrel - 0.01 else "WEAK",
            "channel": f"depthprobe:{row}", "primary": round(absrel, 4),
            "metrics": {"absrel": absrel, "delta125": d125,
                        "floor_absrel": fl_absrel, "floor_delta125": fl_d125,
                        "n_train": len(tr_files), "n_eval": len(ev_files), "row": row},
            "notes": f"depth {row}: AbsRel {absrel:.3f} (floor {fl_absrel:.3f}), "
                     f"d1.25 {d125:.3f} (floor {fl_d125:.3f})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--tier", default="shakedown")
    ap.add_argument("--head-steps", type=int, default=400)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--processed-root",
                    default=os.environ.get("SJEPA_PROCESSED_ROOT", "data_bartu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    try:
        result = run(args)
    except Exception as e:
        result = {"status": "failed", "verdict": "DEAD",
                  "notes": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-2500:]}
    finish(args.out, result, t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Pilot** (same pattern as Task 4 Step 3, module `depthprobe`). Expected: JSON line with `absrel` and floor both finite; probe should beat or tie floor even at pilot tier for `recon` (depth was in its training signal historically — sanity anchor).
- [ ] **Step 3: Commit** — `git commit -m "Add C3 depth probe"`

### Task 6: Reference extraction + wave-1 submission wiring

**Files:**
- Create: `experiments/exp_anatomy/extract_refs.py`, `experiments/exp_anatomy/extract_refs.sbatch`, `experiments/exp_anatomy/submit_wave1.sh`

**Interfaces:**
- Consumes: HF cache from Task 2; `common.to_uint8/scene_files/SCENES`.
- Produces: `results/anatomy_refs/<model_key>/<scene>.pt` with `{features [N,D] fp32, loc [N,3], view_dir [N,3], files [N], model, scene}`; consumed by Tasks 4/5 `--features-dir`.

- [ ] **Step 1: Write `experiments/exp_anatomy/extract_refs.py`**

```python
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

        def enc(imgs):  # list of uint8 HWC
            batch = proc.image_processor(images=imgs, return_tensors="pt")
            with torch.no_grad():
                out = vis(batch["pixel_values"].to(dev).half(),
                          grid_thw=batch["image_grid_thw"].to(dev))
            # out: [total_tokens, D] over the whole batch -> split by per-image counts
            counts = (batch["image_grid_thw"][:, 1] * batch["image_grid_thw"][:, 2] // 4)
            feats = torch.split(out, counts.tolist())
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
    ap.add_argument("--out-root", default="results/anatomy_refs")
    ap.add_argument("--limit", type=int, default=0)   # pilot: cap files/scene
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = _encoder(args.model, dev)
    out_dir = Path(args.out_root) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    for sc in SCENES:
        files = scene_files(args.processed_root, sc)
        if args.limit:
            files = files[:args.limit]
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
        print(f"{args.model}/{sc}: {len(files)} frames", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `experiments/exp_anatomy/extract_refs.sbatch`** — copy of `probe.sbatch` but `--time=01:30:00` and payload `python -m experiments.exp_anatomy.extract_refs --model "${MODEL}" --processed-root "${SJEPA_PROCESSED_ROOT}"`.
- [ ] **Step 3: Write `experiments/exp_anatomy/submit_wave1.sh`**

```bash
#!/usr/bin/env bash
# Submit the wave-1 volley. Usage: submit_wave1.sh [FETCH_JOBID]
# If FETCH_JOBID given, ref-extraction waits on it (afterok) and ref-probes
# wait on the extractions.
set -euo pipefail
cd /cluster/scratch/aleonel/spatial_jepa
FETCH=${1:-}
DEP=""; [ -n "$FETCH" ] && DEP="--dependency=afterok:${FETCH}"

# trained-arm probes: 5 objectives x 2 seeds x 2 probes (shakedown)
for MOD in placerec depthprobe; do
  for OBJ in scratch recon symalign contrastive jepa; do
    for SEED in 0 1; do
      ID="an_${MOD}_${OBJ}_s${SEED}"
      sbatch --job-name "$ID" --export=ALL,MODULE=$MOD,ID=$ID,RESDIR=anatomy,ARGS="--objective ${OBJ} --seed ${SEED} --tier shakedown" \
        experiments/exp_anatomy/probe.sbatch
    done
  done
done

# reference extraction (after weights land), then ref probes (after extraction)
for MODEL in dinov2s dinov2b siglip qwen2vl; do
  EX=$(sbatch --parsable $DEP --job-name "ex_${MODEL}" --export=ALL,MODEL=$MODEL \
       experiments/exp_anatomy/extract_refs.sbatch)
  for MOD in placerec depthprobe; do
    ID="an_${MOD}_ref_${MODEL}"
    sbatch --dependency=afterok:${EX} --job-name "$ID" \
      --export=ALL,MODULE=$MOD,ID=$ID,RESDIR=anatomy,ARGS="--features-dir results/anatomy_refs/${MODEL} --tier shakedown" \
      experiments/exp_anatomy/probe.sbatch
  done
done
squeue -u aleonel -o "%.10i %.28j %.9P %.8T %.10M %R" | head -50
```

- [ ] **Step 4: Sync, pilot on Euler, then launch.** `./to_euler_sync.sh`; single pilot job first (`MODULE=placerec, ARGS="--objective scratch --tier pilot --head-steps 20"`); on `status: ok` in its result.json, run `bash experiments/exp_anatomy/submit_wave1.sh <fetch_jobid>`.
Expected: ~20 trained-arm jobs + 4 extraction jobs + 8 ref-probe jobs queued.
- [ ] **Step 5: Commit** — `git commit -m "Add reference extraction and wave-1 submission wiring"`

---

## Follow-up plans (not in this plan)

- **Plan B:** probes C4 (latent rollout, walk sequences via `rw_<seed>_<step>` ordering), C5 (graph navigation), C6 (correspondence/relative pose); unimodal arms R6/R7 (new objectives in `_pretrain_pool`); MASt3R reference row (vendored repo).
- **Plan C:** Replica Part B (habitat-sim rendering of walks), modality encoders (mesh-graph, BEV-layout CNN, depth CNN) + BEV rasterization; modality-lattice sweep.

## Self-review notes

- Spec coverage: Tasks 1–6 cover the spec's W1–2 items "Replica provisioning", "reference-extraction pipeline R8", probes C3 and C7, and the first sweep jobs; C4–C6/R6/R7/modality encoders explicitly deferred to Plans B/C (spec allows: W1–2 spans two weeks).
- Types: `seed_split` operates on `list[str]` everywhere; feature files carry `files` so probes can re-split identically; `depth_grid` returns `(Tensor[g*g], Tensor[g*g] bool)` and `depthprobe` uses masks consistently.
- Known risk: Qwen2-VL visual-token split arithmetic (`grid_thw` merge factor) must be pilot-verified on 2 images before the full extraction job; the pilot `--limit 2` run exists for exactly this.
