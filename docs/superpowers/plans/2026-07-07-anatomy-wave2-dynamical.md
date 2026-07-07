# Capability-Anatomy Wave-2: Dynamical/Geometric Columns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Add the predictive/dynamical capability columns — C4 latent rollout, C5 spatial-distance (navigation), C6 relative-pose (correspondence) — plus the RGB-only unimodal row (R6) as a valid row across all columns, on the existing ETH3D walks. This is the half of the matrix where predict-objectives should win, i.e. the double-dissociation test.

**Architecture:** New probe modules in `experiments/exp_anatomy/` reuse the Wave-1 pattern (frozen encoder + light head, arm vs `--features-dir` dual mode, result.json contract, generic `probe.sbatch`). A `pretrain_encoder` dispatch wrapper folds in the new `rgb_only` objective so R6 works on every column. Temporal/pair sampling lives in `common.py` with tests guarding the lexicographic-sort trap.

**Tech Stack:** torch 2.4.1+cu121 conda env on scratch, existing `ImgEnc`/`PCDEnc`, SLURM `gpuhe.4h`.

## Global Constraints

- Walk files `data_bartu/<scene>/random_walks/rw_<seed>_<step>.pt`; 26 seeds × 40 steps/scene; keys `img[192,256,3]`, `depth`, `loc[3]`, `view_dir[3]`, graph edges. **Temporal order = sort by (seed,step) numerically; `sorted(glob)` is lexicographic and WRONG.**
- All probes read the **image** latent `vae.img_enc(img)->(_,mu,_)` (query-time is RGB-only), so R7 (PCD-only) is out of scope for this battery — noted, deferred.
- Result contract, tiers (pilot/shakedown/bulk), commit-message hygiene (no trailers), `HF_HUB_OFFLINE=1` on GPU jobs: same as Wave 1.
- Rows this wave: `scratch, recon, symalign, contrastive, jepa, rgb_only`. Reference rows (`--features-dir`) apply to C4–C6 too (frozen features are per-frame; pairs/sequences rebuilt from the saved `files` list + loc/view_dir).

---

### Task 1: Sequence/pair/action utilities + rgb_only pretrainer (TDD)

**Files:**
- Modify: `experiments/exp_anatomy/common.py`
- Test: `tests/test_anatomy_common.py`

**Interfaces (Produced; Tasks 2–4 consume):**
- `walk_sequences(root, scene) -> dict[int, list[str]]` — seed → files ordered by step (numeric).
- `split_seeds(seqs, n_eval=5) -> (train_seqs: dict, eval_seqs: dict)` — highest `n_eval` seeds held out.
- `relative_action(si, sj) -> np.ndarray[6]` — `[Δloc(3, world), Δview_dir(3, world)]` from sample dict i→j.
- `load_walk(f) -> dict` — thin `torch.load(..., weights_only=False)`.
- `pretrain_encoder(vae, autoenc, walk_dirs, objective, steps, dev)` — dispatch: `rgb_only` → augmentation-InfoNCE on `img_enc` only; else delegates to `fewshot._pretrain_pool`.

- [ ] **Step 1: Add failing tests** to `tests/test_anatomy_common.py`:

```python
def test_walk_sequences_numeric_step_order(tmp_path):
    import torch
    from experiments.exp_anatomy.common import walk_sequences
    d = tmp_path / "office" / "random_walks"; d.mkdir(parents=True)
    # steps 0,2,10 — lexicographic would put 10 before 2
    for step in (0, 2, 10):
        torch.save({"loc": [float(step), 0.0, 0.0]}, d / f"rw_0_{step}.pt")
    seqs = walk_sequences(str(tmp_path), "office")
    assert list(seqs.keys()) == [0]
    steps = [int(f.split("_")[-1].split(".")[0]) for f in seqs[0]]
    assert steps == [0, 2, 10]                       # numeric, not ['0','10','2']


def test_split_seeds_holds_out_top():
    from experiments.exp_anatomy.common import split_seeds
    seqs = {s: [f"rw_{s}_{i}.pt" for i in range(3)] for s in range(10)}
    tr, ev = split_seeds(seqs, n_eval=3)
    assert sorted(ev) == [7, 8, 9] and len(tr) == 7


def test_relative_action_world_delta():
    import numpy as np
    from experiments.exp_anatomy.common import relative_action
    si = {"loc": [0.0, 0.0, 0.0], "view_dir": [1.0, 0.0, 0.0]}
    sj = {"loc": [1.0, 2.0, 0.0], "view_dir": [0.0, 1.0, 0.0]}
    a = relative_action(si, sj)
    assert a.shape == (6,)
    assert np.allclose(a[:3], [1.0, 2.0, 0.0])
    assert np.allclose(a[3:], [-1.0, 1.0, 0.0])
```

- [ ] **Step 2: Run** `python -m pytest tests/test_anatomy_common.py -q` on Euler → FAIL (new names missing).
- [ ] **Step 3: Implement** in `common.py`:

```python
def load_walk(f):
    return torch.load(f, map_location="cpu", weights_only=False)


def _step_of(f):
    return int(Path(f).name.rsplit("_", 1)[1].split(".")[0])


def walk_sequences(root, scene):
    seqs = {}
    for f in scene_files(root, scene):
        seed = int(re.search(r"rw_(\d+)_", Path(f).name).group(1))
        seqs.setdefault(seed, []).append(f)
    return {s: sorted(fs, key=_step_of) for s, fs in seqs.items()}


def split_seeds(seqs, n_eval=5):
    ev_seeds = sorted(seqs)[-n_eval:]
    tr = {s: v for s, v in seqs.items() if s not in ev_seeds}
    ev = {s: v for s, v in seqs.items() if s in ev_seeds}
    return tr, ev


def relative_action(si, sj):
    li = np.asarray(si["loc"], dtype=np.float32).reshape(3)
    lj = np.asarray(sj["loc"], dtype=np.float32).reshape(3)
    vi = np.asarray(si["view_dir"], dtype=np.float32).reshape(3)
    vj = np.asarray(sj["view_dir"], dtype=np.float32).reshape(3)
    return np.concatenate([lj - li, vj - vi]).astype(np.float32)


def pretrain_encoder(vae, autoenc, walk_dirs, objective, steps, dev):
    if objective != "rgb_only":
        from experiments.exp_jepa.fewshot import _pretrain_pool
        return _pretrain_pool(vae, autoenc, walk_dirs, objective, steps, dev)
    # RGB-only SSL: InfoNCE between two augmented views (no geometry at all)
    import torch.nn.functional as F
    import torchvision.transforms as T
    from torch.utils.data import ConcatDataset, DataLoader
    from experiments.exp_jepa.jepa import info_nce
    ds = ConcatDataset([autoenc.RandomWalkAutoencoderDataset(str(d)) for d in walk_dirs])
    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        collate_fn=autoenc.collate_random_walk_autoencoder)
    aug = T.Compose([T.RandomResizedCrop((192, 256), scale=(0.6, 1.0)),
                     T.RandomHorizontalFlip(),
                     T.ColorJitter(0.4, 0.4, 0.4, 0.1)])
    opt = torch.optim.AdamW(vae.img_enc.parameters(), lr=1e-4)
    it = iter(loader)
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        img = b.img.permute(0, 3, 1, 2).float().to(dev) / 255.0
        _, z1, _ = vae.img_enc(aug(img)); _, z2, _ = vae.img_enc(aug(img))
        loss = info_nce(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
```

- [ ] **Step 4: Run** the 3 new tests + full suite on Euler → all pass.
- [ ] **Step 5: Commit** — `git commit -m "Add walk-sequence/pair/action utils + rgb_only pretrainer"`

### Task 2: Refactor C3/C7 to the pretrain dispatch (backfill R6 everywhere)

**Files:** Modify `experiments/exp_anatomy/placerec.py`, `experiments/exp_anatomy/depthprobe.py`.

- [ ] **Step 1:** In both, replace `from experiments.exp_jepa.fewshot import PRETRAIN_STEPS, _pretrain_pool` with `from experiments.exp_jepa.fewshot import PRETRAIN_STEPS` and `from experiments.exp_anatomy.common import ... pretrain_encoder`, and swap the `_pretrain_pool(...)` call for `pretrain_encoder(...)` (identical signature).
- [ ] **Step 2:** Add `"rgb_only"` to each `--objective` choices list.
- [ ] **Step 3: Pilot** on Euler: `placerec --objective rgb_only --tier pilot --head-steps 20` → `status: ok`.
- [ ] **Step 4: Commit** — `git commit -m "Route C3/C7 through pretrain_encoder; add rgb_only row"`

### Task 3: C4 latent-rollout probe

**Files:** Create `experiments/exp_anatomy/rollout.py`.

**Design:** Frozen `z_t = img_enc(img_t)`. Train predictor `g([z_t, action_{t->t+k}]) -> ẑ_{t+k}` on train seeds; eval on held-out seeds. Metric = mean cosine distance `1 - cos(ẑ, z_{t+k})`. Floors: **copy-last** (ẑ=z_t) and **shuffled-action** (permute actions at eval; small gap ⇒ latent ignores dynamics). Verdict EXISTS if pred beats copy-last.

- [ ] **Step 1: Write `rollout.py`** (arm + `--features-dir`):

```python
"""C4 latent rollout: predict a future frame's latent from the current latent +
relative camera motion, along random-walk trajectories. Tests whether the space
supports forward dynamics (world-model capability)."""
import argparse, os, sys, time, traceback
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.exp_anatomy.common import (  # noqa: E402
    SCENES, finish, load_walk, pretrain_encoder, relative_action,
    split_seeds, walk_sequences)
from experiments.exp_jepa.fewshot import PRETRAIN_STEPS  # noqa: E402
from experiments.exp_jepa.locate import _load_module, _mlp  # noqa: E402


def _pairs(seqs, k):
    """(file_t, file_tk, sample_t, sample_tk) at horizon k within each walk."""
    out = []
    for fs in seqs.values():
        for t in range(len(fs) - k):
            out.append((fs[t], fs[t + k]))
    return out


def _encode_files(files, encode, cache):
    for f in files:
        if f not in cache:
            cache[f] = encode(f)
    return cache


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.processed_root)

    if args.features_dir:
        feat = {}
        for sc in SCENES:
            d = load_walk(Path(args.features_dir) / f"{sc}.pt")
            for f, z in zip(d["files"], np.asarray(d["features"])):
                feat[f] = z
        dim = next(iter(feat.values())).shape[0]
        def encode(f): return feat[f]
    else:
        autoenc = _load_module("sjepa_autoenc", ROOT / "training_scripts" / "1_autoencoder.py")
        vae = autoenc.ImageGraphVAE(args.latent_dim).to(dev).train()
        if args.objective != "scratch":
            pretrain_encoder(vae, autoenc, [root / s / "random_walks" for s in SCENES],
                             args.objective, PRETRAIN_STEPS.get(args.tier, 200), dev)
        vae.img_enc.eval()
        dim = args.latent_dim
        def encode(f):
            s = load_walk(f)
            img = torch.as_tensor(np.asarray(s["img"]), dtype=torch.float32).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                _, mu, _ = vae.img_enc(img)
            return mu[0].cpu().numpy()

    all_seqs = {}
    for sc in SCENES:
        for seed, fs in walk_sequences(root, sc).items():
            all_seqs[f"{sc}:{seed}"] = fs
    tr_seqs, ev_seqs = split_seeds({i: v for i, v in enumerate(all_seqs.values())},
                                   n_eval=max(2, len(all_seqs) // 6))

    def build(seqs, k):
        Z, A, Y = [], [], []
        for fs in seqs.values():
            for t in range(len(fs) - k):
                si, sj = load_walk(fs[t]), load_walk(fs[t + k])
                Z.append(encode(fs[t])); Y.append(encode(fs[t + k]))
                A.append(relative_action(si, sj))
        return (np.stack(Z), np.stack(A), np.stack(Y)) if Z else (None, None, None)

    K = args.horizon
    if args.tier == "pilot":
        tr_seqs = dict(list(tr_seqs.items())[:3]); ev_seqs = dict(list(ev_seqs.items())[:2])
    Ztr, Atr, Ytr = build(tr_seqs, K)
    Zev, Aev, Yev = build(ev_seqs, K)

    g = _mlp(dim + 6, 2 * dim, dim).to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3)
    X = torch.as_tensor(np.concatenate([Ztr, Atr], 1), dtype=torch.float32, device=dev)
    T_ = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    for _ in range(args.head_steps):
        perm = torch.randperm(len(X), device=dev)[:256]
        loss = F.mse_loss(g(X[perm]), T_[perm])
        opt.zero_grad(); loss.backward(); opt.step()

    def cosd(a, b):
        a = F.normalize(torch.as_tensor(a, dtype=torch.float32), dim=1)
        b = F.normalize(torch.as_tensor(b, dtype=torch.float32), dim=1)
        return float((1 - (a * b).sum(1)).mean())

    with torch.no_grad():
        Xe = torch.as_tensor(np.concatenate([Zev, Aev], 1), dtype=torch.float32, device=dev)
        pred = g(Xe).cpu().numpy()
        Ash = Aev[np.random.permutation(len(Aev))]
        Xsh = torch.as_tensor(np.concatenate([Zev, Ash], 1), dtype=torch.float32, device=dev)
        pred_sh = g(Xsh).cpu().numpy()
    err = cosd(pred, Yev)                    # model
    floor = cosd(Zev, Yev)                   # copy-last
    shuf = cosd(pred_sh, Yev)                # shuffled-action
    row = args.objective if not args.features_dir else f"ref:{Path(args.features_dir).name}"
    return {"status": "ok",
            "verdict": "EXISTS" if err < floor - 0.005 else "WEAK",
            "channel": f"rollout:{row}", "primary": round(err, 5),
            "metrics": {"cosdist": err, "copylast_floor": floor,
                        "shuffled_action": shuf, "action_gap": shuf - err,
                        "horizon": K, "n_train": int(len(Ztr)), "n_eval": int(len(Zev)),
                        "dim": int(dim), "row": row, "seed": args.seed, "tier": args.tier},
            "notes": f"rollout {row} k={K}: cosd {err:.4f} (copy-last {floor:.4f}, "
                     f"shuffled {shuf:.4f}, action-gap {shuf-err:+.4f})"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="scratch",
                    choices=["scratch", "jepa", "symalign", "contrastive", "recon", "rgb_only"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--tier", default="shakedown")
    ap.add_argument("--head-steps", type=int, default=400)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--processed-root", default=os.environ.get("SJEPA_PROCESSED_ROOT", "data_bartu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(); t0 = time.time()
    try:
        result = run(args)
    except Exception as e:
        result = {"status": "failed", "verdict": "DEAD",
                  "notes": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2500:]}
    finish(args.out, result, t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Pilot** on Euler (`rollout --objective scratch --tier pilot --head-steps 40 --horizon 2`) → `status: ok`, finite cosd/floor.
- [ ] **Step 3: Commit** — `git commit -m "Add C4 latent-rollout probe"`

### Task 4: C5 spatial-distance + C6 relative-pose probe (shared pair sampling)

**Files:** Create `experiments/exp_anatomy/spatialpairs.py` (emits two channels: `navdist`, `relpose`).

**Design:** Sample within-scene frame pairs (mixed within-walk horizons + cross-walk). C5 head: regress `‖loc_i-loc_j‖` from `[z_i, z_j]`; floor = train-mean distance (report R²). C6 head: regress relative translation direction (unit `loc_j-loc_i`) + relative heading `cos∠(vd_i,vd_j)` from `[z_i, z_j]`; floor = mean/zero (report cosine acc of direction + heading MAE). `--channel {navdist,relpose}` selects which (one job each) so both fan out.

- [ ] **Step 1: Write `spatialpairs.py`** (arm + `--features-dir`), pair sampler with fixed seed drawing ~4000 train / ~1500 eval pairs from `split_seeds`; two `--channel` heads as above; verdict EXISTS if beats floor (R² > 0 for navdist; direction-cos > floor for relpose).
- [ ] **Step 2: Pilot** both channels on Euler (pilot tier) → `status: ok`.
- [ ] **Step 3: Commit** — `git commit -m "Add C5 navdist + C6 relpose spatial-pair probe"`

### Task 5: Extend submission for Wave-2 columns + rgb_only backfill

**Files:** Create `experiments/exp_anatomy/submit_wave2.sh`.

- [ ] **Step 1:** Submit: (a) C4 `rollout` × 6 objectives × 2 seeds; (b) C5/C6 `spatialpairs --channel {navdist,relpose}` × 6 objectives × 2 seeds; (c) `rgb_only` on C3/C7 × 2 seeds (backfill); (d) reference-row jobs (`--features-dir`) for C4/C5/C6 across the 4 cached models. Mirror `submit_wave1.sh` structure (no new extraction needed — features already cached).
- [ ] **Step 2: Launch** after Task 3/4 pilots are green; `squeue` shows the volley.
- [ ] **Step 3: Commit** — `git commit -m "Add Wave-2 submission wiring"`

### Task 6: Tally Wave-2 into the matrix

- [ ] **Step 1:** Extend `tally_wave1.py` (or add `tally_wave2.py`) to print rollout/navdist/relpose columns (mean over seeds, with floors/gaps) and write `results/anatomy/wave2_tally.jsonl`.
- [ ] **Step 2:** Append a Wave-2 section to `CAMPAIGN_RESULTS.md` with the full 6-row × 5-column picture and an explicit read on whether the **double dissociation** appeared (predict-objectives win C4–C6 while losing C1/C2/C7).
- [ ] **Step 3: Commit.**

## Self-review notes
- Spec coverage: C4/C5/C6 + R6 from the spec's capability battery; R7 explicitly deferred (image-query battery). Modality lattice (Plan C) still separate.
- The lexicographic-sort trap is guarded by `test_walk_sequences_numeric_step_order`.
- Reference rows reuse Wave-1 cached features (per-frame), rebuilding pairs/sequences from saved `files`+loc/view_dir — no re-extraction.
- Known risk: `rgb_only` augmentation InfoNCE may be weak at shakedown step counts; pilot checks `status:ok`, not quality — quality judged in the tally against floors.
