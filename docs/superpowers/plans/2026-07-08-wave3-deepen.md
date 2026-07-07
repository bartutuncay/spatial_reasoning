# Wave-3 "Deepen" Implementation Plan (A1 fusion, A2 branching walks, A3 diagnostics)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Land the three asterisk-removing experiments: the contrastive↔jepa fusion frontier, genuinely action-conditioned rollout via branching walks, and per-row information diagnostics.

**Architecture:** All three ride the existing `exp_anatomy` harness (probe.sbatch, result.json, tally). A1 = trainer extension only. A2 = new CPU data generator (reusing `agent_src` rendering) + a `--walks-root` axis on rollout/extract_refs. A3 = one new probe-style diagnostics worker.

**Tech Stack:** torch 2.4.1+cu121 scratch env; open3d (env has it) for A2 rendering; SLURM `normal.4h` (A2 gen) + `gpuhe.4h` (everything else).

## Global Constraints

- Never write into `data_bartu/` (Bartu's tree via ACL). Branch walks → `/cluster/scratch/aleonel/spatial_jepa/data_branch/<scene>/branch_walks/`.
- Walk `img` is float 0–1 in Bartu's files; `common.to_uint8` handles both ranges — A2 must save the SAME schema (img float 0–1, depth float, loc[3], view_dir[3], graph fields) so every probe/extractor works unchanged.
- Fixed camera height z = −0.2 (office script); per-scene heights/rects extracted from the five `agent_src/rw_small_step/random_walk_<scene>.py` scripts at implementation time.
- Result contract, tiers, commit hygiene (no trailers), `HF_HUB_OFFLINE=1`: unchanged.
- Honesty rule: rollout stays "temporal predictability" language until branching rollout shows positive action-gap.

---

### Task 1: A1 — fused objective rows (`fuse_cj_25/50/75`)

**Files:**
- Modify: `experiments/exp_anatomy/common.py` (pretrain_encoder), the four probe CLIs (`placerec.py`, `depthprobe.py`, `rollout.py`, `spatialpairs.py`)
- Test: `tests/test_anatomy_common.py`

**Interfaces:**
- Produces: objective names `fuse_cj_25|fuse_cj_50|fuse_cj_75` accepted by every probe; `fuse_lambda(name) -> float|None` parser in common.

- [ ] **Step 1: Failing test** in `tests/test_anatomy_common.py`:

```python
def test_fuse_lambda_parser():
    from experiments.exp_anatomy.common import fuse_lambda
    assert fuse_lambda("fuse_cj_25") == 0.25
    assert fuse_lambda("fuse_cj_50") == 0.5
    assert fuse_lambda("fuse_cj_75") == 0.75
    assert fuse_lambda("jepa") is None
```

- [ ] **Step 2: Implement** in `common.py`:

```python
def fuse_lambda(objective):
    """fuse_cj_<pct> -> lambda in (0,1); None for non-fused objectives."""
    if objective.startswith("fuse_cj_"):
        return int(objective.rsplit("_", 1)[1]) / 100.0
    return None
```

and extend `pretrain_encoder` — BEFORE the existing `if objective != "rgb_only"` delegation add:

```python
    lam = fuse_lambda(objective)
    if lam is not None:
        # lam * InfoNCE(mu_i, mu_p)  +  (1-lam) * JEPA(mu_i -> EMA(PCDEnc))
        from experiments.exp_jepa.jepa import (Predictor, clone_as_target,
                                               ema_update, info_nce, jepa_objective)
        from torch.utils.data import ConcatDataset, DataLoader
        ds = ConcatDataset([autoenc.RandomWalkAutoencoderDataset(str(d)) for d in walk_dirs])
        loader = DataLoader(ds, batch_size=4, shuffle=True,
                            collate_fn=autoenc.collate_random_walk_autoencoder)
        predictor = Predictor(vae.pcd_enc.mu_head.out_features).to(dev)
        ema_pcd = clone_as_target(vae.pcd_enc).to(dev)
        opt = torch.optim.AdamW(list(vae.parameters()) + list(predictor.parameters()), lr=1e-5)
        it = iter(loader)
        for _ in range(steps):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader); b = next(it)
            img = b.img.permute(0, 3, 1, 2).float().to(dev)
            pcd = b.pcd.float().to(dev); bvec = b.batch.to(dev); ei = b.edge_index.to(dev)
            ew = autoenc.normalize_edge_weights(b.edge_weights.float()).to(dev)
            _, mu_i, _ = vae.img_enc(img)
            _, mu_p, _ = vae.pcd_enc(pcd, bvec, ei, ew)
            with torch.no_grad():
                _, tgt, _ = ema_pcd(pcd, bvec, ei, ew)
            j_loss, _ = jepa_objective(mu_i, tgt, predictor=predictor, objective="jepa")
            loss = lam * info_nce(mu_i, mu_p) + (1.0 - lam) * j_loss
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(ema_pcd, vae.pcd_enc, 0.996)
        return
```

- [ ] **Step 3:** Add the three names to every probe's `--objective` choices (4 files).
- [ ] **Step 4:** Sync; tests pass on Euler; **pilot** `placerec --objective fuse_cj_50 --tier pilot --head-steps 20` → `status: ok`.
- [ ] **Step 5: Submit the fusion battery** (3 λ × 2 seeds × 5 probes = 30 jobs) via a `submit_wave3_fusion.sh` mirroring submit_wave2 (probes: placerec, depthprobe, rollout, navdist, relpose).
- [ ] **Step 6: Commit.**

### Task 2: A2 — branching-walk generator + action-conditioned rollout

**Files:**
- Create: `experiments/data/gen_branch_walks.py`, `experiments/data/gen_branch_walks.sbatch`
- Modify: `experiments/exp_anatomy/rollout.py` (`--walks branching --walks-root`), `experiments/exp_anatomy/extract_refs.py` (`--walks-root/--out-root` already has out-root; add walks-root), `experiments/exp_anatomy/common.py` (branch sequence util)
- Test: `tests/test_anatomy_common.py`

**Interfaces:**
- Produces: `data_branch/<scene>/branch_walks/bw_<anchor>_<branch>_<step>.pt` (schema-identical samples); `branch_sequences(root, scene) -> dict[(anchor,branch), list[files-by-step]]` in common; rollout `--walks branching` where eval pairs share anchors across branches.

- [ ] **Step 1: Failing test** — `branch_sequences` numeric ordering + grouping:

```python
def test_branch_sequences_grouping(tmp_path):
    import torch
    from experiments.exp_anatomy.common import branch_sequences
    d = tmp_path / "office" / "branch_walks"; d.mkdir(parents=True)
    for a in (0, 1):
        for b in (0, 2):
            for s in (0, 10, 2):
                torch.save({"loc": [0.0, 0.0, 0.0]}, d / f"bw_{a}_{b}_{s}.pt")
    seqs = branch_sequences(str(tmp_path), "office")
    assert set(seqs) == {(0, 0), (0, 2), (1, 0), (1, 2)}
    steps = [int(f.rsplit("_", 1)[1].split(".")[0]) for f in seqs[(0, 0)]]
    assert steps == [0, 2, 10]
```

- [ ] **Step 2: Implement `branch_sequences`** in common.py (regex `bw_(\d+)_(\d+)_`, numeric step sort — mirror `walk_sequences`).
- [ ] **Step 3: Write `gen_branch_walks.py`.** Structure:
  - `SCENE_CONFIGS`: dict per scene {rects, ply_path, z_height, intrinsics} — copy values verbatim from the five `agent_src/rw_small_step/random_walk_<scene>.py` scripts (do this by reading each script during implementation; ply paths point at Bartu's processed tree, fall back to ours).
  - Motion: from anchor pose (xy sampled in rects, heading θ, view_yaw=θ), generate B=3 branches with heading offsets {−45°, 0°, +45°}; each branch rolls K=4 **deterministic** steps (fixed step length 0.05, no noise) with view_yaw pulled to θ_branch (view_pull=0.5, no noise) — futures are action-determined by construction.
  - Rendering + graph + save: same calls as the office script (`raycast_img_with_points`, `rotation_a_to_b`, `safe_knn_connectivity`, `make_graph`) via `sys.path` insertion of `agent_src/` and `agent_src/rw_small_step/`; import those helpers from the office script refactored *by import of its functions* — if the script executes at import (it does), lift the needed pure functions into `gen_branch_walks.py` directly (copy `safe_knn_connectivity`/`make_graph`/spherical helpers; import `raycast_img_with_points` from `agent_step`, `rotation_a_to_b`/`knn_connectivity` from `pcd_slice_methods`).
  - CLI: `--scene`, `--anchors 60`, `--task-index` (SLURM array shard over anchors), `--out-root data_branch`.
- [ ] **Step 4: sbatch** — CPU array (`normal.4h`, `--array=0-<n>`), one scene per submission or scene×shard grid; conda env activated; idempotent per-file (skip existing).
- [ ] **Step 5: Pilot** 1 scene × 2 anchors locally-on-Euler as a single CPU job; verify a saved `bw_0_1_2.pt` loads with the standard keys and `to_uint8` works on its img.
- [ ] **Step 6: Generate** all 5 scenes × 60 anchors × 3 branches × 4 steps (~3.6k samples; CPU array).
- [ ] **Step 7: rollout `--walks branching`:** replace the seq source with `branch_sequences`; build (z_t=anchor-step-0 latent, action, z_{t+k}) triples; eval split holds out whole ANCHORS (no anchor overlap); action-gap as before. Reference rows need features for branch walks: run `extract_refs --walks-root data_branch --out-root results/anatomy_refs_branch` (4 GPU jobs), then `--features-dir results/anatomy_refs_branch/<model>`.
- [ ] **Step 8: Submit** branching-rollout battery: 6 base rows + 3 fusion rows × 2 seeds + 4 refs (~22 jobs, afterok on generation).
- [ ] **Step 9: Commit** (generator, rollout changes, submission).

### Task 3: A3 — mechanism diagnostics worker

**Files:**
- Create: `experiments/exp_anatomy/diagnose.py`
- Modify: `experiments/exp_anatomy/tally.py` (diagnose section)

**Interfaces:**
- Produces: channel `diagnose:<row>` with metrics `{i_pose_nats, i_appear_nats, alignment, uniformity, eff_rank, smoothness}`.

- [ ] **Step 1: Write `diagnose.py`** (arm + `--features-dir`, same skeleton as placerec):
  - Encode all walk frames (or load features) → Z, loc, view_dir; plus per-frame appearance descriptor `a = 48-d color histogram` (16 bins × RGB, computed from img).
  - **I(z; pose) bound:** train InfoNCE critic `f(z) @ g([loc,view_dir])` on train-seed frames, report eval-set InfoNCE bound in nats (log-batch-size cap noted in metrics).
  - **I(z; appearance) bound:** same with `g(a)`.
  - **Alignment/uniformity** (Wang & Isola): alignment = E‖z−z⁺‖² over augmentation pairs (two crops of the same frame through the encoder — arm rows only; refs get NaN + note), uniformity = log E exp(−2‖zᵢ−zⱼ‖²) on normalized z.
  - **eff_rank** via existing `effective_rank`; **smoothness** = mean‖z_{t+1}−z_t‖ / mean-pairwise ‖zᵢ−zⱼ‖ using `walk_sequences`.
  - Critics: 2-layer MLPs, 300 steps, batch 256; seeds from `--seed`.
- [ ] **Step 2: Pilot** on `scratch` (expect: low i_pose, high smoothness≈? — just `status: ok` + finite numbers).
- [ ] **Step 3: Submit** 6 arms + 3 fusion rows + 4 refs (~13 jobs, 1 seed).
- [ ] **Step 4: Tally section** — add `diagnose` block printing the six measures per row.
- [ ] **Step 5: Commit.**

### Task 4: Frontier + mechanism figure data

**Files:**
- Create: `experiments/exp_anatomy/figures.py` (reads matrix_tally.jsonl → writes `results/anatomy/frontier.csv` + `mechanism.csv`; matplotlib PNGs `frontier.png`, `mechanism.png` under `results/figures/`).

- [ ] **Step 1:** Discriminative composite = mean of z-scored (placerec acc, navdist R², relpose cos); predictive = branching-rollout ΔR² (fallback: smooth-walk ΔR² clearly labeled). Scatter all rows + λ-path drawn as a line.
- [ ] **Step 2:** Mechanism: per-row capability coordinates vs each diagnostic, with Spearman ρ annotated.
- [ ] **Step 3:** Run on Euler after batteries drain; rsync PNGs back; commit script (+ CSVs via results are on scratch — PNGs copied into `paper/figs_wip/` locally).

## Verification
- Every new objective/probe passes a pilot (`status: ok`) before its battery.
- Unit tests: fuse parser, branch grouping (+ existing 7) green on Euler.
- Branching rollout sanity: `scratch` row action-gap ≈ 0 stays the null; a positive gap must exceed seed spread across 2 seeds.
- Figures regenerate from tally JSONL alone (no hand numbers).

## Self-review notes
- A2 import strategy avoids executing Bartu's script at import (it runs argparse+IO at module top): pure helpers are copied, heavy renderers imported from `agent_step`/`pcd_slice_methods` which are import-safe (verify at implementation; if `agent_step` also executes on import, copy the single `raycast_img_with_points` function).
- PLY availability per scene checked before generation; scenes lacking `combined_aligned.ply` regenerate via existing preprocessing sbatch first (CPU).
- Fusion rows also get diagnostics (Task 3 includes them) so the λ-path appears in BOTH figures.
