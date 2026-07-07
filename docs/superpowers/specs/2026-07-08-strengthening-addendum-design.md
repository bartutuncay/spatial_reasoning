# Strengthening Addendum — Waves 3–5 (post-matrix design)

**Date:** 2026-07-08 · **Amends:** `2026-07-07-capability-anatomy-design.md` · **Status:** PI-approved bundles A+B+C

## 1. Where the matrix stands (evidence base)

Full 6-row × 5-column matrix on 5 ETH3D scenes (Waves 1–2, 80 jobs, shakedown tier, 2 seeds):

- **Double dissociation, measured:** `contrastive` best at place-rec (0.675) and pairwise geometry
  (navdist R² 0.39, relpose cos 0.23) but *worst* at latent rollout (ΔR² −0.04); `jepa` best at
  rollout (ΔR² 0.34) but mid-pack discriminative. Two axes emerge: a **predictive axis** (rollout;
  won by predict-objectives) and a **discriminative-relational axis** (place-rec + pairwise
  geometry; won by contrastive).
- **Caveats carried:** (i) `action_gap ≈ 0` everywhere ⇒ rollout currently measures temporal
  smoothness, not action-conditioned dynamics; (ii) rollout ΔR² not comparable across latent
  dims (refs 384–1536-d vs ours 128-d); (iii) trained rows near floor on depth; (iv) 5 scenes,
  200-step pretraining, 2 seeds.

The strengthening program removes asterisks or converts them into findings.

## 2. Wave 3 — Deepen (Bundle A)

### A1. Fusion frontier — "can you have both?" (pair: contrastive + jepa, PI-approved)
- `pretrain_encoder` gains fused objective: `loss = λ·InfoNCE(μ_i, μ_p) + (1−λ)·JEPA(μ_i, EMA(PCDEnc))`,
  row name `fuse_cj_<λ>`, λ ∈ {0.25, 0.5, 0.75}.
- Run the full probe battery (placerec, depth, rollout, navdist, relpose) × 2 seeds per λ (~30 jobs).
- Deliverable: frontier in the dissociation plane (discriminative composite vs rollout ΔR²).
  Concave frontier ⇒ fundamental trade-off; dominating point ⇒ practical recipe. Either is a result.

### A2. Branching walks — genuine action-conditioning
- New generator (CPU array per scene) reusing the random-walk machinery: from each of N≈60 anchor
  poses per scene, B=3 distinct actions (heading deltas, e.g. −45°/0°/+45° + step) rolled K≈4 steps
  → `branch_walks/` with filenames `bw_<anchor>_<branch>_<step>.pt`, same sample schema.
- `rollout.py --walks branching`: same-anchor different-action triples make z_{t+k} undetermined
  without the action; **action_gap becomes the column's headline metric** (positive gap = true
  action-conditioned prediction). Rollout column re-run on branching data (6 rows × 2 seeds + refs).
- Prereq check at plan time: read `agent_src/` walk generator to confirm heading control API.

### A3. Mechanism diagnostics — why the dissociation exists
- `diagnose.py`, one job per row (arm or `--features-dir`), emitting per-row:
  1. InfoNCE lower bound on I(z; pose) (small critic, train/eval split);
  2. InfoNCE lower bound on I(z; appearance) — appearance descriptor = downsampled color
     histogram/patch statistics of the image;
  3. alignment & uniformity (Wang & Isola) over augmentation pairs;
  4. effective rank (existing util);
  5. temporal smoothness: mean‖z_{t+1}−z_t‖ / mean pairwise ‖z_i−z_j‖ (scale-free).
- Deliverable: mechanism figure — regress each capability-axis coordinate on the measures.
  Hypothesis: discriminative axis ~ uniformity + I(z;appearance); rollout axis ~ smoothness + low rank.

## 3. Wave 4 — Bulletproof (Bundle B; parallel arrays)

1. **Bulk tier** (1500-step pretraining; current rows are undertrained at 200) + **3 seeds**.
2. **13 ETH3D scenes:** provision the 8 remaining scenes (existing extract/align/graph/walk
   pipeline; CPU array). More scenes attacks floor-mush and restores place-rec difficulty.
3. **Cross-scene probe splits:** `--split cross_scene` in every probe (train probe on 4 scenes,
   eval held-out 5th, rotate) — representation vs scene-memorization.
4. **Dim-fair rollout:** `--project-dim 128` random projection for reference rows; report raw and
   projected. Kills caveat (ii).
5. **C1/C2 columns** (map-conditioned localization ATE; few-shot APR K=50) folded into the tally
   via their existing result.json channels; trained rows only — reference rows marked n/a
   (no map-side encoder), never silently dropped.
6. **Probe-capacity ablation:** linear vs 2-layer head on place-rec (one column suffices).

## 4. Wave 5 — Broaden (Bundle C; ETH3D-first lattice)

- **BEV layouts** derivable now (scene PCD → top-down occupancy raster, `gen_layouts.py`);
  **per-view depth** already in every sample. Mesh/semantics wait for Replica rendering (5b).
- New encoders: 1-ch depth CNN, layout CNN (small). `pretrain_encoder` gains a modality-set
  argument: RGB anchor + sum of pairwise losses to each scene-modality encoder.
- Lattice rows at top-2 objectives (contrastive, jepa): {RGB} → +PCD → +depth → +layout nested,
  plus leave-one-out from full; 2 seeds; full probe battery. In-modality rule applies (depth-
  conditioned rows flagged on the depth column).
- **PID synergy** (`pid.py`, Liang-style discretized estimators) of (z_rgb, z_pcd) w.r.t. each task
  → F6 richness curve.

## 5. Sequencing & effort

- **Tonight:** A1 (trainer-only change) piloted + submitted.
- **This week:** A2 (generator + rollout rerun), A3 (diagnostics), B arrays fire as 8-scene
  provisioning lands; C starts once A code is in (shares the probe battery).
- All work flows through the existing probe.sbatch / result.json / tally.py harness.

## 6. Risks

- **A2 generator control:** if `agent_src` cannot steer headings cleanly, fall back to rejection-
  sampling free walks for same-anchor divergent pairs (weaker but sufficient for a positive gap test).
- **Fused objective collapse:** VICReg terms stay on the JEPA side; collapse diagnostic
  (effective rank) logged per λ-row; a collapsed λ-row is reported as such, not dropped.
- **Bulk-tier cost:** 1500-step pretrain per job ×(rows×columns×seeds) is the largest GPU sink;
  mitigate by pretraining once per (row, seed) and caching encoder checkpoints on scratch for the
  probe battery to share (breaks the self-contained-job pattern deliberately, LEDGERed by checkpoint
  hash). Decide at plan time after costing.
- **Diagnostics circularity:** I(z;pose) uses the same InfoNCE machinery as some training
  objectives; report estimator details and use held-out scenes for critics.

## 7. Approved decisions log

- Bundles A + B + C all green-lit (PI, 2026-07-08).
- Fusion pair: **contrastive + jepa** (PI, 2026-07-08).
- Caveat-honesty rule: rollout is described as "temporal latent predictability" until A2 lands a
  positive action-gap; only then may the paper use action-conditioned/world-model language.
