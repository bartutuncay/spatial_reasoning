# Spatial-JEPA — campaign results (office pilot)

Run 2026-07-06 on ETH Euler (aleonel), branch `jepa`. Single scene (`office`),
120 random-walk samples (3 seeds), shakedown tiers, RTX 4090. **66 training/eval
jobs run in parallel across three campaigns.** All numbers auto-collected from
`results/*/result.json`; reproduce via `experiments/submit_*.sh`.

## Campaign 1 — representation Stage-A (18 jobs)
Four-arm objective ablation × predictor × EMA × collapse × latent-dim × node-feats.
- **No collapse** — every VICReg arm holds effective rank ~107/128.
- **VICReg is necessary (clean ablation):** the one arm without it (`collapse_ema`)
  drops to loss≈0 and **rank halves (107→50)** — textbook partial collapse.
- **SSL loss is NOT comparable across objectives** (different scales) → the arms
  cannot be ranked on loss; needs the downstream pose metric (Campaign 2/3).

## Campaign 2 — localization, clean map (30 jobs)
objective(5) × freeze(2) × {map: head(2) | no_map}. Metric = ATE vs a 9.1 m
centroid floor.
- **Best ATE per objective:** scratch **1.88**, recon 2.00, symalign 2.03,
  jepa 2.12, contrastive 2.26 (m). **scratch (no pretrain) wins; jepa 4th.**
- **Map-conditioning HELPS every objective** (map ~1.9–2.3 vs no_map ~2.3–2.4 m)
  — no inversion.
- **anchor+offset head bug fixed:** was unbounded (144 m) in a ~600 m world frame;
  bounded `tanh*scale` → 1.48 m.

## Campaign 3 — graceful degradation, hard regime (18 jobs)
objective{scratch,jepa,symalign} × map_frac{1,0.5,0.25} × blur{0,3}, frozen.

| map_frac | blur | scratch | jepa | symalign |
|---|---|---|---|---|
| 1.0 | 0 | 1.88 | 2.16 | 2.06 |
| 1.0 | 3 | 1.62 | 2.53 | 1.86 |
| 0.5 | 0 | 1.97 | 2.31 | 2.12 |
| 0.5 | 3 | 1.85 | 2.71 | 1.97 |
| 0.25 | 0 | 1.97 | 2.31 | 2.26 |
| 0.25 | 3 | 1.70 | 2.40 | 2.03 |

- **Degradation is flat** (hard/easy ratio 0.90–1.11×) — the stress axes barely bite.
- **scratch best at every condition; jepa consistently worst.**

## Honest conclusion
The pipeline is validated end-to-end at scale (env, data provisioning, JEPA
trainer with a proven anti-collapse mechanism, map-conditioned localizer, all run
as parallel campaigns). **But the office-only 120-sample data cannot test the
thesis:** the task is memorization/interpolation-solvable, so representation
quality is irrelevant and partial-map/blur do not degrade performance. JEPA shows
no advantage here — and *should not be expected to* in a regime this easy.

**Next move is data, not head engineering:** provision many scenes, generate many
more walks, and evaluate with a **cross-scene split** (train scenes ≠ eval scenes)
so memorization fails and a general representation can matter. Only then is the
jepa-vs-scratch comparison and the graceful-degradation curve meaningful. One
positive that already holds and survives scaling: **map-conditioning helps
(no inversion)**, and **VICReg is empirically necessary** for a high-rank latent.

---

# Capability-Anatomy campaign — Wave 1 (2026-07-07)

New paper direction (see `docs/superpowers/specs/2026-07-07-capability-anatomy-design.md`):
**objective × modality × capability** matrix. Wave 1 = first two capability
**columns** (C7 place recognition, C3 metric depth) across 5 trained-arm rows
(2 seeds) + 4 frozen foundation-model reference rows on Bartu's 5 ETH3D scenes.
28 jobs, **28 ok** (2 initial DEAD re-ran green after the SVD + transient-node
fixes below). Frozen encoder + linear probe; auto-collected via
`experiments/exp_anatomy/tally_wave1.py`. Replica v1 (18 scenes) downloaded to
scratch — ready for the generality/modality waves.

## C7 — place recognition (which of 5 scenes; acc, majority floor 0.20)
| row | linear-probe acc | NN@1 retrieval |
|---|---|---|
| scratch (random enc) | 0.217 | **0.782** |
| recon | 0.647 | 0.508 |
| symalign | 0.555 | 0.420 |
| contrastive | **0.675** | 0.698 |
| jepa | 0.532 | 0.393 |
| ref: DINOv2-S | **1.000** | 0.980 |
| ref: DINOv2-B | 0.990 | 0.980 |
| ref: SigLIP | 1.000 | 0.980 |
| ref: Qwen2-VL-2B tower | 0.970 | 0.975 |

- **A real dissociation, first row:** the **random** encoder is near-worst on the
  *linear* probe (0.22) yet **best-of-the-trained on NN retrieval (0.78)** — its
  features preserve appearance locality but aren't linearly separable. Training
  (any objective) *flips* this: linear separability rises, raw appearance-NN falls.
  Predict/abstract objectives (jepa, symalign) shed the most appearance (NN@1 0.39–0.42).
- **Foundation models saturate this column** (~1.0 acc) — as designed, they anchor
  the dynamic range the small from-scratch rows can't reach. Caveat: 5 very
  different rooms may make place-rec too easy; more (and more similar) scenes needed to
  make the trained-row spread meaningful.

## C3 — metric depth (linear probe → 16×16 log-depth grid; AbsRel↓, train-mean floor 1.65)
| row | AbsRel | δ<1.25 |
|---|---|---|
| scratch | 1.538 | 0.283 |
| recon | 1.795 | 0.313 |
| symalign | 1.698 | 0.299 |
| contrastive | 1.946 | 0.281 |
| jepa | 1.862 | 0.300 |
| ref: DINOv2-B | **0.434** | **0.511** |
| ref: SigLIP | 0.499 | 0.497 |
| ref: DINOv2-S | 0.532 | 0.474 |
| ref: Qwen2-VL-2B tower | 0.569 | 0.448 |

> **SUPERSEDED (2026-07-08):** these numbers used raw (un-standardized) latents as
> the linear-probe input. Un-normalized rows (esp. `rgb_only`) diverged the head and
> masked real signal (one cell overflowed to `inf`). The corrected, **scale-invariant
> protocol** (z-score features by train stats) is in the Wave-2 full-matrix section
> below and **reverses the conclusion**: the trained rows *do* carry modest,
> linearly-decodable depth (AbsRel ~1.08 vs floor 1.65), just far less than
> foundation models (~0.45). Keep only the ordering claim (refs ≫ trained);
> the "trained ≈ floor / carries no depth" claim was a scale artifact.

## Reading so far (2 of 7 columns)
The intended **double dissociation** is not yet visible because the trained rows
cluster near the floor on both columns — the "dynamic-range mush" risk (spec §9.1)
is real at this scale. What *is* already clean: (1) a **linear-vs-NN dissociation**
within place-rec that separates appearance-preserving from appearance-shedding
objectives; (2) foundation-model rows behaving exactly as designed anchors,
saturating discriminative place-rec and dominating geometric depth. The verdict on
whether *objective choice among the small rows* buys distinct capabilities waits on
(a) the predictive/dynamical columns C4–C6 (where predict-objectives should win),
and (b) more scenes / Replica scale to lift the trained rows off the floor.

## Fixes made during Wave 1
- **`effective_rank` SVD crash** (jepa row, `jepa.py:91`): cusolver `svdvals` throws
  `_LinAlgError` on collapsed/ill-conditioned batches → CPU fallback, then rank=1.
  Would have recurred on every jepa job; 19/19 tests pass, collapsed→1.0 verified.
- **NaN/inf depth pixels** poisoned the depth grid → `nan_to_num` before masking (TDD).
- **`eth_proxy`** required for any compute-node internet (both download jobs); GPU
  probes forced `HF_HUB_OFFLINE=1`.
- **Replica `download.sh` needs `pigz`** (absent on nodes) → env `pigz` on PATH.
  Replica v1 (18 scenes) now fully extracted on scratch.
- **SigLIP depth CUDA device-side assert** (1 cell): identical re-run passed
  (AbsRel 0.499) → confirmed transient node fault, not a code path.

---

# Capability-Anatomy — Wave 2 full matrix (2026-07-08)

All 5 capability columns × 6 trained rows (+ rgb_only unimodal control) × 4 frozen
foundation-model refs, 2 seeds, shakedown tier, 5 ETH3D scenes. Depth uses the
corrected scale-invariant probe (supersedes the Wave-1 depth table). Column metric
conventions: placerec acc↑ (floor 0.20); depth AbsRel↓ (floor 1.65); rollout
delta-R²↑ (copy-last floor 0); navdist R²↑; relpose dir-cos↑. Auto: `tally.py`.

| row | placerec↑ | depth↓ | rollout↑ | navdist↑ | relpose↑ |
|---|---|---|---|---|---|
| scratch (random) | 0.217 | 1.080 | 0.190 | 0.031 | −0.015 |
| rgb_only (img SSL) | 0.480 | 1.150 | −0.019 | 0.091 | 0.095 |
| recon | 0.647 | **1.074** | 0.271 | 0.206 | 0.126 |
| symalign | 0.555 | 1.392 | 0.302 | 0.159 | 0.070 |
| **contrastive** | **0.675** | 1.116 | **−0.036** | **0.390** | **0.232** |
| **jepa** | 0.532 | 1.674 | **0.337** | 0.199 | 0.110 |
| ref: DINOv2-S | 1.000 | 0.495 | 0.112 | 0.663 | 0.545 |
| ref: DINOv2-B | 0.990 | **0.435** | 0.033 | 0.630 | 0.488 |
| ref: SigLIP | 1.000 | 0.483 | 0.049 | **0.722** | 0.482 |
| ref: Qwen2-VL-2B | 0.970 | 0.498 | 0.134 | 0.607 | 0.337 |

**The core finding — a double dissociation on the contrastive↔jepa antipode:**
- `contrastive` is **best of the trained rows at place-rec, navdist, relpose** but
  **worst at rollout** (−0.04). `jepa` is **best at rollout** (0.34) but mid-pack
  discriminative. No single objective wins both families.
- **Two capability axes, not one "spatial" axis:** place-rec + navdist + relpose all
  favor contrastive (relating/ID-ing views is a *discrimination* task); only forward
  **rollout** favors the predict-objectives (jepa, symalign). "Spatial" splits into a
  **discriminative-relational** axis and a **predictive** axis.
- **Corrected depth reading:** trained rows now clearly beat the floor (1.07–1.4 vs
  1.65); `recon` best (1.074), `jepa` worst (1.674, at floor) — predictive abstraction
  costs metric-depth readout. Foundation models still dominate (~0.45).
- **Foundation refs** top every column (as designed anchors); they lead rollout only
  modestly (0.03–0.13) and *below* jepa (0.34) — but cross-dim rollout-R² is confounded
  (refs 384–1536-d vs 128-d), so that specific comparison is not claimed.

**Caveats (unchanged):** rollout `action_gap ≈ 0` for all trained rows ⇒ this is
temporal latent *predictability*, not action-conditioned dynamics (A2 branching walks
test this). 5 scenes, 200-step pretraining, 2 seeds.

---

# Wave-3 A1 — the fusion frontier (2026-07-08)

Fused objective `fuse_cj_<λ>` = λ·InfoNCE(μ_i, μ_p) + (1−λ)·JEPA(μ_i→EMA(PCDEnc)),
λ = contrastive weight ∈ {0.25, 0.5, 0.75}. 2 seeds, shakedown tier, 5 ETH3D scenes.
Placed between the measured antipodes (jepa=λ0, contrastive=λ1):

| row (λ=contrastive wt) | place-rec acc ↑ | rollout ΔR² ↑ |
|---|---|---|
| jepa (λ=0) | 0.532 | **0.337** |
| fuse_cj_25 | 0.585 | 0.310 |
| fuse_cj_50 | 0.625 | 0.302 |
| fuse_cj_75 | 0.643 | 0.270 |
| contrastive (λ=1) | **0.675** | −0.036 |

- **Both axes monotone in λ** — the fusion smoothly interpolates the dissociation.
- **Trade-off is asymmetric (the finding):** place-rec climbs the full range (0.53→0.68)
  while rollout barely erodes across the fusion band (0.34→0.27) and **only collapses at
  pure contrastive** (−0.04). `fuse_cj_75` retains ~95% of contrastive's place-rec AND ~80%
  of jepa's rollout — a small predictive term rescues rollout at almost no discriminative cost.
  The dissociation is real but the frontier is *not* a straight Pareto line; it bends.
- navdist/relpose along the frontier: navdist rises with λ (0.185→0.239), relpose flat
  (~0.13) — consistent with those columns patterning discriminative (contrastive-favoring).
- Depth column being re-run under a scale-invariant (z-scored input) probe protocol; numbers
  folded in once that sub-battery drains (fixes the earlier rgb_only inf at its root).

---

# Wave-3 A3 — mechanism diagnostics (2026-07-08)

Per-row information measures (frozen encoder), meant to EXPLAIN the capability
axes. I(z;·) = InfoNCE lower bound in nats (cap log256≈5.5); smoothness = mean
consecutive-step Δ / mean pairwise Δ; alignment = augmentation E‖z1−z2‖² (arms
only); rank = effective rank. 5 scenes, shakedown.

| row | I(z;pose) | I(z;appear) | eff-rank | smooth |
|---|---|---|---|---|
| scratch (random) | 1.45 | 2.14 | 94 | 0.55 |
| rgb_only (img SSL) | 1.35 | 2.12 | **7.4** | 0.19 |
| recon | 0.16 | 0.85 | 116 | 0.85 |
| symalign | −0.25 | 0.55 | 117 | 0.89 |
| **contrastive** | **0.98** | **1.46** | 69 | 0.51 |
| **jepa** | −0.22 | **0.39** | 121 | 0.91 |
| fuse_cj_25 | −0.01 | 0.59 | 119 | 0.89 |
| fuse_cj_50 | 0.16 | 0.76 | 118 | 0.87 |
| fuse_cj_75 | 0.33 | 0.83 | 115 | 0.83 |
| ref: DINOv2-S/B, SigLIP, Qwen | 2.3–2.4 | 2.6–2.8 | 196–352 | 0.39–0.44 |

**Mechanism findings:**
- **The discriminative axis is explained by decodable appearance/pose info.**
  Among cross-modal trained rows, `contrastive` retains the most I(z;appear) 1.46
  and I(z;pose) 0.98; `jepa` the least (0.39 / −0.22). This *is* why contrastive
  wins place-rec/navdist/relpose and jepa wins only rollout — predictive
  abstraction **measurably discards the appearance & pose information discrimination
  needs.** The fusion rows interpolate **monotonically** in both (I(z;appear)
  0.59→0.83, I(z;pose) −0.01→0.33 as λ↑) — the mechanism knob tracks the frontier.
- **Cross-modal conditioning is an anti-collapse regularizer.** Image-only SSL
  (`rgb_only`) collapses to **effective rank 7.4** (vs ~115–121 for the cross-modal
  rows) — this is the root of its earlier depth-probe divergence and its weakness on
  the geometric columns. Adding the point-cloud target prevents that collapse. A
  clean, quotable "why multimodal" result.
- **The predictive (rollout) axis is not a clean single-variable story.** jepa pairs
  high smoothness (0.91) + high rank (121) + low appearance; but contrastive
  (smooth 0.51, worst rollout) breaks a naive smoothness law. Report rollout-vs-
  diagnostics as a Spearman-ρ panel (Task 4), not a one-liner.
- Foundation refs dominate every measure (high I, high rank) — expected anchors.

## Action-conditioned rollout (A2) — status: BLOCKED, not yet measured
Branching walks generated (3600 samples, 4 scenes, same-anchor 3-action). First
battery **timed out** (per-pair `load_walk` of graph-heavy branch files); fixed
with single-pass batched frame caching (11.7 min vs 50-min wall). Re-run then hit
a CUDA illegal-memory-access on the batch-32 encode; dropped to the proven batch-16.
Re-piloting. **The action-gap verdict — whether rollout is genuinely
action-conditioned or just temporal predictability — is still pending this battery.**

---

# Wave-4 BULK matrix + action-gap verdict (2026-07-08)

Full matrix re-run at **bulk tier (1500-step pretrain) × 3 seeds**, 5 ETH3D scenes,
across two GPU pools (4090/gpuhe + a100/gpupr). ~185 jobs, 177 ok. This is the
credibility run; it SHARPENED the dissociation vs shakedown.

| row | placerec↑ | depth↓ | rollout↑ | navdist↑ | relpose↑ |
|---|---|---|---|---|---|
| scratch | 0.217 | 1.061 | 0.196 | 0.118 | 0.010 |
| rgb_only | 0.503 | 1.242 | −0.057 | 0.125 | 0.105 |
| recon | 0.748 | **0.872** | 0.142 | 0.489 | 0.252 |
| symalign | 0.532 | 0.875 | 0.205 | 0.374 | 0.133 |
| **contrastive** | **0.852** | 0.879 | **−0.093** | **0.540** | **0.295** |
| **jepa** | 0.522 | **1.385** | **0.355** | 0.371 | 0.113 |
| fuse_cj_25 | 0.787 | 1.138 | 0.312 | 0.369 | 0.131 |
| fuse_cj_50 | 0.758 | 1.133 | 0.290 | 0.402 | 0.142 |
| fuse_cj_75 | 0.792 | 1.120 | 0.275 | 0.421 | 0.116 |
| ref: DINOv2-B | 0.990 | 0.411 | 0.040 | 0.726 | 0.464 |
| ref: (S/SigLIP/Qwen) | 0.98–1.0 | 0.41–0.51 | 0.05–0.11 | 0.48–0.74 | 0.35–0.52 |

**Double dissociation — robust, sharpened at bulk:**
- `contrastive` wins **4 of 5** trained-row columns (placerec 0.852, depth ~0.88,
  navdist 0.540, relpose 0.295) but is **dead last on rollout (−0.093)**.
- `jepa` wins **rollout alone (0.355)** and is worst on depth (1.385) — predictive
  abstraction costs metric-depth readout most.
- The clean axis: **discriminative-relational** (placerec + navdist + relpose, all
  contrastive-favoring — relating/IDing views is discrimination) vs **predictive**
  (rollout, jepa-favoring). "Spatial" is (at least) two capabilities, not one.
- **Fusion frontier holds at bulk & 3 seeds:** placerec rises with λ (0.52→0.85),
  rollout stays positive across the whole fusion band (0.31→0.28) and only collapses
  at pure contrastive (−0.09). Asymmetric: a little JEPA rescues rollout cheaply.

## Action-gap verdict — "world-model" language NOT earned
Action-conditioned rollout on branching walks (same anchor, 3 divergent actions →
future is action-determined by construction):

| row | delta-R² | **action-gap** (shuffled − model) |
|---|---|---|
| jepa | 0.209 | **+0.000** |
| symalign | 0.176 | −0.000 |
| recon | 0.160 | +0.002 |
| contrastive | −0.309 | +0.021 |
| fuse_cj_25/50/75 | 0.16–0.20 | +0.000 |

**Even when the future is action-determined, shuffling the action does not degrade
prediction (gap ≈ 0 for every trained row).** So the rollout advantage is *temporal
latent predictability* (jepa's latent is autoregressively smooth — delta-cos 0.51),
NOT action-conditioned dynamics. The paper describes this column as "predict-in-
latent temporal structure," and does **not** claim a world model. (Honesty rule from
the strengthening spec, resolved against the strong claim.)

## Notes
- Predictive rows (jepa/symalign) also win rollout on branching walks; contrastive is
  negative there too — the dissociation is consistent across smooth & branching walks.
- A few jepa seeds DEAD on the pre-existing PCD "2 origin nodes" flake (jepa rollout
  n=1 on branching); rerun pending, won't move the qualitative story.

---

# ScanNet generality (2026-07-09, shakedown)

Second, real-scan dataset (20 ScanNet v2 scenes; walks rendered from `_vh_clean_2`
meshes via the ETH3D raycaster with dense surface sampling; ~30% render fill due to
genuine scan incompleteness). Shakedown tier, 2 seeds, `SJEPA_SCENES` scene-set.
~110 jobs; core rows complete. Metric conventions as elsewhere.

| row | placerec↑ | placerec NN@1↑ | rollout↑ | depth↓ | navdist↑ | relpose↑ |
|---|---|---|---|---|---|---|
| scratch | 0.056 (=floor) | 0.685 | 0.371 | **0.515** | 0.021 | 0.003 |
| **contrastive** | **0.204** | **0.322** | **0.283** | 0.851 | 0.012 | 0.055 |
| **jepa** | 0.159 | 0.102 | **0.361** | 0.910 | −0.266 | 0.031 |
| recon | 0.168 | 0.124 | 0.340 | 0.893 | −0.350 | 0.004 |
| symalign | 0.165 | 0.091 | 0.340 | 0.925 | −0.366 | 0.009 |
| fuse_cj_25/50 | 0.15–0.17 | 0.10–0.12 | 0.357–0.359 | ~0.85 | ~−0.13 | 0.02–0.06 |
| ref: SigLIP/DINOv2 | 0.74–0.85 | 0.84–0.93 | 0.14–0.24 | 0.31–0.33 | −0.12 to 0.01 | 0.09–0.14 |

**Core dissociation REPRODUCES cross-dataset:**
- **Discriminative (placerec):** contrastive best trained (0.204 acc, NN@1 0.322);
  predict-objectives lower — same ordering as ETH3D.
- **Predictive (rollout):** contrastive **worst** trained (0.283); jepa/recon/symalign
  higher — same ordering as ETH3D. The contrastive↔jepa antipode holds on ScanNet.

**Honest limits (do not overstate the other columns):**
- **navdist/relpose ≈ floor** on ScanNet — small single rooms → low pairwise-distance
  variance → the regression is near-uninformative (dataset property, not method).
- **depth anomalous** — scratch (0.515) beats trained (0.85–0.91). Likely a shakedown
  transfer artifact: 200-step cross-modal training on sparse ScanNet renders barely
  moves (or scrambles) the low-level depth gradient that random conv features already
  expose to a linear probe. A **bulk ScanNet run** would settle this.
- All shakedown tier / ~30% render fill → lower absolutes than the ETH3D bulk matrix.

**Takeaway:** the headline two-axis dissociation (contrastive wins discrimination,
loses rollout; predict-objectives opposite) transfers to real ScanNet scans; the
pairwise-geometry and depth columns are dataset/tier-limited and want a bulk re-run
before they carry weight in the paper.

---

# ScanNet generality — BULK (2026-07-09, 3 seeds)

The bulk re-run the shakedown asked for. 20 ScanNet scenes, **1500-step**
conditioning, **3 seeds**, reused ref features. 195 jobs, all drained.
Resolves the shakedown's depth anomaly.

| row | placerec↑ | placerec NN@1↑ | depth↓ | rollout↑ | act-gap | navdist↑ | relpose↑ |
|---|---|---|---|---|---|---|---|
| scratch | 0.056 (=floor) | 0.672 | 0.508 | 0.371 | 0.002 | 0.017 | −0.005 |
| rgb_only | 0.084 | 0.519 | 0.856 | 0.312 | 0.028 | 0.033 | 0.073 |
| recon | 0.193 | 0.235 | 0.536 | 0.316 | 0.004 | 0.034 | 0.035 |
| symalign | 0.165 | 0.123 | 0.927 | 0.321 | 0.002 | −0.015 | 0.039 |
| **contrastive** | **0.245** | 0.550 | **0.483** | 0.299 | 0.012 | **0.070** | 0.040 |
| **jepa** | 0.141 | 0.103 | 0.862 | **0.392** | 0.000 | 0.014 | 0.054 |
| fuse_cj_25 | 0.202 | 0.187 | 0.569 | 0.373 | 0.001 | 0.008 | 0.038 |
| fuse_cj_50 | 0.200 | 0.230 | 0.532 | 0.365 | 0.002 | 0.024 | 0.035 |
| fuse_cj_75 | 0.207 | 0.264 | 0.524 | 0.355 | 0.001 | 0.023 | 0.032 |
| ref: DINOv2-S | 0.738 | 0.870 | 0.307 | 0.238 | 0.003 | 0.027 | 0.091 |
| ref: SigLIP | 0.845 | 0.928 | 0.317 | 0.143 | 0.007 | 0.042 | 0.107 |

**What bulk changes vs shakedown:**
- **Double dissociation firms up:** contrastive best trained placerec (0.245),
  worst trained rollout (0.299); jepa best rollout (0.392). Antipode holds, wider gap.
- **Depth anomaly RESOLVED.** Shakedown had scratch (0.515) beating trained rows.
  At bulk, **contrastive wins depth (0.483)** and **jepa is worst (0.862)** — the
  discriminative-axis ordering, matching ETH3D where jepa is also worst at depth.
  Depth sits on the discriminative axis on *both* datasets. No longer a caveat.
- **Action-gap ≈ 0** on ScanNet (jepa 0.000) — the no-world-model negative reproduces
  on real scans.
- **navdist/relpose still ≈ floor** (r²≤0.07, dir_cos≤0.07) — genuine small-room
  artifact of ScanNet single-room crops, not tier. Stays dataset-limited.

**Paper impact:** Generality paragraph upgraded from "core ordering reproduces, depth
confounded, dataset-limited" to a cross-dataset claim spanning place-rec, rollout,
depth, and the action-gap negative. Only pairwise geometry remains near-floor.

---

# Rollout split bug found + fixed (2026-07-10, via oracle positive control)

The review-driven oracle-state positive control (ground-truth [loc, view_dir]
as the "representation") FAILED its pilot: delta-R2 negative, action-gap
exactly 0. A synthetic same-frame repro of the identical head/standardization
passed (delta-R2 0.80, gap +1.5), isolating the bug to data plumbing:
**rollout pooled all scenes into one enumerated dict before split_seeds, which
holds out the highest indices = the last-inserted scene wholesale.** ETH3D
rollout was effectively train-on-4-scenes / eval-on-hospital; ScanNet eval was
the last ~3 scenes. Every other probe splits within-scene. The branching
(action-gap) split had the same flavor (lexicographically-last scene's anchors).

Consequences:
- All prior rollout / rollout_act numbers were measured under an unintended
  cross-scene protocol. Orderings may hold (protocol was identical across rows)
  but absolute values are not the intended within-scene predictability.
- Fixed in rollout.py (per-scene seed split, per-scene anchor split).
- Re-measure waves queued: ETH3D b_rollout_* (anatomy_bulk, overwriting IDs;
  old tallies preserved in git), corrected action-gap ra_rolloutact_* ->
  anatomy_controls, ScanNet sn_rollout_* (anatomy_scannet_bulk), and the oracle
  control alongside.
- Paper impact: rollout column + action-gap numbers must be refreshed after the
  re-measure lands; the oracle control now validates that the harness CAN
  detect action-conditioning (the reviewer's Q10).

---

# Review-sweep fleet results (2026-07-10, ~540 jobs, 3 failures)

All numbers 3 seeds, corrected within-scene split. Full per-cell data in
`experiments/exp_anatomy/tallies/*.jsonl`.

**Corrected rollout (k=2).** ETH3D: rgb_only .420±.035 (BEST — at eff. rank 7.4),
jepa .354±.007, symalign .349±.019, fuses .32–.33, recon .318, scratch .306,
contrastive .285±.015 (worst). ScanNet: jepa .393±.003 top, scratch .359,
contrastive .265±.014 worst (below random!). Contrastive-destroys /
predictive-preserves survives the split fix; "jepa best" is clean on ScanNet,
narrow over symalign on ETH3D.

**Oracle positive control:** ground-truth pose as representation → ΔR² .88/.91,
action-shuffle gap +1.9 (smooth/branching). The harness detects
action-conditioning emphatically.

**Corrected action-gap:** predictive/fused rows ≤ .02 (jepa .008) — the honest
negative reproduces under a VALIDATED harness. Appearance-keeping rows DO use
the action: rgb_only +.36±.10, scratch +.11, contrastive +.10 — smooth latents
extrapolate without the action, detailed ones need it.

**Horizon sweep (k=1/2/4/8):** jepa FLAT (.341/.354/.357/.346) — slow-feature
signature. Everything else rises with k (contrastive .159→.404; rgb_only
.297→.609 steepest). k=1 gives the sharpest dissociation.

**Training curves (1.5k/5k/15k):** relational gaps WIDEN (relpose contrastive
.295→.438 vs jepa .113→.133; navdist .540→.694 vs .371→.435; depth c
.879→.561, j stuck ~1.2–1.4). placerec narrows (jepa .52→.72 vs .86) but does
not close. ROLLOUT CONVERGES: contrastive .285→.346 ≈ jepa .339 at 15k.
→ discriminative-relational axis durable; predictive axis is a
finite-training, short-horizon phenomenon.

**Probe-capacity flips:** discriminative/relational orderings stable under
linear↔MLP (MLP placerec: contrastive .857, jepa .538). Rollout differences
need the MLP readout (linear compresses all rows to .22–.29).

**V-JEPA 2 reference row (ETH3D):** placerec .965, navdist .708, relpose .364,
depth .371, rollout .191 — a PREDICTIVE foundation model behaves like the other
refs on our probes (rollout cross-dim caveat).

**Intervention pilot:** mechanics ok (scratch: no differential drop, as
expected); full 45-job wave queued (anatomy_intervene).

**Headline reframe for the paper:** latent-rollout ΔR² is NOT world-model
evidence — flat in horizon, no action-conditioning under a validated control,
topped by a rank-collapsed encoder, matched by pure contrastive at 10×
compute. Discriminative-relational axis is the durable one.

**Replica status:** quad-mesh → plyfile triangulation path segfaults in the
render job (debug job isolating the faulting step); multi-room dataset still
pending.

---

# Intervention wave + Replica dataset (2026-07-10, late)

**Intervention (42 jobs):** removing the top-k appearance-predictive subspace
(k≈5) drops place-rec consistently more than an equal-dim random subspace
(jepa −.067±.015 vs −.005; fused −.04 vs ~0; refs k≈13, no drop — saturated),
but absolute drops are small: appearance information is redundantly coded and
one linear projection cannot delete it. Reported in the paper as a qualified
interventional check; mechanism stays correlational. (Iterative nullspace
projection would be the stronger tool if reviewers push.)

**Curve reruns:** 5k contrastive rollout now n=3 (.341); 15k contrastive depth
n=3 (.534). Fig 4 regenerated.

**Replica multi-room dataset GENERATED:** 12 scenes × 26 walks × 40 steps =
12,164 frames (97.5% save rate; renders upright, dense, near-full fill).
Root cause of earlier segfault: open3d 0.18 is numpy-2 incompatible — env
restored to numpy 1.26.4 + plyfile 1.0.3. Env note: numpy version matters for
open3d only (data-gen); probes are torch-only.

**Replica matrix wave fired:** submit_replica.sh — 5 ref extractions +
(9 trained + 5 ref rows) × 5 probes × 3 seeds → results/anatomy_replica_bulk
(~215 jobs). This is the multi-room test of nav-dist/rel-pose — the
geometric-evidence answer to the "is place-rec spatial?" critique.
