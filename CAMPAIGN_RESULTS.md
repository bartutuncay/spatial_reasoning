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
