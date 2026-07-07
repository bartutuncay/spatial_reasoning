# The Capability Anatomy of Spatial Embeddings — Paper Design

**Date:** 2026-07-07 · **Target:** CVPR 2027 (deadline ~mid-Nov 2026) · **Status:** approved direction, spec for implementation planning

## 1. Context and motivation

Bartu's core asset is a cross-modal embedding space: an RGB encoder (`models/1_img_enc.py`) and a
point-cloud-graph GNN encoder (`models/1_pcd_enc.py`) trained into a shared 128-d latent on ETH3D
random-walk data. The Spatial-JEPA campaign (June 2026) produced a decisive negative — predictive
(JEPA-style) conditioning *hurts* discriminative localization, losing even to a random frozen
encoder in few-shot transfer — alongside two positives: map-conditioning helps when query and map
share one space, and VICReg prevents collapse.

This paper converts that portfolio into a **theory-framed analysis paper**: instead of betting on
one objective, we make *the relationship between conditioning objective and downstream spatial
capability* the object of study. The JEPA negative becomes one cell of a matrix rather than a
failed bet.

## 2. Thesis

> There is no single "spatial representation." The objective — and the set of spatial modalities
> (point cloud, 3D mesh, 2D layout, depth) — used to condition a multimodal embedding space
> determines *which* spatial capabilities it supports:
> invariance-inducing predictive objectives trade within-scene discriminative power (localization,
> place recognition) for cross-scene geometric and dynamical structure (depth, rollout,
> navigation), and information decomposition (sufficiency/invariance + PID synergy) explains the
> trade-off. We demonstrate this with the first controlled anatomy — same data, same encoders,
> matched compute — across 7 controlled conditioning recipes × 7 spatial capability probes,
> anchored against a row of frozen foundation-model references (not compute-matched; they anchor
> each column's dynamic range).

Expected headline finding: a **double dissociation** — objectives that win discriminative columns
lose predictive/dynamical columns and vice versa. (Cell (jepa, localization) is already measured
and negative; the mirrored cells are the open bet. The paper survives either outcome — see §9.)

## 3. Novelty positioning (checked 2026-07-07, web sweep; scite quota exhausted until Aug 1)

| Prior work | What it does | What it lacks (our whitespace) |
|---|---|---|
| Probing the 3D Awareness of Visual Foundation Models (El Banani et al., CVPR'24, arXiv:2404.08636) | Probes frozen 2D foundation models for depth/normals/correspondence | No control of training objective; no cross-modal geometry conditioning; no localization/rollout/navigation probes |
| From Alignment to Prediction (arXiv:2604.13518, Apr 2026) | Compares BYOL/MAE/I-JEPA under an alignment/reconstruction/prediction taxonomy | Unimodal; generic tasks; zero spatial tasks |
| Quantifying & Modeling Multimodal Interactions (Liang et al., arXiv:2302.12247) | PID framework (redundancy/uniqueness/synergy) with estimators | Applied to sentiment/emotion/regression; never to spatial embeddings or RGB+geometry |
| What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction (arXiv:2606.07687, Jun 2026) | Prediction beats reconstruction *for action-relevance* | Single capability; mirror-image of our localization finding — supporting evidence for the dissociation, not a scoop |
| ATM: Action-Consistency Transfer Matrix (arXiv:2606.09028) | Diagnostic for action-identifiability of latent transitions | A tool we adopt for probe C4, not a competing claim |

**Claim nobody holds:** a controlled, objective-swept, modality-swept anatomy of spatial
capabilities in one testbed, with an information-decomposition account of the trade-offs.
Final citation pass must re-verify all arXiv IDs on arXiv/CVF directly.

## 4. Experimental design — the matrix

### Rows: 8 conditioning recipes (identical data, matched compute, 3 seeds)

| # | Recipe | Status |
|---|---|---|
| R1 | `scratch` — random frozen encoder (floor) | have (`fewshot.py --objective scratch`) |
| R2 | `recon` — Bartu's reconstruction VAE | have |
| R3 | `symalign` — Bartu's symmetric latent alignment | have |
| R4 | `contrastive` — cross-modal InfoNCE | have |
| R5 | `jepa` — predictor + EMA target + VICReg | have |
| R6 | RGB-only SSL (unimodal control; SimCLR-style on our frames) | new, small |
| R7 | PCD-only graph SSL (unimodal control; masked-node prediction) | new, small |
| R8 | Frozen references: DINOv2-S/B, SigLIP/CLIP, MASt3R (or DUSt3R) encoder, Qwen2-VL-2B vision tower | new: feature-extraction pipeline; all fit one RTX 4090 |

R6/R7 isolate "cross-modal" from "trained at all." R8 guarantees every column has dynamic range
above floor and connects the analysis to foundation models (probe3d-successor framing).

### Columns: 7 capability probes (frozen encoder + light head; every probe has an explicit floor and ceiling)

| # | Probe | Metric(s) | Floor / ceiling | Status |
|---|---|---|---|---|
| C1 | Localization-in-map | ATE, pose-recall | predict-centroid / oracle-anchor | have (`locate.py`) |
| C2 | Few-shot cross-scene APR | ATE vs K ∈ {20,50,100} | centroid / full-data APR | have (`fewshot.py`) |
| C3 | Metric depth (linear probe) | AbsRel, δ<1.25 | mean-depth / supervised head | small refactor of `1_pcd_dec.py` |
| C4 | Latent rollout (world-model): predict z_{t+k} from z_t + relative motion along random walks | rollout MSE vs horizon k; ATM-style action-identifiability | shuffled-action / copy-last | new (~1 wk) |
| C5 | Graph navigation (agent): goal-conditioned next-hop classification + geodesic-distance regression between view embeddings | next-hop acc; distance R² | random-hop / GT-graph Dijkstra | new (~1 wk) |
| C6 | Cross-view correspondence / relative pose | matching acc; rel-pose error | random match / MASt3R | new (~1 wk) |
| C7 | Place recognition (which scene) | recall@1 | majority-class / — | new (days) |

Probe protocol: encoders frozen; heads are linear or ≤2-layer MLP with a fixed budget; identical
train/val/test splits per column across all rows; 3 seeds; report mean ± std.

### Lens: diagnostics that make it theory-framed

- **Sufficiency/invariance:** InfoNCE lower bounds on I(z; pose) and I(z; appearance) per row —
  the proposed mechanism for the dissociation.
- **PID:** redundancy / uniqueness / **synergy** of (z_rgb, z_pcd) w.r.t. each task label
  (Liang-style estimators). Synergy is the formalization of "richness of the multimodal space."
- Effective rank (have), alignment/uniformity (Wang & Isola), CKA similarity between row spaces.

### Axis 3: modality lattice — richness vs. modalities of space (added 2026-07-07)

How does the embedding's capability profile change as modalities of space are added to the
conditioning set? RGB stays the **anchor** (only query-time modality); each added modality gets
its own encoder and one pairwise RGB↔modality loss term in the existing trainer — no fusion
architecture needed.

| Modality | ETH3D | Replica | Encoder | Cost |
|---|---|---|---|---|
| RGB (anchor) | ✅ | ✅ | `ImgEnc` | have |
| Point-cloud graph | ✅ | ✅ (mesh-sampled) | `PCDEnc` | have |
| 3D model (mesh) | ✗ (skip; Poisson too noisy) | ✅ native | `PCDEnc` variant: mesh-edge graph + normal features | small |
| 2D layout (floor plan) | ✅ derivable | ✅ derivable | small CNN on top-down BEV occupancy raster | small |
| Per-view depth (egocentric) | ✅ in `.pt` walks | ✅ | 1-channel CNN | ~free |
| Semantics | ✗ | ✅ native | label-map CNN | free, Replica-only |

**Sweep design (combinatorics-controlled):** after Wave 2 fixes the top-2 objectives, run a
**nested chain** RGB → +PCD → +mesh → +layout → +depth (→ +semantics, Replica) for the cumulative
richness curve, plus **leave-one-out from the full set** for per-modality necessity.
≈ 11–12 modality-sets × 2 objectives × 7 probes × 3 seeds ≈ 300–450 short jobs.
Egocentric (depth) vs allocentric (PCD/mesh/layout) contrast is reported explicitly; semantics
tests whether semantic space substitutes for geometric space.

**In-modality rule (honesty):** when a modality is in the conditioning set, the probe that
directly targets it (e.g., depth-conditioned rows on C3) is flagged *in-modality* in every
table/figure and excluded from capability-gain claims.

**PID under >2 sources:** full multivariate PID is intractable; report pairwise-vs-anchor PID and
the **marginal synergy** of each added modality per task (chain differences), with estimator
sensitivity analysis.

### Stress and scale axes (reuse existing switches)

- `--map-frac` ∈ {1.0, 0.75, 0.5, 0.25} and `--blur` on C1 for best/worst rows.
- Data-scale curves: walks-per-scene ∈ {small, medium, full} × {R2..R5}.

## 5. Datasets

1. **ETH3D** (have): 13 indoor scenes, existing random-walk pipeline. Primary testbed for Waves 1–2.
2. **Replica** (Wave-1 provisioning): 18 scenes; download + mesh→point-cloud→graph→walks as
   `normal.4h` CPU batch jobs to `/cluster/scratch/aleonel` (never the laptop). Meshes allow
   unlimited walk rendering → the scale axis.
3. **ScanNet** (conditional): PI files the ToS request now (needs institutional email); if approved
   by ~W4, a 20-scene subset replicates the matrix in Wave 3. Not on the critical path.

## 6. Execution waves on Euler

All jobs go through the existing campaign harness (spec_id → sbatch → LEDGER JSONL/MD with
EXISTS/WEAK/DEAD verdicts); idempotent re-runs; results rsync'd off scratch (15-day purge) via the
workstation↔NAS bridge.

- **W1–2 — Build + pilot.** Probes C3–C7; unimodal arms R6–R7; reference-extraction pipeline R8;
  Replica provisioning (CPU jobs, parallel). Pilot every (row, column) pair once on `office`;
  DEAD-verdict anything broken before the sweep.
- **W3–4 — The wide sweep.** Full matrix on ETH3D: 8 rows × 7 columns × 3 seeds ≈ 200–300 short
  GPU jobs streamed through `gpuhe.4h`/`gpupr.4h`.
- **W5–6 — Generality + modality lattice.** Replica matrix; ScanNet subset if approved;
  **modality-lattice sweep** (nested chain + leave-one-out at top-2 objectives — Replica-first,
  since mesh/semantics only exist there); data-scale curves; stress axes on best/worst cells.
  W1–2 additionally builds the three new modality encoders (mesh-graph, BEV-layout CNN, depth CNN)
  and the BEV rasterization preprocessing.
- **W7–8 — Analysis.** InfoNCE bounds, PID estimation, CKA; dissociation analysis; *stretch:*
  VLM-injection demo (adapter from our embedding into Qwen2-VL-2B, spatial-QA delta) only if
  synergy appears where injection could plausibly help.
- **W9–12 — Writing.** Every figure/table auto-generated from LEDGER; CVPR template; anonymize.
- **W13+ — Buffer** (rebuttal-grade extra seeds, ScanNet completion, demo polish).

## 7. Headline figures

- **F1** Capability-matrix heatmap (rows × columns, normalized between floor and ceiling) — the paper in one image.
- **F2** Dissociation scatter: discriminative composite (C1, C2, C7) vs predictive/dynamical composite (C3, C4, C5); one point per recipe; the "no free lunch" frontier.
- **F3** PID bars per task: where synergy lives (and where it doesn't).
- **F4** Data-scale curves per objective.
- **F5** Stress curves (map-frac, blur).
- **F6** **Modality-richness curve:** capability (per task) and marginal PID synergy vs. the nested
  modality chain RGB → +PCD → +mesh → +layout → +depth (→ +semantics), with leave-one-out
  necessity bars — the "what does each abstraction of space buy?" figure.
- **T1** Full matrix with mean ± std, floors, ceilings, in-modality cells flagged.

## 8. Reuse map (nothing thrown away)

| Asset | Role |
|---|---|
| Four-arm trainer (`experiments/exp_jepa/train.py`) | Rows R2–R5 |
| `locate.py`, `fewshot.py` + map-frac/blur switches | C1, C2 + stress axes |
| `1_pcd_dec.py` depth decoder | C3 linear probe |
| Random walks (`agent_src/`, Bartu's 5×1040 walks) | C4 trajectories, C5 graphs, scale axis |
| `metrics/` (ATE, pose-recall, ECE, AURC, cost-triple) | All columns |
| Campaign harness + LEDGER + sbatch generators | The sweep |
| Effective-rank / VICReg diagnostics | Lens |
| JEPA negative results (CAMPAIGN_RESULTS.md) | Cells (R5, C1) and (R5, C2), already measured |

## 9. Risks and mitigations

1. **Dynamic-range mush** (all from-scratch rows at floor on every column): R8 references guarantee
   column dynamic range; Replica adds data; probes carry explicit floors/ceilings. Graceful
   degradation: "anatomy of spatial embeddings including foundation models" still fills the
   probe3d-successor slot.
2. **No dissociation** (one objective dominates every column): also publishable — "X is all you
   need for spatial capabilities at small scale" with the information account of why; weaker but
   honest. Decision point after Wave 2 (W4): if signal is absent, tighten scope to the
   discriminative columns and reliability story (locate/abstain).
3. **PID estimator fragility** on 128-d continuous latents: use Liang's discretized/clustered
   estimators + report sensitivity; PID is a lens, not the load-bearing claim — F1/F2 stand
   without F3.
4. **Probe head capacity confound** ("the probe learned the task, not the embedding"): fixed
   parameter budget across rows, scratch row as control, report head-capacity ablation on one
   column.
5. **ScanNet approval latency**: not on critical path; Replica carries generality.
6. **Reviewer: "toy scale"**: own it — controlled anatomy requires training all rows from scratch;
   foundation-model rows situate the small-scale findings; probe3d precedent for frozen-reference
   analysis at CVPR.

## 10. Approved decisions log

- Direction: theory-framed analysis (capability anatomy), world-model + agent + VLM folded in as probes/references (PI, 2026-07-07).
- Second dataset: **Replica now, ScanNet request filed in parallel** (PI, 2026-07-07).
- VLM injection demo: **stretch goal**, W7–8, only on evidence of synergy (PI, 2026-07-07).
- Reference rows: **DINOv2-S/B + MASt3R/DUSt3R + SigLIP/CLIP + Qwen2-VL-2B vision tower** (PI, 2026-07-07).
- Modality lattice (Axis 3): **nested chain + leave-one-out at top-2 objectives**; modality set =
  {RGB, PCD, mesh, 2D layout, per-view depth} + semantics (Replica-only) (PI, 2026-07-07).
- Action item (PI): file ScanNet ToS request (institutional email required).
