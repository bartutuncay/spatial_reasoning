# Critical review of `paper/anatomy.tex` for CVPR 2027 (2026-07-10)

Three independent inputs: (a) a fresh-eyes CVPR-reviewer pass over the compiled
draft (no author context), (b) a prior-art/novelty threat scan (web-verified),
(c) a per-seed statistical audit of both bulk tallies. Verdict of the cold
reviewer: **2/5 weak reject** in current form — but every top weakness is
fixable, and the novelty scan found **no scoop**.

---

## 1. Novelty (make-or-break dimension): NOT scooped, but pitch must move

No prior work varies *only* the SSL conditioning objective from scratch at
matched data/compute in a cross-modal (RGB + point-cloud-graph) testbed and
reads out a spatial+predictive capability battery; nobody reports an
objective-level contrastive↔JEPA double dissociation. The conjunction is ours.
But each ingredient has close neighbors that MUST be engaged:

| Threat | Work | Status |
|---|---|---|
| "Objectives determine capabilities" known for CL-vs-MIM on images | Park et al. ICLR'23 (2305.00729); Shekhar et al. (2304.13089) | OVERLAPS — framing level |
| Checkpoint-level spatial probing is crowded | Chen et al. CVPR'25 mid-level probing (2411.17474); Lexicon3D; Feat2GS | OVERLAPS — but all *checkpoint* probing, none objective-controlled |
| Cross-modal 2D-3D SSL → spatial reps | Concerto NeurIPS'25 (2510.23607); CrossJEPA (2511.18424); Sonata CVPR'25 (2503.16429) | OVERLAPS on message, not anatomy |
| **Action-shuffle diagnostic formalized concurrently** | Hansen & Wang 2606.27326 (Jun'26); UWM-JEPA 2605.25313 | **OVERLAPS (concurrent)** — we must cite it as *adopted*, not contributed |
| JEPA→slow-features anticipates our rollout mechanism | Sobal et al. 2211.10831 | Must engage when interpreting the negative |
| Double-dissociation caution | Plaut 1995 (dissociation ≠ modularity) | Cite defensively |

**Positioning pitch:** controlled causal design (vs correlational checkpoint
probing) × cross-modal testbed × discriminative-relational + predictive battery
× the double dissociation with an information/rank mechanism. Contribution (iv)
must be reworded: the shuffle *diagnostic* is concurrent work; the *finding on
our testbed* is ours. ~25 verified citations to add (list in the novelty-scan
appendix below).

## 2. Statistical integrity (audited against tallies)

1. **The paper claims 3 seeds; 8 ETH3D cells had n=2 and one n=4 (duplicate).**
   Duplicate removed; 8 gap-fill jobs queued 2026-07-10; tables regenerate on
   drain. `make_tables.py` now emits mean±std and flags n≠3 / duplicates.
2. **jepa ETH3D depth = 1.385 ± 0.424 (n=2)** — the "jepa worst at depth"
   ETH3D claim currently rests on 2 high-variance seeds. Third seed in flight.
3. **ScanNet depth: contrastive .482±.030 ≈ scratch .508±.012.** "Contrastive
   best" holds only among trained rows; vs random init it is within noise. The
   robust ScanNet depth fact: predictive/image-only rows (.86–.93) are
   *reliably worse than random* — training on the wrong objective actively
   destroys linear depth. (Also correct CAMPAIGN_RESULTS "depth anomaly
   RESOLVED" — partially resolved; the scratch-competitive part persists.)
4. **ScanNet rollout: scratch .371±.001 nearly matches jepa .392±.003** (all 3
   jepa seeds > all 3 scratch seeds, so the gap is real but tiny), and scratch
   beats every other trained row. Honest reframe: random features already carry
   the temporal smoothness; contrastive *destroys* it (.299); jepa preserves
   and slightly sharpens it. The dissociation direction survives, the "jepa
   buys rollout" phrasing does not.
5. **Fusion frontier is NOT monotone in place-rec** (.787→.758→.792, ±.03).
   Abstract says "convex frontier", Results say "monotone" — both falsified by
   our own table. Keep the *asymmetry* claim (rollout survives fusion,
   collapses only at pure contrastive — that part is clean: .312/.290/.275 vs
   −.093), drop monotone/convex.
6. **ETH3D depth is a 3-way statistical tie** (recon .872±.10, symalign
   .875±.01, contrastive .879±.13); bolding recon implies a false winner.

## 3. Prose–table drift (reviewer W3, verified)

- Abstract "wins every discriminative and relational task" — false at ScanNet
  rel-pose (rgb-only .073 best) and ETH3D depth (tie).
- "jepa is worst [ScanNet depth]" — sym-align .927 is worse.
- Fig. 1 caption "Foundation refs dominate all but rollout" contradicts our own
  limitation that rollout is not comparable across latent dims.
- Method promises 4 reference rows; tables show only DINOv2-B (data for all 4
  exists — regenerate tables with all refs).

## 4. Deeper design critiques (reviewer W4–W7)

- **Undertraining artifact risk:** 1500 steps, 128-d, 5 scenes. Different
  objectives converge at different rates → dissociation could be differential
  convergence. Need training-length curves (contrastive+jepa at 5×/10× steps).
- **"Spatial" oversell:** place-rec tracks appearance (our own mechanism
  section says so); on ScanNet the genuinely geometric columns are at floor.
  Either retitle/reframe or lean harder on ETH3D nav-dist/rel-pose as the
  geometric anchors and discuss explicitly.
- **Mechanism is correlational with visible counterexamples:** scratch and
  rgb-only have the *highest* I(z;appearance) (~2.1 nats) and low place-rec.
  ρ=0.60 is inflated by the ref cluster. Honest version: appearance info is
  necessary-not-sufficient; among cross-modally-aligned rows it tracks
  discrimination. Ideal: an intervention (project out the appearance subspace,
  measure the place-rec drop) to earn causal language.
- **Probe protocol unspecified:** head architecture/capacity, floors, splits
  (held-out frames vs held-out scenes), MI estimator, effective-rank
  definition, horizon k — none in the paper. Needs an experimental-details
  section + supplementary; probe-capacity sweep (linear vs 2-layer MLP).
- **Action-shuffle null lacks a positive control** (show the harness detects
  action-conditioning when it exists, e.g. a supervised dynamics head).
- **No predictive foundation reference:** V-JEPA features as a 5th ref row
  would directly test whether the trade-off persists at scale.
- Fig. 3: overprinted ref labels; fuse.50/recon collision.

## 5. Priority plan

**Tier 0 — in flight (done today):** gap-fill 8 seeds; dedupe; ±std tables
with n-flags; drain watcher.

**Tier 1 — text fixes after retally (no new compute):** fix all drift items
(§3), soften frontier claim, confront scratch rows in prose, reword
contribution (iv) + cite concurrent shuffle work, add ~25 verified related-work
citations + positioning paragraphs, all-4-refs tables, fig3 label fix, ρ
recomputed excluding refs, experimental-details section skeleton.

**Tier 2 — cheap Euler experiments (queueable now):**
1. Probe-capacity sweep (linear vs MLP-2) on placerec/depth/rollout.
2. Rollout horizon sweep k∈{1,2,4,8}.
3. Rollout Δ reported over scratch row (recompute from existing caches).
4. Appearance-subspace removal intervention (causal mechanism).
5. Training-length curves: contrastive+jepa at 5k/15k steps (2 rows × 2
   lengths × 3 seeds = 12 pretrain jobs + probes).
6. V-JEPA reference-row extraction + 5 probes × 3 seeds.
7. Action-shuffle positive control (supervised action-conditioned head).

**Tier 3 — structural (needs PI decision):** title/framing response to the
"spatial" critique; growing 4→8 pages (experimental details, discussion,
qualitative figure, supplementary); code-release statement.

---

*Full reviewer report and novelty scan preserved in the session transcript;
citation list with verified arXiv IDs is in the novelty-scan section above.*
