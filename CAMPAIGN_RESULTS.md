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
