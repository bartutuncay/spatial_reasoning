# NeurIPS Submission Checklist

Last updated: March 31, 2026

This checklist is tailored to the current project and is meant to answer one practical question:

> What would need to be true for this work to be competitive at NeurIPS?

There is no guaranteed accept. The goal of this checklist is to make the acceptance bar concrete.

## 1. Claim

- The paper has one headline claim, not several.
- The claim is about global spatial reasoning under partial visibility, not only better reconstruction.
- The title, abstract, introduction, and experiments all support the same claim.

## 2. Track Fit

- Decide early whether the submission is for the Main Track or the Evaluations & Datasets Track.
- If Main Track: the method is the core contribution.
- If Evaluations & Datasets: the benchmark, protocol, or evaluation framework is the core contribution.
- Do not try to split the paper between both tracks.

## 3. Problem Setup

- The task can be explained clearly in two or three sentences.
- Inputs, outputs, and supervision are formally defined.
- The role of visibility metrics is explicit and justified.
- If both reconstruction and localization are included, their relationship is stated cleanly.

## 4. Novelty

- A reviewer can summarize what is new in one sentence.
- The novelty is not just multimodal fusion.
- The novelty is not just applying a GNN to point clouds.
- The work contributes at least one of:
  - a visibility-aware representation,
  - a graph-based spatial reasoning mechanism,
  - a new benchmark or stress test for partial visibility and global structure,
  - or a new empirical finding about failure modes of current methods.

## 5. Baselines

- Comparisons include modern baselines, not only older GNN papers.
- For geometry and reconstruction, compare against some subset of:
  - DUSt3R
  - MUSt3R
  - VGGT
  - MV-DUSt3R+
- For localization or relocalization, compare against some subset of:
  - Scene Coordinate Reconstruction Priors
  - MASt3R-SLAM
  - a strong image-only pose baseline
- For structured reasoning, position against:
  - Multiview Scene Graph
  - EgoSG
  - 3D-Mem

## 6. Evaluation

- The metrics match the central claim.
- If the paper claims global reasoning, geometry-only metrics are not enough.
- Include at least one metric that captures:
  - topology,
  - long-range spatial consistency,
  - visibility reasoning,
  - or localization quality under partial observation.
- Also report standard metrics so reviewers can anchor the results.

## 7. Killer Experiment

- There is one experiment that clearly shows existing methods fail where the proposed method succeeds.
- This experiment is central to the paper, not buried in the appendix.
- Strong examples include:
  - partial visibility sweeps,
  - topology-preserving versus geometry-only comparison,
  - cross-scene generalization under occlusion,
  - or cases where local appearance is ambiguous but global structure resolves the task.

## 8. Ablations

- Remove visibility supervision.
- Remove graph structure.
- Remove cross-modal latent alignment.
- Remove map conditioning.
- Compare image-only, geometry-only, and multimodal variants.
- Show each major component changes results in a coherent way.

## 9. Robustness

- Test different levels of observed map coverage.
- Test unseen scenes or different scene scales.
- Test point-cloud noise or downsampling.
- Test appearance shift if images are part of the model input.

## 10. Evidence Quality

- Results are stable across seeds or splits.
- Tables are not cherry-picked.
- Qualitative results include failures, not only successes.
- Claims stay within what the experiments actually support.

## 11. Writing

- The introduction motivates a real gap in the current literature.
- Related work explains why strong reconstruction methods are still insufficient for the target reasoning task.
- The method section matches the implemented system cleanly.
- Limitations are stated clearly.

## 12. Reproducibility

- Dataset generation is documented end to end.
- Splits are fixed and described.
- Hyperparameters and preprocessing are reproducible.
- Evaluation scripts are stable enough to rerun.

## 13. Acceptance Signal

At minimum, the paper should make a skeptical reviewer think:

> I may not love every design choice, but this is a real problem, the evaluation is careful, and the paper teaches me something new.

## 14. Minimum Competitive Bar

If the project is short on time, this is the minimum package to aim for:

- one crisp claim,
- one killer experiment,
- three to five strong modern baselines,
- one full ablation table,
- one robustness study,
- and a clear failure analysis.

## 15. Practical Reading of the Current Project

For this repository specifically, the strongest NeurIPS path is likely one of:

- Main Track:
  a visibility-aware multimodal graph method for global spatial reasoning under partial observation

- Evaluations & Datasets Track:
  a benchmark or evaluation protocol showing that current reconstruction and localization methods fail under visibility-driven spatial reasoning stress tests

At the current stage of the codebase, the Evaluations & Datasets angle may be the safer path unless the method and experiments are tightened substantially.
