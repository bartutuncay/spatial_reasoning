# Spatial Reasoning with GNNs

## Overview

We introduce **SR-GNN**, a framework for studying *global spatial reasoning in graph-based representation learning*. The core idea is to use *space syntax-derived metrics* on floor plans and 3D scans as *input and benchmark* for message-passing neural networks for reconstruction tasks.

Our aim is to better understand the shortcomings of the GNNs and to *enhance generalization capabilities* using domain-specific injections of information; with the intention of learning representations that preserve *global, path-based structure* induced by geometry.

---

## Motivation

Standard message-passing NNs suffer from:

- **Locality bias**: information propagates only within local neighborhood
- **Oversmoothing**: deeper networks lose node representations
- **Limited expressivity**: inability to distinguish globally different but locally similar graphs


Visibility-based metrics expose these limitations:

- Visibility graphs are *dense, non-planar, and geometrically based*
- Visibility metrics depend on *global features*
- Small geometric changes can cause large global visibility changes

These properties make visibility reasoning an ideal diagnostic task for representation learning.

---

## Core Claim

> *Global visibility metrics expose limitations of locally-biased graph representations, and enforcing visibility-aware reconstruction improves preservation of the global spatial structure.*

---

## Conceptual Abstraction

### Inputs
- Partial or abstract spatial observations:
  - Coarse geometry
  - Region adjacency
  - Point samples or local geometric features

### Latent Representation
- Node-level embeddings
- Optional global graph-level embedding
- Fixed-dimensional bottleneck across models

### Targets
- Visibility graph structure (optional)
- Visibility-based spatial metrics (primary)

---

## Visibility Metrics Used

Visibility metrics are computed on **visibility graphs** derived from floor plans or point clouds.

### Primary Metrics (Recommended)

#### Visual Integration
- Analogous to closeness centrality
- Inverse mean shortest-path distance
- Strongly global and non-local
- Sensitive to overall spatial configuration

#### Visual Mean Depth
- Unnormalized version of integration
- Numerically stable and easier to regress
- Useful as auxiliary supervision

#### Visual Choice (Optional)
- Analogous to betweenness centrality
- Measures flow through space
- Path-combinatorial and highly non-local

---

## Why Visibility Metrics Work Well

| Property | Explanation |
|--------|------------|
| Non-locality | Depend on all-pairs shortest paths |
| Density | Visibility graphs are dense and non-planar |
| Geometry–Topology Coupling | Derived directly from line-of-sight constraints |
| Global Sensitivity | Small geometric changes → large global effects |
| Computational Cost | Classical computation scales poorly |

---

## Canonical Task: Visibility-Aware Autoencoding

### Encoder
- Inputs:
  - Coarse spatial graph (rooms, regions, or sampled cells)
  - Local geometric features (from point clouds or plans)
- Architectures evaluated:
  - Message-passing GNN (baseline)
  - Hierarchical / pooling GNN
  - Graph Transformer (non-local attention)
  - (Optional) higher-order / cell-complex models
- Outputs:
  - Node embeddings
  - Optional global latent

### Decoders / Heads

1. **Visibility Graph Decoder (Optional)**
   - Reconstruct visibility edges
   - Binary cross-entropy or AUC-based loss

2. **Visibility Metric Head (Primary)**
   - Predict visual integration / mean depth / choice
   - MSE for values
   - Rank-based loss (Spearman) for choice

---

## Loss Function

The overall training objective combines reconstruction and global reasoning:

\[
\mathcal{L} =
\lambda_E \mathcal{L}_{edges}
+ \lambda_M \mathcal{L}_{metrics}
+ \lambda_G \mathcal{L}_{geom}
+ \lambda_R \mathcal{L}_{reg}
\]

Where:
- \(\mathcal{L}_{edges}\): visibility edge reconstruction (optional)
- \(\mathcal{L}_{metrics}\): visibility metric prediction (primary)
- \(\mathcal{L}_{geom}\): auxiliary geometric reconstruction (optional)
- \(\mathcal{L}_{reg}\): latent regularization (optional, e.g. VAE or VQ)

---

## What VARL Tests Explicitly

| Property | Tested via Visibility Metrics |
|--------|-------------------------------|
| Locality Bias | Can local message passing infer long-range visibility? |
| Oversmoothing | Does depth collapse global visibility distinctions? |
| Expressivity | Can models distinguish globally different layouts? |
| Geometry–Topology Coupling | Can line-of-sight constraints be preserved? |

---

## Experimental Protocol

### Controlled Comparisons
- Same decoder architecture
- Same latent dimensionality
- Same loss weights
- Encoder is the only variable

### Evaluation Metrics

#### Primary
- MSE / R² on visual integration
- Spearman rank correlation on visual choice

#### Secondary
- Visibility edge reconstruction AUC
- Generalization to:
  - Larger layouts
  - Unseen obstacle distributions

### Key Ablation
- With vs. without visibility-metric supervision

---

## Theoretical Motivation

- Visibility metrics are global shortest-path functionals on dense graphs.
- k-local message-passing GNNs cannot compute such metrics in general.
- Visibility graphs provide **natural counterexamples** to local expressivity assumptions.
- Empirical failures align with known Weisfeiler–Lehman limitations and oversmoothing theory.

---

## Practical Outcome: Fast Visibility Analysis

While VARL is framed as a representation-learning diagnostic, it also yields a practical benefit:

- Learned models act as **amortized surrogates** for visibility analysis.
- Orders-of-magnitude faster than classical all-pairs shortest-path computation.
- Accuracy evaluated via absolute error and rank preservation.

This is a *consequence*, not the primary contribution.

---

## What Is Intentionally Excluded

To keep the concept focused:

- Architectural design theory
- Urban interpretation of syntax values
- Multiple syntax families (axial, segment, VGA combined)
- Heavy geometric or photorealistic reconstruction

---

## Key Contribution (One Sentence)

> **Visibility-Aware Representation Learning uses global visibility metrics as a principled stress test for graph representations, revealing the limitations of locality-biased models and identifying mechanisms required to preserve global spatial structure.**

---

## Suggested Extensions

- Synthetic visibility counterexample datasets
- Out-of-distribution generalization across layout scales
- Formal lower bounds on k-local expressivity
- Integration with neural algorithmic reasoning benchmarks

---

## License

MIT (proposed)

---

## Contact

For questions, benchmarks, or collaboration, please open an issue or contact the authors.

## Applications and Implications.

The ability to preserve global visibility structure has several practical consequences. First, our approach enables amortized approximation of visibility-based spatial metrics, providing near–real-time estimates of visual integration, mean depth, and related measures without repeated all-pairs shortest-path computation. This is particularly relevant in architectural and interior design workflows, where visibility analysis is often recomputed across many design iterations. Second, visibility-aware representations support robust structural reconstruction from incomplete or noisy observations, such as partial floor plans or indoor scans, by enforcing global line-of-sight consistency rather than relying solely on local geometry. Third, the learned representations enable rapid what-if and sensitivity analyses, allowing practitioners to assess how small geometric changes (e.g., adding partitions or openings) affect global visibility patterns. More broadly, because visibility graphs arise in navigation, surveillance, and embodied perception, the proposed framework provides a general mechanism for incorporating global line-of-sight reasoning into downstream systems without explicitly recomputing dense visibility graphs at inference time.