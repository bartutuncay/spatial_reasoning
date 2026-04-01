# Ranked Positioning List

Last updated: March 31, 2026

This note ranks recent papers by how useful they are for positioning the current project.

The current repository is not best framed as a pure feed-forward reconstruction model. It is better framed as:

> a visibility-aware multimodal representation-learning and localization system that aligns egocentric image observations with partial scene geometry and a larger graph-structured map.

Because of that, the ranking below prioritizes papers that help explain where your manuscript sits, not only papers that top a benchmark.

## Tier 1: Most important for positioning

### 1. Scene Coordinate Reconstruction Priors (ICCV 2025)
Link: https://openaccess.thecvf.com/content/ICCV2025/html/Bian_Scene_Coordinate_Reconstruction_Priors_ICCV_2025_paper.html
Why it ranks first:
- It sits directly between reconstruction and relocalization.
- It makes learned geometry priors central to better scene structure and better camera poses.
- It is one of the clearest current references for arguing that better scene representations improve localization.
How to position against it:
- Their representation is scene-coordinate regression with reconstruction priors.
- Your project is multimodal and graph-structured, with partial observed geometry and latent alignment between image and point-cloud views.

### 2. MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors (CVPR 2025)
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html
Why it ranks high:
- It shows the strongest recent bridge from reconstruction priors to localization and mapping.
- It is exactly the kind of paper a reviewer may expect to see in a related-work section for your locator stage.
How to position against it:
- Their system is SLAM-first and built around strong two-view reconstruction priors.
- Your project is representation-first, using graph-structured partial observations and latent matching against a global scene graph.

### 3. Multiview Scene Graph (NeurIPS 2024)
Link: https://openreview.net/forum?id=1ELFGSNBGC
Why it ranks high:
- It gives you a modern scene-representation reference that is explicitly topological.
- It supports framing your latent graph as more than just a pose regressor.
How to position against it:
- Their focus is scene graph construction from image collections.
- Your focus is cross-modal latent alignment plus graph-conditioned localization.

### 4. 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning (CVPR 2025)
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Yang_3D-Mem_3D_Scene_Memory_for_Embodied_Exploration_and_Reasoning_CVPR_2025_paper.html
Why it ranks high:
- It is a strong current reference for structured scene memory and spatial reasoning.
- It helps frame the project as scene understanding rather than only reconstruction quality.
How to position against it:
- Their focus is embodied exploration and memory management.
- Your project is more geometry-grounded and representation-learning-driven.

### 5. EgoSG: Learning 3D Scene Graphs from Egocentric RGB-D Sequences (CVPRW 2024)
Link: https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/html/Zhang_EgoSG_Learning_3D_Scene_Graphs_from_Egocentric_RGB-D_Sequences_CVPRW_2024_paper.html
Why it ranks high:
- Very close to your egocentric observation setting.
- Good for motivating structured reasoning directly from observation sequences.
How to position against it:
- EgoSG is scene-graph prediction from RGB-D streams.
- Your project uses visible point-cloud graphs plus cross-modal autoencoding and downstream localization.

## Tier 2: Important reconstruction references

### 6. DUSt3R: Geometric 3D Vision Made Easy (CVPR 2024)
Link: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html
Why it matters:
- It is one of the main recent reference points for feed-forward geometric reconstruction.
- Reviewers may mentally compare any geometry-first paper to DUSt3R-like methods.
Positioning use:
- Cite as the start of the current reconstruction wave, not as the closest conceptual neighbor.

### 7. MUSt3R: Multi-view Network for Stereo 3D Reconstruction (CVPR 2025)
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Cabon_MUSt3R_Multi-view_Network_for_Stereo_3D_Reconstruction_CVPR_2025_paper.html
Why it matters:
- Important if you want to demonstrate awareness of where multi-view reconstruction has moved after DUSt3R.
Positioning use:
- Cite as a stronger scalable multi-view baseline family.

### 8. VGGT: Visual Geometry Grounded Transformer (CVPR 2025)
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html
Why it matters:
- A broad and strong geometric foundation model.
- Good benchmark-era reference for scene geometry estimation.
Positioning use:
- Cite as a modern geometry-heavy transformer alternative.

### 9. MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds (CVPR 2025)
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Tang_MV-DUSt3R_Single-Stage_Scene_Reconstruction_from_Sparse_Views_In_2_Seconds_CVPR_2025_paper.html
Why it matters:
- Strong sparse-view reconstruction paper.
Positioning use:
- Especially useful if your experiments include sparse or partial map observations.

### 10. Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass (CVPR 2025)
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Fast3R_Towards_3D_Reconstruction_of_1000_Images_in_One_Forward_CVPR_2025_paper.html
Why it matters:
- Important scale reference.
Positioning use:
- Cite if you want to argue your method is not chasing pure reconstruction throughput at web scale.

## Tier 3: Supporting references for explicit spatial reasoning

### 11. Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning (CVPR 2024)
Link: https://openaccess.thecvf.com/content/CVPR2024/html/Li_Know_Your_Neighbors_Improving_Single-View_Reconstruction_via_Spatial_Vision-Language_Reasoning_CVPR_2024_paper.html
Why it matters:
- Good supporting citation for the claim that explicit spatial context helps reconstruction.

### 12. g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks (CVPR 2025)
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_g3D-LF_Generalizable_3D-Language_Feature_Fields_for_Embodied_Tasks_CVPR_2025_paper.html
Why it matters:
- Good if the paper leans toward reusable 3D representations across tasks.

### 13. SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning (NeurIPS 2025)
Link: https://openreview.net/forum?id=hFaXVjRFHI
Why it matters:
- Good for discussion and motivation, but farther from your implementation than the papers above.

### 14. Hierarchical 3D Scene Graphs Construction Outdoors (ICCV 2025)
Link: https://openaccess.thecvf.com/content/ICCV2025/html/Nyffeler_Hierarchical_3D_Scene_Graphs_Construction_Outdoors_ICCV_2025_paper.html
Why it matters:
- Good support for claims about explicit scene hierarchy and graph structure.
- Less central than the indoor / egocentric / latent-alignment papers for your current repo.

## Recommended citation strategy for your paper

If you want a compact, high-signal related-work section, prioritize these first:
- Scene Coordinate Reconstruction Priors
- MASt3R-SLAM
- Multiview Scene Graph
- 3D-Mem
- EgoSG
- DUSt3R
- MUSt3R
- VGGT

If the paper becomes more reconstruction-heavy, move DUSt3R, MUSt3R, VGGT, and MV-DUSt3R+ upward.

If the paper becomes more reasoning-heavy, move Multiview Scene Graph, 3D-Mem, EgoSG, g3D-LF, and SpatialReasoner upward.

## Suggested manuscript angle

The best current positioning is:
- not a pure reconstruction foundation model,
- not a generic embodied reasoning agent,
- but a multimodal graph-based scene representation method that uses partial geometry and visibility-aware supervision for downstream spatial localization and reasoning.
