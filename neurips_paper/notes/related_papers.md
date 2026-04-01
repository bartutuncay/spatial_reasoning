# Related Papers to Add

This note expands the papers already present in `literature/` into a short reading list that is closer to the current codebase.

Last updated: March 31, 2026

## Existing anchors in `literature/`

Best-effort identification from filenames and embedded metadata:
- `1706.02413v1.pdf`: PointNet++
- `2012.09164v2.pdf`: Point Transformer
- `1908.00575v1.pdf`: StructureNet
- `2108.13459v2.pdf`: LSD-StructureNet
- `STD-Net_...pdf`: STD-Net
- `SceneHGN_...pdf`: SceneHGN
- `Learning_Unsupervised_Hierarchical_Part_Decomposition_...pdf`
- `Computer Graphics Forum - 2023 - Hu - SPCNet...pdf`
- `s11263-021-01546-9.pdf`: Learning 3D Semantic Scene Graphs with Instance Embeddings
- `25092-Article Text-29155-1-2-20230626.pdf`: Layout Representation Learning with Spatial and Structural Hierarchies

## Recommended additions

### Point-cloud backbones and pretraining

1. Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling (CVPR 2022)
Why it fits: useful if you want a stronger point-cloud encoder than the current PointNet++ / Point Transformer baseline framing.
Link: https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.html

2. PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies (NeurIPS 2022)
Why it fits: a very practical reference for stronger point-cloud training recipes without abandoning the PointNet++ family.
Link: https://arxiv.org/abs/2206.04670

3. Point Transformer V2: Grouped Vector Attention and Partition-based Pooling
Why it fits: relevant if you want to argue for stronger non-local geometry reasoning than the current GCN-style point-cloud encoder.
Link: https://arxiv.org/abs/2210.05666

### Point-cloud completion and partial-to-complete geometry

4. PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers (ICCV 2021)
Why it fits: directly relevant to the image/partial-point-cloud to geometry reconstruction side of the project.
Link: https://openaccess.thecvf.com/content/ICCV2021/html/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.html

5. SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer (ICCV 2021)
Why it fits: strong completion baseline with coarse-to-fine decoding, useful against the current graph decoder framing.
Link: https://openaccess.thecvf.com/content/ICCV2021/html/Xiang_SnowflakeNet_Point_Cloud_Completion_by_Snowflake_Point_Deconvolution_With_Skip-Transformer_ICCV_2021_paper.html

6. SeedFormer: Patch Seeds Based Point Cloud Completion with Upsample Transformer (ECCV 2022)
Why it fits: another strong completion baseline that emphasizes local detail recovery from partial point sets.
Link: https://arxiv.org/abs/2207.10315

### Scene graphs, topology, and spatial reasoning

7. EgoSG: Learning 3D Scene Graphs from Egocentric RGB-D Sequences (CVPRW 2024)
Why it fits: very close to the repository's egocentric observation setting and useful for positioning the random-walk stage.
Link: https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/html/Zhang_EgoSG_Learning_3D_Scene_Graphs_from_Egocentric_RGB-D_Sequences_CVPRW_2024_paper.html

8. Multiview Scene Graph (NeurIPS 2024)
Why it fits: strong conceptual match for turning sequential observations into a topological scene representation instead of only metric pose predictions.
Link: https://openreview.net/forum?id=1ELFGSNBGC

9. 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning (CVPR 2025)
Why it fits: good forward-looking citation if you want to motivate scene memory and embodied spatial reasoning rather than only reconstruction.
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Yang_3D-Mem_3D_Scene_Memory_for_Embodied_Exploration_and_Reasoning_CVPR_2025_paper.html

### Hierarchical structure and layout reasoning

10. Layout Representation Learning with Spatial and Structural Hierarchies (AAAI 2023)
Why it fits: already in your folder, but worth citing explicitly if you frame the paper around multiscale spatial structure rather than only geometry.
Link: https://research.adobe.com/publication/layout-representation-learning-with-spatial-and-structural-hierarchies/

11. Learning 3D Semantic Scene Graphs with Instance Embeddings (IJCV 2022)
Why it fits: important scene-graph baseline for any claim about structured 3D scene understanding.
Link: https://portal.fis.tum.de/en/publications/learning-3d-semantic-scene-graphs-with-instance-embeddings/

## Suggested citation clusters for the paper

- Encoder baselines: PointNet++, Point Transformer, Point-BERT, PointNeXt
- Geometry reconstruction/completion: STD-Net, SPCNet, PoinTr, SnowflakeNet, SeedFormer
- Structural priors: StructureNet, LSD-StructureNet, SceneHGN
- Scene-level reasoning: Learning 3D Semantic Scene Graphs, EgoSG, Multiview Scene Graph, 3D-Mem

## Most useful papers for this repository specifically

If you only add five for the first draft, I would use:
- PointNeXt
- PoinTr
- SeedFormer
- EgoSG
- Multiview Scene Graph

## Recent frontier: 2024-2026

These are the papers most worth reading if you want the current state of the art around spatial reasoning, reconstruction, localization, and structured scene understanding.

### Reconstruction and geometry foundation models

1. DUSt3R: Geometric 3D Vision Made Easy (CVPR 2024)
Why it matters: this is one of the key papers behind the current feed-forward reconstruction wave and is now a standard reference point for pose-free multi-view reconstruction.
Link: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.html

2. MUSt3R: Multi-view Network for Stereo 3D Reconstruction (CVPR 2025)
Why it matters: extends DUSt3R from pairwise to multi-view prediction in a common frame and is highly relevant for scalable scene reconstruction.
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Cabon_MUSt3R_Multi-view_Network_for_Stereo_3D_Reconstruction_CVPR_2025_paper.html

3. Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass (CVPR 2025)
Why it matters: a strong scaling paper for large image collections, useful if you want to position against modern efficient multi-view reconstruction.
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Fast3R_Towards_3D_Reconstruction_of_1000_Images_in_One_Forward_CVPR_2025_paper.html

4. MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds (CVPR 2025)
Why it matters: strong sparse-view reconstruction paper and especially relevant for partial-observation settings.
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Tang_MV-DUSt3R_Single-Stage_Scene_Reconstruction_from_Sparse_Views_In_2_Seconds_CVPR_2025_paper.html

5. VGGT: Visual Geometry Grounded Transformer (CVPR 2025)
Why it matters: broad, strong visual-geometry model that jointly predicts camera parameters, depth, point maps, and tracks.
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html

### Reconstruction with stronger spatial reasoning

6. Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning (CVPR 2024)
Why it matters: directly connects explicit spatial reasoning to better reconstruction under occlusion and ambiguity.
Link: https://openaccess.thecvf.com/content/CVPR2024/html/Li_Know_Your_Neighbors_Improving_Single-View_Reconstruction_via_Spatial_Vision-Language_Reasoning_CVPR_2024_paper.html

7. Scene Coordinate Reconstruction Priors (ICCV 2025)
Why it matters: very strong positioning paper for any manuscript that sits between reconstruction and relocalization, especially if geometry priors are part of the story.
Link: https://openaccess.thecvf.com/content/ICCV2025/html/Bian_Scene_Coordinate_Reconstruction_Priors_ICCV_2025_paper.html

8. MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors (CVPR 2025)
Why it matters: demonstrates how modern reconstruction priors can become a localization/SLAM system, which is especially relevant to your second-stage locator.
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html

### Spatial reasoning and scene memory

9. Multiview Scene Graph (NeurIPS 2024)
Why it matters: a very good conceptual bridge between image collections, topological scene structure, and reasoning over place/object relations.
Link: https://openreview.net/forum?id=1ELFGSNBGC

10. 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning (CVPR 2025)
Why it matters: one of the strongest recent papers on scene memory and long-horizon embodied spatial reasoning.
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Yang_3D-Mem_3D_Scene_Memory_for_Embodied_Exploration_and_Reasoning_CVPR_2025_paper.html

11. g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks (CVPR 2025)
Why it matters: useful if you want to position your work more broadly as a reusable 3D scene representation rather than only a task-specific pipeline.
Link: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_g3D-LF_Generalizable_3D-Language_Feature_Fields_for_Embodied_Tasks_CVPR_2025_paper.html

12. SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning (NeurIPS 2025)
Why it matters: useful for framing explicit spatial reasoning as a first-class objective, though it is more VLM-oriented than your current project.
Link: https://openreview.net/forum?id=hFaXVjRFHI

### Scene graphs and structured 3D scene understanding

13. EgoSG: Learning 3D Scene Graphs from Egocentric RGB-D Sequences (CVPRW 2024)
Why it matters: very close to your egocentric observation setting and important if you want scene-graph references beyond static reconstructions.
Link: https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/html/Zhang_EgoSG_Learning_3D_Scene_Graphs_from_Egocentric_RGB-D_Sequences_CVPRW_2024_paper.html

14. Hierarchical 3D Scene Graphs Construction Outdoors (ICCV 2025)
Why it matters: a recent structured-scene paper that organizes reconstructions into explicit hierarchies and is good for positioning claims about scene-level structure.
Link: https://openaccess.thecvf.com/content/ICCV2025/html/Nyffeler_Hierarchical_3D_Scene_Graphs_Construction_Outdoors_ICCV_2025_paper.html

## Fast shortlist for this project

If the goal is to position your paper well without overloading the first draft, start here:
- DUSt3R
- MUSt3R
- VGGT
- Scene Coordinate Reconstruction Priors
- MASt3R-SLAM
- Multiview Scene Graph
- 3D-Mem

## Practical positioning takeaway

The recent literature suggests three strong paper lanes:
- Reconstruction-first lane: DUSt3R, MUSt3R, Fast3R, MV-DUSt3R+, VGGT
- Reconstruction-plus-localization lane: Scene Coordinate Reconstruction Priors, MASt3R-SLAM
- Spatial-reasoning / structured-memory lane: Multiview Scene Graph, 3D-Mem, g3D-LF, SpatialReasoner, EgoSG

Your repository is strongest in the overlap between the second and third lanes, with a multimodal graph representation sitting on top of reconstructed geometry.
