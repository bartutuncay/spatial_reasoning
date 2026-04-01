# Pipeline for Processing

> Run from folder $btuncay/cog$

## Scripts to run: small autoencoder

> **extract_dataset_eth3d.py** *$\rightarrow$ generate dataset if unavailable*

> **process_depth_eth3d.py** *$\rightarrow$ generate 2D metrics if CSV missing*

> **pcd.sbatch** $\Rightarrow$ *calls <u>pcd_slice_extractor_camera.py</u> $\rightarrow$ generate 3D metrics and point cloud graphs*

> **write_dataset.sbatch** $\Rightarrow$ *calls <u>combine_data.py</u> $\rightarrow$ generate training data in $.pt$ format*

> **train.sbatch** $\Rightarrow$ *calls <u>autoencoder.py</u> $\rightarrow$ train model*

> **test_model.ipynb** *$\rightarrow$ test model output in notebook*

## Scripts to run: large autoencoder

> **random_walk.sbatch** *$\rightarrow$ generate training data based on a random walk*

> **rw_latents.sbatch** $\Rightarrow$ *calls <u>rw_to_ae.py</u> $\rightarrow$ runs random walk data through autoencoder*

> **pcd_reduced.sbatch** $\Rightarrow$ *calls <u>large_graph_from_pcd.py</u> $\Rightarrow$ generates radius graph from coarsened point cloud*

> **subgraph_lookup.py** $\Rightarrow$ *generates lookup table for clusters in the large point cloud based on spatial adjacency*

> **train_le.sbatch** $\Rightarrow$ *calls <u>large_locator.py</u>*

## Available Data
### Hospital
- Floor plans (synthetic) $\rightarrow$ visible area, closest point, farthest point
- Floor plans (extracted) $\rightarrow$ visible area, distances stdNorm
- Mesh (from floor plan)
- Point cloud

#### ETH3D
- dslr_calibration_undistorted
- dslr_scan_eval
- ground_truth_depth
- images
- masks_for_images
- occlusion
- scan_raw

#### NIDAU

## Scene Graph
- From images:
    - Visible portion of the point cloud
    - Visible portion of the plan
    - Visible area (from plan)
    - Visible volume (from point cloud)
- From scans:
    - Entire point cloud
    - Per-point visible area
    - Per-point visible volume


## Model Architecture
### Autoencoder with Neural Operators / MPNN
- *plans, pcd* $\rightarrow$ generate metrics
- *labeled data* $\rightarrow$ generate metrics

