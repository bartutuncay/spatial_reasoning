# Pipeline for Processing

> Run from folder $btuncay/cog$

## Scripts to run: small autoencoder

> **extract_dataset_eth3d.py** *$\rightarrow$ generate dataset if unavailable*

> **process_depth_eth3d.py** *$\rightarrow$ generate 2D metrics if CSV missing*

> **pcd.sbatch** $\Rightarrow$ *calls <u>pcd_slice_extractor_camera.py</u> $\rightarrow$ generate 3D metrics and point cloud graphs*

> **write_dataset.sbatch** $\Rightarrow$ *calls <u>combine_data.py</u> $\rightarrow$ generate training data in $.pt$ format*

> **train.sbatch** $\Rightarrow$ *calls <u>autoencoder_self_supervised.py</u> $\rightarrow$ train model*

> **test_model.ipynb** *$\rightarrow$ test model output in notebook*

## Scripts to run: navigation stack

> **agent_src/random_walk.sbatch** *$\rightarrow$ generate random-walk observations for navigation episodes*

> **preprocessing_src/rw_latents.sbatch** $\Rightarrow$ *calls <u>rw_to_vae.py</u> $\rightarrow$ caches observation latents as `{latent, loc, viewdir}`*

> **preprocessing_src/pcd_reduced.sbatch** $\Rightarrow$ *calls <u>large_graph_from_pcd.py</u> $\Rightarrow$ generates the coarsened large-map point-cloud graph*

> **preprocessing_src/subgraph_lookup.py** $\Rightarrow$ *generates cluster lookup tables and spatial neighbors for `scan_pcd_graph/clusters`*

> **train_le.sbatch** $\Rightarrow$ *calls <u>large_locator_vae.py</u> $\rightarrow$ trains coarse localization against the cluster bank*

> **train_movement.sbatch** $\Rightarrow$ *calls <u>translation_vae.py</u> $\rightarrow$ trains the short-horizon motion head on adjacent or kNN random-walk pairs*

> **train_recon.sbatch** $\Rightarrow$ *calls <u>registration_operator_vae.py</u> $\rightarrow$ trains fine registration and recovery*

> **experiments/1_observation_encoding.py** *$\rightarrow$ defines the Experiment A evaluation blueprint for latent retrieval and local topological navigation*

> **experiments/2_localization.py** *$\rightarrow$ defines the Experiment B evaluation blueprint for cluster ranking and subgoal selection*

> **experiments/3_recovery.py** *$\rightarrow$ defines the Experiment C evaluation blueprint for docking and kidnapped-robot recovery*

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
