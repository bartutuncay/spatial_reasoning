# spatial_reasoning

Code for **Spatial Reasoning from Monocular RGB with Graph Neural Networks**.

This repo builds training data from point clouds, samples random walks, converts point clouds into graph data, and trains the current models.

## Repository Layout

- `datasets/`: raw downloaded datasets. `datasets/links.txt` lists ETH3D download links. Habitat/Matterport, TUM RGBD and 7-Scenes datasets are supported.
- `datasets_processed/`: extracted and processed data.
- `preprocessing_src/`: dataset extraction, point-cloud graph generation, and random walk helpers.
- `agent_src/`: random walk data generator from GT point clouds.
- `models/`: active model definitions.
- `training_scripts/`: training entry points and Slurm scripts.
- `examples/`: notebooks and small utilities for inspecting outputs.

## Model Status
Currently models are capable of generating depth from monocular images and locating views in globally reconstructed point clouds. The full pipeline is set to include a visual odometry module in addition to a point cloud registration pipeline. 

Our thesis is that GNNs can be made to carry the right biases to enable spatial reasoning, and that their relational modus operandi is more efficient compared to CNN or transformer-based baselines. The framework aims to unify the representation space for all expert models to form a complete SLAM and navigation pipeline with geometric priors.

- `1_model`: autoencoder / depth reconstruction model.
- `2_movement`: *work in progress: visual odometry model.*
- `3_registration`: *work in progress: point cloud registration model.*
- `4_model_locator`: locator model.

## Pipeline

### 1. Download Data

Download the ETH3D archives listed in:

```bash
datasets/links.txt
```

Place each archive under `datasets/<scene_name>/`, e.g:

```text
datasets/relief/relief_scan_raw.7z
datasets/office/office_scan_raw.7z
```

### 2. Extract Data

Use:

```bash
python3 preprocessing_src/extract_dataset_eth3d.py
```

This script currently uses a hard-coded `location` variable. Set it to the scene you want before running.

Extracted data is written under:

```text
datasets_processed/<scene_name>/
```

### 3. Parse Camera Metadata

For ETH3D scenes with DSLR calibration files, parse `images.txt` into CSV:

```bash
python3 preprocessing_src/process_dataset_eth3d.py
```

This writes files such as:

```text
images_parsed.csv
```

Some scripts still have scene paths hard-coded, so check the file before running it on a new scene.

### 4. Generate Random Walk Views

Random-walk scripts sample valid camera locations from scene rectangles, raycast the point cloud, and save `.pt` training samples.

The JSON-based entry point is:

```bash
python3 agent_src/random_walk_json.py --task-index 0 --scene pipes
```

Available scene configs are in:

```bash
agent_src/random_walk_configs.json
```

List configured scenes:

```bash
python3 agent_src/random_walk_json.py --task-index 0 --list-scenes
```

Outputs are saved to each scene's configured `random_walks` directory.

### 5. Build Scene Graphs

For locator training, convert global point clouds into PyTorch Geometric scene graphs:

```bash
python3 preprocessing_src/large_graph_from_pcd.py
```

This writes graph files into:

```text
datasets_processed/<scene_name>/scan_pcd_graph/
```

The matching Slurm script is:

```bash
sbatch preprocessing_src/pcd_reduced.sbatch
```

### 6. Train VAE

Train the image / point-cloud graph autoencoder:

```bash
python3 training_scripts/1_autoencoder.py
```

Or on Slurm:

```bash
sbatch training_scripts/1_train.sbatch
```

Weights and losses are written under:

```text
1_model/<alias>/
```

### 7. Train Locator

Train the locator model after random walks and scene graphs are available:

```bash
python3 training_scripts/4_large_locator.py
```

Or on Slurm:

```bash
sbatch training_scripts/4_train.sbatch
```

Weights and losses are written under:

```text
4_model_locator/<alias>/
```

## Notes

- Some may use hard-coded scene names or paths
- Run scripts from the repo root unless stated otherwise
- Variants of `2_movement` and `3_registration` are functional, however outputs are not reliable yet. Hence, latest dependencies and changes are not pushed to the repo.
