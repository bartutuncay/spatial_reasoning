# Methodology Outline from the Current Repository

This outline is derived from the actual scripts in the repository, especially:
- `preprocessing_src/process_dataset_eth3d.py`
- `preprocessing_src/process_depth_eth3d.py`
- `preprocessing_src/pcd_slice_extractor_camera.py`
- `preprocessing_src/combine_data.py`
- `autoencoder.py`
- `agent_src/random_walk.py`
- `preprocessing_src/rw_to_ae.py`
- `preprocessing_src/large_graph_from_pcd.py`
- `large_locator.py`

## 1. Problem Formulation

The codebase currently supports a two-stage spatial reasoning pipeline.

Stage A:
- Input: an RGB image and the visible portion of the scene point cloud
- Goal: learn a shared latent space between image observations and partial geometric graphs

Stage B:
- Input: latent observations from a random walk and a coarsened graph of the full scene
- Goal: localize the observation sequence with respect to the larger scene representation

That suggests a paper objective like:

> Learn visibility-aware multimodal scene representations that align egocentric observations with a graph-structured global map for reconstruction and localization.

## 2. Data Construction

### 2.1 Camera pose parsing

`process_dataset_eth3d.py` parses COLMAP-style image metadata:
- image ids
- quaternions
- translations
- image names

This creates the camera pose table used by later preprocessing steps.

### 2.2 2D visibility metric generation

`process_depth_eth3d.py` computes directional 2D visibility statistics from the floor-plan geometry:
- visible area
- mean distance
- distance standard deviation
- minimum distance
- maximum distance

These metrics are camera-conditioned and are computed from pose plus estimated viewing direction.

### 2.3 3D visible point-cloud extraction

`pcd_slice_extractor_camera.py` projects the aligned global scan into each camera view and keeps only the visible points by:
- transforming global points into camera coordinates
- discarding points behind the camera
- projecting onto the image plane
- keeping front-most points per pixel

For each observation it also:
- rotates the visible cloud into a canonical camera-aligned frame
- builds a kNN graph over the visible points
- stores edge attributes based on relative spherical coordinates
- saves 3D visibility statistics such as volume and depth summaries

### 2.4 Multimodal training sample assembly

`combine_data.py` merges per-view data into a single training example:
- resized RGB image
- partial point cloud with RGB features
- graph edges and weights
- 2D visibility metrics
- 3D visibility metrics

The output is a per-sample `.pt` file used by the autoencoder training loop.

## 3. Stage A: Cross-Modal Autoencoder

### 3.1 Inputs

From `autoencoder.py` and `models/autoencoder_model.py`, each training sample contains:
- `img`: RGB observation
- `pcd`: partial point cloud with `x,y,z,r,g,b`
- `edge_index`, `edge_weights`: graph structure for the visible cloud
- `vis`: ten-dimensional visibility descriptor

### 3.2 Image encoder

`models/image_encoder_no_cond.py` implements:
- a CNN stem and residual-style convolution blocks
- tokenization of the feature map
- optional 2D positional encoding
- lightweight token processing
- pooled latent output

This produces the image latent `z_img`.

### 3.3 Point-cloud graph encoder

`models/pcd_encoder.py` implements:
- an MLP projection from point features into latent channels
- repeated graph convolutions
- concatenation with graph-level pooled context at every layer
- global mean pooling to produce the graph latent `z_pcd`

This produces the point-cloud latent `z_pcd`.

### 3.4 Decoding and cross-modal reconstruction

The autoencoder reconstructs both modalities from both latents:
- image from image latent
- image from point-cloud latent
- point cloud from image latent
- point cloud from point-cloud latent

This is the cleanest core methodological contribution already present in the code.

### 3.5 Training losses

The current implementation uses:
- image reconstruction loss: L1
- point-cloud reconstruction loss: Chamfer distance plus color consistency
- latent alignment loss: MSE between `z_img` and `z_pcd`

In paper form, this becomes a multimodal reconstruction + alignment objective.

## 4. Stage B: Random-Walk Observation Generation

`agent_src/random_walk.py` simulates agent trajectories inside hand-defined navigable regions.

For each step it generates:
- camera location
- view direction
- rendered RGB observation
- visible point-cloud slice
- graph edges for the visible slice

This produces synthetic sequential observation data that is more suitable for localization than isolated training views.

## 5. Stage C: Latent Trajectory Encoding

`preprocessing_src/rw_to_ae.py` runs the trained autoencoder over random-walk observations and saves:
- image latent
- point-cloud latent
- location
- view direction

This converts raw egocentric observations into compact trajectory nodes.

## 6. Stage D: Global Scene Graph for Localization

`preprocessing_src/large_graph_from_pcd.py` constructs a global scene graph by:
- voxel downsampling the aligned full scan
- building a radius graph
- partitioning the graph into local clusters

This is the map-side representation used during localization.

## 7. Stage E: Latent-to-Map Localization

`large_locator.py` defines a localization network that compares random-walk latents against clustered map geometry.

The current idea is:
- encode each map cluster into a latent using the point-cloud encoder
- build a graph over observation latents
- compare map-cluster embeddings with observation embeddings
- predict observation-to-cluster translations
- predict relative transforms between nearby observation nodes

This suggests a localization formulation based on structured latent matching rather than direct camera-pose regression.

## 8. Recommended Paper Method Section Structure

### 8.1 Scene and observation representation
- Global map as a coarsened point-cloud graph
- Observation as image plus visible partial graph plus visibility descriptor

### 8.2 Visibility-aware preprocessing
- Pose parsing
- 2D plan-based metrics
- 3D raycast and visible-point extraction
- Graph construction for observation slices

### 8.3 Cross-modal representation learning
- Image encoder
- Point-cloud graph encoder
- Shared latent space
- Cross-modal decoders
- Reconstruction and alignment losses

### 8.4 Sequential latent map matching
- Random-walk trajectory generation
- Latent extraction for each step
- Global graph clustering
- Observation-to-cluster matching network
- Relative transform supervision

### 8.5 Training protocol
- Train the autoencoder first
- Freeze or initialize from it for localization
- Train localization on random-walk latents and clustered scene graphs

## 9. Recommended Experimental Story

### 9.1 Main tasks
- cross-modal reconstruction
- latent alignment quality
- localization against the global map

### 9.2 Ablations
- image-only encoder
- point-cloud-only encoder
- no latent alignment loss
- no visibility metrics
- different graph construction methods
- different map coarsening levels

### 9.3 Metrics
- image reconstruction loss
- Chamfer distance
- color consistency
- translation error
- angular error
- localization recall at thresholds
- runtime for preprocessing and inference

## 10. Important Implementation Caveats to Resolve Before Writing Final Claims

These are not blockers for the outline, but they matter for the paper:
- file and folder naming is not fully consistent across scripts
- some code paths still contain TODOs and commented alternatives
- the visibility-conditioning path exists conceptually but is disabled in the main training script
- the localization model is clearly exploratory and will need a cleaner final problem statement

## 11. Short Paper Pitch

The strongest version of the paper is not just “a GNN for localization.”

It is:

> A visibility-aware multimodal graph framework that aligns egocentric image observations with partial scene geometry, then uses the learned latent space for localization inside a coarsened global scene graph.
