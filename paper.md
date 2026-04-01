A. Must-have quantitative experiments
1) Core task: camera pose estimation in a pre-existing map
Metrics: translation (m), rotation (deg), median + percentiles (e.g., 50/75/90), and/or AUC of pose error thresholds
Settings: full map; partial map (see B.3)
Baselines (minimum):
Retrieval + PnP / “image-to-3D” style pipeline (whatever is standard in your area)
Learned features + matching + PnP (a modern feature matcher baseline if feasible)
Image-only pose regressor (same ResNet backbone)
2) Navigation / goal-directed behavior (if you claim it)
Metrics: success rate, path length ratio, SPL (if applicable), collision rate, mean time-to-goal, and trajectory smoothness (curvature / jerk / heading change)
Baselines (minimum):
Image-only RL / policy baseline (your claim mentions this)
Classical pipeline you compare against (define it precisely: SLAM+planner, or reconstruction+planner)
3) Efficiency
Reviewers love hard numbers.
Metrics: inference latency per step (ms), FPS, GPU memory, CPU usage (if relevant)
Comparisons: classical pipeline runtime + your runtime, same hardware
B. Must-have ablations (these are often required for acceptance)
4) Remove map-conditioning (image-only)
Same backbone, same supervision
Shows map actually contributes
5) Remove cross-modal alignment objective
e.g., train pose head directly without the cross-modal autoencoder alignment
Shows representation learning is doing work, not just supervision
6) Graph vs non-graph map encoding
Replace GNN with a simpler map encoder (MLP on pooled points, PointNet-style pooling, or global pooling)
Shows the graph structure matters
7) Partial map robustness sweep
This is a differentiator—make it crisp.
Randomly drop nodes/edges or subsample points/graph regions
Plot performance vs % map observed (25/50/75/100)
Run for both localization and navigation (if you claim both)
C. Generalization & robustness (high value, moderate cost)
8) Cross-scene generalization
Train on some buildings/scenes, test on held-out scenes
If you can do only one thing here: report cross-scene pose accuracy
9) Viewpoint/appearance shift robustness
Different lighting / time / sensor / motion blur (whatever ETH3D supports)
Or synthetic perturbations: brightness/contrast, blur, noise
Show degradation curves
10) Map quality robustness
Add noise to point positions
Downsample density
Add outliers
Show method doesn’t collapse
D. Qualitative experiments (cheap, high persuasive value)
11) Localization visualizations
Show image, predicted camera frustum in map, GT vs pred
Failure cases (at least 2–3, with explanation)
12) Partial reconstruction outputs (only if you keep this claim)
Show predicted local point-cloud completion from a view
Compare to a geometry baseline if possible
Otherwise present as auxiliary capability, not headline
13) Navigation trajectories overlaid on map
Top-down map with path overlays
Include “smoothness” visual cue (heading changes)
E. “If time permits” (nice-to-have, but not essential)
14) Multi-task benefit
Pose-only vs pose+reconstruction vs pose+goal-distance
Show shared embedding improves pose or navigation
15) Edge construction sensitivity
kNN vs radius graph vs learned edges
Show method not brittle
16) Scalability
Map size sweep (small/medium/large scenes)
Runtime + memory scaling

Minimal “ECCV survival set” (if you’re in a 1-week crunch)
If you can only do 6 things, do these:
Pose estimation results + strong baselines
Navigation success + smoothness (if claimed)
Runtime/compute comparison
Ablation: no map-conditioning
Ablation: no cross-modal alignment
Partial map sweep plot
This set alone often determines accept/reject.