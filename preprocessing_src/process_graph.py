import networkx as nx
import pandas as pd

## scene graph: see pipeline/Scene Graph
# high level: scan data
vis_2d = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_2d_360deg.csv'
#vis_3d = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_3d_360deg.csv'

# low level: image data, encoded into high level
images_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/dslr_calibration_undistorted/images_parsed.csv'
results_img_2d = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_2d_camera.csv'
#results_img_3d = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_3d_camera.csv'

images_df = pd.read_csv(images_path)

#image to graph:
# image: construct image graph from keypoints

# scan portion: construct graph from keypoints
# results_img_2d: aggregated output of image/pcd graph
# results_img_3d: aggregated output of image/pcd graph