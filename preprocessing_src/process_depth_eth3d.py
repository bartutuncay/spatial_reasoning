##TODO
# run metrics on dataset: 2D-3D
# embed point cloud data as graph
# embed plan data as graph

import numpy as np
import pandas as pd
from area_methods import compute_area_directional
from scipy.spatial.transform import Rotation as R

images_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/dslr_calibration_undistorted/images_parsed.csv'
images_df = pd.read_csv(images_path)

lines_json = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/anlieferung_plan.json'
lines_df = pd.read_json(lines_json)
lines_2d = []
for row in lines_df.iterrows():
    start = row[1]['start'][:2]
    end = row[1]['end'][:2]
    lines_2d.append([start,end])
lines_2d = np.array(lines_2d)

camera_intrinsics = {'W':6208,'H':4135,'fx':3408.59,'fy':3408.87,'cx':3117.24,'cy':2064.07}
fov_x = 2*np.arctan(camera_intrinsics['W']/(2*camera_intrinsics['fx']))
fov_y = 2*np.arctan(camera_intrinsics['H']/(2*camera_intrinsics['fy']))

N = 360
D = 40
M = 'segments_angle'
print(N,D,M)

results_csv = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_2d_camera.csv'
res_df = pd.DataFrame(columns=['dirs','max_distance','method','img','area_calc','mean_dist','dist_std','min_dist','max_dist',
                            'posX','posY','fov','dir'])
res_df.to_csv(results_csv,index=False)

for _, point in images_df.iterrows():
    qx,qy,qz,qw = point.loc[['qx','qy','qz','qw']].to_numpy()
    R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()

    tx,ty,tz = point.loc[['tx','ty','tz']].to_numpy()
    #translation from quaternion, translation matrices to camera coords
    t_wc = np.array([tx, ty, tz])
    # camera center in world coordinates
    C = -R_wc.T @ t_wc
    d_cam = np.array([0.0, 0.0, 1.0])
    # camera view direction (center)
    d_world = R_wc.T @ d_cam
    d_world = d_world/(np.linalg.norm(d_world))
    area,distances = compute_area_directional(lines_2d,M,N,[C[0],C[1]],D,d_world[:2],fov_x)
    #print(C,d_world)
    img_name = str(point.loc['image_name']).split('.JPG')[0].split('/')[1]
    row = pd.DataFrame([{'dirs': N, 'max_distance' : D, 'method': M, 'img': img_name, 'area_calc': area, 'mean_dist':distances.mean(),
            'dist_std':distances.std(),'min_dist':distances.min(),'max_dist':distances.max(),'posX': C[0], 'posY': C[1],
            'fov': np.rad2deg(fov_x),'dir': np.arctan2(d_world[1],d_world[0])}])
    row.to_csv(results_csv,mode='a',header=False,index=False)
    