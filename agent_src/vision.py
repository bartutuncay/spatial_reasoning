import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import pandas as pd
import open3d.core as o3c
import time
import shapely
from shapely import plotting
from scipy.interpolate import CubicSpline
from sklearn.neighbors import KNeighborsRegressor
import argparse

def raycast_img(trunc,view_dir,fov_x,fov_y,pcd_points,pcd_rgb,vantage,H,W,background=(0, 0, 0),return_depth=True):
    # returns H*W*(RGB)
    
    near, far = float(trunc[0]), float(trunc[1])
    forward = np.asarray(view_dir, dtype=np.float32) #camera basis
    forward /= (np.linalg.norm(forward) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(world_up, forward))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)  # already normalized if right/forward are
    rel = (pcd_points - vantage).astype(np.float32, copy=False) #relative points
    # coordinates in camera frame
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    mask = (z > 1e-6) & (z >= near) & (z <= far) #filter by depth+direction
    if not np.any(mask):
        rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
        rgb_img[:] = np.array(background, dtype=np.uint8)
        depth = np.full((H, W), np.inf, dtype=np.float32) if return_depth else None
        stats = {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return rgb_img, depth, stats
    x = x[mask]; y = y[mask]; z = z[mask]
    cols = pcd_rgb[mask]

    # --- frustum check using normalized image plane coords ---
    tanx = np.tan(0.5 * float(fov_x))
    tany = np.tan(0.5 * float(fov_y))

    xn = x / z
    yn = y / z
    mask_f = (np.abs(xn) <= tanx) & (np.abs(yn) <= tany)
    if not np.any(mask_f):
        rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
        rgb_img[:] = np.array(background, dtype=np.uint8)
        depth = np.full((H, W), np.inf, dtype=np.float32) if return_depth else None
        stats = {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return rgb_img, depth, stats

    xn = xn[mask_f]; yn = yn[mask_f]; z = z[mask_f]
    cols = cols[mask_f]

    # --- map to pixel coordinates ---
    # xn in [-tanx, +tanx] -> u in [0, W-1]
    u = ((xn / tanx) * 0.5 + 0.5) * (W - 1)
    v = (0.5 - (yn / tany) * 0.5) * (H - 1)

    ui = u.astype(np.int32)
    vi = v.astype(np.int32)

    # clamp just in case of boundary numeric issues
    ui = np.clip(ui, 0, W - 1)
    vi = np.clip(vi, 0, H - 1)

    pix = vi * W + ui  # flattened pixel id

    # --- z-buffer: pick closest depth per pixel ---
    # Sort by (pixel, depth) so first occurrence per pixel is nearest
    order = np.lexsort((z, pix))
    pix_s = pix[order]
    z_s = z[order]
    cols_s = cols[order]

    # keep first index for each pixel
    first = np.empty_like(pix_s, dtype=bool)
    first[0] = True
    first[1:] = pix_s[1:] != pix_s[:-1]

    pix_u = pix_s[first]
    z_u = z_s[first]
    cols_u = cols_s[first]

    # --- write outputs ---
    rgb_img = np.zeros((H * W, 3), dtype=np.uint8)
    rgb_img[:] = np.array(background, dtype=np.uint8)

    # handle colors that might be float
    if cols_u.dtype != np.uint8:
        cols_u = np.clip(cols_u, 0, 255).astype(np.uint8)

    rgb_img[pix_u] = cols_u
    rgb_img = rgb_img.reshape(H, W, 3)

    depth = None
    if return_depth:
        depth = np.full((H * W,), np.inf, dtype=np.float32)
        depth[pix_u] = z_u.astype(np.float32)
        depth = depth.reshape(H, W)
        return rgb_img, depth

    return rgb_img

pcd_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/scan_raw/combined_aligned.ply'
images_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/dslr_calibration_undistorted/images_parsed.csv'
images_df = pd.read_csv(images_path)
pcd = o3d.io.read_point_cloud(pcd_path)
pcd_points = np.asarray(pcd.points)
pcd_colors = np.asarray(pcd.colors)

camera_intrinsics = {'W':6208,'H':4135,'fx':3408.59,'fy':3408.87,'cx':3117.24,'cy':2064.07}
fov_x = 2*np.arctan(camera_intrinsics['W']/(2*camera_intrinsics['fx']))
fov_y = 2*np.arctan(camera_intrinsics['H']/(2*camera_intrinsics['fy']))

parser = argparse.ArgumentParser()
parser.add_argument("--task-index",type=int,
                required=True,help="Task index")
args = parser.parse_args()
view_idx = args.task_index

np.random.seed(view_idx)
#coords_x = np.random.uniform()

if False:
    for _, point in images_df.iterrows():
        img_name = str(point.loc['image_name']).split('.JPG')[0].split('/')[1]
        print(img_name)
        t1 = time.time()
        qx,qy,qz,qw = point.loc[['qx','qy','qz','qw']].to_numpy()
        R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
        #translation from quaternion, translation matrices to camera coords
        tx,ty,tz = point.loc[['tx','ty','tz']].to_numpy()
        t_wc = np.array([tx, ty, tz])
        # camera center in world coordinates
        C = -R_wc.T @ t_wc
        d_cam = np.array([0.0, 0.0, 1.0])
        # camera view direction (center)
        d_world = R_wc.T @ d_cam
        d_world = d_world/(np.linalg.norm(d_world))

        img = raycast_img([0,40],d_world,fov_x,fov_y,pcd_points,pcd_colors*255,C,384,512,(0,0,0))
        t2 = time.time()
        print(img)
        print(t2-t1)
        #break

        import cv2
        cv2.imwrite(f'../../../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/agent_view/test_view_{img_name}.png',img)

#plt.savefig('test_view.png')

#module load stack/.2024-06-silent gcc/12.2.0 python/3.11.6