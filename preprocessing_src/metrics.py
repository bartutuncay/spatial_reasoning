from scipy.spatial import KDTree
import os, time, math, argparse
import open3d as o3d
import numpy as np
import pandas as pd
from area_methods import compute_area_combined
from volume_methods import directional_raycast
import argparse


N = 720
D = 40
M = 'segments_angle'
print(N,D,M)

## Floor plan to list
lines_json = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/anlieferung_plan.json'
lines_df = pd.read_json(lines_json)
lines_2d = []
for row in lines_df.iterrows():
    start = row[1]['start'][:2]
    end = row[1]['end'][:2]
    lines_2d.append([start,end])
lines_2d = np.array(lines_2d)

#point_coords = point_coords[0:10]
# downsample ground truth points for faster processing - using grid
points_sampled = []
sample_res = 0.25
xrange = np.arange(-55,15,sample_res)
yrange = np.arange(-5,40,sample_res)
gridX, gridY = np.meshgrid(xrange, yrange, indexing = 'xy')
points_sampled = np.vstack([gridX.ravel(), gridY.ravel()]).T

results = []
results_csv = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_2d_360deg.csv'
res_df = pd.DataFrame(columns=['dirs','max_distance','method','area_calc','mean_dist','dist_std','min_dist','max_dist',
                            'posX','posY','time'])
res_df.to_csv(results_csv,index=False)

for point in points_sampled:
    x = point[0]
    y = point[1]
    start = time.time()
    #try:
    area,dists = compute_area_combined(lines_2d,M,N,[x,y],D)
    end = time.time()
    dur = end-start
    row = pd.DataFrame([{'dirs': N, 'max_distance' : D, 'method': M, 'area_calc': area, 'mean_dist':dists.mean(),
            'dist_std':dists.std(),'min_dist':dists.min(),'max_dist':dists.max(),'posX': x, 'posY': y, 'time': dur}])
    row.to_csv(results_csv,mode='a',header=False,index=False)
    #except:
    #    ValueError
    #error = area/(point[1]['Area']*100)
    #results.append({'dirs': N, 'max_distance' : D, 'method': M, 'area_calc': area, 'error':error,'posX': x, 'posY': y, 'time': dur})

