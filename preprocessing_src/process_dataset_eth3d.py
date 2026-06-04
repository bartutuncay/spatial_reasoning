import open3d as o3d
import numpy as np
import pandas as pd
import glob
from pathlib import Path

#IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
#{QW QX QY QZ}: rotation world -> camera
#{TX TY TZ}: translation world -> camera
#22 0.44485 0.47671 0.548668 -0.523278 -4.21679 -3.41374 -0.282359 0 dslr_images_undistorted/DSC_0696.JPG
#381.773 13.9842 -1 816.757 87.9766 -1 816.757 87.9766 -1 1844.83 -15.1646 -1 1979.84 46.4548 -1 2101.15 106.987 -1 2246.08 21.112 

def parse_img_textfile(path:str):
    images = []
    pts = []
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) %2 != 0:
        print('Odd number of lines')
        return
    
    for i in range(0,len(lines),2):
        metadata = lines[i].split()
        pts = lines[i+1].split()
        if len(metadata) < 10:
            continue
        image_id = int(metadata[0])
        qw, qx, qy, qz = map(float, metadata[1:5])
        tx, ty, tz = map(float, metadata[5:8])
        camera_id = int(metadata[8])
        image_name = " ".join(metadata[9:])
        images.append({"image_id": image_id,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": tx, "ty": ty, "tz": tz,
            "image_name": image_name})
        
        for j in range(0, len(pts), 3):
            x = float(pts[j])
            y = float(pts[j + 1])
            point3d_id = int(pts[j + 2])
            pts.append({"image_id": image_id,
                "x": x,"y": y,"point3d_id": point3d_id})
        images_df = pd.DataFrame(images).sort_values("image_id").reset_index(drop=True)
        pts_df = pd.DataFrame(pts)

    return images_df, pts_df

#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
camera_intrinsics = {'W':6208,'H':4135,'fx':3408.59,'fy':3408.87,'cx':3117.24,'cy':2064.07}

images_path = 'datasets_processed/anlieferung/delivery_area/dslr_calibration_undistorted/images.txt'

images_df, _ = parse_img_textfile(images_path)
images_df.to_csv(images_path.replace('images.txt','images_parsed.csv'),index=False)
#pts_df.to_csv(images_path.replace('images.txt','pts_parsed.csv'),index=False)

