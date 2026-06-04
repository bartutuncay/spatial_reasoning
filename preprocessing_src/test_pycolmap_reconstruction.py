import argparse
import shutil
from pathlib import Path

import pycolmap


def write_ply(reconstruction, path):
    pts = list(reconstruction.points3D.values())
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in pts:
            x, y, z = p.xyz
            r, g, b = p.color
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def stereo_fusion(output_path, workspace_path, input_type):
    print("stereo_fusion order: workspace_path, output_path")
    return pycolmap.stereo_fusion(str(workspace_path), str(output_path), input_type=input_type)


parser = argparse.ArgumentParser()
parser.add_argument("image_dir")
parser.add_argument("--out", default="pycolmap_reconstruction")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
parser.add_argument("--overlap", type=int, default=10)
parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
parser.add_argument("--single-camera", action="store_true")
parser.add_argument("--max-num-features", type=int, default=16384)
parser.add_argument("--max-num-matches", type=int, default=32768)
parser.add_argument("--peak-threshold", type=float, default=0.004)
parser.add_argument("--min-num-matches", type=int, default=8)
parser.add_argument("--min-model-size", type=int, default=2)
parser.add_argument("--dense", action="store_true")
args = parser.parse_args()

image_dir = Path(args.image_dir)
out = Path(args.out)
db = out / "database.db"
sparse = out / "sparse"
mvs = out / "mvs"

if args.overwrite and out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True, exist_ok=True)
sparse.mkdir(exist_ok=True)

pycolmap.extract_features(
    str(db),
    str(image_dir),
    camera_mode=pycolmap.CameraMode.SINGLE if args.single_camera else pycolmap.CameraMode.AUTO,
    reader_options={"camera_model": args.camera_model},
    extraction_options={"sift": {"max_num_features": args.max_num_features, "peak_threshold": args.peak_threshold}},
)

matching_options = {"guided_matching": True, "max_num_matches": args.max_num_matches}

if args.matcher == "sequential":
    pycolmap.match_sequential(str(db), matching_options=matching_options, pairing_options={"overlap": args.overlap})
else:
    pycolmap.match_exhaustive(str(db), matching_options=matching_options)

mapping_options = pycolmap.IncrementalPipelineOptions(
    {
        "min_num_matches": args.min_num_matches,
        "min_model_size": args.min_model_size,
        "mapper": {"abs_pose_min_num_inliers": args.min_num_matches, "max_reg_trials": 5},
    }
)
reconstructions = pycolmap.incremental_mapping(str(db), str(image_dir), str(sparse), options=mapping_options)

if not reconstructions:
    raise RuntimeError("No reconstruction found.")

for model_idx, rec in sorted(reconstructions.items()):
    print(f"model {model_idx}: {len(rec.images)} images, {len(rec.points3D)} points")

idx, reconstruction = max(reconstructions.items(), key=lambda item: len(item[1].images))
ply_path = out / "points.ply"
write_ply(reconstruction, ply_path)
print(f"selected model {idx}: {len(reconstruction.images)} images, {len(reconstruction.points3D)} points")
print(ply_path)

if args.dense:
    model = sparse / str(idx)
    dense_ply = mvs / "dense.ply"
    if mvs.exists():
        shutil.rmtree(mvs)
    reconstruction.write(str(model))
    pycolmap.undistort_images(str(mvs), str(model), str(image_dir))
    pycolmap.patch_match_stereo(str(mvs))
    depth_maps = mvs / "stereo" / "depth_maps"
    input_type = "geometric" if any(depth_maps.glob("*.geometric.bin")) else "photometric"
    if not any(depth_maps.glob(f"*.{input_type}.bin")):
        raise RuntimeError(f"No depth maps found in {depth_maps}")
    stereo_fusion(dense_ply, mvs, input_type)
    print(dense_ply)
