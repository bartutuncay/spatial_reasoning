import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import open3d as o3d


def parse_meshlab_mlp(mlp_path: str | Path) -> dict[str, np.ndarray]:
    """
    Returns { filename_in_mlp : 4x4 numpy array }.
    MeshLab stores matrices row-major as text in <MLMatrix44>.
    """
    mlp_path = Path(mlp_path)
    tree = ET.parse(mlp_path)
    root = tree.getroot()

    transforms: dict[str, np.ndarray] = {}
    for mlmesh in root.findall(".//MLMesh"):
        fname = mlmesh.attrib.get("filename")
        mat_text = mlmesh.findtext("MLMatrix44")
        if fname is None or mat_text is None:
            continue

        vals = [float(x) for x in mat_text.split()]
        if len(vals) != 16:
            raise ValueError(f"Expected 16 floats in MLMatrix44 for {fname}, got {len(vals)}")
        T = np.array(vals, dtype=np.float64).reshape(4, 4)
        transforms[fname] = T

    return transforms


def load_apply_transform(ply_path: Path, T: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    # Open3D applies T to points and rotates normals correctly if present.
    pcd.transform(T)
    return pcd


def combine_aligned_pointclouds(mlp_path: str | Path, ply_dir: str | Path, out_path: str | Path):
    mlp_path = Path(mlp_path)
    ply_dir = Path(ply_dir)
    out_path = Path(out_path)

    transforms = parse_meshlab_mlp(mlp_path)

    combined = o3d.geometry.PointCloud()
    for fname, T in transforms.items():
        ply_path = (ply_dir / fname).resolve()
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY not found: {ply_path}")

        pcd = load_apply_transform(ply_path, T)
        combined += pcd

    # Optional: downsample to remove duplicates / reduce size
    # combined = combined.voxel_down_sample(voxel_size=0.002)  # adjust to your units

    ok = o3d.io.write_point_cloud(str(out_path), combined, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")
    print(f"Wrote combined point cloud: {out_path} (N={np.asarray(combined.points).shape[0]})")

location = os.environ.get('SJEPA_SCENE')  # '<projectfolder>/<eth3dscene>'
if not location:
    raise SystemExit(
        "SJEPA_SCENE is not set. Export it as '<projectfolder>/<eth3dscene>' "
        "(e.g. SJEPA_SCENE=office/office) before running align_pcds.py — "
        "there is no default scene, to avoid silently aligning the wrong one.")


if __name__ == "__main__":
    # Example usage:
    # - mlp contains filename="scan1.ply" etc.
    # - ply_dir is the folder where scan1.ply and scan2.ply live
    combine_aligned_pointclouds(
        mlp_path=f"datasets_processed/{location}/scan_raw/scan_alignment.mlp",
        ply_dir=f"datasets_processed/{location}/scan_raw/",
        out_path=f"datasets_processed/{location}/scan_raw/combined_aligned.ply",
    )
