def directional_raycast(
    trunc, ndirs, view_dir, fov_x, fov_y,
    pcd_points, pcd_rgb, vantage,
    heightlims, lim_height=False
):
    """
    1-1 replacement for your function with fixes:
      - Filters POINTS by FOV (not only ray directions)
      - Correct height clipping via plane intersection (floor/ceiling)
      - Robust stats computed over valid rays only
      - Keeps behavior: returns synthetic ray-hit points (rel_pts) + colors via NN
    Assumes fov_x, fov_y are in radians.
    """

    near, far = trunc[0], trunc[1]

    # --- Relative geometry ---
    rel_positions = pcd_points - vantage
    pcd_distances = np.linalg.norm(rel_positions, axis=1, keepdims=True)

    # Distance shell mask
    mask = (pcd_distances[:, 0] >= near) & (pcd_distances[:, 0] <= far)
    rel_positions = rel_positions[mask]
    pcd_distances = pcd_distances[mask]
    pcd_rgb = pcd_rgb[mask]

    # Unit directions for points (avoid div by 0)
    denom = np.maximum(pcd_distances, 1e-9)
    pcd_dirs = rel_positions / denom

    # --- Sample directions on sphere (Fibonacci) ---
    indices = np.arange(0, ndirs, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / ndirs)
    theta = np.pi * (1 + 5**0.5) * indices
    dirs = np.vstack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ]).T.astype(np.float32)

    # --- Camera basis (forward/right/up) ---
    forward = view_dir / (np.linalg.norm(view_dir) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.abs(np.dot(world_up, forward)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, forward)

    # --- FOV mask for sampled ray directions ---
    x = dirs @ right
    y = dirs @ up
    z = dirs @ forward
    h_ang = np.arctan2(x, z)
    v_ang = np.arctan2(y, z)
    mask_dir = (z > 0) & (np.abs(h_ang) <= fov_x * 0.5) & (np.abs(v_ang) <= fov_y * 0.5)
    dirs = dirs[mask_dir]
    ndirs = dirs.shape[0]

    # --- FOV mask for POINT directions (FIX) ---
    # Prevent out-of-FOV points from contaminating in-FOV bins.
    px = pcd_dirs @ right
    py = pcd_dirs @ up
    pz = pcd_dirs @ forward
    p_h = np.arctan2(px, pz)
    p_v = np.arctan2(py, pz)
    mask_p = (pz > 0) & (np.abs(p_h) <= fov_x * 0.5) & (np.abs(p_v) <= fov_y * 0.5)

    pcd_dirs = pcd_dirs[mask_p]
    rel_positions = rel_positions[mask_p]
    pcd_distances = pcd_distances[mask_p]
    pcd_rgb = pcd_rgb[mask_p]

    # If no rays or no points after masking, return empty but consistent outputs
    if ndirs == 0 or rel_positions.shape[0] == 0:
        rel_pts = np.zeros((0, 3), dtype=np.float32)
        pts_colors = np.zeros((0, 3), dtype=pcd_rgb.dtype) if hasattr(pcd_rgb, "dtype") else np.zeros((0, 3))
        vol = 0.0
        return rel_pts, pts_colors, vol, 0.0, 0.0, 0.0, 0.0

    # Debug prints (kept similar intent)
    print(len(dirs))
    # Print count after point masking (more meaningful than original distance mask count)
    print(rel_positions.shape[0])

    # --- Bin points to nearest ray direction ---
    tree = KDTree(dirs, leafsize=8)
    _, idx = tree.query(pcd_dirs, k=1)

    dist_per_bin = np.ones(ndirs, dtype=np.float64) * 1e5
    np.minimum.at(dist_per_bin, idx, pcd_distances[:, 0])

    # --- Height clipping (FIX) ---
    if lim_height:
        # heightlims are absolute (world Z); convert to relative-to-vantage Z
        ceil_rel = heightlims[1] - vantage[2]
        floor_rel = heightlims[0] - vantage[2]

        dz = dirs[:, 2].astype(np.float64)
        eps = 1e-9

        # Clip against ceiling plane z = ceil_rel
        # Only relevant if ray goes upward and current endpoint exceeds ceiling
        m = (dz > eps) & (dist_per_bin * dz > ceil_rel)
        dist_per_bin[m] = ceil_rel / dz[m]

        # Clip against floor plane z = floor_rel
        # Only relevant if ray goes downward and current endpoint goes below floor
        m = (dz < -eps) & (dist_per_bin * dz < floor_rel)
        dist_per_bin[m] = floor_rel / dz[m]

    # --- Final sanitize and build hit points ---
    dist_per_bin[~np.isfinite(dist_per_bin)] = 0.0
    dist_per_bin[dist_per_bin > far] = 0.0
    dist_per_bin[dist_per_bin < near] = 0.0

    valid_rays = dist_per_bin > 0.0
    rel_pts = dirs[valid_rays] * dist_per_bin[valid_rays, None]

    # Color by nearest neighbor among *masked* original points
    tree_relpts = KDTree(rel_positions, leafsize=8)
    _, rgb_idx = tree_relpts.query(rel_pts, k=1)
    pts_colors = pcd_rgb[rgb_idx]

    # --- Volume and stats (FIX: stats over valid rays only) ---
    alpha = 0.5 * fov_x
    beta = 0.5 * fov_y
    Omega = 4.0 * np.arcsin(np.sin(alpha) * np.sin(beta))

    N = dist_per_bin.size
    vol = (Omega / (3.0 * N)) * np.sum(dist_per_bin**3)

    if np.any(valid_rays):
        dvalid = dist_per_bin[valid_rays]
        min_dist = float(dvalid.min())
        max_dist = float(dvalid.max())
        mean_dist = float(dvalid.mean())
        dist_stdev = float(dvalid.std())
    else:
        min_dist = max_dist = mean_dist = dist_stdev = 0.0

    return rel_pts, pts_colors, vol, min_dist, max_dist, mean_dist, dist_stdev
