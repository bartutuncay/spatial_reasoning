from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - optional at import time
    o3d = None

try:
    from .pcd_slice_methods import knn_connectivity, rotation_a_to_b
except ImportError:
    from pcd_slice_methods import knn_connectivity, rotation_a_to_b

import torch_geometric.typing as pyg_typing

pyg_typing.WITH_INDEX_SORT = False


DEFAULT_CAMERA_INTRINSICS = {"W": 6208, "H": 4135, "fx": 3408.59, "fy": 3408.87}


def wrap_angle(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def cartesian_to_spherical(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    radii = np.linalg.norm(vectors, axis=1)
    xy_norm = np.linalg.norm(vectors[:, :2], axis=1)
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])
    elevation = np.arctan2(vectors[:, 2], xy_norm)
    return np.stack([radii, azimuth, elevation], axis=1).astype(np.float32)


def spherical_edge_attr(points: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.empty((0, 3), dtype=np.float32)

    rel_vecs = points[edge_index[1]] - points[edge_index[0]]
    return cartesian_to_spherical(rel_vecs)


def cartesian_to_norm(points: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.empty((0, 4), dtype=np.float32)

    vectors = points[edge_index[1]] - points[edge_index[0]]
    dist = np.linalg.norm(vectors, axis=1)
    safe_dist = np.maximum(dist, 1e-12)
    dir_vec = vectors / safe_dist[:, None]
    return np.concatenate([dist[:, None], dir_vec], axis=1).astype(np.float32)


def safe_knn_connectivity(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = points.shape[0]
    if num_nodes < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    if num_nodes <= k:
        pairs = np.array(
            [[i, j] for i in range(num_nodes) for j in range(i + 1, num_nodes)],
            dtype=np.int64,
        )
        weights = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1).astype(np.float32)
        return pairs.T, weights

    return knn_connectivity(points, k)


def make_graph(
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
    knn_edge_index: np.ndarray,
    knn_edge_attr: np.ndarray,
    visible_idx: np.ndarray,
) -> Data:
    pcd_points = pcd_points.astype(np.float32, copy=False)
    pcd_colors = pcd_colors.astype(np.float32, copy=False)

    points_norm = np.linalg.norm(pcd_points, axis=1)
    safe_norm = np.maximum(points_norm, 1e-12)
    pcd_points = np.concatenate([points_norm[:, None], pcd_points / safe_norm[:, None]], axis=1)

    num_knn_nodes = pcd_points.shape[0]
    origin = np.zeros((1, 4), dtype=np.float32)
    origin_rgb = np.zeros((1, pcd_colors.shape[1]), dtype=np.float32)
    all_points = np.vstack([origin, pcd_points])
    all_colors = np.vstack([origin_rgb, pcd_colors])

    targets = np.arange(1, num_knn_nodes + 1, dtype=np.int64)
    origin_sources = np.full(num_knn_nodes, 0, dtype=np.int64)
    origin_to_nodes = np.stack([origin_sources, targets], axis=0)
    nodes_to_origin = np.stack([targets, origin_sources], axis=0)
    origin_edge_index = np.concatenate([origin_to_nodes, nodes_to_origin], axis=1)
    origin_edge_attr = cartesian_to_norm(all_points[:, 1:], origin_edge_index)

    data = Data()
    data.pos = torch.from_numpy(all_points)
    data.rgb = torch.from_numpy(all_colors)
    data.edge_index = torch.from_numpy(knn_edge_index + 1).long()
    data.edge_attr = torch.from_numpy(knn_edge_attr).float()
    data.ei_camera = torch.from_numpy(origin_edge_index).long()
    data.ea_camera = torch.from_numpy(origin_edge_attr).float()
    data.origin_node_index = torch.tensor([0], dtype=torch.long)
    data.visible_point_indices = torch.from_numpy(visible_idx).long()
    return data


def raycast_img_with_points(
    trunc: Sequence[float],
    view_dir: np.ndarray,
    fov_x: float,
    fov_y: float,
    pcd_points: np.ndarray,
    pcd_rgb: np.ndarray,
    vantage: np.ndarray,
    H: int,
    W: int,
    background: Sequence[int] = (0, 0, 0),
    return_depth: bool = True,
):
    near, far = float(trunc[0]), float(trunc[1])
    forward = np.asarray(view_dir, dtype=np.float32)
    forward /= np.linalg.norm(forward) + 1e-9
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(world_up, forward))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, forward)

    rel = (pcd_points - vantage).astype(np.float32, copy=False)
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    base_mask = (z > 1e-6) & (z >= near) & (z <= far)

    rgb_img = np.zeros((H * W, 3), dtype=np.uint8)
    rgb_img[:] = np.array(background, dtype=np.uint8)
    depth = np.full((H * W,), np.inf, dtype=np.float32) if return_depth else None

    empty_points = np.empty((0, 3), dtype=pcd_points.dtype)
    empty_colors = np.empty((0, pcd_rgb.shape[1]), dtype=pcd_rgb.dtype)
    empty_idx = np.empty((0,), dtype=np.int64)

    if not np.any(base_mask):
        rgb_img = rgb_img.reshape(H, W, 3)
        if return_depth:
            return rgb_img, depth.reshape(H, W), empty_idx, empty_points, empty_colors
        return rgb_img, empty_idx, empty_points, empty_colors

    base_idx = np.flatnonzero(base_mask)
    x = x[base_mask]
    y = y[base_mask]
    z = z[base_mask]

    tanx = np.tan(0.5 * float(fov_x))
    tany = np.tan(0.5 * float(fov_y))
    xn = x / z
    yn = y / z
    frustum_mask = (np.abs(xn) <= tanx) & (np.abs(yn) <= tany)

    if not np.any(frustum_mask):
        rgb_img = rgb_img.reshape(H, W, 3)
        if return_depth:
            return rgb_img, depth.reshape(H, W), empty_idx, empty_points, empty_colors
        return rgb_img, empty_idx, empty_points, empty_colors

    src_idx = base_idx[frustum_mask]
    xn = xn[frustum_mask]
    yn = yn[frustum_mask]
    z = z[frustum_mask]
    cols = pcd_rgb[src_idx]

    u = ((xn / tanx) * 0.5 + 0.5) * (W - 1)
    v = (0.5 - (yn / tany) * 0.5) * (H - 1)
    ui = np.clip(u.astype(np.int32), 0, W - 1)
    vi = np.clip(v.astype(np.int32), 0, H - 1)
    pix = vi * W + ui

    order = np.lexsort((z, pix))
    pix_s = pix[order]
    z_s = z[order]
    cols_s = cols[order]
    src_idx_s = src_idx[order]

    first = np.empty_like(pix_s, dtype=bool)
    first[0] = True
    first[1:] = pix_s[1:] != pix_s[:-1]

    pix_u = pix_s[first]
    z_u = z_s[first]
    cols_u = cols_s[first]
    visible_idx = src_idx_s[first]

    img_cols = cols_u
    if img_cols.dtype != np.uint8:
        img_cols = np.clip(img_cols * 255.0, 0, 255).astype(np.uint8)

    rgb_img[pix_u] = img_cols
    rgb_img = rgb_img.reshape(H, W, 3)

    if return_depth:
        depth[pix_u] = z_u.astype(np.float32)
        depth = depth.reshape(H, W)

    visible_order = np.argsort(visible_idx)
    visible_idx = visible_idx[visible_order]
    visible_points = pcd_points[visible_idx]
    visible_rgb = pcd_rgb[visible_idx]

    if return_depth:
        return rgb_img, depth, visible_idx, visible_points, visible_rgb
    return rgb_img, visible_idx, visible_points, visible_rgb


@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


DEFAULT_AGENT_RECTS = [
    Rect(xmin=-47, xmax=5, ymin=-4, ymax=0),
    Rect(xmin=-12, xmax=5, ymin=0, ymax=9),
    Rect(xmin=-11, xmax=-7, ymin=9, ymax=15),
    Rect(xmin=-3, xmax=2, ymin=9, ymax=14),
    Rect(xmin=-1, xmax=2, ymin=14, ymax=18),
]


def in_union_rects(xy: np.ndarray, rects: Sequence[Rect]) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    mask = np.zeros(len(xy), dtype=bool)
    for r in rects:
        mask |= (x >= r.xmin) & (x <= r.xmax) & (y >= r.ymin) & (y <= r.ymax)
    return mask


def sample_xy_in_union(rects: Sequence[Rect], n: int, rng: np.random.Generator) -> np.ndarray:
    areas = np.array([r.area for r in rects], dtype=float)
    if areas.sum() <= 0:
        raise ValueError("Total area is zero; check rectangle bounds.")

    probs = areas / areas.sum()
    choices = rng.choice(len(rects), size=n, p=probs)
    pts = np.zeros((n, 2), dtype=float)
    for i, ridx in enumerate(choices):
        r = rects[ridx]
        pts[i, 0] = rng.uniform(r.xmin, r.xmax)
        pts[i, 1] = rng.uniform(r.ymin, r.ymax)
    return pts


@dataclass
class MotionConfig:
    max_step_size: float = 0.2
    max_turn: float | None = np.pi
    sigma_theta: float = 0.0
    sigma_step: float = 0.0
    pitch: float = -0.1
    sigma_view: float = 0.0
    view_pull: float = 0.1
    bounce_on_collision: bool = True
    bounce_noise: float = 0.05

    @classmethod
    def random_walk_defaults(cls) -> "MotionConfig":
        return cls(
            max_step_size=0.2,
            sigma_theta=0.15,
            sigma_step=0.02,
            sigma_view=0.07,
            view_pull=0.1,
            bounce_on_collision=True,
            bounce_noise=0.05,
        )


@dataclass
class RenderConfig:
    render_h: int = 192
    render_w: int = 256
    camera_intrinsics: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CAMERA_INTRINSICS))
    trunc: tuple[float, float] = (0.0, 40.0)
    background: tuple[int, int, int] = (0, 0, 0)
    vantage_z: float = -3.5

    @property
    def fov_x(self) -> float:
        return float(
            2.0 * np.arctan(self.camera_intrinsics["W"] / (2.0 * self.camera_intrinsics["fx"]))
        )

    @property
    def fov_y(self) -> float:
        return float(
            2.0 * np.arctan(self.camera_intrinsics["H"] / (2.0 * self.camera_intrinsics["fy"]))
        )


@dataclass
class AgentState:
    xy: np.ndarray
    motion_yaw: float
    view_yaw: float
    view_dir: np.ndarray
    step_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "xy": self.xy.copy(),
            "motion_yaw": float(self.motion_yaw),
            "view_yaw": float(self.view_yaw),
            "view_dir": self.view_dir.copy(),
            "step_index": int(self.step_index),
        }


def _coerce_action(
    action: Mapping[str, Any] | Sequence[float] | np.ndarray | torch.Tensor,
    current_motion_yaw: float,
    current_view_yaw: float,
) -> dict[str, float | None]:
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    if isinstance(action, np.ndarray):
        action = action.reshape(-1).tolist()

    if isinstance(action, Mapping):
        requested_step = action.get("step", action.get("step_size", action.get("forward", 0.0)))
        theta = action.get("theta")
        turn = action.get("turn")
        view_yaw = action.get("view_yaw")
        view_turn = action.get("view_turn")
        if theta is not None and turn is None:
            turn = wrap_angle(float(theta) - current_motion_yaw)
        if view_yaw is not None and view_turn is None:
            view_turn = wrap_angle(float(view_yaw) - current_view_yaw)
        return {
            "turn": 0.0 if turn is None else float(turn),
            "step": float(requested_step),
            "view_turn": None if view_turn is None else float(view_turn),
        }

    if isinstance(action, Sequence):
        if len(action) < 2:
            raise ValueError("Sequence actions must contain at least [turn, step].")
        return {
            "turn": float(action[0]),
            "step": float(action[1]),
            "view_turn": None if len(action) < 3 else float(action[2]),
        }

    raise TypeError(f"Unsupported action type: {type(action)!r}")


def _inside_rects(point_xy: np.ndarray, rects: Sequence[Rect]) -> bool:
    x, y = point_xy
    for rect in rects:
        if rect.xmin <= x <= rect.xmax and rect.ymin <= y <= rect.ymax:
            return True
    return False


def _view_dir_from_yaw(view_yaw: float, pitch: float) -> np.ndarray:
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(view_yaw), np.sin(view_yaw)
    view_dir = np.array([cp * cy, cp * sy, sp], dtype=np.float32)
    view_dir /= np.linalg.norm(view_dir) + 1e-12
    return view_dir


def walk_nextstep(
    start_xy: Sequence[float],
    rects: Sequence[Rect],
    theta: float,
    step: float = 0.1,
    sigma_theta: float = 0.15,
    sigma_step: float = 0.02,
    pitch: float = -0.1,
    sigma_view: float = 0.07,
    view_pull: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng
    xy = np.asarray(start_xy, dtype=np.float32)
    if not _inside_rects(xy, rects):
        raise ValueError("start_xy must be inside the union.")

    motion_yaw = float(theta) + rng.normal(scale=sigma_theta)
    step_size = max(0.0, float(step) + rng.normal(scale=sigma_step))
    proposal = xy + step_size * np.array([np.cos(motion_yaw), np.sin(motion_yaw)], dtype=np.float32)
    if _inside_rects(proposal, rects):
        xy = proposal
    else:
        motion_yaw = wrap_angle(motion_yaw + np.pi + rng.normal(scale=0.05))
        proposal = xy + step_size * np.array([np.cos(motion_yaw), np.sin(motion_yaw)], dtype=np.float32)
        if _inside_rects(proposal, rects):
            xy = proposal

    view_yaw = wrap_angle(
        (1.0 - view_pull) * float(theta) + view_pull * motion_yaw + rng.normal(scale=sigma_view)
    )
    view_dir = _view_dir_from_yaw(view_yaw, pitch)
    return xy[None, :], view_dir[None, :], np.array([view_yaw], dtype=np.float32)


def record_step(
    point: Sequence[float],
    viewdir: Sequence[float],
    pcd_points: np.ndarray,
    pcd_colors: np.ndarray,
    render_cfg: RenderConfig,
) -> dict[str, Any]:
    point = np.asarray(point, dtype=np.float32)
    viewdir = np.asarray(viewdir, dtype=np.float32)
    vantage = np.array([point[0], point[1], render_cfg.vantage_z], dtype=np.float32)

    view_rgb, view_depth, visible_idx, pcd_visible_world, pcd_rgb = raycast_img_with_points(
        render_cfg.trunc,
        viewdir,
        render_cfg.fov_x,
        render_cfg.fov_y,
        pcd_points,
        pcd_colors,
        vantage,
        render_cfg.render_h,
        render_cfg.render_w,
        render_cfg.background,
    )

    align_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    R_align = rotation_a_to_b(viewdir, align_axis)
    pcd_visible_aligned = (R_align @ (pcd_visible_world - vantage).T).T
    edge_index, edge_weight = safe_knn_connectivity(pcd_visible_aligned, 3)
    data_graph = make_graph(pcd_visible_aligned, pcd_rgb, edge_index, edge_weight, visible_idx)

    return {
        "img": torch.from_numpy(view_rgb / 255.0).to(torch.float32),
        "depth": torch.from_numpy(view_depth).to(torch.float32),
        "graph": data_graph,
        "loc": torch.from_numpy(vantage),
        "xy": torch.from_numpy(point),
        "view_dir": torch.from_numpy(viewdir).to(torch.float32),
        "ei_points": edge_index,
        "ew_points": edge_weight,
        "ei_camera": data_graph.ei_camera,
        "ea_camera": data_graph.ea_camera,
    }


class Capture:
    def __init__(
        self,
        pcd_path: str | Path | None = None,
        rects: Sequence[Rect] | None = None,
        *,
        pcd_points: np.ndarray | None = None,
        pcd_colors: np.ndarray | None = None,
        movement_cfg: MotionConfig | None = None,
        render_cfg: RenderConfig | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.rects = list(DEFAULT_AGENT_RECTS if rects is None else rects)
        self.movement_cfg = MotionConfig() if movement_cfg is None else movement_cfg
        self.render_cfg = RenderConfig() if render_cfg is None else render_cfg
        self.rng = np.random.default_rng(rng_seed)
        self.pcd_points, self.pcd_colors = self._load_point_cloud(
            pcd_path=pcd_path,
            pcd_points=pcd_points,
            pcd_colors=pcd_colors,
        )
        self.state: AgentState | None = None

    def _load_point_cloud(
        self,
        pcd_path: str | Path | None,
        pcd_points: np.ndarray | None,
        pcd_colors: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if pcd_points is not None:
            points = np.asarray(pcd_points, dtype=np.float32)
            if pcd_colors is None:
                colors = np.zeros((points.shape[0], 3), dtype=np.float32)
            else:
                colors = np.asarray(pcd_colors, dtype=np.float32)
            return points, colors

        if pcd_path is None:
            raise ValueError("Provide either `pcd_path` or in-memory `pcd_points`.")
        if o3d is None:
            raise ImportError("open3d is required to load a point cloud from disk.")

        cloud = o3d.io.read_point_cloud(str(Path(pcd_path)))
        return (
            np.asarray(cloud.points, dtype=np.float32),
            np.asarray(cloud.colors, dtype=np.float32),
        )

    def sample_start_xy(self, n: int = 1) -> np.ndarray:
        return sample_xy_in_union(self.rects, n=n, rng=self.rng)

    def inside(self, point_xy: Sequence[float]) -> bool:
        return _inside_rects(np.asarray(point_xy, dtype=np.float32), self.rects)

    def _ensure_state(self) -> AgentState:
        if self.state is None:
            raise RuntimeError("Call reset(...) before step(...).")
        return self.state

    def observe(
        self,
        point: Sequence[float] | None = None,
        viewdir: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        state = self._ensure_state()
        point = state.xy if point is None else point
        viewdir = state.view_dir if viewdir is None else viewdir
        packed = record_step(point, viewdir, self.pcd_points, self.pcd_colors, self.render_cfg)
        packed["state"] = self.get_state()
        return packed

    def get_state(self) -> dict[str, Any]:
        return self._ensure_state().as_dict()

    def reset(
        self,
        start_xy: Sequence[float] | None = None,
        *,
        motion_yaw: float | None = None,
        view_yaw: float | None = None,
        return_observation: bool = True,
    ) -> dict[str, Any] | dict[str, float | np.ndarray]:
        if start_xy is None:
            start_xy = self.sample_start_xy(1)[0]

        xy = np.asarray(start_xy, dtype=np.float32)
        if not self.inside(xy):
            raise ValueError("start_xy must be inside the union.")

        if motion_yaw is None:
            motion_yaw = float(self.rng.uniform(-np.pi, np.pi))
        if view_yaw is None:
            view_yaw = motion_yaw

        self.state = AgentState(
            xy=xy,
            motion_yaw=wrap_angle(float(motion_yaw)),
            view_yaw=wrap_angle(float(view_yaw)),
            view_dir=_view_dir_from_yaw(float(view_yaw), self.movement_cfg.pitch),
            step_index=0,
        )
        if return_observation:
            return self.observe()
        return self.get_state()

    def step(
        self,
        action: Mapping[str, Any] | Sequence[float] | np.ndarray | torch.Tensor,
        *,
        return_observation: bool = True,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        state = self._ensure_state()
        parsed = _coerce_action(action, state.motion_yaw, state.view_yaw)

        requested_turn = float(parsed["turn"])
        applied_turn = requested_turn
        if self.movement_cfg.max_turn is not None:
            applied_turn = float(
                np.clip(applied_turn, -self.movement_cfg.max_turn, self.movement_cfg.max_turn)
            )

        requested_step = float(parsed["step"])
        applied_step = float(np.clip(requested_step, 0.0, self.movement_cfg.max_step_size))

        motion_yaw = wrap_angle(
            state.motion_yaw + applied_turn + self.rng.normal(scale=self.movement_cfg.sigma_theta)
        )
        applied_step = max(0.0, applied_step + self.rng.normal(scale=self.movement_cfg.sigma_step))

        proposal = state.xy + applied_step * np.array(
            [np.cos(motion_yaw), np.sin(motion_yaw)],
            dtype=np.float32,
        )
        collision = not self.inside(proposal)
        bounced = False
        moved = False
        new_xy = state.xy.copy()

        if not collision:
            new_xy = proposal
            moved = True
        elif self.movement_cfg.bounce_on_collision and applied_step > 0.0:
            bounced = True
            motion_yaw = wrap_angle(
                motion_yaw + np.pi + self.rng.normal(scale=self.movement_cfg.bounce_noise)
            )
            bounced_proposal = state.xy + applied_step * np.array(
                [np.cos(motion_yaw), np.sin(motion_yaw)],
                dtype=np.float32,
            )
            if self.inside(bounced_proposal):
                new_xy = bounced_proposal
                moved = True

        target_view_yaw = motion_yaw
        if parsed["view_turn"] is not None:
            target_view_yaw = wrap_angle(state.view_yaw + float(parsed["view_turn"]))

        view_yaw = wrap_angle(
            (1.0 - self.movement_cfg.view_pull) * state.view_yaw
            + self.movement_cfg.view_pull * target_view_yaw
            + self.rng.normal(scale=self.movement_cfg.sigma_view)
        )

        self.state = AgentState(
            xy=new_xy.astype(np.float32, copy=False),
            motion_yaw=motion_yaw,
            view_yaw=view_yaw,
            view_dir=_view_dir_from_yaw(view_yaw, self.movement_cfg.pitch),
            step_index=state.step_index + 1,
        )

        info = {
            "requested_turn": requested_turn,
            "applied_turn": applied_turn,
            "requested_step": requested_step,
            "applied_step": applied_step,
            "moved": moved,
            "collision": collision,
            "bounced": bounced,
            "state": self.get_state(),
        }
        observation = self.observe() if return_observation else None
        return observation, info

    def rollout(
        self,
        actions: Sequence[Mapping[str, Any] | Sequence[float] | np.ndarray | torch.Tensor],
        *,
        return_observations: bool = True,
    ) -> list[tuple[dict[str, Any] | None, dict[str, Any]]]:
        return [self.step(action, return_observation=return_observations) for action in actions]


SimulationStepper = Capture
