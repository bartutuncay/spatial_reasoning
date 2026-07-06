"""Pose-error metrics for single-view localization and latent rollout.

All functions take plain numpy arrays, so they are decoupled from the repo's
sample schema; the campaign harness adapts model outputs into these shapes.

Conventions
-----------
- positions : (N, 3) float arrays in metres, one row per query/frame.
- rotations : (N, 3, 3) rotation matrices under a single consistent
  convention (e.g. camera->world). Only mutual agreement matters.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Translation / rotation error primitives
# --------------------------------------------------------------------------- #
def translation_errors(pred_pos, gt_pos):
    """Per-item Euclidean translation error (metres), shape (N,)."""
    pred_pos = np.asarray(pred_pos, float)
    gt_pos = np.asarray(gt_pos, float)
    if pred_pos.shape != gt_pos.shape:
        raise ValueError(f"shape mismatch: {pred_pos.shape} vs {gt_pos.shape}")
    return np.linalg.norm(pred_pos - gt_pos, axis=-1)


def rotation_errors_deg(pred_R, gt_R):
    """Geodesic angular error (degrees) between rotation matrices, per item.

    Accepts (3, 3) or (N, 3, 3). angle = arccos((trace(pred^T gt) - 1) / 2).
    """
    pred_R = np.asarray(pred_R, float)
    gt_R = np.asarray(gt_R, float)
    rel = np.matmul(np.swapaxes(pred_R, -1, -2), gt_R)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def ate_rmse(pred_pos, gt_pos):
    """RMS translation error over independent queries (metres).

    For single-view localization each query is independent, so this is the
    plain RMS of per-query position errors (no trajectory alignment).
    """
    e = translation_errors(pred_pos, gt_pos)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


# --------------------------------------------------------------------------- #
# Pose recall at (translation, rotation) thresholds
# --------------------------------------------------------------------------- #
def pose_recall(pred_pos, gt_pos, thresholds, pred_R=None, gt_R=None):
    """Fraction of queries within (trans_m, rot_deg) thresholds.

    Parameters
    ----------
    thresholds : list of (trans_metres, rot_degrees). rot_degrees may be None
        (translation-only). Rotation is also skipped if rotations are absent.

    Returns dict mapping a human key -> recall in [0, 1].
    """
    te = translation_errors(pred_pos, gt_pos)
    re = None
    if pred_R is not None and gt_R is not None:
        re = rotation_errors_deg(pred_R, gt_R)
    out = {}
    for t, a in thresholds:
        ok = te <= t
        if a is not None and re is not None:
            ok = ok & (re <= a)
            key = f"{t:g}m/{a:g}deg"
        else:
            key = f"{t:g}m"
        out[key] = float(np.mean(ok)) if te.size else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Trajectory metrics (latent rollout / odometry)
# --------------------------------------------------------------------------- #
def umeyama(src, dst, with_scale=True):
    """Least-squares similarity (Sim3) aligning src -> dst.

    Returns (scale, R, t) such that ``scale * R @ src_i + t ~= dst_i``.
    Reference: Umeyama, 1991.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = src.shape[0]
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
    sigma = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / n
        # degenerate (stationary) source trajectory -> no scale to estimate
        scale = float(np.sum(D * np.diag(S)) / var_s) if var_s > 1e-12 else 1.0
    else:
        scale = 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def ate_rmse_aligned(pred_pos, gt_pos, with_scale=True):
    """Trajectory ATE after Sim3 alignment (for rollout/odometry, metres)."""
    pred_pos = np.asarray(pred_pos, float)
    s, R, t = umeyama(pred_pos, gt_pos, with_scale)
    aligned = (s * (R @ pred_pos.T)).T + t
    return ate_rmse(aligned, gt_pos)


def _se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rpe(pred_pos, pred_R, gt_pos, gt_R, delta=1):
    """Relative pose error over frame pairs at spacing ``delta``.

    Returns (rpe_trans_rmse_metres, rpe_rot_rmse_deg).
    """
    pred_pos = np.asarray(pred_pos, float)
    gt_pos = np.asarray(gt_pos, float)
    pred_R = np.asarray(pred_R, float)
    gt_R = np.asarray(gt_R, float)
    n = pred_pos.shape[0]
    trans, rot = [], []
    for i in range(n - delta):
        rel_pred = np.linalg.inv(_se3(pred_R[i], pred_pos[i])) @ _se3(
            pred_R[i + delta], pred_pos[i + delta]
        )
        rel_gt = np.linalg.inv(_se3(gt_R[i], gt_pos[i])) @ _se3(
            gt_R[i + delta], gt_pos[i + delta]
        )
        E = np.linalg.inv(rel_gt) @ rel_pred
        trans.append(np.linalg.norm(E[:3, 3]))
        cos = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot.append(np.degrees(np.arccos(cos)))
    trans = np.asarray(trans)
    rot = np.asarray(rot)

    def _rmse(a):
        return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")

    return _rmse(trans), _rmse(rot)
