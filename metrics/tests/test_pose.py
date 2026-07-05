import numpy as np
import pytest

from metrics.pose import (
    ate_rmse,
    ate_rmse_aligned,
    pose_recall,
    rotation_errors_deg,
    rpe,
    translation_errors,
    umeyama,
)


def _rz(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_ate_rmse_zero_when_equal():
    p = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6.0]])
    assert ate_rmse(p, p) == pytest.approx(0.0)


def test_translation_errors_known():
    pred = np.array([[0, 0, 0], [3, 4, 0.0]])
    gt = np.array([[0, 0, 0], [0, 0, 0.0]])
    e = translation_errors(pred, gt)
    assert e == pytest.approx([0.0, 5.0])
    # rms of [0, 5] = sqrt(12.5)
    assert ate_rmse(pred, gt) == pytest.approx(np.sqrt(12.5))


def test_translation_shape_mismatch_raises():
    with pytest.raises(ValueError):
        translation_errors(np.zeros((2, 3)), np.zeros((3, 3)))


def test_rotation_error_identity_and_90deg():
    I = np.eye(3)
    assert rotation_errors_deg(I, I) == pytest.approx(0.0, abs=1e-9)
    assert rotation_errors_deg(I, _rz(90)) == pytest.approx(90.0, abs=1e-6)
    # batched
    pred = np.stack([I, I])
    gt = np.stack([_rz(30), _rz(180)])
    errs = rotation_errors_deg(pred, gt)
    assert errs == pytest.approx([30.0, 180.0], abs=1e-6)


def test_pose_recall_translation_and_rotation():
    pred_pos = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0.0]])
    gt_pos = np.array([[0.01, 0, 0], [0.3, 0, 0], [0.02, 0, 0.0]])
    pred_R = np.stack([np.eye(3)] * 3)
    gt_R = np.stack([_rz(1), _rz(1), _rz(30)])  # 3rd fails rotation
    rec = pose_recall(pred_pos, gt_pos, [(0.05, 5.0), (0.5, None)], pred_R, gt_R)
    # (0.05m,5deg): item0 ok, item1 fails trans, item2 fails rot -> 1/3
    assert rec["0.05m/5deg"] == pytest.approx(1 / 3)
    # (0.5m) translation-only: all within 0.5m -> 1.0
    assert rec["0.5m"] == pytest.approx(1.0)


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    s_true, R_true, t_true = 2.0, _rz(37), np.array([1.0, -2.0, 3.0])
    dst = (s_true * (R_true @ src.T)).T + t_true
    s, R, t = umeyama(src, dst, with_scale=True)
    assert s == pytest.approx(s_true, rel=1e-6)
    assert R == pytest.approx(R_true, abs=1e-6)
    assert t == pytest.approx(t_true, abs=1e-6)
    # aligned ATE ~ 0
    assert ate_rmse_aligned(src, dst) == pytest.approx(0.0, abs=1e-6)


def test_rpe_zero_for_identical_trajectories():
    rng = np.random.default_rng(1)
    pos = np.cumsum(rng.normal(size=(10, 3)), axis=0)
    R = np.stack([_rz(a) for a in range(0, 100, 10)])
    t_err, r_err = rpe(pos, R, pos, R, delta=1)
    assert t_err == pytest.approx(0.0, abs=1e-9)
    assert r_err == pytest.approx(0.0, abs=1e-6)
