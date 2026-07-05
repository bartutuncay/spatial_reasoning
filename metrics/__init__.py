"""Evaluation metrics for Spatial-JEPA.

Pure-numpy pose/calibration metrics (no torch) plus a torch-optional
cost-triple. Every campaign result row is produced through these so the
paper's tables regenerate from disk.
"""
from .calibration import aurc, expected_calibration_error, risk_coverage_curve
from .pose import (
    ate_rmse,
    ate_rmse_aligned,
    pose_recall,
    rotation_errors_deg,
    rpe,
    translation_errors,
    umeyama,
)

# Default pose-recall thresholds used across all localization tables.
DEFAULT_RECALL_THRESHOLDS = [(0.05, 5.0), (0.25, 2.0), (0.5, 5.0)]

__all__ = [
    "ate_rmse",
    "ate_rmse_aligned",
    "translation_errors",
    "rotation_errors_deg",
    "pose_recall",
    "umeyama",
    "rpe",
    "expected_calibration_error",
    "risk_coverage_curve",
    "aurc",
    "DEFAULT_RECALL_THRESHOLDS",
]
