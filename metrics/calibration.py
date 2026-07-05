"""Calibration and selective-prediction metrics.

These turn the locator's uncertainty (softmax entropy / logvar-derived
confidence) into the reliability story: expected calibration error, the
risk-coverage curve, and its area (AURC). Pure numpy.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(confidences, correct, n_bins=10):
    """ECE: |accuracy - confidence| averaged over confidence bins.

    Parameters
    ----------
    confidences : (N,) in [0, 1]. correct : (N,) bool (pose within threshold).
    """
    conf = np.asarray(confidences, float)
    cor = np.asarray(correct, bool)
    n = conf.size
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        m = (conf > lo) & (conf <= hi)
        if k == 0:  # include the left edge in the first bin
            m |= conf <= lo
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(cor[m].mean() - conf[m].mean())
    return float(ece)


def risk_coverage_curve(errors, confidences):
    """Selective risk vs coverage; most-confident queries kept first.

    Returns (coverages, risks): coverages in (0, 1], risks = mean error over
    the most-confident ``coverage`` fraction. A well-calibrated model yields a
    monotonically increasing curve.
    """
    err = np.asarray(errors, float)
    conf = np.asarray(confidences, float)
    if err.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(-conf)  # descending confidence
    err_sorted = err[order]
    ks = np.arange(1, err.size + 1)
    risks = np.cumsum(err_sorted) / ks
    coverages = ks / err.size
    return coverages, risks


def aurc(errors, confidences):
    """Area under the risk-coverage curve (lower is better).

    Uses the mean-of-risks convention (Geifman & El-Yaniv, 2017): the average
    selective risk across all coverage levels. For a constant-error predictor
    this equals that constant, as expected.
    """
    _, risk = risk_coverage_curve(errors, confidences)
    if risk.size == 0:
        return float("nan")
    return float(np.mean(risk))
