import numpy as np
import pytest

from metrics.calibration import aurc, expected_calibration_error, risk_coverage_curve


def test_ece_perfect_calibration_is_zero():
    # A bin's empirical accuracy equals its mean confidence -> zero ECE.
    # 100 samples all at confidence 0.5, exactly half correct.
    conf = np.full(100, 0.5)
    correct = np.zeros(100, dtype=bool)
    correct[:50] = True
    assert expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_worst_case():
    # Confident but always wrong -> ECE = 1.0
    conf = np.full(100, 0.95)
    correct = np.zeros(100, dtype=bool)
    assert expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.95, abs=1e-9)


def test_risk_coverage_keeps_confident_first():
    # errors anti-correlated with confidence -> risk rises with coverage
    errors = np.array([3.0, 2.0, 1.0, 0.0])
    conf = np.array([0.1, 0.2, 0.3, 0.4])  # lowest error most confident
    cov, risk = risk_coverage_curve(errors, conf)
    assert cov == pytest.approx([0.25, 0.5, 0.75, 1.0])
    # kept order by confidence: 0.0, 1.0, 2.0, 3.0 -> cumulative means
    assert risk == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert np.all(np.diff(risk) >= 0)


def test_aurc_constant_errors_equals_constant():
    errors = np.full(20, 0.7)
    conf = np.linspace(0, 1, 20)
    # risk is 0.7 at every coverage -> area under (0,1] is 0.7
    assert aurc(errors, conf) == pytest.approx(0.7, abs=1e-6)


def test_empty_inputs_are_safe():
    assert np.isnan(expected_calibration_error([], []))
    cov, risk = risk_coverage_curve([], [])
    assert cov.size == 0 and risk.size == 0
    assert np.isnan(aurc([], []))
