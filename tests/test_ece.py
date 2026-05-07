"""Tests for the calibration error metrics."""
import numpy as np
import pytest
import torch

from src.metrics import CalibrationMetric, ece
from src.metrics.binning_methods import BinEqualMass, BinEqualWidth


# ----------------------------------------------------------------------
# Binning schemes
# ----------------------------------------------------------------------
def test_equal_width_bin_indices():
    bw = BinEqualWidth(num_bins=10)
    scores = np.array([[0.0], [0.5], [0.99], [1.0]])
    idx = bw.compute_bin_indices(scores)
    # 0.0 → bin 0, 0.5 → bin 5, 0.99 → bin 9, 1.0 folds into bin 9
    assert idx[0, 0] == 0
    assert idx[1, 0] == 5
    assert idx[2, 0] == 9
    assert idx[3, 0] == 9


def test_equal_mass_balances_bin_sizes():
    bm = BinEqualMass(num_bins=5)
    scores = np.linspace(0, 1, 100).reshape(-1, 1)
    idx = bm.compute_bin_indices(scores)
    counts = np.bincount(idx.flatten(), minlength=5)
    # Each bin should hold ~20 samples (perfect when N divides evenly)
    assert counts.tolist() == [20, 20, 20, 20, 20]


# ----------------------------------------------------------------------
# CalibrationMetric — top-label setting
# ----------------------------------------------------------------------
def test_perfect_calibration_gives_zero_ece():
    """Predictions that perfectly equal accuracy → ECE = 0."""
    n = 500
    rng = np.random.default_rng(0)
    fx = rng.uniform(0, 1, n).reshape(-1, 1)
    correct = (rng.uniform(0, 1, n) < fx[:, 0]).astype(np.float64).reshape(-1, 1)
    # With many bins under the equal-mass setting, ECE should be small.
    cm = CalibrationMetric(ce_type="em_ece_bin", num_bins=20)
    err, _ = cm.compute_error(fx, correct)
    assert err < 0.1


def test_miscalibration_gives_high_ece():
    """All confidence 0.99, all wrong → ECE ≈ 0.99."""
    n = 200
    fx = np.full((n, 1), 0.99)
    correct = np.zeros((n, 1))
    cm = CalibrationMetric(ce_type="ew_ece_bin", num_bins=10)
    err, _ = cm.compute_error(fx, correct)
    assert err == pytest.approx(0.99, abs=0.01)


# ----------------------------------------------------------------------
# Sweep variants
# ----------------------------------------------------------------------
def test_sweep_returns_monotonic_bins():
    """The monotonic sweep guarantees per-bin accuracy is non-decreasing."""
    rng = np.random.default_rng(1)
    n = 1000
    fx = rng.uniform(0, 1, n).reshape(-1, 1)
    correct = (rng.uniform(0, 1, n) < fx[:, 0]).astype(np.float64).reshape(-1, 1)

    cm = CalibrationMetric(ce_type="em_ece_sweep")
    err, bin_fx_y = cm.compute_error(fx, correct)
    accuracies = bin_fx_y[1]                    # per-bin mean accuracy
    nonzero = accuracies[accuracies > 0]
    diffs = np.diff(nonzero)
    assert np.all(diffs >= -1e-9), accuracies


def test_all_four_variants_produce_finite_ece():
    rng = np.random.default_rng(2)
    n = 200
    fx = rng.uniform(0, 1, n).reshape(-1, 1)
    correct = (rng.uniform(0, 1, n) < fx[:, 0]).astype(np.float64).reshape(-1, 1)

    for variant in ["ew_ece_bin", "em_ece_bin", "ew_ece_sweep", "em_ece_sweep"]:
        cm = CalibrationMetric(ce_type=variant, num_bins=10)
        err, _ = cm.compute_error(fx, correct)
        assert 0.0 <= err <= 1.0, f"{variant}: {err}"


# ----------------------------------------------------------------------
# PyTorch wrapper
# ----------------------------------------------------------------------
def test_ece_wrapper_accepts_int_labels():
    """The torch-tensor wrapper one-hots integer labels automatically."""
    rng = np.random.default_rng(3)
    n, K = 80, 2
    pred = torch.from_numpy(rng.dirichlet(np.ones(K), size=n).astype(np.float64))
    gt = torch.randint(0, K, (n,))
    err, _ = ece(pred, gt, ce_method="em_ece_bin", num_bins=5)
    assert 0.0 <= err <= 1.0


def test_ece_wrapper_unknown_method_raises():
    n, K = 16, 2
    pred = torch.softmax(torch.randn(n, K), dim=-1)
    gt = torch.randint(0, K, (n,))
    with pytest.raises(NotImplementedError):
        ece(pred, gt, ce_method="not_a_method")


def test_ece_with_invalid_inputs_raises():
    """Confidence > 1 should raise."""
    cm = CalibrationMetric(ce_type="ew_ece_bin")
    with pytest.raises(ValueError):
        cm.compute_error(np.array([[1.5]]), np.array([[1.0]]))
