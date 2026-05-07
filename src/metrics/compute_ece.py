"""Calibration error metric.

Faithful port of the original release's ``utils/compute_ece.py``, which
follows Roelofs et al., "Mitigating Bias in Calibration Error Estimation"
(AISTATS 2022) for the monotonic-sweep variants.

Four ECE variants are supported:

- ``ew_ece_bin`` — fixed equal-width bins.
- ``em_ece_bin`` — fixed equal-mass (equal-sample-count) bins.
- ``ew_ece_sweep`` — equal-width bins with monotonic-sweep auto-search
  over the number of bins.
- ``em_ece_sweep`` — equal-mass bins with monotonic-sweep auto-search.

The "sweep" variants are the ones called out in the paper as the
correctly-debiased estimator (§IV-A).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .binning_methods import BinEqualMass, BinEqualWidth


class CalibrationMetric:
    """Compute the calibration error in any of the four supported variants."""

    _SUPPORTED = {"ew_ece_bin", "em_ece_bin", "ew_ece_sweep", "em_ece_sweep",
                  "label_binned"}

    def __init__(
        self,
        ce_type: str = "em_ece_sweep",
        num_bins: int = 10,
        norm: int = 1,
        multiclass_setting: str = "top_label",
    ):
        if ce_type not in self._SUPPORTED:
            raise NotImplementedError(f"ce_type {ce_type!r} not supported.")
        if multiclass_setting not in {"top_label", "marginal"}:
            raise NotImplementedError(
                f"Multiclass setting {multiclass_setting} not supported.")

        if ce_type.startswith("ew"):
            self.bin_method = BinEqualWidth(num_bins)
        elif ce_type.startswith("em"):
            self.bin_method = BinEqualMass(num_bins)
        else:
            self.bin_method = None

        self.ce_type = ce_type
        self.num_bins = num_bins
        self.norm = norm
        self.multiclass_setting = multiclass_setting

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_error(
        self,
        fx: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute the calibration error.

        Parameters
        ----------
        fx
            Predicted confidence scores of shape ``(N, K)``.
        y
            One-hot-encoded labels of shape ``(N, K)``.

        Returns
        -------
        ece : float
        bin_fx_y : np.ndarray of shape ``(2, B)``
            Per-bin ``[mean_score, mean_accuracy]`` arrays.
        """
        if fx.ndim == 1:
            fx = fx.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if fx.max() > 1.0 or fx.min() < 0.0:
            raise ValueError(
                f"fx must be in [0, 1]; got [{fx.min()}, {fx.max()}]."
            )
        if y.max() > 1.0 or y.min() < 0.0:
            raise ValueError(
                f"y must be in [0, 1]; got [{y.min()}, {y.max()}]."
            )

        if self.multiclass_setting == "top_label" and fx.shape[1] > 1:
            fx, y = self._predict_top_label(fx, y)

        if self.ce_type.endswith("bin"):
            binned_fx, binned_y, bin_sizes, bin_indices = self._bin_data(fx, y)
            ce = self._compute_error_all_binned(binned_fx, binned_y, bin_sizes)
            bin_fx_y = self._bin_mean(fx, y, self.num_bins, bin_indices)
        elif self.ce_type.endswith("sweep"):
            ce, bin_fx_y = self._compute_error_monotonic_sweep(fx, y)
        else:                                     # pragma: no cover
            raise NotImplementedError(
                f"Calibration error {self.ce_type} not supported.")

        return ce, bin_fx_y

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _predict_top_label(fx, y):
        picked = np.argmax(fx, axis=1)
        labels = np.argmax(y, axis=1)
        hits = np.array(picked == labels, ndmin=2).T.astype(int)
        return np.max(fx, axis=1, keepdims=True), hits

    def _bin_data(self, fx, y):
        bin_indices = self.bin_method.compute_bin_indices(fx)
        K = fx.shape[1]
        binned_fx = np.zeros((self.num_bins, K))
        binned_y = np.zeros((self.num_bins, K))
        bin_sizes = np.zeros((self.num_bins, K))
        for k in range(K):
            for b in range(self.num_bins):
                idx = np.where(bin_indices[:, k] == b)[0]
                if idx.size > 0:
                    binned_fx[b, k] = fx[idx, k].mean()
                    binned_y[b, k] = y[idx, k].mean()
                    bin_sizes[b, k] = idx.size
        return binned_fx, binned_y, bin_sizes, bin_indices

    def _compute_error_all_binned(self, binned_fx, binned_y, bin_sizes):
        n = bin_sizes[:, 0].sum()
        K = binned_fx.shape[1]
        ce = np.power(np.abs(binned_fx - binned_y), self.norm) * bin_sizes
        return np.power(ce.sum() / (n * K), 1.0 / self.norm)

    @staticmethod
    def _bin_mean(fx, y, n_bins, bins):
        bin_fx, bin_y = [], []
        for b in range(n_bins):
            cur = bins == b
            if cur.any():
                bin_fx.append(fx[cur].mean())
                bin_y.append(y[cur].mean())
            else:
                bin_fx.append(0.0)
                bin_y.append(0.0)
        return np.stack([np.asarray(bin_fx), np.asarray(bin_y)], axis=0)

    def _compute_error_monotonic_sweep(self, fx, y):
        fx = np.squeeze(fx)
        y = np.squeeze(y)
        keep = ~np.isnan(fx)
        fx, y = fx[keep], y[keep]

        if self.ce_type == "em_ece_sweep":
            bins = self._em_monotonic_sweep(fx, y)
        else:                                     # ew_ece_sweep
            bins = self._ew_monotonic_sweep(fx, y)

        n_bins = int(np.max(bins)) + 1
        ce, _ = self._calc_ece_postbin(n_bins, bins, fx, y)
        return ce, self._bin_mean(fx, y, n_bins, bins)

    def _calc_ece_postbin(self, n_bins, bins, fx, y):
        """ECE with monotonicity check."""
        ece = 0.0
        monotonic = True
        last_ym = -np.inf
        for i in range(n_bins):
            cur = bins == i
            if cur.any():
                fxm = fx[cur].mean()
                ym = y[cur].mean()
                if ym < last_ym:
                    monotonic = False
                last_ym = ym
                ece += cur.sum() * np.power(np.abs(ym - fxm), self.norm)
        return np.power(ece / fx.shape[0], 1.0 / self.norm), monotonic

    def _em_monotonic_sweep(self, fx, y):
        """Sweep equal-mass bins from 2 upward; stop at first non-monotonic."""
        sort_ix = np.argsort(fx)
        n = fx.shape[0]
        bins = np.zeros(n, dtype=int)
        prev_bins = np.zeros(n, dtype=int)
        for n_bins in range(2, n):
            bins[sort_ix] = np.minimum(
                n_bins - 1, np.floor(np.arange(n) / n * n_bins)
            ).astype(int)
            _, monotonic = self._calc_ece_postbin(n_bins, bins, fx, y)
            if not monotonic:
                return prev_bins
            prev_bins = bins.copy()
        return bins

    def _ew_monotonic_sweep(self, fx, y):
        """Sweep equal-width bins from 2 upward; stop at first non-monotonic."""
        n = fx.shape[0]
        bins = np.zeros(n, dtype=int)
        prev_bins = np.zeros(n, dtype=int)
        for n_bins in range(2, n):
            bins = np.minimum(n_bins - 1, np.floor(fx * n_bins)).astype(int)
            _, monotonic = self._calc_ece_postbin(n_bins, bins, fx, y)
            if not monotonic:
                return prev_bins
            prev_bins = bins.copy()
        return bins


__all__ = ["CalibrationMetric"]
