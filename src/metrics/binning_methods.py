"""Binning schemes for calibration error computation."""
from __future__ import annotations

import numpy as np


class BinEqualWidth:
    """Equal-width bins on ``[0, 1]``."""

    def __init__(self, num_bins: int):
        self.num_bins = num_bins

    def compute_bin_indices(self, scores: np.ndarray) -> np.ndarray:
        """Assign a bin index for each score.

        Parameters
        ----------
        scores
            Confidence scores of shape ``(N, K)`` (K = number of classes,
            or 1 for top-label setting).

        Returns
        -------
        bin_indices : np.ndarray of shape ``(N, K)``, integer in
            ``[0, num_bins)``.
        """
        edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        bin_indices = np.digitize(scores, edges, right=False) - 1
        # ``digitize`` puts the value 1.0 in bin ``num_bins``; fold it back.
        return np.where(scores == 1.0, self.num_bins - 1, bin_indices)


class BinEqualMass:
    """Equal-sample-count bins (quantile binning)."""

    def __init__(self, num_bins: int):
        self.num_bins = num_bins

    def compute_bin_indices(self, scores: np.ndarray) -> np.ndarray:
        n_examples, n_classes = scores.shape
        bin_indices = np.zeros((n_examples, n_classes), dtype=int)
        for k in range(n_classes):
            sort_ix = np.argsort(scores[:, k])
            bin_indices[:, k][sort_ix] = np.minimum(
                self.num_bins - 1,
                np.floor(np.arange(n_examples) / n_examples * self.num_bins),
            ).astype(int)
        return bin_indices


__all__ = ["BinEqualWidth", "BinEqualMass"]
