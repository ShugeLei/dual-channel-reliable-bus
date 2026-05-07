"""ECE wrapper with PyTorch-friendly inputs.

A tiny wrapper around :class:`CalibrationMetric` that accepts torch
tensors (the form the rest of the pipeline produces) and returns a
``(float, bin_data)`` tuple — matching the original release's API.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.nn.functional import one_hot

from .compute_ece import CalibrationMetric


def ece(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    num_bins: int = 10,
    ce_method: str = "em_ece_sweep",
) -> Tuple[float, np.ndarray]:
    """Compute ECE on PyTorch tensors.

    Parameters
    ----------
    pred
        Predicted scores, shape ``(N, K)`` or ``(N, H, W, K)``.
    gt
        Labels: integer class indices ``(N,)`` (or ``(N, H, W)``), or
        already-one-hot probabilities of the same shape as ``pred``.
    num_bins
        Number of bins for the ``*_bin`` variants. Ignored by the
        ``*_sweep`` variants which auto-search.
    ce_method
        One of ``"ew_ece_bin"``, ``"em_ece_bin"``, ``"ew_ece_sweep"``,
        ``"em_ece_sweep"``.

    Returns
    -------
    (ece_value, bin_fx_y)
    """
    if gt.ndim > 1 and gt.shape != pred.shape:
        # Spatial labels — flatten over batch & spatial dims.
        pred_np = torch.flatten(pred, start_dim=0, end_dim=2).cpu().numpy()
        gt_np = torch.flatten(gt, start_dim=0, end_dim=2).cpu().numpy()
    else:
        pred_np = pred.detach().cpu().numpy()
        if gt.ndim == 1:
            gt_np = one_hot(gt, num_classes=pred.shape[1]).cpu().numpy()
        else:
            gt_np = gt.cpu().numpy()

    if "sweep" in ce_method:
        compute_ce = CalibrationMetric(ce_type=ce_method)
    else:
        compute_ce = CalibrationMetric(ce_type=ce_method, num_bins=num_bins)

    ece_val, bin_fx_y = compute_ce.compute_error(pred_np, gt_np)
    return float(ece_val), bin_fx_y


__all__ = ["ece"]
