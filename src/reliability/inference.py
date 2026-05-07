"""Inference Reliability Score (IRS).

Given a SP-RISA attribution map and a U-Net-predicted lesion mask, IRS
measures how much of the doctor-trusted region of interest (``M_pro``)
the attribution actually highlighted.

``M_pro`` is constructed in two steps:

1. **Spatial enlargement** of the lesion mask by a linear factor of
   ``k = 1.21`` (Chinese breast cancer diagnosis guidelines [31] in the
   paper) — implemented as ``Resize(1.21·H, 1.21·W) + CenterCrop(H, W)``,
   matching the original release.
2. **Below-lesion strip**: the lesion bounding-box height ``h`` is
   measured from contours, and the enlarged mask is shifted *downward*
   by ``h`` pixels (cumulative — i.e. each row gets the rows above it
   added in turn). Their union forms ``M_pro``.

The IRS itself is then **recall** of the doctor-trusted ROI in the
top-``|M_pro|`` pixels of the attribution map:

.. math::

    S \\;=\\; \\text{top-}|M_{pro}|\\,(\\text{attribution}) \\\\
    \\text{IRS} \\;=\\; \\frac{|S \\cap M_{pro}|}{|M_{pro}|}

This is **recall, not IoU** — the original code's denominator is
``m_size = sum(M_pro)``, not the union. The threshold count ``|M_pro|``
is dynamic per-image rather than a fixed top fraction.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from ..attribution.sp_risa import sp_risa


def build_pro_mask(mask: torch.Tensor) -> torch.Tensor:
    """Construct the doctor-trusted ROI ``M_pro``.

    Parameters
    ----------
    mask
        Lesion mask of shape ``(C, H, W)`` or ``(1, H, W)``, dtype float
        in ``{0, 1}`` (as produced by the U-Net segmenter).

    Returns
    -------
    pro_mask : torch.Tensor of shape ``(H, W)``, ``{0, 1}`` (float).
    """
    h, w = mask.size(-2), mask.size(-1)

    # Step 1: scale by k = 1.21 then center-crop back to (H, W).
    scaled_mask = T.Resize((round(1.21 * h), round(1.21 * w)))(mask)
    scaled_mask = T.CenterCrop((h, w))(scaled_mask)

    # Step 2: cumulative downward shift by lesion bbox height.
    mask_behind = scaled_mask[0].cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(mask_behind, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    top_y, bottom_y = h, 0
    for contour in contours:
        top_y = min(top_y, int(np.min(contour[:, :, 1])))
        bottom_y = max(bottom_y, int(np.max(contour[:, :, 1])))
    bbox_h = bottom_y - top_y

    # Cumulative shift: each pass shifts the current state down by 1px.
    # After bbox_h passes, the mask has been "smeared" downward by its
    # own bbox height.
    for _ in range(bbox_h):
        mask_behind = mask_behind + np.concatenate(
            [np.zeros((1, w), dtype=np.uint8), mask_behind[:-1]], axis=0
        )

    pro = scaled_mask.squeeze() + torch.from_numpy(mask_behind).to(mask.device)
    return torch.clamp(pro, 0, 1)


def compute_irs(
    model: torch.nn.Module,
    pred_class: int,
    mask: torch.Tensor,
    image_path: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Compute the inference reliability score for a single image.

    Parameters
    ----------
    model
        Classifier (eval mode).
    pred_class
        Predicted class index (the class whose attribution we score).
    mask
        Predicted lesion mask, shape ``(C, H, W)``, ``{0, 1}`` float.
    image_path
        Path to the original image file (loaded with ``cv2.imread`` for
        SP-RISA, which expects the un-normalized BGR uint8 image).
    args
        Namespace carrying ``dataset`` (str) and ``batch_size`` (int).
        ``dataset`` is used to look up per-dataset normalization stats.

    Returns
    -------
    irs : torch.Tensor (scalar)
        IRS in ``[0, 1]``.
    """
    from ..data.dataset import DATASET_MEAN, DATASET_STD  # avoid circular

    pro_mask = build_pro_mask(mask)
    m_size = round(torch.sum(pro_mask).item())

    if m_size == 0:
        # No predicted lesion — paper §III-C says to fall back. The
        # original release returns -1; the caller in `drs_tester` then
        # uses `max(irs, prob)` so the result is just the model's
        # prediction confidence in that case.
        return torch.tensor(-1.0, device=mask.device)

    # cv2.imread returns a NumPy BGR array; image_path may be a 1-tuple
    # from a DataLoader.
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0]
    image_np = cv2.imread(image_path)

    attribution = sp_risa(
        model, image_np, pred_class,
        mean=DATASET_MEAN[args.dataset],
        std=DATASET_STD[args.dataset],
        batch_size=args.batch_size,
    )
    attribution_flat = torch.flatten(attribution)

    _, indices = torch.topk(attribution_flat, m_size)
    s = torch.zeros_like(attribution_flat)
    s[indices] = 1.0

    pro_flat = torch.flatten(pro_mask).to(s.device)
    irs = torch.sum(s * pro_flat) / m_size
    return irs


__all__ = ["build_pro_mask", "compute_irs"]
