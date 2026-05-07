"""Predictive Reliability Score (PRS).

Test-Time Augmentation reliability: for each of seven hand-picked
augmentation compositions, an augmented image is generated and passed
through the classifier. Augmentations are *retried* up to 10 times if
they shift the lesion out of view (we require the augmented mask to
retain at least ``|M| - 10`` pixels). The class proportions across the
8 predictions (1 original + 7 augmented) are then converted to PRS via:

.. math::

    \\text{PRS} \\;=\\; 1 + \\frac{\\sum_i p_i \\log p_i}{\\log K}
    \\;=\\; 1 - \\frac{H}{\\log K}

so that a stable prediction (low entropy) gives a high reliability
score. The paper writes the literal form ``H / log K`` (uncertainty);
the code writes the equivalent inverted form. We match the code.

The seven transformation compositions are taken verbatim from the
original ``reliability.py`` and capture the dominant geometric
invariances of breast ultrasound: rotation in ``[-30°, 30°]``, random
crop, horizontal flip, and pairwise/triplet compositions of these.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import transforms as T


_TTA_TRANSFORMS = [
    T.Compose([T.RandomRotation((-30, 30)), T.Resize((224, 224))]),
    T.RandomCrop((224, 224)),
    T.Compose([T.RandomHorizontalFlip(1.0), T.Resize((224, 224))]),
    T.Compose([T.RandomRotation((-30, 30)), T.RandomCrop((224, 224))]),
    T.Compose([T.RandomRotation((-30, 30)),
               T.RandomHorizontalFlip(1.0),
               T.Resize((224, 224))]),
    T.Compose([T.RandomHorizontalFlip(1.0), T.RandomCrop((224, 224))]),
    T.Compose([T.RandomRotation((-30, 30)),
               T.RandomHorizontalFlip(1.0),
               T.RandomCrop((224, 224))]),
]


def compute_prs(
    model: torch.nn.Module,
    output_orig: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_retries: int = 10,
    mask_tolerance: int = 10,
) -> torch.Tensor:
    """Compute the predictive reliability score via TTA.

    Parameters
    ----------
    model
        Classifier (eval mode).
    output_orig
        Softmax probabilities of the un-augmented forward pass, shape
        ``(B, K)``. Included as the first prediction.
    image
        Input tensor, shape ``(B, C, H, W)``.
    mask
        Predicted lesion mask, shape ``(B, C, H, W)`` or ``(B, 1, H, W)``.
        Used only to reject augmentations that crop the lesion away.
    max_retries
        Per-transform retry budget when augmentations crop the lesion.
        If all retries fail, that transform is skipped.
    mask_tolerance
        Allowed shrinkage in pixels: an augmentation is accepted if
        ``sum(augmented_mask) > sum(mask) - mask_tolerance``.

    Returns
    -------
    prs : torch.Tensor (scalar)
        PRS in ``[0, 1]``, where 1 = perfectly stable prediction.
    """
    mask_size = torch.sum(mask).item()
    outputs = [output_orig]

    for transform in _TTA_TRANSFORMS:
        for _ in range(max_retries):
            augmented_image = transform(image)
            augmented_mask = transform(mask)
            if torch.sum(augmented_mask).item() > mask_size - mask_tolerance:
                outputs.append(F.softmax(model(augmented_image), dim=-1).data)
                break

    outputs = torch.stack(outputs, dim=1)               # (B, M, K)
    preds = torch.argmax(outputs, dim=-1)               # (B, M)

    one_hot = torch.zeros_like(outputs).scatter_(-1, preds.unsqueeze(-1), 1)
    proportions = torch.sum(one_hot, dim=1) / outputs.size(1)  # (B, K)
    proportions = torch.clamp(proportions, 1e-5, 1.0)

    K = outputs.size(-1)
    log_K = torch.log(torch.tensor(float(K), device=image.device))
    # Equivalent to 1 - H/log K (because Σ p log p = -H)
    prs = 1 + torch.sum(proportions * torch.log(proportions)) / log_K
    return prs


__all__ = ["compute_prs"]
