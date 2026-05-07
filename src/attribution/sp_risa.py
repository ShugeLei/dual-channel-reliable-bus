"""SP-RISA: Super-pixel Random Input Sampling for Attribution.

Faithful reproduction of the algorithm from `sprisa.py` in the original
release. The implementation differs from a reader's first impression of
the paper text in two important ways:

1. **Deletion-based attribution.** Each random mask deletes a subset of
   superpixels (sets their pixels to 0). The attribution at a pixel is
   the average of ``(1 - p_t)`` weighted by ``(1 - m_t)`` — i.e. how much
   *removing* a region drops the prediction score, accumulated over
   masks that include that region in the deletion. This is conceptually
   closer to deletion saliency than to inclusion-based RISE.

2. **Stratified mask generation.** Rather than sampling each superpixel
   independently with ``p = 0.5``, masks are generated in ``n/2`` strata
   indexed by ``i ∈ [0, n/2)``. Stratum ``i`` keeps each superpixel with
   probability ``(n - 2 - i) / n``, sweeping from "drop few" to
   "drop many". For each stratum, ``round((n_mask - n) / (n/2))`` masks
   are drawn. The first ``n`` masks are deterministic single-superpixel
   drops. This produces a more diverse mask population than i.i.d.
   Bernoulli sampling.

SLIC is computed with OpenCV's ``ximgproc.createSuperpixelSLIC``
(``region_size=30``, ``ruler=20``, 10 iterations) — paper Table I.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Internal: Dataset of masked variants
# ----------------------------------------------------------------------
class _MaskedDataset(Dataset):
    """Lazily produces ``image * mask`` for each random mask.

    Masks are smoothed by routing them through a 32×32 → 224×224 resize,
    which mimics RISE's bilinear-upsampled low-resolution masks. This
    softens the boundaries between kept and deleted superpixels.
    """

    def __init__(self, image: np.ndarray, masks: np.ndarray,
                 mean: list[float], std: list[float]):
        self.image = image
        self.masks = masks
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=mean, std=std),
        ])
        self.mask_transform = T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),
            T.Resize((224, 224)),
        ])

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, idx: int):
        img = self.image_transform(self.image)
        mask = self.mask_transform(self.masks[idx])
        return img * mask, mask.squeeze()


# ----------------------------------------------------------------------
# RISE baseline — kept here because the original release ships them
# together and they share the same masked-forward harness
# ----------------------------------------------------------------------
def rise(
    model: nn.Module,
    image: np.ndarray,
    pred_class: int,
    mean: list[float],
    std: list[float],
    *,
    n_mask: int = 5000,
    batch_size: int = 512,
    device: torch.device = DEFAULT_DEVICE,
) -> torch.Tensor:
    """RISE attribution (Petsiuk et al., 2018) — uniform-grid baseline.

    Generates ``n_mask`` random 16×16 binary masks, each upsampled to
    224×224 with smoothing. The attribution at each pixel is the score-
    weighted average of mask values, normalized to ``[0, 1]``.

    Returns
    -------
    attribution : torch.Tensor of shape ``(224, 224)``, on ``device``.
    """
    masks = []
    for _ in range(n_mask):
        m = (np.random.uniform(0, 1, size=(16, 16)) < 0.5)
        masks.append((m * 255).astype(np.uint8))

    model = model.to(device).eval()
    loader = DataLoader(_MaskedDataset(image, masks, mean, std),
                        batch_size=batch_size, shuffle=False)

    mask_tensors = None
    n_pixels = None
    with torch.no_grad():
        for masked_img, mask in loader:
            masked_img, mask = masked_img.to(device), mask.to(device)
            probs = F.softmax(model(masked_img), dim=1).data
            scores = probs[:, pred_class].view(-1, 1, 1)
            if mask_tensors is None:
                mask_tensors = torch.sum(mask * scores, dim=0)
                n_pixels = torch.sum(mask, dim=0)
            else:
                mask_tensors += torch.sum(mask * scores, dim=0)
                n_pixels += torch.sum(mask, dim=0)

    attribution = mask_tensors / n_pixels
    a, b = torch.max(attribution), torch.min(attribution)
    return (attribution - b) / (a - b)


# ----------------------------------------------------------------------
# SP-RISA — paper's main attribution method
# ----------------------------------------------------------------------
def generate_sp_risa_masks(image: np.ndarray, n_mask: int = 4000) -> np.ndarray:
    """Build the stratified set of superpixel-deletion masks.

    The first ``n`` masks each delete a single superpixel (one-out
    masks); the remaining masks are drawn in ``n/2`` strata indexed by
    ``i``. Stratum ``i`` uses the threshold ``(n - 2 - i) / n`` — early
    strata delete few superpixels, late strata delete many. Within each
    stratum, ``ceil((n_mask - n) / (n/2))`` random masks are drawn.

    Generation halts a stratum early if a draw produces fewer than 5
    "kept" superpixels (avoids degenerate fully-deleted images).
    """
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=30, ruler=20)
    slic.iterate(10)
    labels = slic.getLabels()
    palette = list(set(labels.flatten()))
    n = len(palette)

    # Single-superpixel deletion masks (one per superpixel).
    mask_list = [np.where(labels == i, 0, 255).astype(np.uint8) for i in palette]

    # Stratified random masks.
    n_sample = round((n_mask - n) / int(n / 2))
    ss = np.random.uniform(0, 1, size=(int(n / 2), n_sample, n))
    for i in range(int(n / 2)):
        for j in range(n_sample):
            keep = np.asarray(ss[i, j] < (n - 2 - i) / n).nonzero()[0].tolist()
            if len(keep) < 5:
                break
            m = np.full_like(labels, 255, dtype=np.uint8)
            erase = list(set(palette) - set(keep))
            for k in erase:
                m[labels == k] = 0
            mask_list.append(m)

    return np.stack(mask_list, axis=0)


def sp_risa(
    model: nn.Module,
    image: np.ndarray,
    pred_class: int,
    mean: list[float],
    std: list[float],
    *,
    n_mask: int = 3000,
    batch_size: int = 200,
    device: torch.device = DEFAULT_DEVICE,
    use_data_parallel: bool = False,
) -> torch.Tensor:
    """SP-RISA attribution.

    Parameters
    ----------
    model
        Classifier (eval mode).
    image
        Input image as a ``(H, W, 3)`` BGR uint8 array (cv2 convention).
    pred_class
        Class index to attribute (typically the model's argmax).
    mean, std
        Per-channel normalization statistics for the dataset (BUSI / YBUS
        in the original release — see ``src.data.dataset``).
    n_mask
        Target total number of masks. Note: the original release calls
        ``generate_sp_risa_masks(image, 3000)`` (Table I lists 4000).
        Defaulting to 3000 here for fidelity.
    batch_size
        Forward-pass batch size for masked variants.
    device
        Compute device.
    use_data_parallel
        Wrap the model in ``nn.DataParallel`` (matches original code).

    Returns
    -------
    attribution : torch.Tensor of shape ``(224, 224)``, on ``device``,
        normalized to ``[0, 1]``.
    """
    masks = generate_sp_risa_masks(image, n_mask)

    if use_data_parallel:
        model = nn.DataParallel(model)
    model = model.to(device).eval()

    loader = DataLoader(_MaskedDataset(image, masks, mean, std),
                        batch_size=batch_size, shuffle=False)

    mask_tensors = None
    n_pixels = None
    with torch.no_grad():
        for masked_img, mask in loader:
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            probs = F.softmax(model(masked_img), dim=1).data

            # Deletion-based scoring: weight each mask by the *drop* in
            # the predicted-class probability when those superpixels are
            # removed. ``(1 - mask)`` flips kept/deleted so we accumulate
            # over the deleted regions.
            scores = (1 - probs[:, pred_class]).view(-1, 1, 1)
            inv_mask = 1.0 - mask

            if mask_tensors is None:
                mask_tensors = torch.sum(inv_mask * scores, dim=0)
                n_pixels = torch.sum(inv_mask, dim=0)
            else:
                mask_tensors += torch.sum(inv_mask * scores, dim=0)
                n_pixels += torch.sum(inv_mask, dim=0)

    attribution = mask_tensors / n_pixels
    a, b = torch.max(attribution), torch.min(attribution)
    return (attribution - b) / (a - b)


__all__ = ["rise", "sp_risa", "generate_sp_risa_masks"]
