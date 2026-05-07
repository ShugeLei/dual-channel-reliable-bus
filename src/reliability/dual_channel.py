"""Dual-channel reliability — pipeline glue.

The end-to-end test runner from the original ``reliability.py``. For each
test image:

1. Forward the image through the classifier to get ``(prob, pred)``.
2. Compute IRS via SP-RISA + the pro-mask (see ``inference.py``).
3. Compute PRS via TTA (see ``predictive.py``).
4. Combine: ``DRS = 0.5 · IRS + 0.5 · PRS``.

There are two important quirks that come from the original code and are
**not** in the paper text:

- **``irs = max(irs, prob)``**: the IRS for an image is floored by the
  model's softmax confidence on the predicted class. This means a model
  that is very confident in its prediction is treated as at least that
  reliable on the inference side, even if the attribution doesn't fall
  inside the lesion.
- **Temperature scaling for the confidence ECE baseline**: when ECE is
  computed for the "Confidence" baseline, the logits are softmaxed at
  ``T = 8`` rather than ``T = 1``. This significantly softens the
  predicted distribution.

Both are preserved here for faithfulness to the original release.
"""
from __future__ import annotations

import argparse
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..metrics.ece import ece
from .inference import compute_irs
from .predictive import compute_prs


def _classification_metrics(output_all, label_all, num_classes: int):
    """Lightweight accuracy/recall/F1 helpers — torchmetrics version-stable.

    The original used ``torchmetrics.functional.accuracy`` etc. with
    ``num_classes=1, multiclass=False``. We reproduce the binary case
    inline to avoid pinning a torchmetrics version.
    """
    preds = torch.argmax(output_all, dim=-1)
    correct = (preds == label_all).float().mean().item()
    if num_classes == 2:
        tp = ((preds == 1) & (label_all == 1)).sum().item()
        fp = ((preds == 1) & (label_all == 0)).sum().item()
        fn = ((preds == 0) & (label_all == 1)).sum().item()
        recall = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
    else:
        recall, precision, f1 = float("nan"), float("nan"), float("nan")
    return correct, precision, recall, f1


def drs_tester(
    model: torch.nn.Module,
    test_loader: DataLoader,
    args: argparse.Namespace,
) -> Dict:
    """Run the full DRS evaluation loop on a test set.

    Parameters
    ----------
    model
        Classifier (eval mode, weights already loaded). Must already be
        on the correct device.
    test_loader
        DataLoader over a dataset that yields ``(img, label, mask, img_path)``
        tuples. Batch size **must** be 1 for SP-RISA per-image processing.
    args
        Namespace carrying ``dataset``, ``batch_size`` (for SP-RISA),
        ``num_classes``, ``temperature`` (default 8), ``threshold``
        (DRS gating threshold for the screening report).

    Returns
    -------
    summary : dict
        Aggregate metrics: accuracy, precision, recall, f1, mean DRS,
        ECE for {confidence, 1-uncertainty, mDRS} under the four ECE
        variants, and screening accuracy/recall/count above the
        DRS threshold.
    """
    device = next(model.parameters()).device
    model.eval()

    output_all, label_all = [], []
    prs_all, drs_all = [], []
    mdrs = 0.0

    for img, label, mask, img_path in tqdm(test_loader, desc="DRS"):
        img, label, mask = img.to(device), label.to(device), mask.to(device)

        logits = model(img)
        output_orig = F.softmax(logits, dim=-1).data
        prob, pred = torch.max(output_orig.squeeze(), dim=-1)
        prob, pred = prob.item(), pred.item()

        # Temperature-scaled confidence used as the ECE baseline.
        conf_output = F.softmax(logits / args.temperature, dim=-1).data
        output_all.append(conf_output)
        label_all.append(label)

        irs = compute_irs(model, pred, mask, img_path, args)
        irs = max(irs.item() if isinstance(irs, torch.Tensor) else irs, prob)
        prs = compute_prs(model, output_orig, img, mask)
        drs = 0.5 * irs + 0.5 * prs.item() if isinstance(prs, torch.Tensor) else 0.5 * irs + 0.5 * prs

        mdrs += drs

        # Pack reliabilities as one-hot-shaped score vectors so the ECE
        # routines can use them as drop-in replacements for softmax probs.
        prs_vec = torch.zeros_like(output_orig)
        drs_vec = torch.zeros_like(output_orig)
        prs_vec[:, pred] = prs
        drs_vec[:, pred] = drs
        prs_all.append(prs_vec.to(device))
        drs_all.append(drs_vec.to(device))

    output_all = torch.cat(output_all, dim=0)
    label_all = torch.cat(label_all, dim=0)
    prs_all = torch.cat(prs_all, dim=0)
    drs_all = torch.cat(drs_all, dim=0)

    n = len(test_loader)
    mdrs = mdrs / n if n > 0 else 0.0

    # ----------------------------------------------------------------
    # ECE under all four binning schemes for each of three reliability
    # signals: temperature-scaled confidence, 1-uncertainty (PRS), and
    # the dual-channel score.
    # ----------------------------------------------------------------
    def ece_block(scores):
        return {
            "ew_bin":  ece(scores, label_all, num_bins=10, ce_method="ew_ece_bin")[0],
            "em_bin":  ece(scores, label_all, num_bins=10, ce_method="em_ece_bin")[0],
            "ew_sweep": ece(scores, label_all, ce_method="ew_ece_sweep")[0],
            "em_sweep": ece(scores, label_all, ce_method="em_ece_sweep")[0],
        }

    ece_confidence = ece_block(output_all)
    ece_one_minus_unc = ece_block(prs_all)
    ece_mdrs = ece_block(drs_all)

    acc, precision, recall, f1 = _classification_metrics(
        output_all, label_all, args.num_classes
    )

    # Screening gate: keep only predictions whose DRS exceeds the threshold
    keep = torch.max(drs_all, dim=-1).values > args.threshold
    if keep.any():
        acc_screen, _, rec_screen, _ = _classification_metrics(
            output_all[keep], label_all[keep], args.num_classes
        )
        n_screen = int(keep.sum().item())
    else:
        acc_screen, rec_screen, n_screen = float("nan"), float("nan"), 0

    return {
        "n": int(label_all.size(0)),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mDRS": mdrs,
        "ECE": {
            "Confidence (T={})".format(args.temperature): ece_confidence,
            "1-Uncertainty": ece_one_minus_unc,
            "mDRS (ours)": ece_mdrs,
        },
        "screening": {
            "threshold": args.threshold,
            "accuracy": acc_screen,
            "recall": rec_screen,
            "n_kept": n_screen,
        },
    }


__all__ = ["drs_tester"]
