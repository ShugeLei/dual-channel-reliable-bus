"""Reliable Soup (RRS) — DRS-aware model fusion.

A model-soup variant that uses the dual-channel reliability score (DRS)
as the fusion criterion rather than validation accuracy. Faithful to
``RRS.py`` in the original release.

The fusion proceeds as follows. Given a pool of trained checkpoints:

1. Load each checkpoint, compute its mean DRS on the **validation** set,
   and sort checkpoints by mean DRS ascending. (The lowest-DRS model
   is the seed — counterintuitive, but matches the original code.)
2. Iterate: for each candidate, do a *random partial* parameter fusion
   (each parameter independently included with probability ``threshold``)
   and recompute mean DRS. Keep the fusion if DRS improves; reject
   otherwise.
3. Repeat the full pass over the pool ``floor(1/threshold) + 1`` times.

The "random partial" fusion (``fuse_partial``) makes RRS distinct from
greedy soup: instead of averaging *all* parameters of an accepted
candidate, only a random subset of parameters is averaged in. This adds
stochasticity and lets multiple checkpoints contribute different layers.
"""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..reliability.inference import compute_irs
from ..reliability.predictive import compute_prs


logger = logging.getLogger(__name__)


def fuse_partial(
    cur_state_dict: dict,
    new_state_dict: dict,
    cur_num: torch.Tensor,
    threshold: float,
):
    """Random partial parameter fusion.

    Each parameter is independently included in the fusion with
    probability ``threshold``. Per-parameter inclusion counts (``cur_num``)
    are tracked separately so that included parameters are correctly
    averaged.
    """
    if new_state_dict.keys() != cur_state_dict.keys():
        raise ValueError("Checkpoint key mismatch.")

    ave_idx = torch.rand(len(new_state_dict))
    for i, (k, v) in enumerate(new_state_dict.items()):
        if cur_state_dict[k].dtype != v.dtype:
            if k.endswith("num_batches_tracked"):
                v = v.to(dtype=torch.long)
        if ave_idx[i].item() < threshold:
            cur_state_dict[k] = cur_state_dict[k] * cur_num[i] + v
            cur_num[i] += 1
            if cur_state_dict[k].dtype == torch.int64:
                cur_state_dict[k].div_(cur_num[i], rounding_mode="trunc")
            else:
                cur_state_dict[k].div_(cur_num[i])

    return cur_state_dict, cur_num


def compute_mdrs(model: nn.Module, dataset, args) -> float:
    """Mean DRS over a dataset (used as the soup-selection criterion)."""
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    mdrs = 0.0
    for img, label, mask, img_path in loader:
        model.eval()
        img, label, mask = img.to(device), label.to(device), mask.to(device)

        output_orig = F.softmax(model(img), dim=-1).data
        prob, pred = torch.max(output_orig.squeeze(), dim=-1)
        prob, pred = prob.item(), pred.item()

        irs = compute_irs(model, pred, mask, img_path, args)
        irs_val = irs.item() if isinstance(irs, torch.Tensor) else irs
        if irs_val < 0:
            # No predicted lesion — fall back to PRS-only.
            drs = compute_prs(model, output_orig, img, mask).item()
        else:
            irs_val = max(irs_val, prob)
            prs = compute_prs(model, output_orig, img, mask).item()
            drs = 0.5 * irs_val + 0.5 * prs
        mdrs += drs

    return mdrs / max(len(loader), 1)


def reliable_soup(
    model: nn.Module,
    checkpoint_dir: str | Path,
    val_data,
    test_data,
    args,
):
    """Run the RRS algorithm on a pool of trained checkpoints.

    Parameters
    ----------
    model
        A model instance (architecture only — weights will be loaded
        per-checkpoint inside this function).
    checkpoint_dir
        Directory containing ``.pth`` files to fuse.
    val_data, test_data
        Validation set (used as the DRS-selection objective) and test
        set (used to report final performance).
    args
        Namespace carrying ``dataset``, ``batch_size``, ``threshold``
        (the partial-fusion probability — paper sweeps over this).

    Returns
    -------
    (state_dict, num_ingredients)
        Final fused weights and the number of distinct checkpoints that
        contributed at least one parameter.
    """
    checkpoint_dir = Path(checkpoint_dir)
    files = sorted(checkpoint_dir.iterdir(), reverse=True)
    if not files:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")

    # 1. Per-checkpoint mean DRS on val
    drs_dict: dict[Path, float] = {}
    for fp in files:
        model.load_state_dict(torch.load(fp, map_location="cpu"))
        drs_dict[fp] = compute_mdrs(model, val_data, args)
        logger.info("  %s  mDRS=%.5f", fp.name, drs_dict[fp])

    # 2. Sort ASC by DRS and seed with the lowest-DRS model
    files.sort(key=lambda x: drs_dict[x])
    cur = torch.load(files[0], map_location="cpu")
    cur_num = torch.ones(len(cur), dtype=torch.long)
    model.load_state_dict(cur)
    best_drs = drs_dict[files[0]]

    n_ingredients = 1
    n_passes = int(1 / args.threshold) + 1
    for _ in range(n_passes):
        for fp in files[1:]:
            cand = torch.load(fp, map_location="cpu")
            new_state, new_num = fuse_partial(
                deepcopy(cur), cand, cur_num.clone(), args.threshold
            )
            model.load_state_dict(new_state)
            drs = compute_mdrs(model, val_data, args)
            if drs > best_drs:
                cur, cur_num, best_drs = new_state, new_num, drs
                n_ingredients += 1

    model.load_state_dict(cur)
    return cur, n_ingredients


__all__ = ["fuse_partial", "compute_mdrs", "reliable_soup"]
