"""Model-soup baselines (Wortsman et al., 2022).

Two non-DRS-aware soup recipes used as baselines for the Reliable Soup
contribution (``src.soup.reliable``):

- **Uniform soup**: average all checkpoints in a directory.
- **Greedy soup**: start from the best-on-val checkpoint, and only
  fold in additional checkpoints that *don't* hurt validation accuracy.

Faithful to ``base_soups.py`` in the original release; rewritten with
``Path``, configurable args, and no torchmetrics dependency.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from ..training.evaluate import evaluation


logger = logging.getLogger(__name__)


def fuse_uniform(cur_state_dict: dict, new_state_dict: dict, cur_num: int):
    """Running-average parameter fusion.

    Each call folds ``new_state_dict`` into the running average represented
    by ``(cur_state_dict, cur_num)``. After the call, ``cur_state_dict``
    is the mean of ``cur_num + 1`` checkpoints.
    """
    if new_state_dict.keys() != cur_state_dict.keys():
        raise ValueError("Checkpoint key mismatch — different model architectures?")

    for k, v in new_state_dict.items():
        if cur_state_dict[k].dtype != v.dtype:
            if k.endswith("num_batches_tracked"):
                v = v.to(dtype=torch.long)
        cur_state_dict[k] = cur_state_dict[k] * cur_num + v
        if cur_state_dict[k].dtype == torch.int64:
            cur_state_dict[k].div_(cur_num + 1, rounding_mode="trunc")
        else:
            cur_state_dict[k].div_(cur_num + 1)

    return cur_state_dict, cur_num + 1


def uniform_soup(
    model: nn.Module,
    checkpoint_dir: str | Path,
    test_data,
    *,
    batch_size: int = 256,
):
    """Average all checkpoints in ``checkpoint_dir`` and evaluate on ``test_data``.

    Returns
    -------
    (state_dict, output_all, label_all)
        ``state_dict`` is the averaged model parameters; the latter two
        are the test-set softmax outputs and labels for downstream
        metrics (accuracy, ECE, etc.).
    """
    checkpoint_dir = Path(checkpoint_dir)
    files = sorted(checkpoint_dir.iterdir())
    if not files:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")

    cur = torch.load(files[0], map_location="cpu")
    cur_num = 1
    for fp in files[1:]:
        new = torch.load(fp, map_location="cpu")
        cur, cur_num = fuse_uniform(cur, new, cur_num)

    model.load_state_dict(cur)
    output_all, label_all = evaluation(model, test_data, batch_size)
    return cur, output_all, label_all


def greedy_soup(
    model: nn.Module,
    checkpoint_dir: str | Path,
    val_data,
    test_data,
    *,
    batch_size: int = 256,
):
    """Greedy-soup recipe (Wortsman et al., 2022).

    Sort checkpoints by validation accuracy descending; greedily fold in
    each next checkpoint only if it improves validation accuracy.

    Returns
    -------
    (state_dict, output_all, label_all)
    """
    checkpoint_dir = Path(checkpoint_dir)
    files = sorted(checkpoint_dir.iterdir(), reverse=True)
    if not files:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")

    cur = torch.load(files[0], map_location="cpu")
    cur_num = 1
    model.load_state_dict(cur)

    output_all, label_all = evaluation(model, val_data, batch_size)
    best_acc = (output_all.argmax(dim=-1) == label_all).float().mean().item()
    logger.info("Greedy soup seed: %s  val_acc=%.5f", files[0].name, best_acc)

    for fp in files[1:]:
        candidate = torch.load(fp, map_location="cpu")
        new_state, new_num = fuse_uniform(deepcopy(cur), candidate, cur_num)
        model.load_state_dict(new_state)
        out, lab = evaluation(model, val_data, batch_size)
        acc = (out.argmax(dim=-1) == lab).float().mean().item()
        if acc > best_acc:
            cur, cur_num, best_acc = new_state, new_num, acc
            logger.info("  + %s -> val_acc=%.5f", fp.name, acc)

    model.load_state_dict(cur)
    output_all, label_all = evaluation(model, test_data, batch_size)
    return cur, output_all, label_all


__all__ = ["uniform_soup", "greedy_soup", "fuse_uniform"]
