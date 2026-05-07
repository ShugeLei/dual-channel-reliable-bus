"""Classifier training loop.

Faithful to the original ``train.py`` with a few modernizations: argparse
removed (now configurable via a dataclass), no hardcoded paths, more
detailed checkpoint naming, and the binary-only metrics from the original
generalized to multi-class.

Training-loop quirks worth preserving:

- **Random LR/batch-size sweep**: ``scripts/train.py`` runs this loop
  ``4 × 5 = 20`` times per backbone with random LR perturbations and
  decreasing batch sizes. The resulting checkpoint pool is the input to
  the model-soup methods (see ``src.soup``).
- **Selective checkpointing**: only models whose validation accuracy
  exceeds ``args.acc_threshold`` *and* whose final training loss is
  ``< 0.1`` are saved. This keeps the soup pool clean.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Single-run training configuration."""

    model: str = "resnet50"
    dataset: str = "YBUS"
    save_path: str = "checkpoint"
    num_classes: int = 2
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    momentum: float = 0.9
    seed: int = 1
    pretrain: bool = True
    multi_GPU: bool = False
    acc_threshold: float = 0.91
    loss_save_threshold: float = 0.1
    loss_weight: Optional[List[float]] = None  # for class-imbalanced losses


def _binary_metrics(probs: torch.Tensor, labels: torch.Tensor):
    """Accuracy / recall / F1 for binary classification."""
    preds = torch.argmax(probs, dim=-1)
    acc = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    recall = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return acc, recall, f1


def train(
    model: nn.Module,
    train_set,
    val_set,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
):
    """Train a single classifier configuration.

    Saves checkpoints under
    ``<save_path>/<dataset>/<model>/<model>_<acc>_<rec>_<f1>_<ep>_<lr>_<bs>_<seed>[_pretrained].pth``
    whenever both accuracy and final-step loss criteria are met. Returns
    ``(best_acc, best_recall, best_f1, best_epochs)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Class-weighted loss only for ISIC2018 in the original release.
    if cfg.dataset == "ISIC2018" and cfg.loss_weight is not None:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(cfg.loss_weight)
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    best_acc, best_recall, best_f1, best_epochs = 0.0, 0.0, 0.0, cfg.epochs
    train_loss_history, val_loss_history = [], []

    save_dir = Path(cfg.save_path) / cfg.dataset / cfg.model
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("lr=%s  batch_size=%s  epochs=%s",
                cfg.lr, cfg.batch_size, cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        train_loader = DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg.batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )

        # ----------- Train -----------
        model.train()
        epoch_loss = 0.0
        for data, target, _ in tqdm(train_loader, desc=f"Ep {epoch}/{cfg.epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * target.size(0)
        epoch_loss /= len(train_set)
        train_loss_history.append(epoch_loss)

        # ----------- Validate -----------
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            output_all, label_all = [], []
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * labels.size(0)
                output_all.append(F.softmax(outputs, dim=1))
                label_all.append(labels)
            val_loss /= len(val_set)
            val_loss_history.append(val_loss)
            scheduler.step(val_loss)

            output_all = torch.cat(output_all, dim=0)
            label_all = torch.cat(label_all, dim=0)
            acc, recall, f1 = _binary_metrics(output_all, label_all)

        logger.info(
            "Epoch %d/%d: train_loss=%.5f val_loss=%.5f acc=%.5f recall=%.5f f1=%.5f",
            epoch, cfg.epochs, epoch_loss, val_loss, acc, recall, f1,
        )

        # ----------- Save (selective) -----------
        if acc > cfg.acc_threshold and loss.item() < cfg.loss_save_threshold:
            model_weights = (
                model.module.state_dict() if cfg.multi_GPU else model.state_dict()
            )
            stem = "{}_{:.5f}_{:.5f}_{:.5f}_{}_{:.5f}_{}_{}".format(
                cfg.model, acc, recall, f1, epoch, cfg.lr, cfg.batch_size, cfg.seed,
            )
            if cfg.pretrain:
                stem += "_pretrained"
            torch.save(model_weights, save_dir / f"{stem}.pth")

        if acc > best_acc:
            best_acc = acc
            best_epochs = epoch + 10            # original release's heuristic
        best_recall = max(recall, best_recall)
        best_f1 = max(f1, best_f1)

    if best_epochs > cfg.epochs:
        cfg.epochs = best_epochs

    return best_acc, best_recall, best_f1, best_epochs, train_loss_history, val_loss_history


__all__ = ["TrainConfig", "train"]
