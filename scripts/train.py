"""Train classifiers on BUSI / YBUS.

Mirrors the original ``train.py`` CLI, including the random LR/batch
sweep that produces the checkpoint pool used by the model-soup methods.

Example
-------
.. code-block:: bash

    python scripts/train.py \\
        --model resnet50 \\
        --root /data/YBUS \\
        --dataset YBUS \\
        --epochs 100 --batch-size 64 \\
        --save-path checkpoint
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data import choose_dataset
from src.models import build_model
from src.training import TrainConfig, train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--root", type=str, required=True,
                   help="Path to dataset root containing images/ and list/")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dataset", type=str, default="YBUS")
    p.add_argument("--multi-gpu", action="store_true")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--acc-threshold", type=float, default=0.91,
                   help="Minimum val accuracy required to save a checkpoint")
    p.add_argument("--save-path", type=str, default="checkpoint")
    p.add_argument("--log-dir", type=str, default="log")
    p.add_argument("--n-sweep", type=int, default=20,
                   help="Number of (lr, batch_size) random combinations to try "
                        "(20 = 4 batch sizes * 5 lr perturbations, paper default)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_data = choose_dataset(args.dataset, args.root, mode="train")
    val_data = choose_dataset(args.dataset, args.root, mode="val")

    log_dir = Path(args.log_dir) / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.model}.txt"

    # Random LR/batch sweep — paper default is 4 × 5 = 20.
    n_lr = 5
    n_bs = max(1, args.n_sweep // n_lr)
    lr_bs_search = np.random.rand(n_bs, n_lr)

    for i in range(n_bs * n_lr):
        lr = lr_bs_search[i // n_lr, i % n_lr] * 1.5 * args.lr + 0.1 * args.lr
        batch_size = max(1, args.batch_size // (2 ** (i // n_lr)))
        logging.info("[%d/%d] lr=%.6f batch_size=%d", i + 1, n_bs * n_lr, lr, batch_size)

        model = build_model(args.model, args.num_classes,
                            pretrained=not args.no_pretrain)
        if args.multi_gpu:
            model = nn.DataParallel(model)

        cfg = TrainConfig(
            model=args.model, dataset=args.dataset, save_path=args.save_path,
            num_classes=args.num_classes, epochs=args.epochs,
            batch_size=batch_size, lr=lr, momentum=args.momentum,
            seed=args.seed, pretrain=not args.no_pretrain,
            multi_GPU=args.multi_gpu, acc_threshold=args.acc_threshold,
        )
        best_acc, best_recall, best_f1, best_epochs, train_hist, val_hist = train(
            model, train_data, val_data, cfg
        )

        with log_file.open("a+") as f:
            f.write(
                f"{time.strftime('%m/%d %H:%M')} lr={lr:.6f} "
                f"batch_size={batch_size} best_epochs={best_epochs} "
                f"best_acc={best_acc:.5f} best_recall={best_recall:.5f} "
                f"best_f1={best_f1:.5f}\n"
            )

        # Loss curve
        loss_dir = Path("result") / args.dataset
        loss_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(train_hist)
        plt.plot(val_hist)
        plt.legend(["train_loss", "val_loss"], loc="upper right")
        plt.savefig(
            loss_dir / f"{args.model}_{lr:.5f}_{batch_size}_{time.strftime('%m_%d_%H_%M')}.png"
        )
        plt.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
