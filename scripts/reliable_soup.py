"""Reliable Soup (RRS) — DRS-aware model fusion.

Faithful port of the original ``RRS.py`` CLI.

Example
-------
.. code-block:: bash

    python scripts/reliable_soup.py \\
        --model resnet50 \\
        --root /data/YBUS \\
        --dataset YBUS \\
        --model-path checkpoint/YBUS/resnet50 \\
        --mask-root /data/predicted_masks \\
        --save-path checkpoint/soup \\
        --samples 50
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data import choose_dataset
from src.models import build_model
from src.soup import reliable_soup
from src.training.evaluate import evaluation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="YBUS")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.4,
                   help="Per-parameter inclusion probability for partial fusion")
    p.add_argument("--model-path", type=str, required=True,
                   help="Directory of candidate checkpoints")
    p.add_argument("--mask-root", type=str, required=True)
    p.add_argument("--samples", type=int, default=50,
                   help="Number of random threshold values to sweep")
    p.add_argument("--save-path", type=str, default="checkpoint/soup")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_data = choose_dataset(
        args.dataset, args.root,
        task="reliability", mode="val", mask_root=args.mask_root,
    )
    test_data = choose_dataset(
        args.dataset, args.root,
        task="classification", mode="test",
    )

    model = build_model(args.model, args.num_classes, pretrained=False).to(device)

    # Random sweep over threshold values — paper reports best across this sweep.
    best_acc, best_rec, best_f1 = 0.0, 0.0, 0.0
    threshold_sweep = sorted(np.random.rand(args.samples).tolist())

    for i in tqdm(range(args.samples), desc="threshold sweep"):
        args.threshold = 0.1 + 0.4 * threshold_sweep[i]
        try:
            cur_state, n_ingredients = reliable_soup(
                model, args.model_path, val_data, test_data, args
            )
        except FileNotFoundError as e:
            logging.error(str(e))
            return

        if n_ingredients <= 1:
            continue

        model.load_state_dict(cur_state)
        out, lab = evaluation(model, test_data, args.batch_size)
        preds = out.argmax(dim=-1)
        acc = (preds == lab).float().mean().item()
        tp = ((preds == 1) & (lab == 1)).sum().item()
        fp = ((preds == 1) & (lab == 0)).sum().item()
        fn = ((preds == 0) & (lab == 1)).sum().item()
        recall = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        save_dir = Path(args.save_path) / args.dataset / args.model
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{args.model}_{acc:.5f}_{recall:.5f}_{f1:.5f}_RRsoup.pth"
        torch.save(cur_state, save_path)

        logging.info(
            "n_ingredients=%d threshold=%.5f acc=%.5f recall=%.5f f1=%.5f -> %s",
            n_ingredients, args.threshold, acc, recall, f1, save_path,
        )

        if acc > best_acc:
            best_acc, best_rec, best_f1 = acc, recall, f1

    logging.info("Best across sweep: acc=%.5f recall=%.5f f1=%.5f",
                 best_acc, best_rec, best_f1)


if __name__ == "__main__":
    main()
