"""Uniform and greedy soup baselines for comparison.

Faithful port of the original ``base_soups.py`` CLI.

Example
-------
.. code-block:: bash

    python scripts/base_soup.py \\
        --model resnet50 \\
        --root /data/YBUS \\
        --dataset YBUS \\
        --model-path checkpoint/YBUS/resnet50
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.data import choose_dataset
from src.metrics import ece
from src.models import build_model
from src.soup import greedy_soup, uniform_soup
from src.training.evaluate import evaluation


def _binary_metrics(probs, labels):
    preds = probs.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    recall = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return acc, precision, recall, f1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="YBUS")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--model-path", type=str, required=True,
                   help="Directory of candidate checkpoints")
    p.add_argument("--save-path", type=str, default="log/baselines.txt")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_data = choose_dataset(args.dataset, args.root,
                              task="classification", mode="val")
    test_data = choose_dataset(args.dataset, args.root,
                               task="classification", mode="test")
    model = build_model(args.model, args.num_classes, pretrained=False).to(device)

    # ---- Uniform soup
    uniform_state, out, lab = uniform_soup(
        model, args.model_path, test_data, batch_size=args.batch_size,
    )
    u_acc, u_prec, u_rec, u_f1 = _binary_metrics(out, lab)
    u_em_sweep, _ = ece(out, lab, ce_method="em_ece_sweep")
    u_ew_sweep, _ = ece(out, lab, ce_method="ew_ece_sweep")
    logging.info(
        "Uniform soup: acc=%.5f recall=%.5f f1=%.5f ece(em_sweep)=%.5f ece(ew_sweep)=%.5f",
        u_acc, u_rec, u_f1, u_em_sweep, u_ew_sweep,
    )

    # ---- Greedy soup
    greedy_state, out, lab = greedy_soup(
        model, args.model_path, val_data, test_data, batch_size=args.batch_size,
    )
    g_acc, g_prec, g_rec, g_f1 = _binary_metrics(out, lab)
    g_em_sweep, _ = ece(out, lab, ce_method="em_ece_sweep")
    g_ew_sweep, _ = ece(out, lab, ce_method="ew_ece_sweep")
    logging.info(
        "Greedy soup:  acc=%.5f recall=%.5f f1=%.5f ece(em_sweep)=%.5f ece(ew_sweep)=%.5f",
        g_acc, g_rec, g_f1, g_em_sweep, g_ew_sweep,
    )

    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.save_path).open("a+") as f:
        f.write(
            f"uniform: acc={u_acc:.5f} rec={u_rec:.5f} f1={u_f1:.5f} "
            f"ece_em_sw={u_em_sweep:.5f} ece_ew_sw={u_ew_sweep:.5f}\n"
            f"greedy:  acc={g_acc:.5f} rec={g_rec:.5f} f1={g_f1:.5f} "
            f"ece_em_sw={g_em_sweep:.5f} ece_ew_sw={g_ew_sweep:.5f}\n"
        )


if __name__ == "__main__":
    main()
