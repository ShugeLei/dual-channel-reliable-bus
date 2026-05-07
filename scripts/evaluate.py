"""Evaluate every checkpoint in a directory on the test set.

Faithful port of the original ``evaluate.py`` ``__main__`` block. Useful
for inspecting a soup-ingredient pool: it reports accuracy, recall, and
F1 for every ``.pth`` file in ``--model-path`` and writes them as a CSV
log.

Example
-------
.. code-block:: bash

    python scripts/evaluate.py \\
        --model resnet50 \\
        --root /data/YBUS --dataset YBUS \\
        --model-path checkpoint/YBUS/resnet50 \\
        --log log/model_metrics.txt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from src.data import choose_dataset
from src.models import build_model
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
    return acc, recall, f1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="YBUS")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--model-path", type=str, required=True,
                   help="Either a single .pth file or a directory of them")
    p.add_argument("--log", type=str, default="log/model_metrics.txt")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = choose_dataset(args.dataset, args.root,
                               task="classification", mode="test")
    model = build_model(args.model, args.num_classes, pretrained=False).to(device)

    model_path = Path(args.model_path)
    if model_path.is_dir():
        files = sorted(model_path.iterdir(), reverse=True)
    else:
        files = [model_path]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a+") as f:
        for fp in tqdm(files):
            state = torch.load(fp, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            output_all, label_all = evaluation(model, test_data, args.batch_size)
            acc, rec, f1 = _binary_metrics(output_all, label_all)
            f.write(f"{fp.name},{acc:.5f},{rec:.5f},{f1:.5f}\n")
            logging.info("%s -> acc=%.5f rec=%.5f f1=%.5f", fp.name, acc, rec, f1)


if __name__ == "__main__":
    main()
