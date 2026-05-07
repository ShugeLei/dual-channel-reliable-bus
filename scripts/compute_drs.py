"""Compute DRS and ECE metrics on the test set — reproduces Table III.

Loads a trained classifier checkpoint and a directory of pre-computed
U-Net lesion masks, runs the dual-channel reliability pipeline on every
test image, and reports accuracy / mDRS / ECE under all four binning
schemes for {Confidence, 1-Uncertainty, mDRS}.

Example
-------
.. code-block:: bash

    python scripts/compute_drs.py \\
        --model resnet50 \\
        --root /data/YBUS \\
        --dataset YBUS \\
        --model-path checkpoint/YBUS/resnet50/<weights>.pth \\
        --mask-root /data/predicted_masks \\
        --batch-size 256 \\
        --temperature 8 --threshold 0.8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import choose_dataset
from src.models import build_model
from src.reliability import drs_tester


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="YBUS")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256,
                   help="SP-RISA forward batch size")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to a trained classifier checkpoint .pth file")
    p.add_argument("--mask-root", type=str, required=True,
                   help="Directory containing predicted lesion masks "
                        "(<mask-root>/<dataset>/<id>_pred.png)")
    p.add_argument("--temperature", type=float, default=8,
                   help="Softmax temperature for the confidence ECE baseline "
                        "(matches original release)")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="DRS threshold for the screening report")
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON output path")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model, args.num_classes, pretrained=False)
    state = torch.load(args.model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.to(device).eval()

    test_data = choose_dataset(
        args.dataset, args.root,
        task="reliability", mode="test", mask_root=args.mask_root,
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    summary = drs_tester(model, test_loader, args)
    print(json.dumps(summary, indent=2, default=float))

    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(summary, f, indent=2, default=float)
        logging.info("Wrote results to %s", out)


if __name__ == "__main__":
    main()
