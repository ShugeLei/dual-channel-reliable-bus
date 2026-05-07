"""Generate SP-RISA / RISE attribution maps for a test set.

Faithful CLI port of the original ``sprisa.py`` ``__main__`` block.

Example
-------
.. code-block:: bash

    python scripts/sprisa_demo.py \\
        --model resnet50 \\
        --root /data/YBUS \\
        --dataset YBUS \\
        --model-path checkpoint/YBUS/resnet50/<weights>.pth \\
        --save-dir result/sprisa
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm

from src.attribution import rise, sp_risa
from src.data import DATASET_MEAN, DATASET_STD
from src.models import build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--root", type=str, required=True,
                   help="Path to dataset root (with images/ and list/test.txt)")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dataset", type=str, default="YBUS")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="result/sprisa")
    p.add_argument("--include-rise", action="store_true",
                   help="Also compute and save RISE attribution maps")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    (save_dir / "sprisa").mkdir(parents=True, exist_ok=True)
    if args.include_rise:
        (save_dir / "rise").mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, args.num_classes, pretrained=False)
    state = torch.load(args.model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.to(device).eval()

    # Load test list
    list_root = Path(args.root) / "list"
    if "BUSI" in str(args.root):
        list_root = list_root / "classification"
    test_list = (list_root / "test.txt").read_text().strip().splitlines()

    mean, std = DATASET_MEAN[args.dataset], DATASET_STD[args.dataset]

    eval_transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=mean, std=std),
    ])

    image_dir = Path(args.root) / "images"

    for record in tqdm(test_list):
        filename, label = record.split(",")
        label = int(label)
        img_path = image_dir / filename
        # cv2 returns BGR uint8
        image = cv2.resize(cv2.imread(str(img_path), 1), (256, 256))

        # Forward to determine predicted class
        x = eval_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=-1)[0]
        pred = int(probs.argmax().item())

        if pred != label:
            continue                                       # original: skip mispredictions

        attribution = sp_risa(model, image, pred, mean, std,
                              batch_size=args.batch_size, device=device)
        attr_map = (attribution * 255).cpu().numpy().astype(np.uint8)
        out_name = filename.replace(".jpg", "_sprisa.png")
        cv2.imwrite(str(save_dir / "sprisa" / out_name), attr_map)

        if args.include_rise:
            rise_attr = rise(model, image, pred, mean, std,
                             batch_size=args.batch_size, device=device)
            rise_map = (rise_attr * 255).cpu().numpy().astype(np.uint8)
            out_name = filename.replace(".jpg", "_rise.png")
            cv2.imwrite(str(save_dir / "rise" / out_name), rise_map)


if __name__ == "__main__":
    main()
