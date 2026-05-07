# Dual-Channel Reliable Breast Ultrasound Image Classification

[![tests](https://github.com/<your-handle>/dual-channel-reliable-bus/actions/workflows/tests.yml/badge.svg)](https://github.com/<your-handle>/dual-channel-reliable-bus/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

PyTorch implementation accompanying **"Dual-Channel Reliable Breast Ultrasound Image Classification Based on Explainable Attribution and Uncertainty Quantification"** (Hu, Lei, et al.).

> **Reproducing the paper's results?** See [`REPRODUCING.md`](REPRODUCING.md).
> **Method walkthrough?** See [`docs/method.md`](docs/method.md).

This repository is a reorganized version of the original release ([`uncertainty_classification`](https://github.com/...)) — same algorithms, refactored into an installable package with documentation, type hints, tests, and CI.

---

## What it does

For each test image, the framework produces a single **Dual-channel Reliability Score** (DRS) intended to flag predictions a clinician should look at more carefully:

```
DRS = 0.5 · IRS + 0.5 · PRS
```

- **IRS (Inference Reliability Score)** — runs **SP-RISA**, a superpixel-based variant of RISE, to attribute the prediction to image regions; then measures recall of a *doctor-trusted* region of interest (lesion mask + 1.21× enlargement + below-lesion strip) within the top-`|M_pro|` attribution pixels.
- **PRS (Predictive Reliability Score)** — runs the classifier under **7 hand-picked TTA compositions** (rotation, flip, crop, and combinations) and reports `1 − H/log K`, where `H` is the entropy of the class-vote distribution. High = stable.

A second contribution, **Reliable Soup (RRS)**, uses DRS as the model-soup fusion criterion instead of validation accuracy — see `src/soup/reliable.py`.

Calibration quality is reported via **ECE under four binning schemes** (Roelofs et al., 2022): equal-width and equal-mass bins, each with both fixed `B=10` and monotonic-sweep auto-search.

## Repository layout

```
dual-channel-reliable-bus/
├── src/
│   ├── attribution/      SP-RISA + RISE baseline
│   ├── reliability/      IRS, PRS, end-to-end DRS pipeline
│   ├── soup/             Uniform/greedy baselines + Reliable Soup (RRS)
│   ├── metrics/          ECE: 4 variants (ew/em × bin/sweep)
│   ├── models/           torchvision factory (resnet50, vgg16, vit_b_16, ...)
│   ├── data/             BUS dataset (txt-list-based) + per-dataset stats
│   ├── training/         Classifier loop + eval helper
│   └── utils/            Reliability-diagram plotting
├── scripts/              CLI: train / evaluate / compute_drs / sprisa_demo /
│                              reliable_soup / base_soup / compute_dataset_stats
├── configs/              YAML defaults
├── tests/                pytest suite
└── docs/                 method walkthrough
```

## Installation

```bash
git clone https://github.com/<your-handle>/dual-channel-reliable-bus.git
cd dual-channel-reliable-bus
pip install -e .
```

Requires Python ≥ 3.9, PyTorch ≥ 1.13, OpenCV with `ximgproc` (install `opencv-contrib-python`, not just `opencv-python`).

## Quick start

### Data layout

Both datasets follow the same convention:

```
<root>/
├── images/
│   ├── <id_1>.jpg
│   └── ...
└── list/
    ├── train.txt        # YBUS, or BUSI/reliability — one record per line
    ├── val.txt          # filename,label
    └── test.txt
    # (BUSI only) list/classification/{train,val,test}.txt
```

For the **reliability evaluation step**, lesion masks come from a separately-trained U-Net; pass the mask root via `--mask-root`. The expected file is `<mask-root>/<dataset>/<image_stem>_pred.png`.

### 1. Train classifiers

```bash
python scripts/train.py \
    --model resnet50 \
    --root /data/YBUS --dataset YBUS \
    --epochs 100 --batch-size 64 \
    --save-path checkpoint
```

The script runs a random LR/batch-size sweep producing **multiple checkpoints per backbone**, which serve as ingredients for the soup methods. See [`REPRODUCING.md`](REPRODUCING.md) for details.

Backbones used in the paper: `resnet50`, `vgg16`, `vit_b_16`. Any other torchvision classifier name also works (passed straight through to `torchvision.models.<name>`).

### 2. Compute DRS and ECE on the test set

```bash
python scripts/compute_drs.py \
    --model resnet50 \
    --root /data/YBUS --dataset YBUS \
    --model-path checkpoint/YBUS/resnet50/<weights>.pth \
    --mask-root /data/predicted_masks \
    --batch-size 256 --temperature 8 --threshold 0.8 \
    --output runs/table3.json
```

This reproduces the **Table III** result: per-image IRS / PRS / DRS, mean DRS, and ECE under all four binning schemes for {Confidence, 1-Uncertainty, mDRS}.

### 3. Reliable Soup (RRS)

```bash
python scripts/reliable_soup.py \
    --model resnet50 \
    --root /data/YBUS --dataset YBUS \
    --model-path checkpoint/YBUS/resnet50 \
    --mask-root /data/predicted_masks \
    --samples 50 \
    --save-path checkpoint/soup
```

Uses DRS as the fusion criterion to produce a stronger ensemble from the checkpoint pool.

### 4. SP-RISA visualizations

```bash
python scripts/sprisa_demo.py \
    --model resnet50 --root /data/YBUS --dataset YBUS \
    --model-path checkpoint/YBUS/resnet50/<weights>.pth \
    --save-dir result/sprisa --include-rise
```

## Citation

```bibtex
@article{hu2023dualchannel,
  title   = {Dual-Channel Reliable Breast Ultrasound Image Classification
             Based on Explainable Attribution and Uncertainty Quantification},
  author  = {Hu, Haonan and Lei, Shuge and Sun, Dasheng and Zhang, Huabin
             and Yuan, Kehong and Dai, Jian and Tang, Jijun and Tong, Yan},
  year    = {2023}
}
```

## License

MIT
