# Reproducing the paper

A step-by-step recipe from a fresh clone to the numbers in Table III.

## 1. Environment

Tested on Ubuntu 22.04 + CUDA 11.8 with NVIDIA A100 / RTX 3090.

```bash
git clone https://github.com/<your-handle>/dual-channel-reliable-bus.git
cd dual-channel-reliable-bus

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # pinned versions
pip install -e .
pytest tests/ -v                        # should pass cleanly
```

> **Critical**: install `opencv-contrib-python`, *not* `opencv-python`. SP-RISA
> uses `cv2.ximgproc.createSuperpixelSLIC` which only ships in the contrib
> distribution. The lock file already specifies this.

## 2. Data

Both BUSI and YBUS share the same on-disk layout:

```
<root>/
├── images/
│   ├── <id>.jpg
│   └── ...
└── list/                    # for YBUS
    ├── train.txt            # filename,label  per line
    ├── val.txt
    └── test.txt
└── list/classification/     # for BUSI
    ├── train.txt
    ...
```

### BUSI (public)

Download from <https://scholar.cu.edu.eg/?q=afahmy/pages/dataset>. The release
ships images in three subfolders (`benign/`, `malignant/`, `normal/`) — flatten
them into `images/` and write your own `list/{train,val,test}.txt` with an 80/10/10
stratified split.

### YBUS (private)

The YBUS dataset is the proprietary clinical collection described in the paper
(Peking University Shenzhen Hospital + Shenzhen Baoan Maternity and Child Health
Hospital + Beijing Tsinghua ChangGung Hospital). It is **not** publicly
redistributed; the splits used in the paper are not shipped with this repo.
Researchers with IRB-approved access should contact the authors.

### Predicted lesion masks

The IRS pipeline expects U-Net-predicted masks rather than ground-truth
annotations. Train a U-Net (or U-Net++) on BUSI's image+ground-truth-mask pairs
and write predictions to:

```
<mask-root>/<dataset>/<image_stem>_pred.png
```

These are loaded at evaluation time via `--mask-root`.

## 3. Per-dataset normalization stats

If you're working with a new dataset, derive its RGB mean and std once:

```bash
python scripts/compute_dataset_stats.py /data/<dataset>/images
```

Then add the values to `DATASET_MEAN` / `DATASET_STD` in
`src/data/dataset.py`. BUSI and YBUS are already pre-populated.

## 4. Train classifiers

```bash
python scripts/train.py \
    --model resnet50 \
    --root /data/YBUS --dataset YBUS \
    --epochs 100 --batch-size 64 \
    --save-path checkpoint --log-dir log \
    --seed 1
```

The training loop runs a **random LR/batch-size sweep** (4 × 5 = 20
combinations by default) and saves a checkpoint whenever both
`val_acc > 0.91` and `final_step_loss < 0.1`. The resulting pool of
checkpoints is the input to the model-soup step.

For VGG16 / ViT-b, swap `--model resnet50` for `--model vgg16` /
`--model vit_b_16`. Any other torchvision classifier name also works.

## 5. Evaluate every individual checkpoint

```bash
python scripts/evaluate.py \
    --model resnet50 \
    --root /data/YBUS --dataset YBUS \
    --model-path checkpoint/YBUS/resnet50 \
    --log log/per_checkpoint_metrics.txt
```

This produces one CSV row per checkpoint. Useful to inspect the soup pool.

## 6. Reproduce Table III (DRS + ECE)

For each backbone, pick a representative checkpoint and run the full
DRS pipeline:

```bash
python scripts/compute_drs.py \
    --model resnet50 \
    --root /data/YBUS --dataset YBUS \
    --model-path checkpoint/YBUS/resnet50/<weights>.pth \
    --mask-root /data/predicted_masks \
    --batch-size 256 \
    --temperature 8 --threshold 0.8 \
    --output runs/table3_ybus_resnet50.json
```

The JSON output contains accuracy, mDRS, and ECE under all four binning
schemes for {Confidence, 1-Uncertainty, mDRS} — the three rows of Table III.

## 7. Run the model-soup baselines

```bash
# Uniform + greedy soup baselines (Wortsman et al.)
python scripts/base_soup.py \
    --model resnet50 --root /data/YBUS --dataset YBUS \
    --model-path checkpoint/YBUS/resnet50

# Reliable Soup (RRS, the paper's contribution)
python scripts/reliable_soup.py \
    --model resnet50 --root /data/YBUS --dataset YBUS \
    --model-path checkpoint/YBUS/resnet50 \
    --mask-root /data/predicted_masks \
    --samples 50 \
    --save-path checkpoint/soup
```

`reliable_soup.py` sweeps the partial-fusion threshold across 50 random
values and saves a soup checkpoint for each successful fusion. The best
test accuracy across the sweep is what's reported.

## 8. Hyperparameter knobs

All paper hyperparameters are exposed as CLI flags. The defaults match
the original release.

| Flag                    | Default | Meaning                                    |
|-------------------------|---------|--------------------------------------------|
| `--temperature`         | 8       | Softmax temperature for confidence ECE     |
| `--threshold` (DRS)     | 0.8     | DRS gating threshold for the screening report |
| `--threshold` (RRS)     | random  | Per-parameter inclusion probability for partial fusion |
| `--samples` (RRS)       | 50      | Number of random thresholds to sweep       |
| SP-RISA `n_mask`        | 3000    | Total masks per image (paper Table I says 4000; original code calls with 3000) |
| SLIC `region_size`      | 30      | Target superpixel area                     |
| SLIC `ruler`            | 20      | Compactness                                |
| SLIC iterations         | 10      | Refinement iterations                      |

## 9. Releasing your trained checkpoints

After verifying the numbers, attach checkpoints to a GitHub Release so
others can skip training:

```bash
git tag v0.1.0 && git push origin v0.1.0
gh release create v0.1.0 \
    checkpoint/YBUS/resnet50/<weights>.pth \
    checkpoint/soup/YBUS/resnet50/<RRsoup>.pth \
    --title "v0.1.0" \
    --notes "Trained on YBUS / BUSI with seed=1."
```

## 10. Archiving a DOI

Once a tag is published, link the GitHub repo to Zenodo
(<https://zenodo.org/account/settings/github/>). The next tag mints a
DOI you can put in the README and the paper's BibTeX.
