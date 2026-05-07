# Method walkthrough

This document maps the paper's contributions to the modules that implement
them and explains the key implementation details.

## Overview

```
                    ┌──────────────────────┐
                    │   Image x            │
                    └─────────┬────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
   classifier f         lesion mask M           classifier f
   ──────────────       (pre-computed           + 7 TTA
   p, ŷ                 by separate U-Net)      transforms
        │                     │                      │
        │                     ▼                      │
        │             build_pro_mask                 │
        │             (1.21× scale +                 │
        │              cumulative                    │
        │              downward shift                │
        │              by lesion bbox h)             │
        │                     │                      │
        │                     ▼                      │
        │             ┌──── M_pro ────┐              │
        │             │               │              │
        ▼             ▼               ▼              ▼
   SP-RISA  ───►   top-|M_pro|    intersection   class-vote
   attribution     attribution    / |M_pro|      proportions
        │             │               │              │
        │             └─── threshold ─┘              ▼
        │                     │                  H = -Σ p log p
        │                     ▼                      │
        │                 IRS_raw                    │
        │                     │                      ▼
        │                     ▼                  PRS = 1 - H/log K
        │             IRS = max(IRS_raw, p)
        │                     │
        │                     ▼
        └────►  DRS = 0.5·IRS + 0.5·PRS
```

## §III-B — SP-RISA

**File**: `src/attribution/sp_risa.py`

SP-RISA is a *deletion-based* attribution method. It generates a stratified
collection of random binary masks where each mask deletes a subset of
SLIC superpixels, runs the masked images through the classifier, and
accumulates the predicted-class probability *drop* over the deleted regions.

The mask generation is non-trivial:

1. The first `n` masks are deterministic single-superpixel deletions
   (one per superpixel, `n = number of superpixels`).
2. The remaining masks are stratified into `n/2` strata indexed by
   `i ∈ [0, n/2)`. Stratum `i` keeps each superpixel with probability
   `(n - 2 - i) / n`, sweeping from "delete few" to "delete many". For
   each stratum, `ceil((n_mask - n) / (n/2))` random masks are drawn.

The attribution at each pixel is then:

```python
A[h, w] = Σ_t (1 - p_t) · (1 - m_t)[h, w]   ÷   Σ_t (1 - m_t)[h, w]
```

normalized to `[0, 1]` by min-max scaling.

**SLIC parameters** (paper Table I):
- `cv2.ximgproc.createSuperpixelSLIC(image, region_size=30, ruler=20)`
- `slic.iterate(10)`

**Mask smoothing**: each binary mask is downsampled to `32×32` and resampled
to `224×224` via bilinear, mirroring RISE's smoothed low-resolution masks.

## §III-C — Inference Reliability Score

**File**: `src/reliability/inference.py`

Given the SP-RISA attribution map `A` and a U-Net-predicted lesion mask
`M`, IRS measures recall of the *doctor-trusted ROI* `M_pro`:

```python
S = top-|M_pro|(A)                          # binary
IRS = |S ∩ M_pro| / |M_pro|                # ∈ [0, 1]
```

Note: the threshold count `|M_pro|` is dynamic per image, not a fixed
top fraction.

`M_pro` is constructed in `build_pro_mask`:

1. **Spatial enlargement**: `Resize(1.21·H, 1.21·W) + CenterCrop(H, W)`.
2. **Below-lesion strip**: the lesion bounding-box height `h_bbox` is
   computed from `cv2.findContours`. The enlarged mask is then smeared
   downward by `h_bbox` cumulative 1-pixel shifts, simulating the
   posterior-acoustic region radiologists also read in ultrasound.

Whenever `|M_pro| = 0` (i.e. the U-Net failed to predict any lesion),
`compute_irs` returns `-1` and the caller falls back to the model's
prediction confidence as IRS — see `src/reliability/dual_channel.py`,
where `irs = max(irs, prob)` is applied unconditionally.

## §III-D — Predictive Reliability Score

**File**: `src/reliability/predictive.py`

PRS is a TTA-based stability score. Seven hand-picked transformation
compositions are applied to the input, and a forward pass per
augmentation is collected (plus the original = 8 total predictions).

Augmentation composition list (in order):

1. `RandomRotation(±30°) + Resize(224×224)`
2. `RandomCrop(224×224)`
3. `HorizontalFlip + Resize(224×224)`
4. `RandomRotation(±30°) + RandomCrop(224×224)`
5. `RandomRotation(±30°) + HorizontalFlip + Resize(224×224)`
6. `HorizontalFlip + RandomCrop(224×224)`
7. `RandomRotation(±30°) + HorizontalFlip + RandomCrop(224×224)`

**Mask preservation rejection**: each augmentation is retried up to 10
times if it crops the lesion away (`sum(augmented_mask) ≤ |M| - 10`).

The class-vote proportions `p_i = (#predictions = i) / 8` are converted
to PRS via:

```
PRS = 1 + Σ pᵢ log pᵢ / log K   =   1 - H / log K
```

Note this is the *negation* of the literal text in the paper, which
writes `PRS = H / log K`. The implementation form is the natural
"reliability" interpretation: high PRS = stable prediction.

## §III-A — DRS combination

**File**: `src/reliability/dual_channel.py`

```
DRS = 0.5 · IRS + 0.5 · PRS
```

with two important quirks preserved from the original code:

- `irs = max(irs, prob)` — IRS is floored by the model's softmax
  confidence on the predicted class.
- `conf_output = softmax(logits / T)` with `T = 8` by default — the
  "Confidence" baseline ECE is computed on temperature-scaled
  probabilities, *not* vanilla softmax.

## §IV-A — ECE

**File**: `src/metrics/{compute_ece,binning_methods,ece}.py`

Implements four ECE variants following Roelofs et al. (2022):

| Method            | Bin scheme   | Bin count       |
|-------------------|--------------|-----------------|
| `ew_ece_bin`      | equal-width  | fixed `B = 10`  |
| `em_ece_bin`      | equal-mass   | fixed `B = 10`  |
| `ew_ece_sweep`    | equal-width  | monotonic search |
| `em_ece_sweep`    | equal-mass   | monotonic search |

The `*_sweep` variants search upward over `B` and stop just before
per-bin accuracy ceases to be monotonic in confidence. This is the
"automatic search for the number of bins" variant the paper calls out
as the correctly-debiased estimator (§IV-A).

## RRS — Reliable Soup

**File**: `src/soup/reliable.py`

A model-soup variant that uses mean DRS on validation as the fusion
criterion (instead of validation accuracy). Distinguishing features
from greedy soup:

1. **Inverse seed selection**: starts from the *lowest*-DRS checkpoint
   in the pool (matches the original `RRS.py`).
2. **Random partial fusion**: each candidate is folded in via per-
   parameter Bernoulli mixing with probability `threshold`, rather
   than a uniform full average.
3. **Multi-pass**: the full pool is iterated `floor(1/threshold) + 1`
   times.

The hyperparameter `threshold ∈ [0.1, 0.5]` is swept randomly across
50 samples (`scripts/reliable_soup.py --samples 50`); the best test
accuracy across the sweep is reported.

## Module index

```
src/
├── attribution/sp_risa.py        SP-RISA + RISE
├── reliability/
│   ├── inference.py              IRS, build_pro_mask
│   ├── predictive.py             PRS via 7-aug TTA
│   └── dual_channel.py           drs_tester() — full eval pipeline
├── soup/
│   ├── base.py                   uniform_soup, greedy_soup
│   └── reliable.py               reliable_soup (RRS)
├── metrics/
│   ├── ece.py                    PyTorch-friendly wrapper
│   ├── compute_ece.py            CalibrationMetric + monotonic sweep
│   └── binning_methods.py        BinEqualWidth, BinEqualMass
├── data/
│   ├── dataset.py                BUS dataset (txt-list-based)
│   └── statistics.py             compute_dataset_stats
├── models/factory.py             torchvision wrapper
├── training/
│   ├── train.py                  classifier loop
│   └── evaluate.py               test-set forward helper
└── utils/reliability_diagram.py  calibration plot helpers
```
