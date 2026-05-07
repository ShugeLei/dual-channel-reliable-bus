"""Tests for IRS pro-mask construction and PRS entropy."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.reliability.inference import build_pro_mask
from src.reliability.predictive import compute_prs


# ----------------------------------------------------------------------
# build_pro_mask
# ----------------------------------------------------------------------
def test_pro_mask_contains_lesion():
    """The pro mask must always contain the original lesion."""
    h, w = 100, 100
    mask = torch.zeros(1, h, w)
    mask[0, 40:60, 40:60] = 1.0
    pro = build_pro_mask(mask)
    # Lesion must be a subset of pro mask.
    assert torch.all((mask[0] - pro) <= 0)
    # Pro mask must be strictly larger.
    assert pro.sum() > mask.sum()


def test_pro_mask_with_empty_lesion():
    """An empty lesion produces an empty pro mask (no contours found)."""
    h, w = 50, 50
    mask = torch.zeros(1, h, w)
    pro = build_pro_mask(mask)
    assert pro.sum() == 0


def test_pro_mask_includes_below_lesion_strip():
    """Pro mask should extend below the lesion (posterior region)."""
    h, w = 100, 100
    mask = torch.zeros(1, h, w)
    # Lesion positioned in the upper half so there's room below to extend.
    mask[0, 20:40, 40:60] = 1.0
    pro = build_pro_mask(mask)
    # There should be active pixels below the original lesion's bottom row.
    below_strip = pro[40:, :]
    assert below_strip.sum() > 0


def test_pro_mask_clamped_to_zero_one():
    """The output pro_mask must be in {0, 1} despite additive shifts."""
    h, w = 60, 60
    mask = torch.zeros(1, h, w)
    mask[0, 10:30, 10:30] = 1.0
    pro = build_pro_mask(mask)
    assert pro.min() >= 0
    assert pro.max() <= 1


# ----------------------------------------------------------------------
# PRS
# ----------------------------------------------------------------------
class _ConstantClassifier(nn.Module):
    """Always predicts the same class with high confidence."""

    def __init__(self, target: int = 0, num_classes: int = 2):
        super().__init__()
        self.target = target
        self.num_classes = num_classes

    def forward(self, x):
        b = x.shape[0]
        out = torch.full((b, self.num_classes), -10.0, device=x.device)
        out[:, self.target] = 10.0
        return out


def test_prs_is_one_for_perfectly_stable_classifier():
    """When all augmented predictions agree, PRS = 1."""
    torch.manual_seed(0)
    model = _ConstantClassifier(target=0).eval()
    image = torch.randn(1, 3, 224, 224)
    mask = torch.ones(1, 1, 224, 224)
    output_orig = F.softmax(model(image), dim=-1).data
    prs = compute_prs(model, output_orig, image, mask)
    assert prs.item() == pytest.approx(1.0, abs=1e-4)


def test_prs_is_in_unit_interval():
    """PRS must always lie in [0, 1] for any classifier output."""
    torch.manual_seed(1)
    # Random untrained classifier — its predictions will be non-trivially
    # variable across augmentations.
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
    ).eval()
    image = torch.randn(1, 3, 224, 224)
    mask = torch.ones(1, 1, 224, 224)
    output_orig = F.softmax(model(image), dim=-1).data
    prs = compute_prs(model, output_orig, image, mask)
    assert 0.0 <= prs.item() <= 1.0
