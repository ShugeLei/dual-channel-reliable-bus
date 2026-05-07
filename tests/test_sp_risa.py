"""Tests for SP-RISA mask generation and the RISE/SP-RISA forward path."""
import numpy as np
import pytest
import torch
import torch.nn as nn

# cv2.ximgproc requires opencv-contrib-python. Skip the whole module gracefully
# if the user has plain opencv-python installed.
cv2 = pytest.importorskip("cv2")
ximgproc = pytest.importorskip("cv2.ximgproc")

from src.attribution.sp_risa import generate_sp_risa_masks, rise, sp_risa


@pytest.fixture
def synthetic_image():
    """A 256×256 BGR image with a bright square in the upper-left."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[40:120, 40:120] = 220
    return img


class _TinyClassifier(nn.Module):
    """3-channel global-average-pool + linear: cheap forward pass."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        return self.fc(self.gap(x).flatten(1))


# ----------------------------------------------------------------------
# Mask generation
# ----------------------------------------------------------------------
def test_generate_masks_returns_correct_shape(synthetic_image):
    masks = generate_sp_risa_masks(synthetic_image, n_mask=200)
    assert masks.ndim == 3
    assert masks.shape[1:] == synthetic_image.shape[:2]
    assert masks.dtype == np.uint8


def test_generate_masks_first_n_are_single_drops(synthetic_image):
    """The deterministic prefix is one mask per superpixel, each
    deleting just that superpixel."""
    masks = generate_sp_risa_masks(synthetic_image, n_mask=400)
    # Each of the first n masks should be 255 everywhere except one
    # superpixel (set to 0).
    first = masks[0]
    assert first.max() == 255
    assert first.min() == 0
    # The deleted region should be a connected blob (one superpixel).
    deleted = (first == 0).astype(np.uint8)
    n_components, _ = cv2.connectedComponents(deleted)
    assert n_components <= 2  # background + 1 deleted superpixel


# ----------------------------------------------------------------------
# Forward integration
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n_mask", [100])
def test_sp_risa_returns_normalized_attribution(synthetic_image, n_mask):
    torch.manual_seed(0)
    np.random.seed(0)
    model = _TinyClassifier().eval()
    attr = sp_risa(
        model, synthetic_image, pred_class=0,
        mean=[0.5] * 3, std=[0.5] * 3,
        n_mask=n_mask, batch_size=32,
        device=torch.device("cpu"),
    )
    assert attr.shape == (224, 224)
    # min-max normalized to [0, 1]
    assert torch.isclose(attr.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(attr.max(), torch.tensor(1.0), atol=1e-6)


def test_rise_returns_normalized_attribution(synthetic_image):
    torch.manual_seed(0)
    np.random.seed(0)
    model = _TinyClassifier().eval()
    attr = rise(
        model, synthetic_image, pred_class=0,
        mean=[0.5] * 3, std=[0.5] * 3,
        n_mask=64, batch_size=32,
        device=torch.device("cpu"),
    )
    assert attr.shape == (224, 224)
    assert torch.isclose(attr.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(attr.max(), torch.tensor(1.0), atol=1e-6)
