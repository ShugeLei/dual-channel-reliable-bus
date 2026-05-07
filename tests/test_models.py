"""Tests for the model factory."""
import pytest
import torch

from src.models import build_model, KNOWN_MODELS


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("not_a_real_torchvision_name", num_classes=2)


@pytest.mark.parametrize("name", ["resnet50"])
def test_resnet_builds_and_forwards(name):
    """Smoke-test the most common backbone end-to-end."""
    model = build_model(name, num_classes=2, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2)


def test_known_models_listed():
    assert "resnet50" in KNOWN_MODELS
    assert "vgg16" in KNOWN_MODELS
    assert "vit_b_16" in KNOWN_MODELS
