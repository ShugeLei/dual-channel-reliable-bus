"""Model factory.

The original release creates classifiers via ``eval('torchvision.models.{}'.format(args.model))``
which is convenient but brittle. We wrap that with a registry so any
torchvision classifier name is supported, plus a small whitelist of the
ones used in the paper (VGG16, ResNet50, ViT-b).

For pretrained weights, the original release loads ImageNet-pretrained
state into a head-stripped model (skipping ``fc`` and ``classifier``
keys), then trains the new head from scratch. We replicate that logic
in :func:`build_model`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


# Convenience alias list — any other torchvision classifier name will
# also work (passed straight through to ``getattr(torchvision.models, ...)``).
KNOWN_MODELS = ("vgg16", "resnet50", "vit_b_16", "efficientnet_b0",
                "mobilenet_v2")


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a torchvision classifier with a custom-classes head.

    Parameters
    ----------
    name
        Any model factory name on ``torchvision.models``, e.g.
        ``"resnet50"``, ``"vgg16"``, ``"vit_b_16"``.
    num_classes
        Number of output classes.
    pretrained
        If ``True``, ImageNet weights are loaded into the backbone
        (skipping the head, which is replaced for ``num_classes``).

    Returns
    -------
    model : nn.Module
    """
    if not hasattr(torchvision.models, name):
        raise ValueError(f"Unknown torchvision model: {name!r}")

    factory = getattr(torchvision.models, name)
    model = factory(weights=None, num_classes=num_classes)

    if pretrained:
        # Match the original release's manual transfer-learning path:
        # load ImageNet weights with the original 1000-class head, then
        # filter that head out so it doesn't clash with our new head.
        pretrained_weights = factory(weights="DEFAULT").state_dict()
        filtered = {k: v for k, v in pretrained_weights.items()
                    if "fc" not in k and "classifier" not in k
                    and "heads" not in k}
        model.load_state_dict(filtered, strict=False)

    return model


__all__ = ["build_model", "KNOWN_MODELS"]
