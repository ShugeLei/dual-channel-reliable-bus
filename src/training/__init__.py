"""Training and evaluation utilities."""

from .evaluate import evaluation
from .train import TrainConfig, train

__all__ = ["TrainConfig", "train", "evaluation"]
