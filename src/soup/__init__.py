"""Model soup methods: uniform / greedy baselines and Reliable Soup (RRS)."""

from .base import fuse_uniform, greedy_soup, uniform_soup
from .reliable import compute_mdrs, fuse_partial, reliable_soup

__all__ = [
    "uniform_soup",
    "greedy_soup",
    "fuse_uniform",
    "reliable_soup",
    "fuse_partial",
    "compute_mdrs",
]
