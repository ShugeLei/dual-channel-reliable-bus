"""Compute per-channel RGB mean and std for a dataset.

Use this to derive the ``DATASET_MEAN`` / ``DATASET_STD`` constants in
``src/data/dataset.py`` for a new dataset. The values printed for BUSI
and YBUS by the original release are already wired in.

Example
-------
.. code-block:: bash

    python scripts/compute_dataset_stats.py /data/YBUS/images
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.statistics import compute_dataset_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image_dir", type=Path,
                   help="Directory of images (e.g. <root>/images)")
    args = p.parse_args()

    mean, std = compute_dataset_stats(args.image_dir)
    print(f"mean = [{mean[0]:.8f}, {mean[1]:.8f}, {mean[2]:.8f}]")
    print(f"std  = [{std[0]:.8f}, {std[1]:.8f}, {std[2]:.8f}]")


if __name__ == "__main__":
    main()
