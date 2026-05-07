"""Test-set evaluation helper — wraps a forward pass over a DataLoader."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluation(
    model: torch.nn.Module,
    test_data,
    batch_size: int,
    num_workers: int = 4,
):
    """Run softmax forwards over a dataset; return concatenated probs & labels.

    Reproduces the helper from the original ``evaluate.py``. Returned
    tensors live on the same device as ``model``.
    """
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)

    output_all, label_all = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            output_all.append(F.softmax(model(images), dim=-1).data)
            label_all.append(labels)

    return torch.cat(output_all, dim=0), torch.cat(label_all, dim=0)


__all__ = ["evaluation"]
