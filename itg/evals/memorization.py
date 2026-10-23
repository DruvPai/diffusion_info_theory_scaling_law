"""Nearest-neighbor memorization ratio in pixel space, against a streamed training set."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def memorization_ratio(samples: Tensor, train_batches: Iterable[Tensor]) -> float:
    """Fraction of samples whose nearest train neighbour is within a third of the second."""
    features = samples.reshape(samples.shape[0], -1)
    best = torch.full(
        (features.shape[0], 2), torch.inf, dtype=features.dtype, device=features.device
    )
    for batch in train_batches:
        train = batch.reshape(batch.shape[0], -1)
        distances_sq = (
            features.pow(2).sum(1, keepdim=True)
            + train.pow(2).sum(1)
            - 2.0 * features @ train.T
        ).clamp(min=0.0)
        best = torch.cat([best, distances_sq], dim=1).sort(dim=1).values[:, :2]
    return float((best[:, 0] < best[:, 1] / 9.0).float().mean())
