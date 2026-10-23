"""Reference image sets: draws from the training pool, and fixed arrays."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
from torch import Tensor

from itg.data import Dataset
from itg.runner.artifacts import draw_indices
from itg.runner.case import Case
from itg.seeds import stable_seed

ReferenceBatches = Callable[[int], Iterator[Tensor]]


def from_dataset(dataset: Dataset, ids: np.ndarray) -> ReferenceBatches:
    def batches(batch_size: int) -> Iterator[Tensor]:
        for start in range(0, len(ids), batch_size):
            yield dataset.get(ids[start : start + batch_size])

    return batches


def from_arrays(arrays: Sequence[Tensor]) -> ReferenceBatches:
    def batches(batch_size: int) -> Iterator[Tensor]:
        for array in arrays:
            for start in range(0, array.shape[0], batch_size):
                yield array[start : start + batch_size]

    return batches


def pool_reference(case: Case, count: int, seed: int) -> ReferenceBatches:
    """`count` images drawn without replacement from the whole pool."""
    dataset = case.dataset()
    ids = draw_indices(stable_seed(seed, "full"), pool_size=dataset.size, count=min(count, dataset.size))
    return from_dataset(dataset, ids)
