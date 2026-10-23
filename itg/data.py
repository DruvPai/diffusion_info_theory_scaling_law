"""Datasets and replayable index streams.

A dataset maps stable integer ids to samples; a sampler draws positions into
a loader's id list. Index streams stay in numpy keyed on `(seed, position)`,
so a resumed loader replays its predecessor exactly.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from itg.config import Config
from itg.seeds import stable_seed

IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp")
CACHE_DIR = Path.home() / ".cache" / "itg"
# Shared machine: leave cores for everyone else.
DECODE_WORKERS = 16

class Dataset(Protocol):
    def get(self, ids: np.ndarray) -> Tensor: ...

    @property
    def size(self) -> int: ...


@dataclass(frozen=True)
class TensorDataset:
    """An in-memory pool, for training on other models' generated samples."""

    data: Tensor

    def get(self, ids: np.ndarray) -> Tensor:
        return self.data[torch.from_numpy(np.asarray(ids, dtype=np.int64))]

    @property
    def size(self) -> int:
        return int(self.data.shape[0])


def _decode(job: tuple[Path, tuple[int, int]]) -> np.ndarray:
    """Center crop to a square, then bicubic resize."""
    path, size = job
    with Image.open(path) as image:
        image = ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


class ImageDataset:
    """Image folder decoded once into a uint8 memmap cache, then sliced.

    Samples are RGB, CHW, normalized to [-1, 1].
    """

    def __init__(self, root: str | Path, *, image_size: int) -> None:
        directory = Path(root).expanduser()
        self.paths = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"no image files found under {directory}")
        self.image_size = image_size
        self._images = self._load_cache(directory)

    @property
    def size(self) -> int:
        return len(self.paths)

    def _load_cache(self, directory: Path) -> np.ndarray:
        """Decode every image once into a uint8 memmap keyed by root and size."""
        key = json.dumps(
            [str(directory.resolve()), self.image_size, "center_crop", len(self.paths)],
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        path = CACHE_DIR / f"{directory.name}_{self.image_size}_{digest}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npy.tmp")
        size = (self.image_size, self.image_size)
        images = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.uint8,
            shape=(len(self.paths), self.image_size, self.image_size, 3),
        )
        work = ((path, size) for path in self.paths)
        with ProcessPoolExecutor(max_workers=DECODE_WORKERS) as pool:
            for index, decoded in enumerate(pool.map(_decode, work, chunksize=64)):
                images[index] = decoded
        images.flush()
        del images
        tmp.replace(path)
        return np.load(path, mmap_mode="r")

    def get(self, ids: np.ndarray) -> Tensor:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.min(initial=0) < 0 or (ids.size and ids.max() >= self.size):
            raise IndexError(f"image id out of range [0, {self.size})")
        raw = torch.from_numpy(np.ascontiguousarray(self._images[ids]))
        return (raw.float() / 127.5 - 1.0).permute(0, 3, 1, 2).contiguous()


def make_dataset(config: Config) -> Dataset:
    return ImageDataset(config.image_root, image_size=config.image_size)


# --- samplers -------------------------------------------------------------


class Sampler(Protocol):
    position: int

    def sample(self, size: int, batch_size: int) -> np.ndarray: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


@dataclass
class UniformSampler:
    """Uniform draws with replacement, pure in (seed, position)."""

    seed: int
    position: int = 0

    def sample(self, size: int, batch_size: int) -> np.ndarray:
        uniforms = np.random.default_rng((self.seed, self.position)).random(batch_size)
        self.position += batch_size
        return (uniforms * size).astype(np.int64)

    def state_dict(self) -> dict:
        return {"kind": "uniform", "seed": self.seed, "position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.seed, self.position = int(state["seed"]), int(state["position"])


@dataclass
class EpochShuffleSampler:
    """One permutation per epoch, dropping each epoch's short final batch."""

    seed: int
    epoch: int = 0
    offset: int = 0
    position: int = 0
    _order: np.ndarray | None = None

    def sample(self, size: int, batch_size: int) -> np.ndarray:
        if not 0 < batch_size <= size:
            raise ValueError("batch_size must be positive and at most the dataset size")
        if self.offset + batch_size > size:
            self.epoch += 1
            self.offset = 0
            self._order = None
        if self._order is None or self._order.shape[0] != size:
            self._order = np.random.default_rng(self.seed + self.epoch).permutation(size)
        local = self._order[self.offset : self.offset + batch_size]
        self.offset += batch_size
        self.position += batch_size
        return local

    def state_dict(self) -> dict:
        return {
            "kind": "epoch_shuffle",
            "seed": self.seed,
            "epoch": self.epoch,
            "offset": self.offset,
            "position": self.position,
        }

    def load_state_dict(self, state: dict) -> None:
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])
        self.position = int(state["position"])
        self._order = None


def make_sampler(config: Config, seed: int) -> Sampler:
    if config.regime == "epoch_shuffle":
        return EpochShuffleSampler(seed=seed)
    # iid and streaming both draw uniformly; streaming draws from the whole pool.
    return UniformSampler(seed=seed)


# --- loader ---------------------------------------------------------------


class Loader:
    """A dataset restricted to `ids`, batched through a replayable sampler."""

    def __init__(
        self,
        dataset: Dataset,
        ids: np.ndarray | None,
        sampler: Sampler,
        batch_size: int,
        *,
        flip: FlipSpec | None = None,
        labels: Tensor | None = None,
    ) -> None:
        self.dataset = dataset
        # `None` means the identity id map, i.e. the whole pool.
        self.ids = None if ids is None else np.asarray(ids, dtype=np.int64)
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.flip = flip
        # Set only for the classifier's labelled pool; `batch` then yields a pair.
        self.labels = labels

    def __len__(self) -> int:
        return self.dataset.size if self.ids is None else int(self.ids.shape[0])

    def source_ids(self, local: np.ndarray) -> np.ndarray:
        return local if self.ids is None else self.ids[local]

    def batch(self) -> tuple[Tensor, ...]:
        """`(samples,)`, or `(samples, labels)` for the classifier's pool."""
        position = self.sampler.position
        local = self.sampler.sample(len(self), self.batch_size)
        ids = self.source_ids(local)
        samples = self.dataset.get(ids)
        if self.flip is not None:
            samples = self.flip(samples, position)
        if self.labels is None:
            return (samples,)
        return samples, self.labels[torch.from_numpy(ids)]

    def state_dict(self) -> dict:
        return {"sampler": self.sampler.state_dict(), "batch_size": self.batch_size}

    def load_state_dict(self, state: dict) -> None:
        self.sampler.load_state_dict(state["sampler"])
        self.batch_size = int(state["batch_size"])


@dataclass(frozen=True)
class FlipSpec:
    """Replayable random horizontal flips, seeded by the sampler position."""

    seed: int

    def __call__(self, batch: Tensor, position: int) -> Tensor:
        rng = np.random.default_rng(
            stable_seed(self.seed, "horizontal_flip", int(position))
        )
        mask = torch.from_numpy(rng.random(batch.shape[0]) < 0.5)
        return torch.where(mask[:, None, None, None], batch.flip(-1), batch)


def make_flip(config: Config, seed: int) -> FlipSpec | None:
    return FlipSpec(seed) if config.image_random_horizontal_flip else None


def make_loader(
    config: Config, dataset: Dataset, ids: np.ndarray | None, *, dataset_seed: int
) -> Loader:
    return Loader(
        dataset,
        ids,
        make_sampler(config, stable_seed(dataset_seed, "sampler")),
        config.batch_size,
        flip=make_flip(config, dataset_seed),
    )
