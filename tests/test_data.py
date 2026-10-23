from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

import itg.data as data_module
from itg.data import (
    EpochShuffleSampler,
    FlipSpec,
    ImageDataset,
    Loader,
    Sampler,
    TensorDataset,
    UniformSampler,
    make_flip,
)
from tests.tiny import tiny_config


def write_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def image_folder(root: Path, images: list[np.ndarray]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index, array in enumerate(images):
        write_image(root / f"{index:03d}.png", array)
    return root


@pytest.mark.parametrize(
    ("original", "replay"),
    [
        (lambda: UniformSampler(seed=99), lambda: UniformSampler(seed=0)),
        (lambda: EpochShuffleSampler(seed=3), lambda: EpochShuffleSampler(seed=0)),
    ],
    ids=["uniform", "epoch_shuffle"],
)
def test_sampler_state_replays_the_next_draw(
    original: Callable[[], Sampler], replay: Callable[[], Sampler]
) -> None:
    sampler = original()
    sampler.sample(17, 4)
    state = sampler.state_dict()
    expected = sampler.sample(17, 4)
    resumed = replay()
    resumed.load_state_dict(state)
    np.testing.assert_array_equal(resumed.sample(17, 4), expected)


def test_epoch_shuffle_covers_support_and_drops_short_batch() -> None:
    sampler = EpochShuffleSampler(seed=3)
    epoch = np.concatenate([sampler.sample(10, 5) for _ in range(2)])
    assert sorted(epoch.tolist()) == list(range(10))

    partial = EpochShuffleSampler(seed=3)
    drawn = np.concatenate([partial.sample(10, 4) for _ in range(2)])
    assert len(set(drawn.tolist())) == 8
    partial.sample(10, 4)
    assert partial.epoch == 1


def test_image_dataset_range_and_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path / "cache")
    rng = np.random.default_rng(0)
    root = image_folder(tmp_path / "images", [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)])
    chw = ImageDataset(root, image_size=8).get(np.array([0]))
    assert chw.shape == (1, 3, 8, 8)
    assert float(chw.min()) >= -1.0 and float(chw.max()) <= 1.0


def test_image_dataset_center_crop_preserves_aspect_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path / "cache")
    array = np.zeros((10, 20, 3), dtype=np.uint8)
    array[:, :5, 0] = 255
    array[:, 5:15, 1] = 255
    array[:, 15:, 2] = 255
    root = image_folder(tmp_path / "images", [array])

    image = ImageDataset(root, image_size=8).get(np.array([0]))[0]
    assert float(image[1].mean()) > 0.9
    assert float(image[[0, 2]].mean()) < -0.9


def test_image_dataset_out_of_range_id_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path / "cache")
    root = image_folder(tmp_path / "images", [np.zeros((8, 8, 3), dtype=np.uint8)])
    dataset = ImageDataset(root, image_size=8)
    with pytest.raises(IndexError):
        dataset.get(np.array([1]))


def test_image_cache_matches_a_direct_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path / "cache")
    rng = np.random.default_rng(1)
    arrays = [rng.integers(0, 256, (11, 19, 3), dtype=np.uint8) for _ in range(3)]
    root = image_folder(tmp_path / "images", arrays)

    cached = ImageDataset(root, image_size=8).get(np.arange(3))
    for index in range(3):
        with Image.open(root / f"{index:03d}.png") as handle:
            fitted = ImageOps.fit(
                handle.convert("RGB"), (8, 8), method=Image.Resampling.BICUBIC
            )
        decoded = np.array(fitted, dtype=np.uint8)
        expected = torch.from_numpy(decoded).float() / 127.5 - 1.0
        torch.testing.assert_close(cached[index], expected.permute(2, 0, 1))


def test_loader_state_replays_the_next_batch() -> None:
    dataset = TensorDataset(torch.randn(16, 3, generator=torch.Generator().manual_seed(7)))
    ids = np.arange(16)

    def build(seed: int) -> Loader:
        return Loader(dataset, ids, UniformSampler(seed=seed), batch_size=4)

    loader = build(11)
    loader.batch()
    state = loader.state_dict()
    (expected,) = loader.batch()

    replay = build(0)
    replay.load_state_dict(state)
    assert torch.equal(replay.batch()[0], expected)


def test_flip_spec_flips_and_replays_from_loader_state() -> None:
    flip = make_flip(tiny_config(image_random_horizontal_flip=True), seed=5)
    assert isinstance(flip, FlipSpec)

    batch = torch.randn(8, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    flipped = flip(batch, position=0)
    reference = batch.flip(-1)
    kept = torch.tensor([bool(torch.equal(flipped[i], batch[i])) for i in range(8)])
    turned = torch.tensor([bool(torch.equal(flipped[i], reference[i])) for i in range(8)])
    assert bool((kept | turned).all()) and bool(turned.any())

    loader = Loader(
        TensorDataset(batch), np.arange(8), UniformSampler(seed=2), batch_size=4, flip=flip
    )
    loader.batch()
    state = loader.state_dict()
    (expected,) = loader.batch()
    replay = Loader(
        TensorDataset(batch), np.arange(8), UniformSampler(seed=0), batch_size=1, flip=flip
    )
    replay.load_state_dict(state)
    assert torch.equal(replay.batch()[0], expected)
