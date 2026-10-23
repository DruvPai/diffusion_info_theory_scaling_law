from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from itg.config import Config
from itg.data import Loader, TensorDataset, make_loader
from itg.models import build_model
from itg.optim import EMA
from itg.processes import Process, make_process
from itg.seeds import generator
from itg.train import Batch, Task, denoising_loss, train_model
from tests import tiny

TRAIN_SEED = 4321


def tiny_config(**overrides: Any) -> Config:
    return tiny.tiny_config(**{"dataset_size": 32, "num_steps": 8, **overrides})


def tiny_loader(config: Config) -> Loader:
    assert config.dataset_size is not None
    images = torch.rand(config.dataset_size, *config.data_shape, generator=generator(5)) * 2 - 1
    return make_loader(
        config, TensorDataset(images), np.arange(config.dataset_size), dataset_seed=5
    )


def run(config: Config, directory: Path, **overrides: Any) -> None:
    train_model(
        config,
        make_process(config),
        tiny_loader(config),
        train_seed=TRAIN_SEED,
        checkpoint_dir=directory,
        log_interval=0,
        **{"checkpoint_interval": 0, **overrides},
    )


def load(directory: Path) -> dict[str, Any]:
    return torch.load(directory / "state.pt", map_location="cpu", weights_only=False)


def assert_identical(left: Any, right: Any, path: str = "") -> None:
    if isinstance(left, Tensor):
        assert torch.equal(left, right), f"{path} differs"
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), f"{path} keys differ"
        for key in left:
            assert_identical(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"{path} length differs"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_identical(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"{path} differs"


def count_batches(loader: Loader, limit: int | None = None) -> list[int]:
    """Count `loader.batch` calls, raising once `limit` of them have been served."""
    calls = [0]
    original: Callable[[], Batch] = loader.batch

    def batch() -> Batch:
        if limit is not None and calls[0] >= limit:
            raise RuntimeError("interrupted")
        calls[0] += 1
        return original()

    loader.batch = batch  # type: ignore[method-assign]
    return calls


def test_resume_is_bit_exact(tmp_path: Path) -> None:
    config = tiny_config(num_steps=8)
    run(config, tmp_path / "straight")

    resumed = tmp_path / "resumed"
    interrupted = tiny_loader(config)
    count_batches(interrupted, limit=4)
    with pytest.raises(RuntimeError, match="interrupted"):
        train_model(
            config,
            make_process(config),
            interrupted,
            train_seed=TRAIN_SEED,
            checkpoint_dir=resumed,
            log_interval=0,
            checkpoint_interval=4,
        )
    assert load(resumed)["completed_steps"] == 4

    tail = tiny_loader(config)
    drawn = count_batches(tail)
    train_model(
        config,
        make_process(config),
        tail,
        train_seed=TRAIN_SEED,
        checkpoint_dir=resumed,
        log_interval=0,
        checkpoint_interval=0,
    )
    assert drawn[0] == 4

    straight, restarted = load(tmp_path / "straight"), load(resumed)
    assert straight["optimizer"]["state"]
    for key in ("model", "ema_model", "optimizer", "loader"):
        assert_identical(straight[key], restarted[key], key)


def recording_task(records: list[tuple[Tensor, Tensor]]) -> Task:
    def loss(
        config: Config,
        model: nn.Module,
        process: Process,
        batch: Batch,
        t: Tensor,
        eps: Tensor,
        gen: torch.Generator | None,
    ) -> tuple[Tensor, dict[str, float]]:
        records.append((t.clone(), eps.clone()))
        return denoising_loss(config, model, process, batch, t, eps, gen)

    return Task(build=build_model, loss=loss)


def test_microbatching_draws_identical_times_and_noise(tmp_path: Path) -> None:
    config = tiny_config(num_steps=4)
    full: list[tuple[Tensor, Tensor]] = []
    micro: list[tuple[Tensor, Tensor]] = []
    run(config, tmp_path / "full", task=recording_task(full))
    run(
        replace(config, microbatch_size=config.batch_size // 2),
        tmp_path / "micro",
        task=recording_task(micro),
    )

    assert (len(full), len(micro)) == (4, 8)
    for step, (t, eps) in enumerate(full):
        halves = micro[2 * step : 2 * step + 2]
        assert torch.equal(torch.cat([half[0] for half in halves]), t)
        assert torch.equal(torch.cat([half[1] for half in halves]), eps)


def test_microbatching_leaves_the_update_unchanged(tmp_path: Path) -> None:
    """Not bit-exact: accumulation reorders the float sum and Muon's Newton-Schulz amplifies it to ~1e-6."""
    config = tiny_config()
    run(config, tmp_path / "full")
    run(replace(config, microbatch_size=config.batch_size // 2), tmp_path / "micro")

    full, micro = load(tmp_path / "full")["model"], load(tmp_path / "micro")["model"]
    for name, value in full.items():
        torch.testing.assert_close(value, micro[name], rtol=0.0, atol=1e-5)


def test_resume_with_changed_weight_config_raises(tmp_path: Path) -> None:
    config = tiny_config()
    run(config, tmp_path / "case")
    with pytest.raises(ValueError, match="different weight-affecting config"):
        run(replace(config, dropout_p=0.1), tmp_path / "case")


def test_loss_is_the_edm_weighted_x_0_error() -> None:
    config = tiny_config()
    process = make_process(config)
    model = build_model(config, process, generator(0))
    gen = generator(2)
    x_0 = torch.randn(3, *config.data_shape, generator=gen)
    eps = torch.randn(3, *config.data_shape, generator=gen)
    t = torch.tensor([0.02, 0.4, 0.9])
    x_t = process.forward(x_0, eps, t)

    weight = (1 / config.edm.sigma_data**2 + 1 / process.sigma(t) ** 2).sqrt()[:, None, None, None]
    expected = F.mse_loss(weight * model(x_t, t).value, weight * x_0)
    loss, _ = denoising_loss(config, model, process, (x_0,), t, eps, None)
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(("decay", "tracks"), [(0.0, True), (1.0, False)])
def test_ema_extremes(decay: float, tracks: bool) -> None:
    config = tiny_config()
    model = build_model(config, make_process(config), generator(0))
    ema = EMA(model, decay)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update(model)

    reference: dict[str, Tensor] = model.state_dict() if tracks else initial
    for name, value in ema.model.state_dict().items():
        assert torch.equal(value, reference[name])


def test_ema_model_is_detached_from_the_live_model() -> None:
    config = tiny_config()
    model: nn.Module = build_model(config, make_process(config), generator(0))
    ema = EMA(model, 0.9)
    assert ema.model is not model


def test_compiled_training_runs_and_is_reproducible(tmp_path: Path) -> None:
    """The default path. Compiled runs match each other bit-for-bit but not
    eager ones, which is why compile_model is in the weight fingerprint."""
    config = tiny_config(num_steps=2, compile_model=True)
    run(config, tmp_path / "a")
    run(config, tmp_path / "b")
    left, right = load(tmp_path / "a")["model"], load(tmp_path / "b")["model"]
    assert all(torch.equal(value, right[name]) for name, value in left.items())
    assert all(torch.isfinite(value).all() for value in left.values())
    assert config.weight_fingerprint() != replace(config, compile_model=False).weight_fingerprint()
