"""The training loop, checkpointing, and resume.

Per-step randomness is a pure function of `(train_seed, step)`: the step
generator yields, in this fixed order, the times, the forward noise, and a
dropout seed — always all three, and always at the full batch size before
microbatch slicing, so gradient accumulation does not move the stream.
(The one exception is the dropout masks themselves, which are drawn per
microbatch because a full-batch mask set would not fit in memory.)
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from itg.config import Config
from itg.data import Loader
from itg.models import build_classifier, build_model, count_params
from itg.optim import EMA, make_optimizer, tensor_norm
from itg.processes import Process
from itg.seeds import draw_seed, generator, stable_seed

# (samples,) for denoising, (samples, labels) for the classifier.
Batch = tuple[Tensor, ...]
LossFn = Callable[
    [Config, nn.Module, Process, Batch, Tensor, Tensor, torch.Generator | None],
    tuple[Tensor, dict[str, float]],
]


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [
            {name: float(value) if value else None for name, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    """Step telemetry lives beside the checkpoint, not inside it: it embeds
    wall-clock throughput, which would make `state.pt` non-reproducible."""
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_times(
    process: Process, gen: torch.Generator, batch_size: int, mean: float, std: float
) -> Tensor:
    logsnr = torch.randn(batch_size, generator=gen) * std + mean
    return process.logsnr_inverse(logsnr).clamp(1e-4, 1.0 - 1e-4)


def loss_weight(config: Config, process: Process, t: Tensor) -> Tensor:
    """EDM's weighting: unit-scale targets for the preconditioner's raw output."""
    return (1.0 / config.edm.sigma_data**2 + 1.0 / process.sigma(t) ** 2).sqrt()


def denoising_loss(
    config: Config,
    model: nn.Module,
    process: Process,
    batch: Batch,
    t: Tensor,
    eps: Tensor,
    gen: torch.Generator | None,
) -> tuple[Tensor, dict[str, float]]:
    """MSE on the x_0 error under EDM's weighting."""
    (x_0,) = batch
    x_t = process.forward(x_0, eps, t)
    pred = process.convert_prediction(model(x_t, t, gen), x_t, t, "x_0").value
    weight = loss_weight(config, process, t).reshape(-1, *([1] * (x_0.ndim - 1)))
    return F.mse_loss(weight * pred, weight * x_0), {}


def classifier_loss(
    config: Config,
    model: nn.Module,
    process: Process,
    batch: Batch,
    t: Tensor,
    eps: Tensor,
    gen: torch.Generator | None,
) -> tuple[Tensor, dict[str, float]]:
    """Multinomial log-loss over which source model produced a noised sample."""
    del config
    x_0, labels = batch
    logits = model(process.forward(x_0, eps, t), t, gen)
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return F.cross_entropy(logits, labels), {"accuracy": float(accuracy)}


@dataclass(frozen=True)
class Task:
    """The two things that differ between the networks we train."""

    build: Callable[[Config, Process, torch.Generator], nn.Module]
    loss: LossFn


DENOISING = Task(build=build_model, loss=denoising_loss)


def classifier_task(num_classes: int) -> Task:
    return Task(
        build=lambda config, process, gen: build_classifier(
            config, process, num_classes, gen
        ),
        loss=classifier_loss,
    )


@dataclass
class TrainResult:
    model: nn.Module
    metrics: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def content_fingerprint(config: Config, train_seed: int, steps: int) -> str:
    """Identity of a checkpoint's weights.

    Training is deterministic in `(config, train_seed, completed_steps)`, so
    this is exact and costs no file hashing.
    """
    raw = f"{config.weight_fingerprint()}|{train_seed}|{steps}"
    return hashlib.sha256(raw.encode()).hexdigest()


def train_model(
    config: Config,
    process: Process,
    loader: Loader,
    *,
    train_seed: int,
    checkpoint_dir: Path,
    task: Task = DENOISING,
    device: torch.device | str = "cpu",
    log_interval: int = 1_000,
    checkpoint_interval: int = 1_000,
    force_retrain: bool = False,
    metadata: dict[str, Any] | None = None,
    logger: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    device = torch.device(device)
    state_path = checkpoint_dir / "state.pt"
    model = task.build(config, process, generator(stable_seed(train_seed, "init"))).to(
        device
    )
    optimizer, scheduler = make_optimizer(model, config)
    ema = EMA(model, config.ema_decay)
    ema.model.to(device)

    base_metadata: dict[str, Any] = {
        "parameter_count": count_params(model),
        "train_seed": int(train_seed),
        **(metadata or {}),
    }
    completed = 0
    metrics: list[dict[str, Any]] = []
    metrics_path = checkpoint_dir / "metrics.csv"

    if state_path.exists() and not force_retrain:
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["config_hash"] != config.weight_fingerprint():
            raise ValueError(
                f"{state_path} was trained under a different weight-affecting config; "
                "pass force_retrain to retrain from scratch"
            )
        model.load_state_dict(state["model"])
        ema.model.load_state_dict(state["ema_model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        loader.load_state_dict(state["loader"])
        completed = int(state["completed_steps"])
        metrics = read_metrics(metrics_path)
        base_metadata = {**state["metadata"], **(metadata or {})}

    def persist(step: int) -> None:
        payload = {
            "model": model.state_dict(),
            "ema_model": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loader": loader.state_dict(),
            "completed_steps": step,
            "metadata": base_metadata,
            "config_hash": config.weight_fingerprint(),
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(state_path)
        sidecar = {
            **base_metadata,
            "completed_steps": step,
            "num_steps": config.num_steps,
            "config_hash": config.weight_fingerprint(),
            "fingerprint": content_fingerprint(config, train_seed, step),
        }
        (checkpoint_dir / "state.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
        write_metrics(metrics_path, metrics)

    if completed >= config.num_steps:
        return TrainResult(ema.model, metrics, {**base_metadata, "completed_steps": completed})

    stepper = torch.compile(model, dynamic=False) if config.compile_model else model
    micro = config.microbatch_size or config.batch_size
    num_micro = config.batch_size // micro
    last_log = (time.perf_counter(), completed)
    for step in range(completed, config.num_steps):
        batch = tuple(part.to(device) for part in loader.batch())
        gen = generator(stable_seed(train_seed, "step", step))
        t = sample_times(
            process, gen, config.batch_size, config.time_logsnr_mean, config.time_logsnr_std
        ).to(device)
        eps = torch.randn(
            (config.batch_size, *config.data_shape), generator=gen
        ).to(device)
        dropout_seed = draw_seed(gen)

        optimizer.zero_grad(set_to_none=True)
        total_loss, aux = 0.0, {}
        for index in range(num_micro):
            start, end = index * micro, (index + 1) * micro
            dropout_gen = (
                generator(stable_seed(dropout_seed, index), device)
                if config.dropout_p > 0
                else None
            )
            loss, extra = task.loss(
                config,
                stepper,
                process,
                tuple(part[start:end] for part in batch),
                t[start:end],
                eps[start:end],
                dropout_gen,
            )
            (loss / num_micro).backward()
            total_loss += float(loss.detach()) / num_micro
            aux = {name: aux.get(name, 0.0) + value / num_micro for name, value in extra.items()}

        done = step + 1
        is_log_step = log_interval > 0 and (
            done % log_interval == 0 or done == config.num_steps
        )
        before = [param.detach().clone() for param in model.parameters()] if is_log_step else []
        # The headline LR: the Muon group when there is one, else AdamW's.
        learning_rate = max(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if is_log_step:
            now = time.perf_counter()
            steps_per_sec = (done - last_log[1]) / max(now - last_log[0], 1e-9)
            last_log = (now, done)
            record = {
                "step": done,
                "loss": total_loss,
                "batch_size": config.batch_size,
                "grad_norm": tensor_norm(
                    param.grad for param in model.parameters() if param.grad is not None
                ),
                "update_norm": tensor_norm(
                    new - old
                    for new, old in zip(model.parameters(), before, strict=True)
                ),
                "param_norm": tensor_norm(model.parameters()),
                "lr": learning_rate,
                "steps_per_sec": steps_per_sec,
                "samples_per_sec": steps_per_sec * config.batch_size,
                "sampler_position": loader.sampler.position,
                **aux,
            }
            metrics.append(record)
            if logger is not None:
                logger(record)
        if checkpoint_interval > 0 and done % checkpoint_interval == 0 and done < config.num_steps:
            persist(done)

    persist(config.num_steps)
    return TrainResult(
        ema.model, metrics, {**base_metadata, "completed_steps": config.num_steps}
    )
