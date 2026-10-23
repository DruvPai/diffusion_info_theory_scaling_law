"""Per-k networks trained on the pooled samples of train source models 0..k-1.

The distillation model and the k-way posterior classifier are the same object
trained on the same pool with different objectives, so they share one config
derivation, one cache key, and one training entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from itg.config import Config, EvalConfig
from itg.data import Loader, TensorDataset, UniformSampler
from itg.runner.artifacts import (
    ModelRef,
    TableRow,
    aux_dir,
    normalize_spec,
    read_array,
    read_json,
    samples_path,
    spec_path,
    stale_against,
    write_json,
)
from itg.runner.case import Case
from itg.runner.events import log_event
from itg.seeds import stable_seed
from itg.train import DENOISING, classifier_task, train_model


@dataclass(frozen=True)
class AuxHyperparameters:
    """`classifier_*` falls back to `mixture_*`, then to the case config."""

    num_steps: int
    batch_size: int
    lr: float
    num_layers: int | None


def _first[T](*values: T | None, default: T) -> T:
    return next((value for value in values if value is not None), default)


def aux_hyperparameters(
    evals: EvalConfig, config: Config, kind: str
) -> AuxHyperparameters:
    own = kind == "classifier"
    return AuxHyperparameters(
        num_steps=_first(
            evals.classifier_steps if own else None, evals.mixture_steps, default=config.num_steps
        ),
        batch_size=_first(
            evals.classifier_batch_size if own else None,
            evals.mixture_batch_size,
            default=config.batch_size,
        ),
        lr=_first(
            evals.classifier_lr if own else None, evals.mixture_lr, default=config.lr
        ),
        num_layers=_first(
            evals.classifier_num_layers if own else None,
            evals.mixture_num_layers,
            default=None,
        ),
    )


def aux_config(
    config: Config, evals: EvalConfig, hyper: AuxHyperparameters, *, pool_size: int
) -> Config:
    """The config the auxiliary network actually trains under."""
    config = replace(
        config,
        regime="iid",
        dataset_size=pool_size,
        batch_size=hyper.batch_size,
        num_steps=hyper.num_steps,
        lr=hyper.lr,
        image_random_horizontal_flip=False,
    )
    if hyper.num_layers is None:
        return config
    ratio = evals.mixture_model_aspect_ratio or config.hidden_dim / max(config.num_layers, 1)
    hidden = max(1, round(hyper.num_layers * ratio))
    return replace(config, num_layers=hyper.num_layers, hidden_dim=hidden)


def train_aux_network(
    case: Case, kind: str, k: int, train_indices: list[int]
) -> tuple[ModelRef, list[TableRow]]:
    # The pooled `mixture_train` samples, labelled by which model produced each.
    paths = [samples_path(case.directory, idx, "mixture_train") for idx in train_indices[:k]]
    batches = [read_array(path) for path in paths]
    pool = torch.cat(batches)
    labels = torch.cat(
        [torch.full((b.shape[0],), label, dtype=torch.long) for label, b in enumerate(batches)]
    )
    sidecars = [read_json(spec_path(path)) for path in paths]
    hyper = aux_hyperparameters(case.evals, case.config, kind)
    config = aux_config(case.config, case.evals, hyper, pool_size=pool.shape[0])
    seed = case.seeds.aux_network_train_seed(kind, k)
    directory = aux_dir(case.directory, kind, k)
    state_path = directory / "state.pt"
    spec = normalize_spec(
        {
            "kind": kind,
            "model_indices": train_indices[:k],
            "samples_per_model": case.evals.mixture_samples_per_model,
            "config_hash": config.weight_fingerprint(),
            "upstream": sidecars,
        }
    )
    stale = stale_against(state_path, spec)
    if stale:
        log_event(f"train_{kind}:cache_stale", mixture_size=k)
    write_json(spec_path(state_path), spec)

    loader = Loader(
        TensorDataset(pool),
        np.arange(pool.shape[0]),
        UniformSampler(seed=stable_seed(seed, "sampler")),
        hyper.batch_size,
        labels=labels if kind == "classifier" else None,
    )
    result = train_model(
        config,
        case.process,
        loader,
        train_seed=seed,
        checkpoint_dir=directory,
        task=DENOISING if kind == "mixture" else classifier_task(k),
        device=case.device,
        log_interval=case.evals.log_interval,
        checkpoint_interval=case.evals.checkpoint_interval,
        force_retrain=stale or case.forced(f"train_{kind}"),
        metadata={
            "role": kind,
            "model_indices": train_indices[:k],
            "mixture_size": k,
            "samples_per_model": case.evals.mixture_samples_per_model,
            "training_samples": int(pool.shape[0]),
            "model_hidden_dim": config.hidden_dim,
            "model_num_layers": config.num_layers,
        },
        logger=lambda record: log_event(f"train_{kind}:metric", mixture_size=k, **record),
    )
    metrics = [{"case": case.name, "aux_kind": kind, "mixture_size": k, **row} for row in result.metrics]
    return ModelRef.load(directory), metrics


def train_aux_networks(
    case: Case, kind: str
) -> tuple[dict[int, ModelRef], list[TableRow]]:
    sizes = case.evals.sizes
    train_indices = list(case.evals.mixture_train_models[: max(sizes)])
    log_event(f"train_{kind}s:start", sizes=sizes)
    refs: dict[int, ModelRef] = {}
    metrics: list[TableRow] = []
    for k in sizes:
        ref, rows = train_aux_network(case, kind, k, train_indices)
        refs[k] = ref
        metrics.extend(rows)
        log_event(f"train_{kind}:finish", mixture_size=k)
    return refs, metrics


def aux_model_config(case: Case, kind: str, k: int) -> Config:
    """The config a stored auxiliary checkpoint was built with, for reloading."""
    metadata = read_json(aux_dir(case.directory, kind, k) / "state.json")
    return aux_config(
        case.config,
        case.evals,
        aux_hyperparameters(case.evals, case.config, kind),
        pool_size=int(metadata["training_samples"]),
    )
