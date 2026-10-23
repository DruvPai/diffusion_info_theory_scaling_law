"""One case: its dataset manifest, its source models, and their samples."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from itg.config import Config, EvalConfig
from itg.data import Dataset, Loader, make_dataset, make_loader
from itg.models import make_module
from itg.processes import Process, make_process
from itg.runner.artifacts import (
    JsonDict,
    ModelRef,
    TableRow,
    draw_indices,
    ensure_artifact,
    model_dir,
    read_json,
    samples_path,
    write_array,
    write_json,
)
from itg.runner.events import log_event
from itg.sampling import sample
from itg.seeds import CaseSeeds, stable_seed
from itg.train import train_model


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


SAMPLE_ROLES = ("eval", "mixture_train")


@dataclass
class Case:
    """Everything one case is, with no mutable stage state."""

    name: str
    directory: Path
    config: Config
    evals: EvalConfig
    fields: TableRow = field(default_factory=dict)
    device: torch.device = field(default_factory=default_device)
    seed_case_name: str | None = None

    @property
    def seeds(self) -> CaseSeeds:
        return CaseSeeds(self.evals.seed, self.seed_case_name or self.name)

    @property
    def process(self) -> Process:
        return make_process(self.config)

    def forced(self, stage: str) -> bool:
        return stage in self.evals.force

    def dataset(self) -> Dataset:
        return make_dataset(self.config)


# --- dataset manifest ----------------------------------------------------


def prepare_dataset(case: Case) -> JsonDict:
    """Per-model training id lists, drawn without replacement from the image pool."""
    config, evals, seeds = case.config, case.evals, case.seeds
    manifest: JsonDict = {
        "dataset": "image_folder",
        "num_train_samples": config.dataset_size,
        "models": {},
    }
    if config.dataset_size is not None:
        pool = case.dataset().size
        manifest["index_scheme"] = "pool"
        manifest["pool_size"] = pool
        for idx in evals.source_model_indices:
            manifest["models"][f"{idx:03d}"] = {
                "indices": draw_indices(
                    seeds.model_dataset_seed(idx), pool_size=pool, count=config.dataset_size
                ).tolist()
            }
    write_json(case.directory / "dataset" / "manifest.json", manifest)
    return manifest


def training_ids(case: Case, model_idx: int) -> np.ndarray | None:
    if case.config.dataset_size is None:
        return None
    manifest = read_json(case.directory / "dataset" / "manifest.json")
    return np.asarray(manifest["models"][f"{model_idx:03d}"]["indices"], dtype=np.int64)


def training_loader(case: Case, dataset: Dataset, model_idx: int) -> Loader:
    return make_loader(
        case.config,
        dataset,
        training_ids(case, model_idx),
        dataset_seed=case.seeds.model_dataset_seed(model_idx),
    )


# --- source models -------------------------------------------------------


def train_source_models(
    case: Case, only: tuple[int, ...] | None = None
) -> tuple[dict[int, ModelRef], list[TableRow]]:
    """Train the source models in `only`, or all of them, with their metric rows."""
    dataset = case.dataset()
    process = case.process
    indices = only or case.evals.source_model_indices
    refs: dict[int, ModelRef] = {}
    metrics: list[TableRow] = []
    for idx in indices:
        log_event("train_model:start", model_index=idx)
        directory = model_dir(case.directory, idx)
        result = train_model(
            case.config,
            process,
            training_loader(case, dataset, idx),
            train_seed=case.seeds.model_train_seed(idx),
            checkpoint_dir=directory,
            device=case.device,
            log_interval=case.evals.log_interval,
            checkpoint_interval=case.evals.checkpoint_interval,
            force_retrain=case.forced("train"),
            metadata={"model_index": idx},
            logger=lambda record, idx=idx: log_event(
                "train_model:metric", model_index=idx, **record
            ),
        )
        metrics.extend({"case": case.name, "model_index": idx, **row} for row in result.metrics)
        refs[idx] = ModelRef.load(directory)
        log_event("train_model:finish", **result.metadata)
    case.fields["parameter_count"] = int(
        read_json(model_dir(case.directory, indices[0]) / "state.json")[
            "parameter_count"
        ]
    )
    return refs, metrics


def load_model(
    case: Case, ref: ModelRef, config: Config | None = None, num_classes: int | None = None
) -> nn.Module:
    """The EMA (eval) weights of a checkpoint, on the case device."""
    model = make_module(config or case.config, case.process, num_classes)
    state = torch.load(ref.directory / "state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["ema_model"])
    return model.to(case.device).eval()


# --- samples -------------------------------------------------------------


def write_samples(
    case: Case,
    path: Path,
    load: Callable[[], nn.Module],
    *,
    seed: int,
    count: int,
    identity: JsonDict,
) -> None:
    """Draw `count` samples into `path`, cached on `identity` plus the sampling
    knobs. `load` is deferred so a cache hit never reads a checkpoint."""
    evals = case.evals
    spec = {
        **identity,
        "samples": count,
        "seed": seed,
        "steps": evals.sample_steps,
        "order": evals.sample_order,
    }

    def produce() -> None:
        log_event("generate_samples:draw", path=str(path))
        write_array(
            path,
            sample(
                case.process,
                load(),
                num_samples=count,
                data_shape=case.config.data_shape,
                seed=seed,
                steps=evals.sample_steps,
                order=evals.sample_order,
                batch_size=evals.eval_batch_size,
                device=case.device,
            ),
        )

    if not ensure_artifact(path, spec, produce, force=case.forced("samples")):
        log_event("generate_samples:cache_hit", path=str(path))


def generate_samples(case: Case, role_name: str, refs: dict[int, ModelRef]) -> None:
    evals = case.evals
    # The shared seed is the load-bearing difference between the roles: probes
    # use one seed across models so per-model metrics are comparable under
    # common random numbers, while a pool concatenated across models must be
    # independent.
    probes = role_name == "eval"
    models = evals.source_model_indices if probes else tuple(evals.mixture_train_models)
    count = evals.eval_samples_per_model if probes else evals.mixture_samples_per_model
    base_seed = case.seeds.sample_seed(role_name)
    log_event("generate_samples:start", role=role_name, samples_per_model=count)
    for idx in models:
        write_samples(
            case,
            samples_path(case.directory, idx, role_name),
            lambda ref=refs[idx]: load_model(case, ref),
            seed=base_seed if probes else stable_seed(base_seed, idx),
            count=count,
            identity={"role": role_name, "model_index": idx, "upstream": refs[idx].spec()},
        )
    log_event("generate_samples:finish", role=role_name)
