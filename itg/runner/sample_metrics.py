"""Cached per-sample-set metrics: FID, memorization, preview grid."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from itg.evals import write_sample_grid
from itg.evals.memorization import memorization_ratio
from itg.evals.quality import clean_features, clean_fid
from itg.runner.artifacts import (
    JsonDict,
    TableRow,
    ensure_artifact,
    read_array,
    read_json,
    write_array,
    write_csv,
    write_json,
)
from itg.runner.case import Case
from itg.runner.events import log_event
from itg.runner.references import ReferenceBatches


def _row(case: Case, sample_file: Path, tags: dict[str, int | None]) -> TableRow:
    return {
        "case": case.name,
        "sample_file": str(sample_file.relative_to(case.directory)),
        **{name: value for name, value in tags.items() if value is not None},
    }


def reference_features(case: Case, reference: ReferenceBatches, count: int) -> "np.ndarray":
    """Real-data features at training resolution, cached across the grid."""
    path = case.directory.parent / f"fid_reference_{case.config.image_size}_{count}.npy"
    if path.exists():
        return read_array(path).numpy()
    batch_size = case.evals.eval_batch_size
    features = np.concatenate([clean_features(batch) for batch in reference(batch_size)])
    write_array(path, torch.from_numpy(features))
    return features


def reference_floor(case: Case, features: "np.ndarray", count: int) -> float:
    """FID a perfect model would score: disjoint real draws, same sample count.

    The estimator is biased at finite samples, and at 8192 that bias is a
    large fraction of a good model's score. It is identical for every case,
    so curves stay comparable, but the absolute number is meaningless
    without it.
    """
    path = case.directory.parent / f"fid_floor_{case.config.image_size}_{count}.json"
    if path.exists():
        return float(read_json(path)["fid_floor"])
    value = clean_fid(features[:count], features[count:])
    write_json(path, {"fid_floor": value, "num_samples": count})
    return value


def write_fid(
    case: Case,
    sample_file: Path,
    generated: Tensor,
    reference: ReferenceBatches | None,
    *,
    generated_by: str,
    sample_spec: JsonDict,
    tags: dict[str, int | None],
) -> None:
    """clean-FID against real data. Meaningful only well above the 2048 feature dim."""
    count = case.evals.fid_reference_samples
    path = sample_file.parent / "fid.csv"
    spec = {"kind": "fid", "reference_samples": count, "sample_spec": sample_spec}

    def produce() -> None:
        log_event("fid:start", sample_file=str(sample_file), num_samples=generated.shape[0])
        features = reference_features(case, reference, count) if reference is not None else None
        num_generated = int(generated.shape[0])
        value = clean_fid(clean_features(generated), features) if features is not None else None
        floor = (
            reference_floor(case, features, num_generated)
            if features is not None and len(features) > 2 * num_generated
            else None
        )
        write_csv(
            path,
            [
                {
                    **_row(case, sample_file, tags),
                    "generated_by": generated_by,
                    "fid": value,
                    "fid_floor": floor,
                    "fid_num_samples": num_generated,
                    "fid_reference_samples": count,
                }
            ],
        )

    if not ensure_artifact(path, spec, produce, force=case.forced("sample_metrics")):
        log_event("fid:cache_hit", sample_file=str(sample_file))


def write_memorization(
    case: Case,
    sample_file: Path,
    samples: Tensor,
    train_batches: ReferenceBatches | None,
    *,
    sample_spec: JsonDict,
    train_spec: JsonDict | None,
    tags: dict[str, int | None],
) -> None:
    path = sample_file.parent / "memorization.csv"
    batch_size = case.evals.eval_batch_size
    spec = {
        "kind": "memorization",
        "encoders": ["identity"],
        "sample_spec": sample_spec,
        "train_spec": train_spec,
    }

    def produce() -> None:
        log_event("memorization:start", sample_file=str(sample_file))
        ratio = (
            memorization_ratio(samples, train_batches(batch_size))
            if train_batches is not None
            else math.nan
        )
        write_csv(path, [{**_row(case, sample_file, tags), "encoder": "identity", "memorization_ratio": ratio}])

    if not ensure_artifact(path, spec, produce, force=case.forced("sample_metrics")):
        log_event("memorization:cache_hit", sample_file=str(sample_file))


def write_sample_figure(case: Case, sample_file: Path, samples: Tensor, sample_spec: JsonDict) -> None:
    path = sample_file.with_suffix(".png")

    def produce() -> None:
        write_sample_grid(samples, path)

    ensure_artifact(
        path,
        {"kind": "sample_grid", "sample_spec": sample_spec},
        produce,
        force=case.forced("sample_metrics"),
    )
