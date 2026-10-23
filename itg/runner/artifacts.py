"""Artifact paths, atomic IO, and the spec-sidecar cache.

Staleness has exactly one mechanism: a `<file>.spec.json` sidecar holding the
desired spec, including fingerprints of the artifact's inputs. Regeneration
happens iff the stored spec differs. No mtime or content sniffing.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

JsonDict = dict[str, Any]
TableRow = dict[str, Any]


def staging(path: Path) -> Path:
    """A writer-private staging name.

    One path can have concurrent writers -- the per-model jobs of a case all
    write the case config and dataset manifest -- and a shared staging name
    makes the second rename fail, because the first already moved the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def write_json(path: Path, payload: JsonDict) -> None:
    tmp = staging(path)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[TableRow]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    tmp = staging(path)
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_array(path: Path, array: Tensor) -> None:
    tmp = staging(path)
    with tmp.open("wb") as handle:  # a path handle: np.save would append .npy
        np.save(handle, array.detach().cpu().float().numpy())
    tmp.replace(path)


def read_array(path: Path) -> Tensor:
    return torch.from_numpy(np.load(path))


def normalize_spec(spec: JsonDict) -> JsonDict:
    return json.loads(json.dumps(spec, sort_keys=True, default=str))


def spec_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.spec.json")


def stored_spec(path: Path) -> JsonDict | None:
    """An artifact's sidecar, or None if it has none."""
    sidecar = spec_path(path)
    return read_json(sidecar) if sidecar.exists() else None


def ensure_artifact(
    path: Path, spec: JsonDict, produce: Callable[[], None], *, force: bool = False
) -> bool:
    """Produce `path` unless its stored spec already equals `spec`."""
    if not force and path.exists() and stored_spec(path) == normalize_spec(spec):
        return False
    produce()
    write_json(spec_path(path), normalize_spec(spec))
    return True


def stale_against(path: Path, spec: JsonDict) -> bool:
    """Whether a stored sidecar disagrees with `spec`.

    A missing sidecar is not stale: it means an interrupted run that should
    resume, not restart. Callers write the sidecar before training.
    """
    stored = stored_spec(path)
    return stored is not None and stored != normalize_spec(spec)


def draw_indices(seed: int, *, pool_size: int, count: int) -> np.ndarray:
    if count > pool_size:
        raise ValueError(f"cannot draw {count} unique indices from {pool_size}")
    return np.random.default_rng(seed).choice(pool_size, count, replace=False)


# --- case layout ---------------------------------------------------------


def model_dir(case_dir: Path, model_idx: int) -> Path:
    return case_dir / "models" / f"model_{model_idx:03d}"


def aux_dir(case_dir: Path, kind: str, k: int) -> Path:
    return case_dir / f"{kind}_models" / f"k_{k:03d}"


def samples_path(case_dir: Path, model_idx: int, role: str) -> Path:
    return case_dir / "samples" / f"model_{model_idx:03d}" / role / "samples.npy"


def aux_samples_path(case_dir: Path, kind: str, k: int) -> Path:
    return case_dir / f"{kind}_samples" / f"k_{k:03d}" / "samples.npy"


def information_curve_path(case_dir: Path, method: str, k: int) -> Path:
    return case_dir / f"model_information_curve_{method}_from_{k:03d}_source_models.csv"


def information_summaries_path(case_dir: Path) -> Path:
    return case_dir / "model_information_summaries.json"


@dataclass(frozen=True)
class ModelRef:
    """A trained checkpoint: where it is and what identifies its weights."""

    directory: Path
    config_hash: str
    fingerprint: str

    @classmethod
    def load(cls, directory: Path) -> "ModelRef":
        state = read_json(directory / "state.json")
        return cls(directory, state["config_hash"], state["fingerprint"])

    def spec(self) -> JsonDict:
        return {"config_hash": self.config_hash, "fingerprint": self.fingerprint}
