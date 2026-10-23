from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import itg.data as data_module
import itg.runner.pipeline as pipeline_module
from itg.config import EvalConfig, InformationMethod
from itg.runner.artifacts import (
    draw_indices,
    ensure_artifact,
    spec_path,
    stale_against,
)
from itg.runner.case import Case, prepare_dataset
from itg.runner.pipeline import run_case
from tests.tiny import image_folder, tiny_config

EVALS = EvalConfig(
    mixture_train_models=(0, 1),
    mixture_eval_models=(2,),
    mixture_sizes=(1, 2),
    eval_samples_per_model=8,
    mixture_samples_per_model=16,
    sample_steps=3,
    eval_batch_size=4,
    eval_num_times=3,
    immse_num_quadrature_nodes=3,
    immse_mc_samples=2,
    immse_num_probes=4,
    mixture_steps=4,
    mixture_num_layers=1,
    mixture_model_aspect_ratio=8.0,
    log_interval=3,
    checkpoint_interval=100,
)


@pytest.fixture(autouse=True)
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A private decode cache, and no FID: it needs Inception weights."""
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pipeline_module, "write_fid", lambda *args, **kwargs: None)


def tiny_case(directory: Path, **eval_overrides: object) -> Case:
    root = directory / "images"
    if not root.exists():
        image_folder(root, 64)
    return Case(
        name="tiny",
        directory=directory,
        config=tiny_config(root),
        evals=replace(EVALS, **eval_overrides),  # type: ignore[arg-type]
        fields={"n_samples": "16", "num_layers": 2},
        device=torch.device("cpu"),
    )


def events(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def sample_mtimes(directory: Path) -> dict[Path, int]:
    return {path: path.stat().st_mtime_ns for path in sorted(directory.rglob("*.npy"))}


def test_ensure_artifact_caches_on_the_spec(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    calls = [0]

    def produce() -> None:
        calls[0] += 1
        path.write_text(str(calls[0]))

    assert ensure_artifact(path, {"v": 1}, produce) is True
    assert ensure_artifact(path, {"v": 1}, produce) is False
    assert ensure_artifact(path, {"v": 2}, produce) is True
    assert ensure_artifact(path, {"v": 2}, produce, force=True) is True
    spec_path(path).unlink()
    assert ensure_artifact(path, {"v": 2}, produce) is True
    assert calls[0] == 4


def test_stale_against_treats_a_missing_sidecar_as_resumable(tmp_path: Path) -> None:
    path = tmp_path / "state.pt"
    assert stale_against(path, {"v": 1}) is False
    ensure_artifact(path, {"v": 1}, path.touch)
    assert stale_against(path, {"v": 1}) is False
    assert stale_against(path, {"v": 2}) is True


def test_prepare_dataset_writes_pool_indices(tmp_path: Path) -> None:
    case = tiny_case(tmp_path / "case")
    manifest = prepare_dataset(case)
    assert (manifest["index_scheme"], manifest["pool_size"]) == ("pool", 64)
    for idx in (0, 1, 2):
        indices = manifest["models"][f"{idx:03d}"]["indices"]
        expected = draw_indices(
            case.seeds.model_dataset_seed(idx), pool_size=64, count=16
        )
        assert indices == expected.tolist()
        assert len(set(indices)) == 16
    assert json.loads((tmp_path / "case" / "dataset" / "manifest.json").read_text()) == manifest


def test_run_case_end_to_end_then_all_cache_hits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="itg")
    summary = run_case(tiny_case(tmp_path))
    assert summary["case"] == "tiny"
    assert "information_evals:start" in events(caplog)

    expected = [
        "dataset/manifest.json",
        "config.json",
        "training_metrics.csv",
        "model_information_summaries.json",
        "models/model_002/state.pt",
        "samples/model_000/eval/samples.npy",
        "samples/model_000/eval/memorization.csv",
        "samples/model_001/mixture_train/samples.npy",
        "mixture_models/k_002/state.pt",
        "classifier_models/k_002/state.pt",
        "mixture_samples/k_002/samples.npy",
    ]
    for method in EVALS.information_methods:
        for k in EVALS.sizes:
            expected.append(
                f"model_information_curve_{method}_from_{k:03d}_source_models.csv"
            )
    for name in expected:
        assert (tmp_path / name).exists(), name

    caplog.clear()
    run_case(tiny_case(tmp_path))
    replayed = events(caplog)
    assert "information_evals:cache_hit" in replayed
    assert "information_evals:start" not in replayed
    assert "generate_samples:cache_hit" in replayed


def test_extra_information_method_recomputes_only_the_information_stage(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    methods: tuple[InformationMethod, ...] = ("distilled_mixture",)
    run_case(tiny_case(tmp_path, information_methods=methods))
    before = sample_mtimes(tmp_path)

    caplog.set_level(logging.INFO, logger="itg")
    run_case(
        tiny_case(
            tmp_path,
            information_methods=(*methods, "classifier_model_free"),
        )
    )
    assert "information_evals:start" in events(caplog)
    assert sample_mtimes(tmp_path) == before


def test_unknown_stage_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        run_case(tiny_case(tmp_path), ("train", "nonsense"))


def test_case_failure_is_logged_with_a_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="itg")
    case = replace(tiny_case(tmp_path), config=tiny_config(tmp_path / "missing"))
    with pytest.raises(ValueError, match="no image files"):
        run_case(case)

    logged = [
        json.loads(line) for line in (tmp_path / "experiment.log").read_text().splitlines()
    ]
    failures = [entry for entry in logged if entry["event"] == "case:error"]
    assert len(failures) == 1
    assert "Traceback" in failures[0]["traceback"]
    assert failures[0]["exception"] == "ValueError"
