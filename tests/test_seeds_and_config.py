from __future__ import annotations

from dataclasses import replace

import pytest

from itg.config import EvalConfig
from itg.seeds import CaseSeeds, stable_seed
from tests.tiny import tiny_config


def test_stable_seed_is_pure_and_order_sensitive() -> None:
    assert stable_seed(0, "a", "b") == stable_seed(0, "a", "b")
    assert stable_seed(0, "a", "b") != stable_seed(0, "b", "a")
    assert stable_seed(0, "a") != stable_seed(1, "a")
    assert 0 <= stable_seed(0, "a") < 2**31


def test_stable_seed_regression_anchors() -> None:
    assert stable_seed(0, "a") == 845039730
    assert stable_seed(123, "case", "dataset_source") == 1073640698


def test_case_seeds_are_pure_and_distinct() -> None:
    seeds = CaseSeeds(0, "case")
    assert seeds.case_eval_seed == CaseSeeds(0, "case").case_eval_seed
    assert seeds.model_train_seed(0) != seeds.model_train_seed(1)
    assert seeds.model_train_seed(0) != seeds.model_dataset_seed(0)
    assert seeds.sample_seed("eval") != seeds.sample_seed("mixture_train")
    assert seeds.aux_network_train_seed("mixture", 2) != seeds.aux_network_train_seed(
        "classifier", 2
    )
    assert seeds.aux_network_train_seed("mixture", 2) != seeds.aux_network_train_seed(
        "mixture", 3
    )
    assert seeds.case_eval_seed != seeds.immse_seed
    other = CaseSeeds(1, "case")
    assert seeds.case_eval_seed != other.case_eval_seed
    assert seeds.model_train_seed(0) != other.model_train_seed(0)
    assert CaseSeeds(0, "other").case_eval_seed != seeds.case_eval_seed


def test_weight_fingerprint_ignores_the_image_root_only() -> None:
    base = tiny_config()
    assert replace(base, image_root="/a").weight_fingerprint() == base.weight_fingerprint()
    for field, value in (
        ("hidden_dim", 256),
        ("num_steps", 11),
        ("seed", 1),
        ("dropout_p", 0.1),
    ):
        assert replace(base, **{field: value}).weight_fingerprint() != base.weight_fingerprint()


def test_data_shape_is_rgb_chw() -> None:
    assert tiny_config().data_shape == (3, 8, 8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dropout_p": 1.0},
        {"batch_size": 6, "microbatch_size": 4},
        {"regime": "iid", "dataset_size": None},
    ],
)
def test_config_rejects_bad_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        tiny_config(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mixture_train_models": (0, 1), "mixture_eval_models": (1, 2)},
        {"mixture_train_models": (0, 1), "mixture_eval_models": (2,), "mixture_sizes": (3,)},
        {"mixture_train_models": (0, 1), "mixture_eval_models": (2,), "mixture_sizes": (0,)},
    ],
)
def test_eval_config_rejects_bad_model_sets(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EvalConfig(**overrides)  # ty: ignore[invalid-argument-type]
