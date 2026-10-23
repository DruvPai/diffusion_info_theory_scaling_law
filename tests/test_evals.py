from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from itg.evals.immse import GaussianReference, immse_log_density, quadrature
from itg.evals.information import (
    cumulative_trapezoid,
    information_rate,
    make_log_snr_eval_grid,
    model_score,
)
from itg.evals.memorization import memorization_ratio
from itg.processes import Prediction, Process
from itg.seeds import CaseSeeds, stable_seed

PROCESS = Process(0.002, 80.0, 7.0)


class GaussianScore(nn.Module):
    """Exact score of the time-t marginal of N(0, variance I) under `process`."""

    def __init__(self, process: Process, variance: float) -> None:
        super().__init__()
        self.process = process
        self.variance = variance

    def forward(
        self, x_t: Tensor, t: Tensor, gen: torch.Generator | None = None
    ) -> Prediction:
        marginal = self.process.alpha(t) ** 2 * self.variance + self.process.sigma(t) ** 2
        return Prediction(-x_t / marginal.reshape(-1, *([1] * (x_t.ndim - 1))), "score")


def test_memorization_ratio_over_chunked_train_batches() -> None:
    train = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
    ratio = memorization_ratio(torch.tensor([[0.0], [10.0]]), (train[:2], train[2:]))
    assert ratio == 1.0


def test_cumulative_trapezoid_matches_a_known_integral() -> None:
    x = np.linspace(0.0, 1.0, 101)
    integral = cumulative_trapezoid(x, x)
    assert integral[0] == 0.0
    np.testing.assert_allclose(integral, x**2 / 2, atol=1e-4)


def test_information_rate_uses_g_squared_over_two() -> None:
    mean, se, per_probe = information_rate(
        torch.tensor([[1.0, 2.0], [4.0, 7.0]]),
        torch.tensor([[0.0, 2.0], [1.0, 3.0]]),
        2.0,
    )
    np.testing.assert_allclose(per_probe.numpy(), [1.0, 25.0], rtol=1e-6)
    assert mean == pytest.approx(13.0)
    assert se > 0.0


def test_information_grid_is_finite() -> None:
    grid = make_log_snr_eval_grid(PROCESS, num_times=5, t_min=0.05, t_max=0.95)
    assert grid.time.shape == (5,)
    assert bool((grid.g_squared >= 0).all())
    for field in (grid.g_squared, grid.log_snr, grid.positive_log_snr_weight):
        assert bool(torch.isfinite(field).all())


def test_probes_are_common_random_numbers_across_methods() -> None:
    grids = [
        make_log_snr_eval_grid(PROCESS, num_times=6, t_min=0.05, t_max=0.95)
        for _ in range(2)
    ]
    for left, right in zip(grids[0].__dict__.values(), grids[1].__dict__.values(), strict=True):
        assert torch.equal(left, right)

    seed = CaseSeeds(0, "case").case_eval_seed
    per_time = [stable_seed(seed, "time", index) for index in range(4)]
    assert per_time == [stable_seed(seed, "time", index) for index in range(4)]
    assert len(set(per_time)) == 4


@pytest.mark.parametrize(("reference_variance", "tolerance"), [(1.0, 1e-6), (2.0, 0.03)])
def test_immse_recovers_a_gaussian_log_density(
    reference_variance: float, tolerance: float
) -> None:
    process = PROCESS
    # sigma(0.4) is about one, so the marginal is about twice the data variance.
    dim, variance, t = 2, 1.0, 0.4
    model = GaussianScore(process, variance)
    time = torch.tensor([t])
    marginal = float(process.alpha(time) ** 2 * variance + process.sigma(time) ** 2)

    probes = torch.randn(8, dim, generator=torch.Generator().manual_seed(0)) * math.sqrt(
        marginal
    )
    truth = -0.5 * (
        probes.pow(2).sum(-1) / marginal + dim * math.log(2 * math.pi * marginal)
    )

    gammas, weights = quadrature(128, 1e-4, 1e4)
    estimate = immse_log_density(
        process,
        lambda model, y, s: model_score(model, process, y, s, 256),
        model,
        GaussianReference(
            mean=torch.zeros(dim), variances=torch.full((dim,), reference_variance)
        ),
        probes,
        t,
        gammas=gammas,
        weights=weights,
        seed=3,
        mc_samples=128,
        data_shape=(dim,),
    )
    assert float(((estimate - truth).abs() / truth.abs()).max()) < tolerance
