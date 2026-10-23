from __future__ import annotations

import pytest
import torch

from itg.processes import Prediction, Process

PROCESS = Process(0.002, 80.0, 7.0)
TIMES = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)


def test_dlogsnr_dt_matches_finite_difference() -> None:
    h = 1e-6
    central = (PROCESS.logsnr(TIMES + h) - PROCESS.logsnr(TIMES - h)) / (2 * h)
    torch.testing.assert_close(PROCESS.dlogsnr_dt(TIMES), central, rtol=1e-5, atol=1e-5)


def test_logsnr_inverse_round_trips() -> None:
    torch.testing.assert_close(PROCESS.logsnr_inverse(PROCESS.logsnr(TIMES)), TIMES)


def test_g_squared_is_non_negative() -> None:
    assert bool((PROCESS.g_squared(TIMES) >= 0).all())


@pytest.mark.parametrize("kind", ["eps", "score"])
def test_prediction_conversion_round_trips(kind: str) -> None:
    gen = torch.Generator().manual_seed(0)
    x_0 = torch.randn(4, 3, generator=gen)
    eps = torch.randn(4, 3, generator=gen)
    t = torch.full((4,), 0.5)
    x_t = PROCESS.forward(x_0, eps, t)
    other = PROCESS.convert_prediction(Prediction(x_0, "x_0"), x_t, t, kind)
    back = PROCESS.convert_prediction(other, x_t, t, "x_0")
    torch.testing.assert_close(back.value, x_0, rtol=1e-4, atol=1e-4)
