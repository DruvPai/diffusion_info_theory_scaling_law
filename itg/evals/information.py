"""Information-rate evaluation: the log-SNR grid and the rate identity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from itg.processes import Process
from itg.sampling import predict


@torch.no_grad()
def model_score(
    model: nn.Module, process: Process, x_t: Tensor, t: Tensor, batch_size: int
) -> Tensor:
    return process.convert_prediction(
        predict(model, x_t, t, batch_size), x_t, t, "score"
    ).value


def cumulative_trapezoid(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral against monotone `x`, starting at 0."""
    if values.shape[0] == 0:
        return values
    areas = 0.5 * (values[:-1] + values[1:]) * np.abs(np.diff(x))
    return np.concatenate([[0.0], np.cumsum(areas)])


def mean_and_se(values: Tensor) -> tuple[float, float]:
    count = values.shape[0]
    mean = values.mean()
    variance = (values - mean).pow(2).sum() / max(count - 1, 1)
    return float(mean), float((variance / max(count, 1)).sqrt())


@dataclass(frozen=True)
class InformationEvalGrid:
    time: Tensor
    g_squared: Tensor
    log_snr: Tensor
    positive_log_snr_weight: Tensor


def make_log_snr_eval_grid(
    process: Process, *, num_times: int, t_min: float, t_max: float
) -> InformationEvalGrid:
    """Uniform-in-t grid with the analytic g^2 and log-SNR change of variables.

    g^2(t) = -sigma^2 dlogSNR/dt holds for any linear-SDE process, so every
    process is a first-class citizen of the estimators.
    """
    time = torch.linspace(t_min, t_max, num_times, dtype=torch.float64)
    derivative = process.dlogsnr_dt(time)
    return InformationEvalGrid(
        time=time,
        g_squared=process.g_squared(time),
        log_snr=process.logsnr(time),
        positive_log_snr_weight=1.0 / derivative.abs().clamp(min=1e-12),
    )


def information_rate(
    eval_scores: Tensor, mixture_scores: Tensor, g_squared: float
) -> tuple[float, float, Tensor]:
    """(g^2/2) E||s_eval - s_hat||^2, per probe and aggregated."""
    gap = (eval_scores.flatten(1) - mixture_scores.flatten(1)).pow(2).sum(dim=-1)
    per_probe = (g_squared / 2.0) * gap
    mean, se = mean_and_se(per_probe)
    return mean, se, per_probe
