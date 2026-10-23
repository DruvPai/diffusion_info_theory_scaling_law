"""Model-free log-density estimation via the I-MMSE identity.

For a model with score `s(y, s)` and time-t marginal `p_t` (Guo-Shamai-Verdu;
Kong et al. 2023),

    log p_t(x) = log p_{G,t}(x)
                 + (1/2) integral_0^inf (m_G(x; gamma) - m(x; gamma)) dgamma,

where gamma is the SNR of the Gaussian channel X_t -> X_s, m is the
reconstruction error of x from the further-noised y via Tweedie, and m_G is
the same for a moment-matched Gaussian reference. The reference is a control
variate: no bias, only variance.

Everything here is in flattened data vectors; probes are reshaped to the
model's native shape only for score evaluations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from itg.processes import Process
from itg.seeds import generator, stable_seed

ScoreFn = Callable[[nn.Module, Tensor, Tensor], Tensor]

MOMENT_CHUNK_ROWS = 256


@dataclass(frozen=True)
class GaussianReference:
    """Diagonal moment-matched Gaussian for a model's clean-data law."""

    mean: Tensor
    variances: Tensor


def fit_gaussian_reference(samples: Tensor) -> GaussianReference:
    """Per-dimension moments, accumulated in chunks over a large sample pool."""
    count = samples.shape[0]
    dim = int(np.prod(samples.shape[1:]))
    total = torch.zeros(dim, dtype=torch.float64)
    total_sq = torch.zeros(dim, dtype=torch.float64)
    for start in range(0, count, MOMENT_CHUNK_ROWS):
        chunk = samples[start : start + MOMENT_CHUNK_ROWS].reshape(-1, dim).double()
        total += chunk.sum(dim=0)
        total_sq += chunk.pow(2).sum(dim=0)
    mean = total / count
    variances = (total_sq - count * mean.pow(2)).clamp(min=0.0) / max(count - 1, 1)
    return GaussianReference(
        mean=mean.float(), variances=variances.clamp(min=1e-12).float()
    )


def quadrature(
    num_nodes: int, gamma_min: float, gamma_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced channel-SNR nodes and trapezoid weights for (1/2) f dgamma."""
    if num_nodes < 2 or not 0.0 < gamma_min < gamma_max:
        raise ValueError("expected num_nodes >= 2 and 0 < gamma_min < gamma_max")
    gammas = np.exp(np.linspace(np.log(gamma_min), np.log(gamma_max), num_nodes))
    weights = np.zeros(num_nodes)
    deltas = np.diff(gammas)
    weights[:-1] += deltas / 2.0
    weights[1:] += deltas / 2.0
    return gammas, weights / 2.0


def channel_params(
    process: Process, t: float, gamma: float
) -> tuple[Tensor, float, float]:
    """(s, a, sigma_ts) of the channel X_t -> X_s with SNR gamma."""
    time = torch.tensor([t])
    alpha_t = process.alpha(time)
    inv_snr_s = torch.exp(-process.logsnr(time)) + 1.0 / (gamma * alpha_t**2)
    s = process.logsnr_inverse(-torch.log(inv_snr_s))
    a = float(process.alpha(s) / alpha_t)
    variance = float(process.sigma(s) ** 2 - a**2 * process.sigma(time) ** 2)
    return s, a, float(np.sqrt(max(variance, 1e-20)))


def immse_log_density(
    process: Process,
    score_fn: ScoreFn,
    model: nn.Module,
    reference: GaussianReference,
    x_flat: Tensor,
    t: float,
    *,
    gammas: np.ndarray,
    weights: np.ndarray,
    seed: int,
    mc_samples: int,
    data_shape: tuple[int, ...],
) -> Tensor:
    """log p_t^model(x) per probe.

    Channel noise is common across the models a caller compares (shared
    `seed`), so the per-model log-densities entering the softmax are paired.
    """
    device = x_flat.device
    num_probes, dim = x_flat.shape
    time = torch.tensor([t])
    alpha_t = float(process.alpha(time))
    sigma_t = float(process.sigma(time))
    mean_t = alpha_t * reference.mean.to(device)
    var_t = alpha_t**2 * reference.variances.to(device) + sigma_t**2

    delta = x_flat - mean_t
    log_density = -0.5 * (
        delta.pow(2) / var_t + torch.log(2.0 * torch.pi * var_t)
    ).sum(dim=-1)

    for index, (gamma, weight) in enumerate(zip(gammas, weights, strict=True)):
        s, a, sigma_ts = channel_params(process, t, float(gamma))
        eps = torch.randn(
            (mc_samples, num_probes, dim),
            generator=generator(stable_seed(seed, "node", index)),
        ).to(device)
        y_flat = a * x_flat[None] + sigma_ts * eps
        scores = score_fn(
            model,
            y_flat.reshape(mc_samples * num_probes, *data_shape),
            s.to(device).expand(mc_samples * num_probes),
        ).reshape(mc_samples, num_probes, dim)
        # Tweedie for the channel X_t -> X_s, and the reference's exact
        # posterior mean on the same noise draws (a paired difference).
        x_hat = (y_flat + sigma_ts**2 * scores) / a
        gain = a * var_t / (a**2 * var_t + sigma_ts**2)
        x_hat_reference = mean_t + gain * (y_flat - a * mean_t)
        gap = (x_flat - x_hat_reference).pow(2).sum(dim=-1) - (x_flat - x_hat).pow(
            2
        ).sum(dim=-1)
        log_density = log_density + float(weight) * gap.mean(dim=0)
    return log_density
