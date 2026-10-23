"""The probability-flow ODE sampler.

Randomness is drawn at the full `num_samples` shape, so the generated set does
not depend on `batch_size` (which only chunks the forward pass).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from itg.processes import Prediction, Process
from itg.seeds import generator, stable_seed


@torch.no_grad()
def predict(model: nn.Module, x: Tensor, t: Tensor, batch_size: int) -> Prediction:
    """Model prediction over `x`, chunked into `batch_size` forward passes."""
    parts = [
        model(x[start : start + batch_size], t[start : start + batch_size])
        for start in range(0, x.shape[0], batch_size)
    ]
    return Prediction(torch.cat([part.value for part in parts]), parts[0].kind)


@torch.no_grad()
def sample(
    process: Process,
    model: nn.Module,
    *,
    num_samples: int,
    data_shape: tuple[int, ...],
    seed: int,
    steps: int = 200,
    order: int = 1,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Probability-flow ODE from t=1 over `steps` uniform times: DDIM at order 1,
    Karras Heun at order 2."""
    x = process.terminal_std() * torch.randn(
        (num_samples, *data_shape), generator=generator(stable_seed(seed, "x_T"))
    ).to(device)
    grid = torch.linspace(1.0, 0.0, steps, device=x.device)
    broadcast = (-1, *([1] * (x.ndim - 1)))
    for index in range(steps):
        t = grid[index].expand(num_samples)
        pred = predict(model, x, t, batch_size)
        if index == steps - 1:
            return process.convert_prediction(pred, x, t, "x_0").value
        s = grid[index + 1].expand(num_samples)
        sigma_s = process.sigma(s).reshape(broadcast)
        eps = process.convert_prediction(pred, x, t, "eps").value
        moved = process.convert_prediction(pred, x, t, "x_0").value + sigma_s * eps
        if order == 2:
            eps_next = process.convert_prediction(
                predict(model, moved, s, batch_size), moved, s, "eps"
            ).value
            moved = moved + (sigma_s - process.sigma(t).reshape(broadcast)) * (eps_next - eps) / 2
        x = moved
    raise AssertionError("unreachable")
