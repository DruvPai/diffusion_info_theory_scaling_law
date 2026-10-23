"""The EDM diffusion process: x_t = x_0 + sigma(t) eps on t in [0, 1].

Prediction kinds are `x_0`, `eps`, and `score` (= -eps / sigma).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from itg.config import Config


class Prediction(NamedTuple):
    value: Tensor
    kind: str


def _bcast(v: Tensor, ndim: int) -> Tensor:
    return v.reshape(v.shape[0], *([1] * (ndim - 1)))


class Process:
    """alpha = 1, sigma = ((1-t) sigma_min^(1/rho) + t sigma_max^(1/rho))^rho."""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float) -> None:
        self.rho = rho
        self.min_root = sigma_min ** (1.0 / rho)
        self.max_root = sigma_max ** (1.0 / rho)

    def _u(self, t: Tensor) -> Tensor:
        return (1.0 - t) * self.min_root + t * self.max_root

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return self._u(t) ** self.rho

    def logsnr(self, t: Tensor) -> Tensor:
        return -2.0 * self.rho * torch.log(self._u(t))

    def dlogsnr_dt(self, t: Tensor) -> Tensor:
        return -2.0 * self.rho * (self.max_root - self.min_root) / self._u(t)

    def logsnr_inverse(self, lam: Tensor) -> Tensor:
        return (torch.exp(-lam / (2.0 * self.rho)) - self.min_root) / (
            self.max_root - self.min_root
        )

    def forward(self, x_0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        return x_0 + _bcast(self.sigma(t), x_0.ndim) * eps

    def g_squared(self, t: Tensor) -> Tensor:
        """Reverse-SDE diffusion coefficient, -sigma^2 dlogSNR/dt."""
        return torch.clamp(-self.sigma(t) ** 2 * self.dlogsnr_dt(t), min=0.0)

    def terminal_std(self) -> float:
        return float(self.sigma(torch.ones(1)).item())

    def convert_prediction(
        self, pred: Prediction, x_t: Tensor, t: Tensor, to_kind: str
    ) -> Prediction:
        if pred.kind == to_kind:
            return pred
        sigma = _bcast(self.sigma(t), x_t.ndim)
        if pred.kind == "x_0":
            eps = (x_t - pred.value) / sigma
        elif pred.kind == "score":
            eps = -sigma * pred.value
        else:
            eps = pred.value
        if to_kind == "x_0":
            return Prediction(x_t - sigma * eps, to_kind)
        if to_kind == "eps":
            return Prediction(eps, to_kind)
        if to_kind == "score":
            return Prediction(-eps / sigma, to_kind)
        raise ValueError(f"unknown prediction kind {to_kind!r}")


def make_process(config: Config) -> Process:
    return Process(config.edm.sigma_min, config.edm.sigma_max, config.edm.rho)
