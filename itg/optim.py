"""Muon + AdamW, the LR schedule, and the EMA eval model."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from itg.config import Config
from itg.models import adam_params, muon_params
from itg.models.common import mup_lr_scales, mup_wd_scales

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def adam_peak_lr(config: Config) -> float:
    return config.lr if config.optimizer_adam_lr is None else config.optimizer_adam_lr


def muon_peak_lr(config: Config) -> float:
    return config.lr * config.muon_lr_multiplier


def schedule_fraction(config: Config, step: int) -> float:
    """Peak-LR multiplier at `step`; 0 at step 0 (optax `count` semantics)."""
    num_steps = config.num_steps
    if num_steps == 1:
        return 1.0
    warmup = max(1, min(int(round(num_steps * config.lr_warmup_fraction)), num_steps - 1))
    final_step = num_steps - 1
    position = min(step, final_step)
    if position <= warmup:
        return position / warmup
    return max(0.0, 1.0 - (position - warmup) / max(1, final_step - warmup))


def orthogonalize(matrix: Tensor, ns_steps: int, eps: float = 1e-8) -> Tensor:
    """Newton-Schulz orthogonalization, Frobenius-preconditioned.

    Shape (m, n), or (batch, m, n) to run a stack of equally shaped matrices as
    one set of batched matmuls.
    """
    transposed = matrix.shape[-2] > matrix.shape[-1]
    x = matrix.mT if transposed else matrix
    x = x / (torch.linalg.matrix_norm(x, dim=(-2, -1), keepdim=True) + eps)
    a_coeff, b_coeff, c_coeff = NS_COEFFS
    for _ in range(ns_steps):
        gram = x @ x.mT
        x = a_coeff * x + (b_coeff * gram + c_coeff * gram @ gram) @ x
    return x.mT if transposed else x


def orthogonalize_all(matrices: list[Tensor], ns_steps: int) -> list[Tensor]:
    """Orthogonalize a heterogeneous list, batching over equal shapes.

    Muon's cost is kernel launches over many small matrices, and a deep model
    repeats a handful of shapes across its blocks, so one batched call per
    distinct shape removes most of them.
    """
    flat = [m.reshape(m.shape[0], -1).float() for m in matrices]
    shapes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, matrix in enumerate(flat):
        shapes[(matrix.shape[0], matrix.shape[1])].append(index)
    updates: list[Tensor] = [torch.empty(0)] * len(flat)
    for indices in shapes.values():
        batch = orthogonalize(torch.stack([flat[index] for index in indices]), ns_steps)
        for slot, index in enumerate(indices):
            updates[index] = batch[slot].reshape(matrices[index].shape).to(matrices[index].dtype)
    return updates


class MuonAdamW(Optimizer):
    """Muon on every matrix, AdamW on every vector, in one step.

    Muon: momentum with Nesterov blend, Frobenius pre-normalization, Newton-
    Schulz orthogonalization, then decoupled weight decay. The orthogonalized
    update already has spectral norm ~ 1, so `lr_scales` (muP's `sqrt(n)` on
    input and output weights) is the only shape factor; `wd_scales` carries the
    reciprocal, keeping the decay `lr * weight_decay` on every matrix.
    """

    def __init__(
        self,
        matrices: Iterable[nn.Parameter],
        others: Iterable[nn.Parameter],
        *,
        lr_scales: Iterable[float] = (),
        wd_scales: Iterable[float] = (),
        muon_lr: float,
        adam_lr: float,
        beta: float,
        ns_steps: int,
        weight_decay: float,
        adam_betas: tuple[float, float],
        adam_eps: float,
        adam_weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            [
                {
                    "params": list(matrices),
                    "lr": muon_lr,
                    "kind": "muon",
                    "beta": beta,
                    "ns_steps": ns_steps,
                    "weight_decay": weight_decay,
                    "lr_scales": list(lr_scales),
                    "wd_scales": list(wd_scales),
                },
                {
                    "params": list(others),
                    "lr": adam_lr,
                    "kind": "adam",
                    "betas": adam_betas,
                    "eps": adam_eps,
                    "weight_decay": adam_weight_decay,
                },
            ],
            {},
        )

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        """Every stage runs over the whole group at once: the per-parameter
        loop below is Python-only bookkeeping and launches no kernels."""
        for group in self.param_groups:
            size = len(group["params"])
            lr_scales = group.get("lr_scales") or [1.0] * size
            wd_scales = group.get("wd_scales") or [1.0] * size
            params, grads, mus, nus, lrs, decays = [], [], [], [], [], []
            count = 0
            for param, lr_scale, wd_scale in zip(group["params"], lr_scales, wd_scales, strict=True):
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["mu"] = torch.zeros_like(param)
                    if group["kind"] == "adam":
                        state["nu"] = torch.zeros_like(param)
                state["step"] += 1
                count = state["step"]
                params.append(param)
                grads.append(param.grad)
                mus.append(state["mu"])
                if group["kind"] == "adam":
                    nus.append(state["nu"])
                lrs.append(group["lr"] * lr_scale)
                decays.append(group["weight_decay"] * wd_scale)
            if not params:
                continue
            updates = (
                self._muon_updates(grads, mus, count, group)
                if group["kind"] == "muon"
                else self._adam_updates(grads, mus, nus, count, group)
            )
            torch._foreach_mul_(updates, lrs)
            if group["weight_decay"]:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(params, [lr * decay for lr, decay in zip(lrs, decays, strict=True)]),
                )
            torch._foreach_sub_(params, updates)

    @staticmethod
    def _muon_updates(
        grads: list[Tensor], mus: list[Tensor], count: int, group: dict
    ) -> list[Tensor]:
        beta = group["beta"]
        torch._foreach_mul_(mus, beta)
        torch._foreach_add_(mus, grads, alpha=1 - beta)
        blended = list(
            torch._foreach_add(
                torch._foreach_mul(mus, beta / (1 - beta ** (count + 1))),
                torch._foreach_mul(grads, (1 - beta) / (1 - beta**count)),
            )
        )
        return orthogonalize_all(blended, group["ns_steps"])

    @staticmethod
    def _adam_updates(
        grads: list[Tensor], mus: list[Tensor], nus: list[Tensor], count: int, group: dict
    ) -> list[Tensor]:
        b1, b2 = group["betas"]
        torch._foreach_mul_(mus, b1)
        torch._foreach_add_(mus, grads, alpha=1 - b1)
        torch._foreach_mul_(nus, b2)
        torch._foreach_addcmul_(nus, grads, grads, value=1 - b2)
        denominator = torch._foreach_sqrt(torch._foreach_div(nus, 1 - b2**count))
        torch._foreach_add_(denominator, group["eps"])
        return list(torch._foreach_div(torch._foreach_div(mus, 1 - b1**count), denominator))


def make_optimizer(model: nn.Module, config: Config) -> tuple[Optimizer, LambdaLR]:
    optimizer = MuonAdamW(
        muon_params(model),
        adam_params(model),
        lr_scales=mup_lr_scales(model, config),
        wd_scales=mup_wd_scales(model, config),
        muon_lr=muon_peak_lr(config),
        adam_lr=adam_peak_lr(config),
        beta=config.muon_beta,
        ns_steps=config.muon_ns_steps,
        weight_decay=config.weight_decay,
        adam_betas=(config.adam_b1, config.adam_b2),
        adam_eps=config.adam_eps,
    )
    return optimizer, LambdaLR(optimizer, lambda step: schedule_fraction(config, step))


# EMA half-life as a fraction of the run: a fixed decay leaves short runs
# evaluating partly random weights. Anchored so a 78k-step run resolves to
# exactly 0.9999 and keeps its checkpoints.
EMA_ANCHOR_STEPS = 78_000
EMA_ANCHOR_DECAY = 0.9999
EMA_HALFLIFE_FRACTION = math.log(2.0) / -math.log(EMA_ANCHOR_DECAY) / EMA_ANCHOR_STEPS


def half_life_ema_decay(num_steps: int) -> float:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    return 0.5 ** (1.0 / (EMA_HALFLIFE_FRACTION * num_steps))


class EMA:
    """Exponential moving average over parameters and buffers, seeded at init."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0:
            self.model.load_state_dict(model.state_dict())
            return
        if self.decay >= 1:
            return
        pairs = list(zip(self.model.state_dict().values(), model.state_dict().values(), strict=True))
        blended = [(ema, live) for ema, live in pairs if ema.is_floating_point()]
        torch._foreach_lerp_(
            [ema for ema, _ in blended], [live for _, live in blended], 1.0 - self.decay
        )
        for ema, live in pairs:
            if not ema.is_floating_point():
                ema.copy_(live)


def tensor_norm(tensors: Iterable[Tensor]) -> float:
    total = sum(float(tensor.detach().float().pow(2).sum()) for tensor in tensors)
    return math.sqrt(total)
