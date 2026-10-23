"""Shared model pieces: muP scaling, explicit init, embeddings.

Width-depth muP, in absolute width and depth — there is no base model:

                  input             hidden            output
    multiplier    alpha             alpha / L         alpha / n
    init variance sigma^2 / d0      sigma^2 / n       sigma^2
    LR            eta sqrt(n)       eta               eta sqrt(n)
    weight decay  lambda / sqrt(n)  lambda            lambda / sqrt(n)

`n` is the model width (`hidden_dim`) and `L` the residual-branch count. Layer
widths need not equal `n` -- the UNet's level `i` is `n * channel_mult[i]` --
only be proportional to it, so `n` is the quantity you scale, not each layer's
own width. `d0` is the input data dimension, i.e. an input weight's own
`fan_in`.

`sigma`, `eta`, `lambda` and `alpha` are per-weight, not one global constant:
each weight scales the config values by its own `WeightHP`, which defaults to
1 throughout.

The learning-rate row *is* the shape scaling. Muon's orthogonalized update
already has spectral norm ~ its learning rate, so it carries no
`sqrt(fan_out / fan_in)` factor of its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from itg.config import Config

Role = Literal["input", "hidden", "output"]


@dataclass(frozen=True)
class WeightHP:
    """One weight's own hyperparameters, as multiples of the config values."""

    sigma: float = 1.0
    lr: float = 1.0
    weight_decay: float = 1.0
    alpha: float = 1.0


class Linear(nn.Linear):
    zero_init: bool
    role: Role
    hp: WeightHP

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        zero_init: bool = False,
        role: Role = "hidden",
        hp: WeightHP = WeightHP(),
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.zero_init, self.role, self.hp = zero_init, role, hp


class Conv2d(nn.Conv2d):
    zero_init: bool
    role: Role
    hp: WeightHP

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        zero_init: bool = False,
        role: Role = "hidden",
        hp: WeightHP = WeightHP(),
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.zero_init, self.role, self.hp = zero_init, role, hp


Weighted = (Linear, Conv2d)


def fan(module: Linear | Conv2d) -> tuple[int, int]:
    return module.weight.shape[0], module.weight[0].numel()


def model_width(config: Config) -> int:
    """`n`: the width every layer's own width is proportional to."""
    return config.hidden_dim


def _width_scale(module: Linear | Conv2d, config: Config) -> float:
    """`sqrt(n)` on input and output weights, 1 on hidden."""
    if module.role == "hidden":
        return 1.0
    return math.sqrt(model_width(config))


def mup_lr_scale(module: Linear | Conv2d, config: Config) -> float:
    return module.hp.lr * _width_scale(module, config)


def mup_wd_scale(module: Linear | Conv2d, config: Config) -> float:
    """The reciprocal width scale, so `lr * wd` stays `eta lambda`."""
    return module.hp.weight_decay / _width_scale(module, config)


def input_multiplier(module: Linear | Conv2d, config: Config) -> float:
    """The `alpha` block multiplier on an input weight."""
    return config.mup_multiplier * module.hp.alpha


def output_multiplier(module: Linear | Conv2d, config: Config) -> float:
    """The `alpha / n` block multiplier on an output weight."""
    alpha = config.mup_multiplier * module.hp.alpha
    return alpha / model_width(config)


def taps(module: Linear | Conv2d) -> int:
    """Kernel positions per input channel: 1 for a matrix, `k^2` for a conv."""
    return module.weight[0][0].numel() if module.weight.ndim > 2 else 1


def init_std(module: Linear | Conv2d, config: Config) -> float:
    """`sigma / sqrt(d0)` on an input weight (d0 is its own fan_in),
    `sigma / sqrt(n)` on hidden, plain `sigma` on output.

    Convolutions take the same rows against their own geometry. A conv's RMS
    operator norm is `sqrt(Cin / Cout) ||A||_2` -- channels only, the spatial
    dimensions cancel -- and for iid entries of std `c` its Fourier blocks have
    entry std `c k`, so `||A||_R = c k sqrt(Cin) (1 + sqrt(Cin / Cout))`.
    Hence `Theta(1)` needs `c = sigma / sqrt(Cin k^2)`, the layer's own fan_in
    with the taps included, and an output conv needs `sigma / k`. The model
    width is a poor stand-in there: it drops both the taps and the level's
    channel multiplier.
    """
    sigma = config.mup_init_sigma * module.hp.sigma
    fan_in = fan(module)[1]
    if module.role == "input":
        return sigma / math.sqrt(fan_in)
    if isinstance(module, Conv2d):
        return sigma / math.sqrt(fan_in if module.role == "hidden" else taps(module))
    return sigma if module.role == "output" else sigma / math.sqrt(model_width(config))


def residual_branches(config: Config) -> int:
    """`L`: residual additions in the UNet trunk, muP's depth."""
    edm = config.edm
    levels = len(edm.unet_channel_mult)
    attention = sum(
        config.image_size >> level in edm.unet_attn_resolutions for level in range(levels)
    )
    # Down: num_blocks per level. Middle: two blocks and one attention.
    # Up: num_blocks + 1 per level, one extra for the skip from `conv_in`.
    return (
        levels * (2 * edm.unet_num_blocks + 1)
        + 3
        + attention * (2 * edm.unet_num_blocks + 1)
    )


def mup_residual_scale(config: Config) -> float:
    """The `alpha / L` block multiplier on a hidden (residual) branch."""
    if not config.mup_residual_depth_scaling:
        return config.mup_multiplier
    return config.mup_multiplier / max(residual_branches(config), 1)


def dropout(h: Tensor, p: float, gen: torch.Generator | None) -> Tensor:
    """Inverted dropout from an explicit generator; identity when `gen` is None."""
    if p <= 0.0 or gen is None:
        return h
    keep = torch.rand(h.shape, generator=gen, device=h.device, dtype=h.dtype) >= p
    return h * keep / (1.0 - p)


class SinusoidalEmbedding(nn.Module):
    """DDPM-style scalar embedding, `[cos(t f), sin(t f)]`."""

    def __init__(self, emb_dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        half = emb_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        args = t.float()[:, None] * self.freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def muon_weights(model: nn.Module) -> list[Linear | Conv2d]:
    """Every matrix layer. The table's learning rates are for Muon, and muP
    under Adam would instead need a per-layer `1 / fan_in` rate, so matrices
    all go to Muon and AdamW keeps only the vectors (biases, norm gains)."""
    return [module for module in model.modules() if isinstance(module, Weighted)]


def muon_params(model: nn.Module) -> list[nn.Parameter]:
    return [module.weight for module in muon_weights(model)]


def mup_lr_scales(model: nn.Module, config: Config) -> list[float]:
    """Aligned with `muon_params`, so the optimizer can zip them."""
    return [mup_lr_scale(module, config) for module in muon_weights(model)]


def mup_wd_scales(model: nn.Module, config: Config) -> list[float]:
    return [mup_wd_scale(module, config) for module in muon_weights(model)]


def adam_params(model: nn.Module) -> list[nn.Parameter]:
    tagged = {id(param) for param in muon_params(model)}
    return [param for param in model.parameters() if id(param) not in tagged]


def init_model(model: nn.Module, config: Config, gen: torch.Generator) -> None:
    """Gaussian init at `init_std` from `gen`; no ambient RNG, no torch defaults."""
    for module in model.modules():
        if not isinstance(module, Weighted):
            continue
        with torch.no_grad():
            if module.zero_init:
                module.weight.zero_()
            else:
                module.weight.copy_(
                    torch.randn(module.weight.shape, generator=gen)
                    * init_std(module, config)
                )
            if module.bias is not None:
                module.bias.zero_()
    for module in model.modules():
        if (
            isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm))
            and module.weight is not None
        ):
            with torch.no_grad():
                module.weight.fill_(1.0)
