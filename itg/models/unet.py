"""DDPM++/SongUNet-style UNet under EDM preconditioning, as an `x_0` predictor."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from itg.config import Config
from itg.models.common import (
    Conv2d,
    Linear,
    SinusoidalEmbedding,
    dropout,
    input_multiplier,
    mup_residual_scale,
    output_multiplier,
)
from itg.processes import Prediction, Process


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, max(1, channels // 4)), channels, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        dropout_p: float,
        residual_scale: float,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.residual_scale = residual_scale
        self.norm1 = _group_norm(in_ch)
        self.conv1 = Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = Linear(emb_dim, 2 * out_ch)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch, 3, padding=1, zero_init=zero_init)
        self.skip = (
            Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, h: Tensor, emb: Tensor, gen: torch.Generator | None) -> Tensor:
        out = self.conv1(F.silu(self.norm1(h)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        out = F.silu(self.norm2(out) * (1 + scale) + shift)
        return self.skip(h) + self.residual_scale * self.conv2(
            dropout(out, self.dropout_p, gen)
        )


class SelfAttention(nn.Module):
    def __init__(self, channels: int, residual_scale: float, zero_init: bool) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.norm = _group_norm(channels)
        self.qkv = Conv2d(channels, 3 * channels, 1)
        self.proj = Conv2d(channels, channels, 1, zero_init=zero_init)

    def forward(self, h: Tensor, emb: Tensor, gen: torch.Generator | None) -> Tensor:
        del emb, gen
        batch, channels, height, width = h.shape
        qkv = self.qkv(self.norm(h)).reshape(batch, 3, 1, channels, height * width)
        q, k, v = qkv.permute(1, 0, 2, 4, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(batch, channels, height, width)
        return h + self.residual_scale * self.proj(out)


class Upsample(nn.Module):
    """Nearest-neighbour 2x, with the trunk's uniform call signature."""

    def forward(self, h: Tensor, emb: Tensor, gen: torch.Generator | None) -> Tensor:
        del emb, gen
        return F.interpolate(h, scale_factor=2.0, mode="nearest")


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, h: Tensor, emb: Tensor, gen: torch.Generator | None) -> Tensor:
        del emb, gen
        return self.conv(h)


class SongUNet(nn.Module):
    """Unconditional UNet: residual blocks with attention at set resolutions."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        channels, resolution, _ = config.data_shape
        base, edm = config.hidden_dim, config.edm
        emb_dim = 4 * base
        attn = set(edm.unet_attn_resolutions)
        scale = mup_residual_scale(config)
        zero = config.residual_zero_init
        self.time_embed = SinusoidalEmbedding(base)
        self.emb_proj = nn.Sequential(
            Linear(base, emb_dim), nn.SiLU(), Linear(emb_dim, emb_dim)
        )
        self.conv_in = Conv2d(channels, base, 3, padding=1, role="input")

        self.down = nn.ModuleList()
        skip_channels = [base]
        width, size = base, resolution
        for level, mult in enumerate(edm.unet_channel_mult):
            for _ in range(edm.unet_num_blocks):
                stage = nn.ModuleList([ResBlock(width, base * mult, emb_dim, config.dropout_p, scale, zero)])
                width = base * mult
                if size in attn:
                    stage.append(SelfAttention(width, scale, zero))
                self.down.append(stage)
                skip_channels.append(width)
            if level < len(edm.unet_channel_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(width)]))
                skip_channels.append(width)
                size //= 2

        self.mid = nn.ModuleList(
            [
                ResBlock(width, width, emb_dim, config.dropout_p, scale, zero),
                SelfAttention(width, scale, zero),
                ResBlock(width, width, emb_dim, config.dropout_p, scale, zero),
            ]
        )

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(edm.unet_channel_mult))):
            for block in range(edm.unet_num_blocks + 1):
                stage = nn.ModuleList(
                    [
                        ResBlock(
                            width + skip_channels.pop(),
                            base * mult,
                            emb_dim,
                            config.dropout_p,
                            scale,
                            zero,
                        )
                    ]
                )
                width = base * mult
                if size in attn:
                    stage.append(SelfAttention(width, scale, zero))
                if level and block == edm.unet_num_blocks:
                    stage.append(Upsample())
                    size *= 2
                self.up.append(stage)

        self.norm_out = _group_norm(width)
        self.conv_out = Conv2d(width, channels, 3, padding=1, role="output")
        self.output_scale = output_multiplier(self.conv_out, config)
        self.input_scale = input_multiplier(self.conv_in, config)

    def forward(self, x: Tensor, noise_level: Tensor, gen: torch.Generator | None) -> Tensor:
        emb = self.emb_proj(self.time_embed(noise_level))
        h = self.input_scale * self.conv_in(x)
        skips = [h]
        for stage in self.down:
            for layer in stage:
                h = layer(h, emb, gen)
            skips.append(h)
        for layer in self.mid:
            h = layer(h, emb, gen)
        for stage in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in stage:
                h = layer(h, emb, gen)
        return self.output_scale * self.conv_out(F.silu(self.norm_out(h)))


class EDMPrecond(nn.Module):
    """Karras preconditioning around `SongUNet`, yielding an `x_0` prediction."""

    # logits are [batch, class, height, width]
    class_dim = 1

    def __init__(self, config: Config, process: Process) -> None:
        super().__init__()
        self.net = SongUNet(config)
        self.process = process
        self.sigma_data = config.edm.sigma_data
        self.raw_output = False

    def swap_in_classifier_head(self, config: Config, num_classes: int) -> None:
        net = self.net
        net.conv_out = Conv2d(
            net.conv_out.in_channels, num_classes, 3, padding=1, zero_init=True, role="output"
        )
        self.raw_output = True

    def forward(
        self, x_t: Tensor, t: Tensor, gen: torch.Generator | None = None
    ) -> Prediction:
        sigma = self.process.sigma(t).reshape(-1, 1, 1, 1).clamp(min=1e-8)
        variance = self.sigma_data**2 + sigma**2
        c_skip = self.sigma_data**2 / variance
        c_out = sigma * self.sigma_data / variance.sqrt()
        c_in = 1.0 / variance.sqrt()
        c_noise = sigma.flatten().log() / 4.0
        raw = self.net(c_in * x_t, c_noise, gen)
        if self.raw_output:
            return Prediction(raw, "x_0")
        return Prediction(c_skip * x_t + c_out * raw, "x_0")
