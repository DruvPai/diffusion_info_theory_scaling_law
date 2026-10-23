"""Time-conditioned k-way classifier over a diffusion backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from itg.config import Config


class Classifier(nn.Module):
    """A backbone whose output head emits `num_classes` logits, mean-pooled.

    Each architecture swaps its own head and declares `class_dim`, the axis
    the classes land on; the head is linear, so pooling before or after it is
    equivalent.
    """

    def __init__(self, backbone: nn.Module, config: Config, num_classes: int) -> None:
        super().__init__()
        backbone.swap_in_classifier_head(config, num_classes)
        self.class_dim = backbone.class_dim
        self.backbone = backbone
        self.num_classes = num_classes

    def forward(
        self, x_t: Tensor, t: Tensor, gen: torch.Generator | None = None
    ) -> Tensor:
        logits = self.backbone(x_t, t, gen).value
        pooled = [dim for dim in range(1, logits.ndim) if dim != self.class_dim]
        return logits.mean(dim=pooled) if pooled else logits
