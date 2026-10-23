"""Model factory."""

from __future__ import annotations

import torch
from torch import nn

from itg.config import Config
from itg.models.classifier import Classifier
from itg.models.common import adam_params, init_model, muon_params
from itg.models.unet import EDMPrecond
from itg.processes import Process

__all__ = [
    "Classifier",
    "adam_params",
    "build_classifier",
    "build_model",
    "count_params",
    "make_module",
    "muon_params",
]


def make_module(
    config: Config, process: Process, num_classes: int | None = None
) -> nn.Module:
    """Architecture only, uninitialized: for loading a checkpoint into."""
    model = EDMPrecond(config, process)
    return model if num_classes is None else Classifier(model, config, num_classes)


def build_model(config: Config, process: Process, gen: torch.Generator) -> nn.Module:
    model = make_module(config, process)
    init_model(model, config, gen)
    return model


def build_classifier(
    config: Config, process: Process, num_classes: int, gen: torch.Generator
) -> Classifier:
    model = make_module(config, process, num_classes)
    init_model(model, config, gen)
    return model


def count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
