"""PNG grids for image samples."""

from __future__ import annotations

from math import ceil, isqrt
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

CHANNELS = 3


def _square_root(value: int) -> int:
    root = isqrt(value)
    if root * root != value:
        raise ValueError(f"expected square dimension, got {value}")
    return root


def as_images(samples: Tensor) -> np.ndarray:
    """uint8 (N,H,W,3) from [-1,1] samples."""
    values = samples.detach().float().cpu()
    n = values.shape[0]
    if values.ndim == 4:
        if values.shape[-1] == CHANNELS:
            images = values
        elif values.shape[1] == CHANNELS:
            images = values.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"unsupported sample shape {tuple(values.shape)}")
    elif values.ndim == 3:
        seq, token = values.shape[1:]
        grid, patch = _square_root(seq), _square_root(token // CHANNELS)
        images = (
            values.reshape(n, grid, grid, patch, patch, CHANNELS)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(n, grid * patch, grid * patch, CHANNELS)
        )
    elif values.ndim == 2:
        side = _square_root(values.shape[1] // CHANNELS)
        images = values.reshape(n, side, side, CHANNELS)
    else:
        raise ValueError(f"unsupported sample shape {tuple(values.shape)}")
    return ((images + 1.0) * 127.5).clamp(0.0, 255.0).to(torch.uint8).numpy()


def write_sample_grid(
    samples: Tensor,
    out: Path,
    *,
    max_images: int = 64,
    cols: int = 8,
) -> Path:
    images = as_images(samples)[:max_images]
    rows = ceil(len(images) / cols)
    h, w = images.shape[1:3]
    canvas = np.zeros((rows * h, cols * w, CHANNELS), dtype=np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out)
    return out
