"""A tiny UNet config over a small image folder, fast enough for CPU tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from itg.config import Config, EDMConfig


def image_folder(root: Path, count: int, size: int = 8, seed: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        array = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        Image.fromarray(array).save(root / f"{index:03d}.png")
    return root


def tiny_config(root: Path | str = "/nonexistent", /, **overrides: Any) -> Config:
    base = dict(
        image_root=str(root),
        image_size=8,
        regime="iid",
        dataset_size=16,
        hidden_dim=8,
        num_layers=1,
        batch_size=4,
        num_steps=6,
        # Compiling a tiny model on CPU is pure overhead; the compiled path
        # has its own test.
        compile_model=False,
        edm=EDMConfig(unet_channel_mult=(1, 2), unet_num_blocks=1, unet_attn_resolutions=(4,)),
    )
    return Config(**{**base, **overrides})  # ty: ignore[invalid-argument-type]
