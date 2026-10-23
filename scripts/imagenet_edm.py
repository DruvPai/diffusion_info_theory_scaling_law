"""ImageNet EDM/UNet (N, P) grid: 2 train / 2 eval source models, model-free
estimator only. Log-spaced dataset sizes to 100,000 plus the whole corpus,
which trains for the same number of steps as 100,000 and is the endpoint where
every source model sees the same data.

    uv run python scripts/imagenet_edm.py
    uv run python scripts/imagenet_edm.py --image-size 128
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itg.experiments import MODEL_FREE_EVALS, Experiment, edm_base
from itg.runner.cli import add_arguments, run

N_VALUES = ("10", "63", "400", "630", "1000", "1600", "2500", "4000",
            "6300", "16000", "40000", "100000", "inf")
# Depth 1 underfits ImageNet-64; depth 6 needs --microbatch-size 64 to fit 80GB.
# Depth 4 is run over a subset of N -- the knee plus the endpoint -- so the grid
# is ragged there; the report skips cases with no artifacts.
DEPTHS = [2, 3, 4, 5]
EDM_EVALS = replace(MODEL_FREE_EVALS, sample_order=2, sample_steps=40)


def channel_mult(image_size: int) -> tuple[int, ...]:
    """Halve to a 16x16 trunk so the attention block sees the same map at any size."""
    levels = image_size // 16
    assert levels >= 2 and levels & (levels - 1) == 0
    return (1,) + (2,) * (levels.bit_length() - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--microbatch-size", type=int)
    parser.set_defaults(n_values=N_VALUES, depths=DEPTHS)
    args = parser.parse_args()
    size = args.image_size
    base = replace(edm_base("imagenet", image_size=size), microbatch_size=args.microbatch_size)
    run(
        Experiment(
            name="imagenet_edm",
            output_dir=Path(f"experiments/imagenet_edm_{size}"),
            base=replace(base, edm=replace(base.edm, unet_channel_mult=channel_mult(size))),
            evals=EDM_EVALS,
            n_values=(),
            depths=(),
            case_suffix=f"_edm_img{size}",
            unet_blocks_follow_depth=True,
            ema_half_life=True,
            # Pinned so extending the grid past 100000 neither rebudgets the
            # streaming case nor moves the bias anchor it supplies.
            streaming_steps_from=100_000,
        ),
        args,
    )


if __name__ == "__main__":
    main()
