"""FFHQ 64x64 EDM/UNet (N, P) grid: 2 train / 2 eval source models, model-free
estimator only.

    uv run python scripts/ffhq_edm.py
    uv run python scripts/ffhq_edm.py --case n_10_h32_l1_edm_img64
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itg.experiments import MODEL_FREE_EVALS, Experiment, edm_base
from itg.runner.cli import add_arguments, run

N_VALUES = ("10", "43", "180", "780", "3300", "14000", "60000")
EDM_EVALS = replace(MODEL_FREE_EVALS, sample_order=2, sample_steps=40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(n_values=N_VALUES, depths=[1, 2, 4])
    args = parser.parse_args()
    run(
        Experiment(
            name="ffhq_edm",
            output_dir=Path("experiments/ffhq_edm"),
            base=edm_base("ffhq", image_size=64),
            evals=EDM_EVALS,
            n_values=(),
            depths=(),
            case_suffix="_edm_img64",
            unet_blocks_follow_depth=True,
            ema_half_life=True,
        ),
        args,
    )


if __name__ == "__main__":
    main()
