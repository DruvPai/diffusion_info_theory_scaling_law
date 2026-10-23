"""CIFAR-10 EDM/UNet (N, P) grid: 2 train / 2 eval source models, model-free
estimator only.

    uv run python scripts/cifar10_edm.py
    uv run python scripts/cifar10_edm.py --case n_10_h32_l1_edm_img32
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itg.experiments import MODEL_FREE_EVALS, Experiment, edm_base
from itg.runner.cli import add_arguments, run

N_VALUES = ("10", "41", "170", "710", "2900", "6500", "9000", "12000",
            "16000", "25000", "35000", "40000", "45000", "50000")
EDM_EVALS = replace(MODEL_FREE_EVALS, sample_order=2, sample_steps=18)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(n_values=N_VALUES, depths=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    run(
        Experiment(
            name="cifar10_edm",
            output_dir=Path("experiments/cifar10_edm"),
            base=edm_base("cifar10", image_size=32),
            evals=EDM_EVALS,
            n_values=(),
            depths=(),
            case_suffix="_edm_img32",
            unet_blocks_follow_depth=True,
            ema_half_life=True,
        ),
        args,
    )


if __name__ == "__main__":
    main()
