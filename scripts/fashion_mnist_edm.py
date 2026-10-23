"""FashionMNIST EDM/UNet (N, P) grid: 5 train / 5 eval source models, all three
information estimators on identical probes.

    uv run python scripts/fashion_mnist_edm.py
    uv run python scripts/fashion_mnist_edm.py --case n_10_h32_l1_edm_img32
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itg.experiments import ALL_ESTIMATORS_EVALS, Experiment, edm_base
from itg.runner.cli import add_arguments, run

N_VALUES = ("10", "43", "180", "780", "3300", "14000", "60000")
EDM_EVALS = replace(ALL_ESTIMATORS_EVALS, sample_order=2, sample_steps=18)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.set_defaults(n_values=N_VALUES, depths=[1, 2, 4])
    args = parser.parse_args()
    run(
        Experiment(
            name="fashion_mnist_edm",
            output_dir=Path("experiments/fashion_mnist_edm"),
            base=edm_base("fashion_mnist", image_size=32),
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
