"""FashionMNIST matched-bank sweep: K_train = K_val = k, one k per run.

The main grid fixes the held-out count at five and sweeps the bank alone. This
runs the diagonal instead, with the model-free estimator only. Source models
and their samples are symlinked from `experiments/fashion_mnist_edm`, so only
the information stage does work.

    uv run python scripts/fashion_mnist_edm_matched_bank.py --k 2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itg.experiments import ALL_ESTIMATORS_EVALS, Experiment, edm_base
from itg.runner.cli import add_arguments, parse_n, run

N_VALUES = ("10", "43", "180", "780", "3300", "14000", "60000")
SOURCE_DIR = Path("experiments/fashion_mnist_edm")
LINKED = ("models", "samples")


def link_artifacts(experiment: Experiment) -> None:
    for name, _n, _depth in experiment.cases():
        target = experiment.output_dir / name
        target.mkdir(parents=True, exist_ok=True)
        for child in LINKED:
            link = target / child
            if not link.exists():
                link.symlink_to((SOURCE_DIR / name / child).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--k", type=int, required=True, choices=(2, 3, 4, 5))
    parser.set_defaults(n_values=N_VALUES, depths=[1, 2, 4], stages=["information"])
    args = parser.parse_args()
    experiment = Experiment(
        name="fashion_mnist_edm_matched_bank",
        output_dir=args.output_dir or Path(f"experiments/fashion_mnist_matched/k{args.k}"),
        base=edm_base("fashion_mnist", image_size=32),
        evals=replace(
            ALL_ESTIMATORS_EVALS,
            sample_order=2,
            sample_steps=18,
            information_methods=("classifier_model_free",),
            mixture_sizes=(args.k,),
            mixture_eval_models=tuple(range(5, 5 + args.k)),
        ),
        n_values=tuple(parse_n(v) for v in args.n_values),
        depths=tuple(args.depths),
        case_suffix="_edm_img32",
        unet_blocks_follow_depth=True,
        ema_half_life=True,
    )
    link_artifacts(experiment)
    run(experiment, args)


if __name__ == "__main__":
    main()
