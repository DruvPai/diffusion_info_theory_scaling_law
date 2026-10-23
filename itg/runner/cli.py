"""Shared command line for the grid scripts.

Each experiment is its own script that builds an `Experiment` and calls `run`;
there is no registry. Dispatch re-executes the calling script per case with the
same arguments plus `--case`, so a grid always runs exactly what was asked for.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch

from itg.experiments import Experiment
from itg.runner.aggregate import write_experiment_report
from itg.runner.case import Case
from itg.runner.dispatch import dispatch, jobs
from itg.runner.events import configure_console
from itg.runner.pipeline import FORCE_STAGES, PER_MODEL_STAGES, STAGES, run_case


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", help="run one case instead of dispatching the grid")
    parser.add_argument("--model", type=int, help="train one source model of --case, then stop")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-values", nargs="+", help="dataset sizes; 'inf' for streaming")
    parser.add_argument("--depths", type=int, nargs="+")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--per-model-only", action="store_true", help=f"stages {PER_MODEL_STAGES}")
    parser.add_argument("--force", nargs="*", default=[], choices=FORCE_STAGES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sequential", action="store_true", help="run the grid in-process")
    parser.add_argument("--report-only", action="store_true")


def parse_n(value: str) -> int | None:
    return None if value.lower() in {"inf", "none"} else int(value)


def set_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: a few reduction kernels have no deterministic implementation;
    # they are logged rather than made fatal.
    torch.use_deterministic_algorithms(True, warn_only=True)


def run(experiment: Experiment, args: argparse.Namespace) -> None:
    configure_console()
    set_determinism()
    if args.output_dir is not None:
        experiment = replace(experiment, output_dir=args.output_dir)
    if args.n_values:
        experiment = replace(experiment, n_values=tuple(parse_n(v) for v in args.n_values))
    if args.depths:
        experiment = replace(experiment, depths=tuple(args.depths))

    if args.report_only:
        write_experiment_report(experiment)
        return

    lookup = {name: (n, depth) for name, n, depth in experiment.cases()}
    evals = replace(experiment.evals, force=frozenset(args.force))
    if args.seed is not None:
        evals = replace(evals, seed=args.seed)
    stages = PER_MODEL_STAGES if args.per_model_only else tuple(args.stages)

    def case(name: str) -> Case:
        if name not in lookup:
            raise ValueError(f"unknown case {name!r} in {experiment.name}")
        n, depth = lookup[name]
        return Case(
            name=name,
            directory=experiment.output_dir / name,
            config=experiment.case_config(n, depth),
            evals=evals,
            fields=experiment.case_fields(n, depth),
        )

    if args.case:
        if args.model is None:
            run_case(case(args.case), stages)
        else:
            run_case(case(args.case), ("train",), (args.model,))
        return
    if args.sequential:
        for name in experiment.scheduling_order():
            run_case(case(name), stages)
    else:
        # Every source model first, packed across GPUs independently of its case,
        # then the per-case evals over the finished models. The second round
        # restates --force without "train" so a forced retrain happens once.
        order = experiment.scheduling_order()
        dispatch(jobs(order, experiment.output_dir,
                      [["--model", str(i)] for i in evals.source_model_indices]))
        dispatch(jobs(order, experiment.output_dir,
                      [["--force", *sorted(evals.force - {"train"})]]))
    write_experiment_report(experiment)
