"""Experiment-level aggregation: per-case artifacts into report tables.

Per-case CSVs carry only a `case` column; the grid fields live once in
`cases.csv` and are joined back in here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from itg.experiments import Experiment
from itg.runner.artifacts import (
    aux_samples_path,
    information_curve_path,
    information_summaries_path,
    model_dir,
    read_json,
    samples_path,
)
from itg.runner.report import read_case_csvs, write_report


def case_table(experiment: Experiment) -> pd.DataFrame:
    rows = []
    for name, n, depth in experiment.cases():
        state = model_dir(experiment.output_dir / name, experiment.evals.source_model_indices[0]) / "state.json"
        rows.append(
            {
                "case": name,
                **experiment.case_fields(n, depth),
                "parameter_count": read_json(state)["parameter_count"] if state.exists() else None,
            }
        )
    return pd.DataFrame(rows)


def dataset_pool_size(case_dirs: list[Path]) -> float | None:
    """Images every source model draws from, or None when they get disjoint blocks."""
    for case_dir in case_dirs:
        manifest = case_dir / "dataset" / "manifest.json"
        if manifest.exists():
            pool = read_json(manifest).get("pool_size")
            if pool is not None:
                return float(pool)
    return None


def write_experiment_report(experiment: Experiment) -> Path:
    output_dir = experiment.output_dir
    report_dir = output_dir / "report"
    evals = experiment.evals
    sizes = evals.sizes
    case_dirs = [output_dir / name for name, _n, _depth in experiment.cases()]
    cases = case_table(experiment)

    summaries = pd.DataFrame(
        [
            row
            for case_dir in case_dirs
            if information_summaries_path(case_dir).exists()
            for row in read_json(information_summaries_path(case_dir))["summaries"]
        ]
    )
    curves = read_case_csvs(
        [
            information_curve_path(case_dir, method, k)
            for case_dir in case_dirs
            for method in evals.information_methods
            for k in sizes
        ]
    )
    metric_dirs = [
        samples_path(case_dir, idx, "eval").parent
        for case_dir in case_dirs
        for idx in evals.source_model_indices
    ] + [
        aux_samples_path(case_dir, "mixture", k).parent for case_dir in case_dirs for k in sizes
    ]
    tables = {
        "information_summary": summaries,
        "training_metrics": read_case_csvs([d / "training_metrics.csv" for d in case_dirs]),
        "information_curves": curves,
        "memorization": read_case_csvs([d / "memorization.csv" for d in metric_dirs]),
        "fid": read_case_csvs(
            [
                samples_path(case_dir, idx, "mixture_train").parent / "fid.csv"
                for case_dir in case_dirs
                for idx in evals.mixture_train_models
            ]
        ),
    }
    # Per-case CSVs carry only a `case` column; the grid fields join back here.
    for name, frame in tables.items():
        if not frame.empty:
            overlap = [c for c in frame.columns if c in cases.columns and c != "case"]
            tables[name] = frame.drop(columns=overlap).merge(cases, on="case", how="left")
    report_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(report_dir / "cases.csv", index=False)
    write_report(report_dir, tables, sizes, dataset_pool_size(case_dirs))
    print(f"[report] written: {report_dir}", flush=True)
    return report_dir
