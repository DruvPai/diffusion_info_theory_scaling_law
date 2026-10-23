from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itg.runner.report import write_report
from itg.scaling_law import fit_scaling_law, scaling_law_prediction

TRUE = {
    "group_count_ref": 8.0, "group_count_exponent": 0.4, "critical_size_ref": 600.0,
    "critical_size_exponent": 0.35, "sigma": 0.5, "ell": 13.0, "floor_constant": 200.0,
    "floor_scale": 50.0, "floor_exponent": 0.7, "P_ref": 4e5, "pool": 1e6,
}


def scaling_law_grid(
    n_values: np.ndarray, p_values: np.ndarray
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The grid `fit_scaling_law` is meant to recover, generated from its own model.

    nan is the run on the whole pool.
    """
    n = np.repeat(np.append(n_values, np.nan), len(p_values))
    p = np.tile(p_values, len(n_values) + 1)
    information = scaling_law_prediction(TRUE, np.nan_to_num(n, nan=TRUE["pool"]), p)
    return pd.Series(n), pd.Series(p), pd.Series(information)


def test_fit_scaling_law_recovers_its_own_parameters() -> None:
    n, p, information = scaling_law_grid(
        np.geomspace(10, 1e5, 9), np.array([1e5, 4e5, 1.6e6])
    )
    fit = fit_scaling_law(n, p, information, TRUE["pool"])
    assert fit is not None
    assert fit["critical_size_exponent"] == pytest.approx(TRUE["critical_size_exponent"], rel=0.05)
    assert fit["sigma"] == pytest.approx(TRUE["sigma"], rel=0.05)
    assert fit["log_residual_rms"] < 1e-3
    assert fit["num_points"] == 27


def test_fit_scaling_law_returns_none_on_a_small_grid() -> None:
    n, p, information = scaling_law_grid(np.geomspace(1e2, 1e4, 3), np.array([1e5, 4e5]))
    assert fit_scaling_law(n, p, information, TRUE["pool"]) is None


def report_tables() -> dict[str, pd.DataFrame]:
    rows = [
        {
            "case": f"n_{n}_l{depth}",
            "n_samples": str(n),
            "num_layers": depth,
            "method": method,
            "mixture_size": k,
            "model_information_estimate_log_snr": float(k * depth) / n,
            "parameter_count": 100 * depth,
        }
        for n in (10, 100)
        for depth in (2, 4)
        for method in ("distilled_mixture", "classifier_model_free")
        for k in (1, 2)
    ]
    summaries = pd.DataFrame(rows)
    curves = pd.DataFrame(
        [
            {
                **{key: row[key] for key in ("case", "n_samples", "num_layers", "method", "mixture_size")},
                "time": time,
                "model_information_rate_time": 1.0 + time,
                "weight_ess_mean": 1.5,
            }
            for row in rows
            for time in (0.2, 0.8)
        ]
    )
    training = pd.DataFrame(
        [
            {
                "n_samples": str(n),
                "num_layers": depth,
                "model_index": index,
                "step": step,
                # The aggregator always joins num_steps in from cases.csv.
                "num_steps": 2,
                "loss": 1.0 / step,
                "grad_norm": 0.5,
            }
            for n in (10, 100)
            for depth in (2, 4)
            for index in (0, 1)
            for step in (1, 2)
        ]
    )
    memorization = pd.DataFrame(
        [
            {
                "n_samples": str(n),
                "num_layers": depth,
                "encoder": "identity",
                "memorization_ratio": 0.25,
            }
            for n in (10, 100)
            for depth in (2, 4)
        ]
    )
    return {
        "information_summary": summaries,
        "information_curves": curves,
        "training_metrics": training,
        "memorization": memorization,
    }


def test_write_report_writes_tables_and_figures(tmp_path: Path) -> None:
    tables = report_tables()
    write_report(tmp_path, tables, sizes=[1, 2])

    for name, frame in tables.items():
        assert len(pd.read_csv(tmp_path / f"{name}.csv")) == len(frame)
    for figure in (
        "information_vs_n.png",
        "information_vs_mixture_size.png",
        "method_agreement.png",
        "information_rate_curves.png",
        "posterior_weight_ess.png",
        "training_health.png",
        "memorization.png",
    ):
        assert (tmp_path / figure).stat().st_size > 0, figure
