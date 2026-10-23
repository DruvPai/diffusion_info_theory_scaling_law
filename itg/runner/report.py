"""Aggregated tables and figures for a finished grid."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from itg.scaling_law import fit_scaling_law, scaling_law_prediction  # noqa: E402


PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")

DISTILLED_METHOD = "distilled_mixture"



def n_sort_key(value: object) -> float:
    return math.inf if str(value) == "inf" else float(str(value))


def read_case_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_lines(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    out: Path,
    col: str | None = None,
    row: str | None = None,
    unit: str | None = None,
    logx: bool = False,
    logy: bool = False,
    hline: float | None = None,
    hline_label: str = "",
    xsort: bool = True,
    title: str = "",
) -> None:
    """Faceted line chart: one line per hue level (per `unit`), fixed palette."""
    df = df.dropna(subset=[y])
    if not pd.api.types.is_numeric_dtype(df[x]):
        df = df.assign(**{x: df[x].map(n_sort_key)})
    if logx:
        df = df[df[x] != math.inf]
    if df.empty:
        return
    col_levels = sorted(df[col].unique()) if col else [None]
    row_levels = sorted(df[row].unique()) if row else [None]
    hue_levels = sorted(df[hue].unique(), key=n_sort_key)
    colors = {level: PALETTE[idx % len(PALETTE)] for idx, level in enumerate(hue_levels)}
    fig, axes = plt.subplots(
        len(row_levels),
        len(col_levels),
        figsize=(4.2 * len(col_levels), 3.2 * len(row_levels)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for row_idx, row_level in enumerate(row_levels):
        for col_idx, col_level in enumerate(col_levels):
            ax = axes[row_idx][col_idx]
            cell = df
            if row:
                cell = cell[cell[row] == row_level]
            if col:
                cell = cell[cell[col] == col_level]
            for hue_level in hue_levels:
                series = cell[cell[hue] == hue_level]
                if series.empty:
                    continue
                units = series[unit].unique() if unit else [None]
                for unit_idx, unit_level in enumerate(units):
                    line = series[series[unit] == unit_level] if unit else series
                    if xsort:
                        line = line.sort_values(x)
                    ax.plot(
                        line[x],
                        line[y],
                        color=colors[hue_level],
                        linewidth=1.2,
                        alpha=0.9 if len(units) == 1 else 0.45,
                        marker="o" if len(line) <= 12 and len(units) == 1 else None,
                        markersize=3.5,
                        label=str(hue_level) if unit_idx == 0 else None,
                    )
            if hline is not None:
                ax.axhline(hline, color="0.4", linestyle="--", linewidth=1.0, label=hline_label)
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            parts = []
            if row and len(row_levels) > 1:
                parts.append(f"{row}={row_level}")
            if col and len(col_levels) > 1:
                parts.append(f"{col}={col_level}")
            ax.set_title(", ".join(parts), fontsize=9)
            if row_idx == len(row_levels) - 1:
                ax.set_xlabel(x)
            if col_idx == 0:
                ax.set_ylabel(y)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        for ax_row in axes:
            for ax in ax_row:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    break
            if handles:
                break
    if handles:
        fig.legend(handles, labels, title=hue, loc="upper right", fontsize=8, title_fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_report(
    report_dir: Path,
    tables: dict[str, pd.DataFrame],
    sizes: list[int],
    pool_size: float | None = None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if not frame.empty:
            frame.to_csv(report_dir / f"{name}.csv", index=False)

    empty = pd.DataFrame()
    summaries = tables.get("information_summary", empty)
    training = tables.get("training_metrics", empty)
    curves = tables.get("information_curves", empty)
    memorization = tables.get("memorization", empty)

    if not summaries.empty:
        save_lines(
            summaries[summaries.mixture_size == max(sizes)],
            x="n_samples",
            y="model_information_estimate_log_snr",
            hue="num_layers",
            col="method",
            logx=True,
            out=report_dir / "information_vs_n.png",
            title=f"Model information vs dataset size (k={max(sizes)})",
        )
        save_lines(
            summaries,
            x="mixture_size",
            y="model_information_estimate_log_snr",
            hue="num_layers",
            col="method",
            row="n_samples",
            out=report_dir / "information_vs_mixture_size.png",
            title="Model information vs mixture size k",
        )
    if not summaries.empty and "parameter_count" in summaries:
        grid = summaries[summaries.mixture_size == max(sizes)].copy()
        grid["n_value"] = grid.n_samples.map(n_sort_key)
        fits: dict[str, Any] = {}
        for method, group in grid.groupby("method"):
            fit = fit_scaling_law(
                group.n_value,
                group.parameter_count,
                group.model_information_estimate_log_snr,
                pool_size,
            )
            if fit is not None:
                fits[str(method)] = fit
        if fits:
            (report_dir / "scaling_law_fit.json").write_text(
                json.dumps(fits, indent=2, sort_keys=True)
            )
            methods = sorted(fits)
            fig, axes = plt.subplots(
                1, len(methods), figsize=(4.6 * len(methods), 3.6), squeeze=False, sharey=True
            )
            for ax, method in zip(axes[0], methods, strict=True):
                fit = fits[method]
                # inf anchors the fit but has no place on a log axis.
                group = grid[(grid.method == method) & np.isfinite(grid.n_value)]
                p_levels = sorted(group.parameter_count.unique())
                colors = {level: PALETTE[i % len(PALETTE)] for i, level in enumerate(p_levels)}
                grid_n = np.geomspace(group.n_value.min(), group.n_value.max(), 200)
                for p_value in p_levels:
                    points = group[group.parameter_count == p_value]
                    ax.scatter(points.n_value, points.model_information_estimate_log_snr,
                               s=18, color=colors[p_value], label=f"P={int(p_value):,}")
                    ax.plot(grid_n,
                            scaling_law_prediction(fit, grid_n, np.full_like(grid_n, p_value)),
                            color=colors[p_value], linewidth=1.2)
                    ax.axvline(fit["critical_size"][str(int(p_value))], color=colors[p_value],
                               linewidth=0.8, linestyle="--", alpha=0.6)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(
                    f"{method}\na={fit['critical_size_exponent']:.2f} sigma={fit['sigma']:.2f} "
                    f"ell={fit['ell']:.1f}",
                    fontsize=9,
                )
                ax.set_xlabel("N")
                ax.grid(True, alpha=0.25, linewidth=0.5)
                ax.spines[["top", "right"]].set_visible(False)
            axes[0][0].set_ylabel("model information (log-SNR integral)")
            axes[0][0].legend(fontsize=8)
            fig.suptitle("Model-information scaling law (dashed: N_c(P))", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(report_dir / "scaling_law_fit.png", dpi=150)
            plt.close(fig)

    if not summaries.empty and DISTILLED_METHOD in set(summaries.method):
        merged = summaries[summaries.method != DISTILLED_METHOD].merge(
            summaries[summaries.method == DISTILLED_METHOD][
                ["case", "mixture_size", "model_information_estimate_log_snr"]
            ],
            on=["case", "mixture_size"],
            suffixes=("", "_distilled"),
        )
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(4.8, 4.4))
            methods = sorted(merged.method.unique())
            colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
            for method in methods:
                points = merged[merged.method == method]
                ax.scatter(
                    points.model_information_estimate_log_snr_distilled,
                    points.model_information_estimate_log_snr,
                    s=18,
                    color=colors[method],
                    label=method,
                )
            span = (
                min(merged.model_information_estimate_log_snr_distilled.min(),
                    merged.model_information_estimate_log_snr.min()),
                max(merged.model_information_estimate_log_snr_distilled.max(),
                    merged.model_information_estimate_log_snr.max()),
            )
            ax.plot(span, span, color="#888888", linewidth=0.8, linestyle="--")
            ax.set_xlabel("distilled mixture estimate")
            ax.set_ylabel("posterior-weight estimate")
            ax.set_title("Estimator agreement per (case, k)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(report_dir / "method_agreement.png", dpi=150)
            plt.close(fig)

    if not curves.empty:
        save_lines(
            curves[curves.mixture_size == max(sizes)],
            x="time",
            y="model_information_rate_time",
            hue="n_samples",
            col="method",
            row="num_layers",
            logy=True,
            out=report_dir / "information_rate_curves.png",
            title=f"Information rate over diffusion time (k={max(sizes)})",
        )
        if "weight_ess_mean" in curves:
            save_lines(
                curves[curves.mixture_size == max(sizes)],
                x="time",
                y="weight_ess_mean",
                hue="n_samples",
                col="method",
                row="num_layers",
                out=report_dir / "posterior_weight_ess.png",
                title=f"Posterior weight ESS over time (k={max(sizes)}; low = degenerate weights)",
            )
    if not training.empty:
        # Fraction of the run, not raw steps: num_steps varies ~10x across the
        # N axis, so a step axis puts the short cases in the left margin.
        training = training.assign(training_progress=training.step / training.num_steps)
        health = training.melt(
            id_vars=[
                c
                for c in ("n_samples", "num_layers", "model_index", "training_progress")
                if c in training
            ],
            value_vars=[
                c
                for c in ("loss", "grad_norm", "update_norm", "param_norm", "steps_per_sec")
                if c in training
            ],
            var_name="metric",
        )
        save_lines(
            health,
            x="training_progress",
            y="value",
            hue="n_samples",
            col="num_layers",
            row="metric",
            unit="model_index",
            logy=True,
            xsort=False,
            out=report_dir / "training_health.png",
            title="Training health (per model)",
        )
    fid = tables.get("fid", empty)
    if not fid.empty:
        floor = fid.fid_floor.dropna()
        save_lines(
            fid.groupby(["n_samples", "num_layers"], as_index=False).fid.mean(),
            x="n_samples",
            y="fid",
            hue="num_layers",
            logx=True,
            logy=True,
            hline=float(floor.iloc[0]) if not floor.empty else None,
            hline_label="real data, same sample count",
            out=report_dir / "fid.png",
            title="clean-FID vs dataset size (mean over models)",
        )
    if not memorization.empty:
        per_model = (
            memorization[memorization.mixture_size.isna()]
            if "mixture_size" in memorization
            else memorization
        )
        if not per_model.empty:
            save_lines(
                per_model.groupby(["n_samples", "num_layers", "encoder"], as_index=False)
                .memorization_ratio.mean(),
                x="n_samples",
                y="memorization_ratio",
                hue="num_layers",
                col="encoder",
                logx=True,
                out=report_dir / "memorization.png",
                title="Memorization ratio vs dataset size (mean over models)",
            )
