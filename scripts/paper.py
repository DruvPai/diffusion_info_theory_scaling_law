"""Every figure CSV and every number in the paper, from the grids' reports.

    uv run python scripts/paper.py ../paper/figures/pgfplots/data

Reads `experiments/<grid>/report/*.csv` (written by each grid script's
`--report-only`), writes the pgfplots CSVs into the given directory, and prints
the numbers quoted in the text and tables, section by section.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares, minimize
from scipy.stats import pearsonr, spearmanr

from itg.experiments import SIGMA_DATA
from itg.processes import Process
from itg.runner.report import n_sort_key
from itg.scaling_law import (
    Fit,
    critical_size,
    excess,
    fit_scaling_law,
    floor,
    scaling_law_prediction,
    total_dimension,
)

EXPERIMENTS = Path("experiments")
# name: (experiment directory, pool size, input dimension, sigma_data key,
# finite-population correction). The correction 1/N - 1/pool is applied on the
# grids that train on the whole pool; ImageNet's grid stops at 8% of it.
GRIDS = {
    "imagenet64": ("imagenet_edm_64", 1_271_167, 12_288, "imagenet", False),
    "cifar10": ("cifar10_edm", 50_000, 3_072, "cifar10", True),
    "fashion_mnist": ("fashion_mnist_edm", 60_000, 3_072, "fashion_mnist", True),
    "ffhq": ("ffhq_edm", 60_000, 12_288, "ffhq", True),
}
DEPTH = {1_118_531: 1, 6_654_019: 2, 19_893_507: 3, 44_124_035: 4, 82_632_643: 5}
ESTIMATE = "model_information_estimate_log_snr"
# The 250,000-image run at depth 5, outside the ImageNet grid.
OUTSIDE_RUN = EXPERIMENTS / "imagenet_edm_64" / "n_250000_h160_l5_edm_img64"
# sigma at the estimator's t_min = 0.05 on the EDM schedule.
SIGMA_T_MIN = float(Process(0.002, 80.0, 7.0).sigma(torch.tensor([0.05]))[0])


# --- data -------------------------------------------------------------------


def model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of trained models; rows without a model index are mixture draws."""
    frame = frame[frame.model_index.notna()]
    if "generated_by" in frame:
        frame = frame[frame.generated_by == "model"]
    if "encoder" in frame:
        frame = frame[frame.encoder == "identity"]
    return frame


def with_n(frame: pd.DataFrame, pool: int) -> pd.DataFrame:
    n = frame.n_samples.map(n_sort_key).fillna(pool).replace(np.inf, pool)
    return frame.assign(N=n, P=frame.parameter_count)


def grid_summary(name: str) -> pd.DataFrame:
    """Model-free estimates at the largest bank, every row."""
    experiment, pool, *_ = GRIDS[name]
    summary = pd.read_csv(EXPERIMENTS / experiment / "report" / "information_summary.csv")
    summary = summary[summary.method == "classifier_model_free"]
    return with_n(summary, pool)


def conditions(name: str) -> pd.DataFrame:
    """One row per (P, N): model information, memorization ratio, FID."""
    experiment, pool, *_ = GRIDS[name]
    report = EXPERIMENTS / experiment / "report"
    summary = grid_summary(name)
    summary = summary[summary.mixture_size == summary.mixture_size.max()]
    columns = [summary.groupby(["P", "N"])[ESTIMATE].mean().rename("info")]
    for file, value, label in (("memorization.csv", "memorization_ratio", "mem"), ("fid.csv", "fid", "fid")):
        frame = with_n(model_rows(pd.read_csv(report / file)), pool)
        columns.append(frame.groupby(["P", "N"])[value].mean().rename(label))
    return pd.concat(columns, axis=1).reset_index()


def law_fit(
    frame: pd.DataFrame, pool: int | None, *, finite_pool: bool = True,
    include_pool_runs: bool = False, fixed: dict[str, float] | None = None,
) -> Fit:
    fit = fit_scaling_law(
        frame.N, frame.P, frame["info"], pool, include_pool_runs=include_pool_runs,
        finite_pool=finite_pool, fixed=fixed,
    )
    assert fit is not None
    return fit


def rms(values: object) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def log_error(fit: Fit, frame: pd.DataFrame) -> float:
    return rms(np.log(scaling_law_prediction(fit, frame.N, frame.P) / frame["info"]))


def plateau(fit: Fit, p: object) -> np.ndarray:
    return total_dimension(fit, np.asarray(p, dtype=np.float64)) * fit["ell"]


def memorization_map(excess_values: object, plateaus: object, ratios: object) -> Callable:
    """Logistic in log(excess / plateau), fitted by maximum likelihood."""
    x = np.log(np.asarray(excess_values) / np.asarray(plateaus))
    m = np.asarray(ratios)

    def nll(w: np.ndarray) -> float:
        prob = np.clip(1 / (1 + np.exp(-(w[0] + w[1] * x))), 1e-9, 1 - 1e-9)
        return -float(np.sum(m * np.log(prob) + (1 - m) * np.log(1 - prob)))

    w = minimize(nll, [0, 1], method="Nelder-Mead",
                 options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10}).x
    return lambda e, pl: 1 / (1 + np.exp(-(w[0] + w[1] * np.log(np.asarray(e) / np.asarray(pl)))))


def fid_design(x: object, p: object) -> np.ndarray:
    """Columns of Eq. (fid_map): log FID = alpha + beta log x + gamma log P."""
    x, p = np.asarray(x, dtype=np.float64), np.asarray(p, dtype=np.float64)
    return np.c_[np.ones(len(x)), np.log(x), np.log(p)]


def generalization_regime(fit: Fit, frame: pd.DataFrame) -> pd.DataFrame:
    """N >= e^{2 sigma} N_c(P): the N^-1 regime where the FID map applies."""
    return frame[frame.N >= np.exp(2 * fit["sigma"]) * critical_size(fit, frame.P)]


# --- figures ----------------------------------------------------------------


def series(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean and standard error across the trained models of each condition."""
    grouped = frame[frame.model_index.notna()].groupby("x")[value]
    return pd.DataFrame({"mean": grouped.mean(), "se": grouped.sem().fillna(0.0)}).reset_index()


def grid_figures(name: str, fit: Fit, out: Path) -> None:
    """Law curves, measurements, memorization, FID, and bank ratios, at x = N / N_c(P)."""
    experiment, pool, *_ = GRIDS[name]
    report = EXPERIMENTS / experiment / "report"

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.assign(n_samples=frame.n_samples.map(n_sort_key))
        frame = frame[np.isfinite(frame.n_samples)]
        return frame.assign(x=frame.n_samples / critical_size(fit, frame.parameter_count))

    information = normalized(pd.read_csv(report / "information_summary.csv"))
    information = information[information.method == "classifier_model_free"]
    top = information.mixture_size.max()
    for (depth, k), part in information.groupby(["num_layers", "mixture_size"]):
        part = part.sort_values("x")
        pd.DataFrame({"x": part.x, "observed": part[ESTIMATE]}).to_csv(
            out / f"{name}_information_l{depth}_k{k}.csv", index=False
        )
        if k < top:
            reference = information[(information.mixture_size == top) & (information.num_layers == depth)]
            ratio = part[ESTIMATE].to_numpy() / reference.set_index(
                "x"
            )[ESTIMATE].reindex(part.x).to_numpy()
            pd.DataFrame({"x": part.x, "ratio": ratio}).to_csv(
                out / f"{name}_kratio_l{depth}_k{k}.csv", index=False
            )

    memorization = normalized(pd.read_csv(report / "memorization.csv"))
    for depth, part in memorization[memorization.encoder == "identity"].groupby("num_layers"):
        series(part, "memorization_ratio").to_csv(out / f"{name}_memorization_pixel_l{depth}.csv", index=False)
    for depth, part in normalized(pd.read_csv(report / "fid.csv")).groupby("num_layers"):
        series(part, "fid").assign(floor=part.fid_floor.iloc[0]).to_csv(
            out / f"{name}_fid_l{depth}.csv", index=False
        )

    for depth, part in information[information.mixture_size == top].groupby("num_layers"):
        part = part.sort_values("n_samples")
        p_value = float(part.parameter_count.iloc[0])
        grid = np.geomspace(8.0, pool * 0.9995, 500)
        law = scaling_law_prediction(fit, grid, np.full_like(grid, p_value))
        law_floor = float(floor(fit, np.array([p_value]))[0])
        pd.DataFrame({
            "n": grid, "x": grid / critical_size(fit, np.array([p_value]))[0],
            "fit": law, "excess": law - law_floor,
        }).to_csv(out / f"{name}_lawfit_l{depth}.csv", index=False)
        pd.DataFrame({
            "n": part.n_samples, "x": part.x, "observed": part[ESTIMATE],
            "excess": (part[ESTIMATE] - law_floor).clip(lower=0.05),
        }).to_csv(out / f"{name}_raw_l{depth}.csv", index=False)


def budget_figures(fit: Fit, out: Path) -> None:
    """Fashion-MNIST at K_train = K_val = k, against the ten-model grid."""
    budget = pd.concat(
        pd.read_csv(path)
        for path in sorted((EXPERIMENTS / "fashion_mnist_matched").glob("k*/report/information_summary.csv"))
    )
    budget = budget[budget.method == "classifier_model_free"]
    budget = budget.assign(x=budget.n_samples / critical_size(fit, budget.parameter_count))
    top = budget.mixture_size.max()
    reference = budget[budget.mixture_size == top].set_index(["num_layers", "x"])[ESTIMATE]
    for (depth, k), part in budget.groupby(["num_layers", "mixture_size"]):
        part = part.sort_values("x")
        pd.DataFrame({"x": part.x, "observed": part[ESTIMATE]}).to_csv(
            out / f"fashion_mnist_budget_l{depth}_m{2 * k}.csv", index=False
        )
        if k < top:
            ratio = part[ESTIMATE].to_numpy() / reference.loc[depth].reindex(part.x).to_numpy()
            pd.DataFrame({"x": part.x, "ratio": ratio}).to_csv(
                out / f"fashion_mnist_budgetratio_l{depth}_m{2 * k}.csv", index=False
            )


def correlation_figures(fit: Fit, frame: pd.DataFrame, pool: int, out: Path) -> None:
    """ImageNet: memorization against relative model information, FID against the excess."""
    body = frame[frame.N < pool]
    relative = body["info"] / plateau(fit, body.P)
    memorization = memorization_map(body["info"], plateau(fit, body.P), body.mem)
    curve = np.geomspace(1e-3, 1.5, 200)
    pd.DataFrame({"rel": curve, "mem": memorization(curve, 1.0)}).to_csv(out / "imagenet64_corr_memcurve.csv", index=False)
    regime = generalization_regime(fit, body).assign(
        exc=lambda f: excess(fit, f.N, f.P)
    )
    w = np.linalg.lstsq(fid_design(regime.exc, regime.P), np.log(regime.fid), rcond=None)[0]
    for p_value, part in body.assign(rel=relative).groupby("P"):
        depth = DEPTH[int(p_value)]
        part[["N", "info", "rel", "mem", "fid"]].to_csv(out / f"imagenet64_corr_l{depth}.csv", index=False)
        tail = regime[regime.P == p_value]
        tail[["N", "info", "exc", "fid"]].to_csv(out / f"imagenet64_corr_regime_l{depth}.csv", index=False)
        line = np.geomspace(tail.exc.min(), tail.exc.max(), 40)
        pd.DataFrame({"exc": line, "fid": np.exp(fid_design(line, np.full_like(line, p_value)) @ w)}).to_csv(
            out / f"imagenet64_corr_fidline_l{depth}.csv", index=False
        )


def withheld_size_maps(
    frame: pd.DataFrame, pool: int, train: pd.DataFrame, finite_pool: bool
) -> tuple[Fit, Callable, np.ndarray]:
    """The law, the memorization map, and the FID map, fitted on `train`."""
    fit = law_fit(train, pool, finite_pool=finite_pool)
    body = train[train.N < pool]
    memorization = memorization_map(excess(fit, body.N, body.P), plateau(fit, body.P), body.mem)
    regime = generalization_regime(fit, body)
    w = np.linalg.lstsq(fid_design(excess(fit, regime.N, regime.P), regime.P), np.log(regime.fid), rcond=None)[0]
    return fit, memorization, w


def outside_run() -> dict[str, float]:
    """Model information, memorization, and FID of the 250,000-image run."""
    summary = pd.read_json(OUTSIDE_RUN / "model_information_summaries.json")
    rows = pd.DataFrame(list(summary.summaries))
    rows = rows[(rows.method == "classifier_model_free") & (rows.mixture_size == rows.mixture_size.max())]
    memorization = pd.concat(pd.read_csv(path) for path in OUTSIDE_RUN.glob("samples/model_*/eval/memorization.csv"))
    fid = pd.concat(pd.read_csv(path) for path in OUTSIDE_RUN.glob("samples/model_*/mixture_train/fid.csv"))
    return {
        "n": 250_000,
        "info": float(rows[ESTIMATE].mean()),
        "mem": float(model_rows(memorization).memorization_ratio.mean()),
        "fid": float(model_rows(fid).fid.mean()),
    }


def extrapolation_figures(name: str, frame: pd.DataFrame, pool: int, out: Path) -> None:
    """Predictions at the largest model size from fits to the others."""
    largest = frame.P.max()
    fit, memorization, w = withheld_size_maps(frame, pool, frame[frame.P != largest], GRIDS[name][4])
    grid = np.exp(np.linspace(math.log(8), math.log(min(pool * 0.9, 3e5)), 300))
    threshold = np.exp(2 * fit["sigma"])
    for p_value in sorted(frame.P.unique()):
        depth = DEPTH[int(p_value)]
        sizes = np.full_like(grid, p_value)
        e = excess(fit, grid, sizes)
        pd.DataFrame({
            "n": grid, "info": scaling_law_prediction(fit, grid, sizes),
            "mem": memorization(e, plateau(fit, [p_value])[0]),
        }).to_csv(out / f"{name}_extrap_law_l{depth}.csv", index=False)
        regime = grid >= threshold * critical_size(fit, np.array([p_value]))[0]
        pd.DataFrame({"n": grid[regime], "fid": np.exp(fid_design(e[regime], sizes[regime]) @ w)}).to_csv(
            out / f"{name}_extrap_fid_l{depth}.csv", index=False
        )
        measured = frame[(frame.P == p_value) & (frame.N < pool)]
        measured[["N", "info", "mem", "fid"]].rename(columns={"N": "n"}).to_csv(
            out / f"{name}_extrap_obs_l{depth}.csv", index=False
        )
    if name == "imagenet64":
        pd.DataFrame([outside_run()]).to_csv(out / "imagenet64_extrap_250k.csv", index=False)


# --- the sigmoid alternatives of the functional-form comparison --------------


def sigmoid_law(kind: str, theta: np.ndarray, n: np.ndarray, p: np.ndarray, p_ref: float, n_ref: float, pool: int) -> np.ndarray:
    """Eq. (appendix_sigmoid_ramp) for "ramp", Eq. (appendix_sigmoid_power) for "power"."""
    # Written term by term as fitted for the paper: the floor is nearly flat
    # in (c0, c1, beta), so reordering the float operations moves the optimum.
    log_p = np.log(p / p_ref)
    law_floor = np.exp(theta[0]) + np.exp(theta[1]) * np.exp(-theta[2] * log_p)
    if kind == "ramp":
        log_a, log_c, delta, s, log_lambda, gamma = theta[3:]
        critical = n_ref * np.exp(log_lambda) * np.exp(gamma * log_p)
        ramp = np.clip(np.log(pool / n) / np.log(pool / critical), 0, 1)
        return law_floor + ramp * (np.exp(log_c) * np.exp(-delta * log_p) / (1 + (n / critical) ** s) + np.exp(log_a))
    log_c, delta, s, log_lambda, gamma, log_b, alpha = theta[3:]
    critical = n_ref * np.exp(log_lambda) * np.exp(gamma * log_p)
    inverse_n = np.clip(1 / n - 1 / pool, 0, None)
    return law_floor + np.exp(log_c) * np.exp(-delta * log_p) / (1 + (n / critical) ** s) + np.exp(log_b) * (n_ref * inverse_n) ** alpha


def sigmoid_fit(kind: str, frame: pd.DataFrame, pool: int) -> dict:
    body = frame[frame.N < pool]
    n, p, info = body.N.to_numpy(float), body.P.to_numpy(float), body["info"].to_numpy()
    p_ref, n_ref, scale = float(np.median(np.unique(p))), float(np.median(np.unique(n))), info.min()
    if kind == "ramp":
        starts = [[math.log(.6 * scale), math.log(.3 * scale), b, la, math.log(2e4), .1, s, ll, .3]
                  for b in (.3, 1.5) for la in (math.log(scale), math.log(5 * scale)) for s in (1., 3.) for ll in (-1., 1.)]
        low, high = [-20, -20, -3, -20, -20, -5, .05, -10, -3], [20, 20, 3, 20, 20, 5, 20, 10, 3]
    else:
        starts = [[math.log(.6 * scale), math.log(.3 * scale), b, math.log(2e4), .1, s, ll, .3, lb, al]
                  for b in (.3, 1.5) for lb in (math.log(scale), math.log(10 * scale)) for s in (1., 3.)
                  for ll in (-1., 1.) for al in (.5, 1.)]
        low, high = [-20, -20, -3, -20, -5, .05, -10, -3, -20, .01], [20, 20, 3, 20, 5, 20, 10, 3, 20, 5]

    def residuals(theta: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(sigmoid_law(kind, theta, n, p, p_ref, n_ref, pool), 1e-12)) - np.log(info)

    best = min(
        (least_squares(residuals, np.clip(x0, np.array(low) + 1e-6, np.array(high) - 1e-6), bounds=(low, high))
         for x0 in starts),
        key=lambda result: result.cost,
    )
    return {"kind": kind, "theta": best.x, "p_ref": p_ref, "n_ref": n_ref, "pool": pool}


def law_comparison(name: str, frame: pd.DataFrame, pool: int) -> None:
    """Table appendix_law_comparison: log RMSE, BIC, floor, and two withheld tests."""
    body, runs_on_pool = frame[frame.N < pool], frame[frame.N >= pool]
    largest = frame.P.max()
    median_n = np.median(np.unique(body.N))

    def fit_form(kind: str, train: pd.DataFrame) -> tuple[Callable, Callable]:
        """The fitted law and its floor F(P)."""
        if kind == "MP":
            fit = law_fit(train, pool, finite_pool=GRIDS[name][4])
            return lambda f: scaling_law_prediction(fit, f.N, f.P), lambda f: floor(fit, f.P)
        fit = sigmoid_fit(kind, train, pool)
        theta = fit["theta"]
        return (
            lambda f: sigmoid_law(kind, theta, f.N.to_numpy(float), f.P.to_numpy(float), fit["p_ref"], fit["n_ref"], pool),
            lambda f: np.exp(theta[0]) + np.exp(theta[1]) * (f.P.to_numpy(float) / fit["p_ref"]) ** -theta[2],
        )

    for kind, params in (("MP", 9), ("ramp", 9), ("power", 10)):
        predict, law_floor = fit_form(kind, frame)
        residual = np.log(predict(body) / body["info"])
        bic = len(residual) * math.log(np.sum(residual**2) / len(residual)) + params * math.log(len(residual))
        floor_error = rms(np.log(law_floor(runs_on_pool) / runs_on_pool["info"]))
        withheld_p = body[body.P == largest]
        held_p = rms(np.log(fit_form(kind, frame[frame.P != largest])[0](withheld_p) / withheld_p["info"]))
        line = (f"  {kind:<6} rms {rms(residual):.3f} BIC {bic:.1f} floor {floor_error:.3f} "
                f"largest P {held_p:.3f} / {rms(residual[body.P == largest]):.3f}")
        train_n = body[body.N <= median_n]
        if train_n.N.nunique() >= 4:
            withheld_n = body[body.N > median_n]
            held_n = rms(np.log(fit_form(kind, train_n)[0](withheld_n) / withheld_n["info"]))
            line += f"  large N {held_n:.3f} / {rms(residual[body.N > median_n]):.3f}"
        if kind == "power" and name == "imagenet64":
            line += f"  alpha {sigmoid_fit(kind, frame, pool)['theta'][-1]:.2f}"
        print(line)


# --- numbers ----------------------------------------------------------------


def coefficients(name: str, fit: Fit, frame: pd.DataFrame, pool: int) -> None:
    """Tables appendix_scaling_law_coefficients, appendix_critical_sample_sizes, and the plateau."""
    _, _, input_dim, data_key, finite_pool = GRIDS[name]
    # Over the whole grid, the runs on the whole pool included.
    measured = frame.dropna()
    correlation = pearsonr(measured["info"], measured.mem)[0]
    print(f"  k {fit['group_count_ref']:.3g}  b {fit['group_count_exponent']:.3f}  "
          f"N_c(P_ref) {fit['critical_size_ref']:.0f}  a {fit['critical_size_exponent']:.3f}  "
          f"sigma {fit['sigma']:.3f}  ell {fit['ell']:.2f}  c0 {fit['floor_constant']:.1f}  "
          f"c1 {fit['floor_scale']:.3g}  beta {fit['floor_exponent']:.2f}  P_ref {fit['P_ref']:.2g}  "
          f"points {fit['num_points']}  rms {fit['log_residual_rms']:.3f}  floor {fit['log_residual_rms_floor']:.3f}")
    description_length = input_dim * math.log(SIGMA_DATA[data_key] / SIGMA_T_MIN)
    print(f"  plateau Lambda = {description_length:.3g}; info-mem Pearson {correlation:.3f}")
    for key in sorted(fit["critical_size"], key=int):
        width = 32 * DEPTH[int(key)]
        print(f"    L={DEPTH[int(key)]}  N_c {fit['critical_size'][key]:.0f}  N_c/h {fit['critical_size'][key] / width:.2f}  "
              f"d_tot {fit['total_dimension'][key]:.0f} ({fit['total_dimension'][key] / input_dim:.2f} of input)  "
              f"plateau {fit['plateau'][key]:.0f} ({fit['plateau'][key] / description_length:.2f} Lambda)  "
              f"F {fit['floor'][key]:.1f}  measured {fit['floor_measured'][key]:.1f}  "
              f"floor/plateau {100 * fit['floor_measured'][key] / fit['plateau'][key]:.1f}%")
    fixed_ell = law_fit(
        frame, pool, finite_pool=finite_pool, fixed={"ell": math.log(SIGMA_DATA[data_key] / SIGMA_T_MIN)}
    )
    print(f"  with ell fixed at log(sigma_data / sigma_tmin) = {fixed_ell['ell']:.2f}: rms {fixed_ell['log_residual_rms']:.3f}")


def imagenet_details(fit: Fit, frame: pd.DataFrame, pool: int) -> None:
    """Appendix sub:appendix_imagenet_details and Table imagenet_fit."""
    body = frame[frame.N < pool]
    sigma_zero = law_fit(frame, pool, finite_pool=False, fixed={"sigma": 1e-6})
    corrected = law_fit(frame, pool, finite_pool=True)
    print(f"  sigma = 0: rms {sigma_zero['log_residual_rms']:.3f}; "
          f"1/N - 1/N_pool in place of 1/N: rms {corrected['log_residual_rms']:.3f}")
    for p_value, part in body.groupby("P"):
        part = part.sort_values("N")
        key, n_c = str(int(p_value)), critical_size(fit, np.array([p_value]))[0]
        measured_floor = fit["floor_measured"][key]
        tail = part[(part.N >= 4000) & (part.N <= 40_000)]
        slope = np.polyfit(np.log(tail.N), np.log(tail["info"] - measured_floor), 1)[0]
        m, n = part.mem.to_numpy(), part.N.to_numpy()
        below = np.where(m < m[0] / 2)[0][0]
        half = math.exp(np.interp(m[0] / 2, [m[below], m[below - 1]], [math.log(n[below]), math.log(n[below - 1])]))
        past = part[part.N >= part.N[part.mem < 0.5].min()]
        peak = past.N[past.fid.idxmax()]
        print(f"    L={DEPTH[int(p_value)]}  N_c {n_c:.0f}  d_tot/d {fit['total_dimension'][key] / 12_288:.2f}  "
              f"F {fit['floor'][key]:.0f}  measured {measured_floor:.0f}  tail {slope:.2f}  "
              f"mem half {half / n_c:.2f}  first <1% {n[m < 0.01].min() / n_c:.2f}  "
              f"FID peak {peak / n_c:.2f}  mem at N=10 {m[0]:.2f}")
    memorization = memorization_map(body["info"], plateau(fit, body.P), body.mem)
    regime = generalization_regime(fit, body)
    within = [spearmanr(part["info"], part.fid)[0] for _, part in regime.groupby("P")]
    print(f"  Pearson info-mem {pearsonr(body['info'], body.mem)[0]:.2f}; logistic MAE "
          f"{np.abs(memorization(body['info'], plateau(fit, body.P)) - body.mem).mean():.3f}; "
          f"{len(regime)} conditions in the N^-1 regime; min within-size Spearman {min(within):.2f}")
    for label, x in (("excess E", excess(fit, regime.N, regime.P)), ("measured I", regime["info"])):
        design, y = fid_design(x, regime.P), np.log(regime.fid)
        w = np.linalg.lstsq(design, y, rcond=None)[0]
        r2 = 1 - np.var(y - design @ w) / np.var(y)
        print(f"  FID map on {label}: R^2 {r2:.2f}, beta {w[1]:.2f}, gamma {w[2]:.2f}")
    measured_excess = regime["info"] - floor(fit, regime.P)
    print(f"  measured I minus fitted floor at N=100000: {measured_excess[regime.N == 100_000].round(1).tolist()}")


def bank_size(frame: pd.DataFrame, pool: int) -> None:
    """ImageNet with K_train = 1 against 2, on the same evaluated models."""
    summary = grid_summary("imagenet64")
    estimates = summary.groupby(["P", "N", "mixture_size"])[ESTIMATE].mean().unstack()
    ratio = (estimates[1] / estimates[2])[estimates.index.get_level_values("N") < pool]
    print(f"  K=1 / K=2: median {ratio.median():.3f}, {100 * (ratio <= 1.05).mean():.0f}% at most 1.05, "
          f"largest {ratio.max():.2f} at (P, N) = {ratio.idxmax()}")
    fits = {k: law_fit(estimates[k].rename("info").reset_index(), pool, finite_pool=False) for k in (1, 2)}
    sizes = np.array(sorted(frame.P.unique()))
    change = np.abs(critical_size(fits[1], sizes) / critical_size(fits[2], sizes) - 1).max()
    print(f"  refit: N_c within {100 * change:.1f}%, a {fits[1]['critical_size_exponent']:.3f} vs "
          f"{fits[2]['critical_size_exponent']:.3f}, sigma {fits[1]['sigma']:.3f} vs {fits[2]['sigma']:.3f}")


def withheld_size(name: str, frame: pd.DataFrame, pool: int, pooled: list[pd.DataFrame]) -> None:
    """Table appendix_extrapolation, and Table imagenet_prediction for ImageNet."""
    largest = frame.P.max()
    test = frame[(frame.P == largest) & (frame.N < pool)]
    full_fit = law_fit(frame, pool, finite_pool=GRIDS[name][4])
    fit, memorization, w = withheld_size_maps(frame, pool, frame[frame.P != largest], GRIDS[name][4])
    train = frame[(frame.P != largest) & (frame.N < pool)]
    predicted_mem = memorization(excess(fit, test.N, test.P), plateau(fit, test.P))
    train_mem = memorization(excess(fit, train.N, train.P), plateau(fit, train.P))
    train_regime, test_regime = generalization_regime(fit, train), generalization_regime(fit, test)

    def fid_error(part: pd.DataFrame) -> float:
        return rms(fid_design(excess(fit, part.N, part.P), part.P) @ w - np.log(part.fid))

    print(f"  information {log_error(fit, test):.3f} / {log_error(full_fit, test):.3f}  "
          f"memorization MAE {np.abs(predicted_mem - test.mem).mean():.3f} / {np.abs(train_mem - train.mem).mean():.3f}  "
          f"Pearson {pearsonr(predicted_mem, test.mem)[0]:.3f}  FID {fid_error(test_regime):.3f} / "
          f"{fid_error(train_regime):.3f} ({len(test_regime)})")
    pooled.append(pd.DataFrame({"mem": test.mem, "pred": predicted_mem}))
    if name != "imagenet64":
        return
    threshold = np.exp(2 * fit["sigma"]) * critical_size(fit, np.array([largest]))[0]
    print(f"  N_c at L=5 {critical_size(fit, np.array([largest]))[0]:.0f} (all sizes "
          f"{critical_size(full_fit, np.array([largest]))[0]:.0f}); FID threshold {threshold:.0f}")
    every_n = train
    w_all = np.linalg.lstsq(fid_design(excess(fit, every_n.N, every_n.P), every_n.P), np.log(every_n.fid), rcond=None)[0]
    print(f"  FID map fitted on every N: {rms(fid_design(excess(fit, test.N, test.P), test.P) @ w_all - np.log(test.fid)):.3f}")
    w_n = np.linalg.lstsq(fid_design(train_regime.N, train_regime.P), np.log(train_regime.fid), rcond=None)[0]
    print(f"  regression on log N and log P: {rms(fid_design(test_regime.N, test_regime.P) @ w_n - np.log(test_regime.fid)):.3f}")
    outside = outside_run()
    rows = pd.concat([test, pd.DataFrame([{**outside, "N": outside["n"], "P": largest}])])
    e = excess(fit, rows.N, rows.P)
    fid = np.where(rows.N >= threshold, np.exp(fid_design(e, rows.P) @ w), np.nan)
    for (_, row), info, mem, f in zip(rows.iterrows(), scaling_law_prediction(fit, rows.N, rows.P),
                                       memorization(e, plateau(fit, rows.P)), fid, strict=True):
        print(f"    N={row.N:>7.0f}  information {info:,.0f} / {row['info']:,.0f}  "
              f"memorization {mem:.3f} / {row.mem:.3f}  FID {f:.1f} / {row.fid:.1f}")
    print(f"  250k with the fit to all sizes: {scaling_law_prediction(full_fit, np.array([250_000.]), np.array([largest]))[0]:.1f}")


def bank_sweeps() -> None:
    """Tables appendix_bank_sensitivity and appendix_budget_sensitivity (pool runs fitted)."""
    def refit(label: str, part: pd.DataFrame, pool: int) -> None:
        fit = law_fit(with_n(part, pool).rename(columns={ESTIMATE: "info"}), pool, include_pool_runs=True)
        print(f"  {label}: a {fit['critical_size_exponent']:.3f}  sigma {fit['sigma']:.3f}  ell {fit['ell']:.2f}  "
              f"b {fit['group_count_exponent']:.3f}  plateau {plateau(fit, [fit['P_ref']])[0]:.4g}  "
              f"rms {fit['log_residual_rms']:.4f}  N_c {[round(v) for v in fit['critical_size'].values()]}")

    for name in ("cifar10", "ffhq", "fashion_mnist"):
        experiment, pool, *_ = GRIDS[name]
        summary = pd.read_csv(EXPERIMENTS / experiment / "report" / "information_summary.csv")
        for (method, k), part in summary.groupby(["method", "mixture_size"]):
            refit(f"{name} {method} K={k}", part, pool)
    for path in sorted((EXPERIMENTS / "fashion_mnist_matched").glob("k*/report/information_summary.csv")):
        summary = pd.read_csv(path)
        summary = summary[(summary.method == "classifier_model_free") & (summary.mixture_size == summary.mixture_size.max())]
        refit(f"budget {2 * summary.mixture_size.max()} models", summary, 60_000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="the paper's figures/pgfplots/data directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pooled: list[pd.DataFrame] = []
    for name, (_, pool, *_, finite_pool) in GRIDS.items():
        frame = conditions(name)
        fit = law_fit(frame, pool, finite_pool=finite_pool)
        grid_figures(name, fit, args.out)
        extrapolation_figures(name, frame, pool, args.out)
        print(f"\n== {name}: coefficients")
        coefficients(name, fit, frame, pool)
        print(f"== {name}: predictions at the withheld largest model size")
        withheld_size(name, frame, pool, pooled)
        print(f"== {name}: functional forms (ramp = sigmoid-log, power = sigmoid-power)")
        law_comparison(name, frame, pool)
        if name == "imagenet64":
            correlation_figures(fit, frame, pool, args.out)
            print("== imagenet64: details")
            imagenet_details(fit, frame, pool)
            bank_size(frame, pool)
        if name == "fashion_mnist":
            budget_figures(fit, args.out)
    pooled_frame = pd.concat(pooled)
    print(f"\n== all grids, withheld largest size: memorization Pearson "
          f"{pearsonr(pooled_frame.pred, pooled_frame.mem)[0]:.3f}, MAE "
          f"{np.abs(pooled_frame.pred - pooled_frame.mem).mean():.3f} over {len(pooled_frame)} conditions")
    print("\n== bank-size and budget refits")
    bank_sweeps()


if __name__ == "__main__":
    main()
