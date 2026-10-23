"""The model-information scaling law of the paper (Eq. 14), from the linear-Gaussian model.

    I(P, N) = F(P) + m(P) E_z[d_z I_ell(N / d_z)],
    F(P) = c0 + c1 P~^-beta,  m(P) = k P~^-b,  d_z = N_c(P) e^{sigma z},
    N_c(P) = N_c(P_ref) P~^a,  P~ = P / P_ref,

where I_ell is the per-dimension model information of the linear-Gaussian
model as rho -> 0, with ell = log(1/rho) / 2 nats per memorized direction.
On grids whose largest N is close to the pool, 1/N becomes the
finite-population 1/N - 1/N_pool (`finite_pool`).
A fit is a plain dict of these nine parameters plus `P_ref`, `pool`, and
`finite_pool`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Bound on |beta|; at the bound the floor is a step, not a power law.
FLOOR_EXPONENT_MAX = 3.0

# Gauss-Hermite nodes for the average over z ~ N(0, 1).
SPREAD_NODES, SPREAD_WEIGHTS = np.polynomial.hermite_e.hermegauss(31)
SPREAD_WEIGHTS = SPREAD_WEIGHTS / SPREAD_WEIGHTS.sum()

Fit = dict[str, Any]


def linear_gaussian_information(ratio: np.ndarray, ell: float) -> np.ndarray:
    """Per-dimension model information of the linear-Gaussian model as rho -> 0.

    `ratio` is d / N = 1 / eta. Below N = d each empty direction carries `ell`
    nats; above it the Marchenko--Pastur log-determinant leaves
    ratio / 4 + O(ratio^2).
    """
    c = np.asarray(ratio, dtype=np.float64)
    over = np.clip(c, 1e-3, 1 - 1e-12)
    above = np.where(
        c < 1e-3, c / 4 + c**2 / 12 + c**3 / 24, 0.5 * (1 + (1 / over - 1) * np.log1p(-over))
    )
    eta = 1 / np.maximum(c, 1.0)
    below = ell * (1 - eta) + 0.5 * (
        eta * np.log(eta) + eta + (1 - eta) * np.log1p(-np.minimum(eta, 1 - 1e-16))
    )
    return np.where(c <= 1.0, above, below)


def _log_p(fit: Fit, p: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(p, dtype=np.float64) / fit["P_ref"])


def floor(fit: Fit, p: np.ndarray) -> np.ndarray:
    """F(P), the estimate when every model trains on the same data."""
    return fit["floor_constant"] + fit["floor_scale"] * np.exp(-fit["floor_exponent"] * _log_p(fit, p))


def critical_size(fit: Fit, p: np.ndarray) -> np.ndarray:
    """N_c(P), the median group size: where a typical group stops memorizing."""
    return fit["critical_size_ref"] * np.exp(fit["critical_size_exponent"] * _log_p(fit, p))


def total_dimension(fit: Fit, p: np.ndarray) -> np.ndarray:
    """d_tot(P), the summed group size; the plateau is d_tot(P) * ell."""
    groups = fit["group_count_ref"] * np.exp(-fit["group_count_exponent"] * _log_p(fit, p))
    return groups * critical_size(fit, p) * math.exp(fit["sigma"] ** 2 / 2)


def excess(fit: Fit, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The law minus its floor."""
    n, p = np.asarray(n, dtype=np.float64), np.asarray(p, dtype=np.float64)
    # Subsets drawn from one pool fluctuate as 1/N - 1/pool, so N = pool is exact.
    pool = fit["pool"] if fit.get("finite_pool", True) else math.inf
    inverse_n = np.clip(1 / n - 1 / pool, 0.0, None)
    sizes = critical_size(fit, p)[:, None] * np.exp(fit["sigma"] * SPREAD_NODES)
    per_group = sizes * linear_gaussian_information(sizes * inverse_n[:, None], fit["ell"])
    groups = fit["group_count_ref"] * np.exp(-fit["group_count_exponent"] * _log_p(fit, p))
    return groups * (per_group @ SPREAD_WEIGHTS)


def scaling_law_prediction(fit: Fit, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    return floor(fit, p) + excess(fit, n, p)


# (name, fitted in log space, lower, upper) for the nine parameters, in order.
PARAMETERS = (
    ("group_count_ref", True, -20, 30),
    ("group_count_exponent", False, -5, 5),
    ("critical_size_ref", True, -5, 25),
    ("critical_size_exponent", False, -3, 3),
    ("sigma", True, -8, 2),
    ("ell", True, -5, 5),
    ("floor_constant", True, -20, 20),
    ("floor_scale", True, -20, 20),
    ("floor_exponent", False, -FLOOR_EXPONENT_MAX, FLOOR_EXPONENT_MAX),
)


def fit_scaling_law(
    n_column: pd.Series,
    p_column: pd.Series,
    information_column: pd.Series,
    pool_size: float | None = None,
    *,
    include_pool_runs: bool = False,
    finite_pool: bool = True,
    fixed: Mapping[str, float] | None = None,
) -> Fit | None:
    """Least squares in log-information over the grid points with N < pool.

    Runs on the whole pool (N = pool, nan, or inf) only check the fitted floor,
    unless `include_pool_runs`. `finite_pool` applies the finite-population
    correction 1/N - 1/pool. `fixed` holds named parameters at given values.
    """
    fixed = dict(fixed or {})
    n = np.asarray(n_column, dtype=np.float64)
    p = np.asarray(p_column, dtype=np.float64)
    information = np.asarray(information_column, dtype=np.float64)
    pool = math.inf if pool_size is None else float(pool_size)
    keep = np.isfinite(p) & np.isfinite(information) & (information > 0)
    n, p, information = np.nan_to_num(n[keep], nan=pool, posinf=pool), p[keep], information[keep]
    body = n < pool
    fitted = np.ones_like(body) if include_pool_runs else body
    if len(np.unique(n[body])) < 4 or body.sum() < 10:
        return None
    p_ref = float(np.median(np.unique(p)))
    free = [param for param in PARAMETERS if param[0] not in fixed]

    def unpack(theta: np.ndarray) -> Fit:
        values = {
            name: math.exp(value) if log else float(value)
            for (name, log, _, _), value in zip(free, theta, strict=True)
        }
        return {**values, **fixed, "P_ref": p_ref, "pool": pool, "finite_pool": finite_pool}

    def residuals(theta: np.ndarray) -> np.ndarray:
        prediction = scaling_law_prediction(unpack(theta), n, p)
        return np.log(np.maximum(prediction, 1e-12)) - np.log(information)

    # The floor starts below the smallest fitted estimate.
    scale = float(information[fitted & ~body].min() if (fitted & ~body).any() else information[body].min())
    best = None
    for log_nc in np.log(np.median(n[body])) + np.array([-1.0, 1.0]):
        for log_k in (0.0, 3.0):
            for sigma in (0.2, 0.8):
                for beta in (0.3, 1.5):
                    start = dict(zip(
                        (name for name, *_ in PARAMETERS),
                        (log_k, 0.3, log_nc, 0.3, math.log(sigma), math.log(10.0),
                         math.log(0.6 * scale), math.log(0.3 * scale), beta),
                        strict=True,
                    ))
                    fit = least_squares(
                        lambda theta: residuals(theta)[fitted],
                        x0=np.array([start[name] for name, *_ in free]),
                        bounds=([low for *_, low, _ in free], [high for *_, high in free]),
                    )
                    if best is None or fit.cost < best.cost:
                        best = fit
    assert best is not None
    fit = unpack(best.x)
    sizes = np.array(sorted(set(p)))

    def per_size(values: np.ndarray) -> dict[str, float]:
        return {str(int(s)): float(v) for s, v in zip(sizes, values, strict=True)}

    residual = residuals(best.x)
    return {
        **fit,
        "critical_size": per_size(critical_size(fit, sizes)),
        "total_dimension": per_size(total_dimension(fit, sizes)),
        "plateau": per_size(total_dimension(fit, sizes) * fit["ell"]),
        "floor": per_size(floor(fit, sizes)),
        "floor_measured": {
            str(int(s)): float(information[~body & (p == s)].mean())
            for s in sizes if (~body & (p == s)).any()
        },
        "floor_exponent_at_bound": abs(fit["floor_exponent"]) >= FLOOR_EXPONENT_MAX - 1e-6,
        "log_residual_rms": float(np.sqrt(np.mean(residual[body] ** 2))),
        "log_residual_rms_floor": float(np.sqrt(np.mean(residual[~body] ** 2)))
        if (~body).any() else None,
        "num_points": int(body.sum()),
    }
