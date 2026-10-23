"""Model information from the eval-source / mixture score gap.

Every method answers the same question and differs only in how it
approximates the k-source mixture score `s_K` at the probes:

* `distilled_mixture`: the trained distillation model's own score.
* `classifier_model_based`: posterior weights `W_j(x, t)` from a
  time-conditioned k-way classifier, applied to the train source models'
  own scores.
* `classifier_model_free`: the same weights from per-model log-densities
  computed with the I-MMSE identity, softmaxed.

The rate is `(g^2/2) E||s_eval - s_hat||^2` for all of them; the
posterior-weight methods additionally report the anchored pairwise form
`(g^2/4) E sum_j W_j ||s_eval - s_j||^2` and weight diagnostics.

One pass over the log-SNR grid serves every method, so all curves come from
the same forward-noised probes with identical noise — which is what makes
the cheap estimators comparable against the explicit one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from itg.evals.immse import (
    GaussianReference,
    fit_gaussian_reference,
    immse_log_density,
    quadrature,
)
from itg.evals.information import (
    InformationEvalGrid,
    cumulative_trapezoid,
    information_rate,
    make_log_snr_eval_grid,
    mean_and_se,
    model_score,
)
from itg.runner.artifacts import (
    JsonDict,
    ModelRef,
    TableRow,
    draw_indices,
    information_curve_path,
    information_summaries_path,
    normalize_spec,
    read_array,
    read_json,
    samples_path,
    spec_path,
    write_csv,
    write_json,
)
from itg.runner.aux_networks import aux_model_config
from itg.runner.case import Case, load_model
from itg.runner.events import log_event
from itg.seeds import generator, stable_seed


@dataclass(frozen=True)
class MethodRequirements:
    """`aux_network` is the per-k network the method reads (None = trains
    nothing); `source_scores` marks methods that build `s_K` as `sum_j W_j s_j`."""

    aux_network: str | None
    source_scores: bool


METHOD_REQUIREMENTS = {
    "distilled_mixture": MethodRequirements("mixture", False),
    "classifier_model_based": MethodRequirements("classifier", True),
    "classifier_model_free": MethodRequirements(None, True),
}

INFORMATION_METHODS = tuple(METHOD_REQUIREMENTS)


def validated_methods(methods: Sequence[str]) -> tuple[str, ...]:
    unknown = sorted(set(methods) - set(METHOD_REQUIREMENTS))
    if unknown:
        raise ValueError(f"unknown information method(s) {unknown}")
    return tuple(methods)


def required_aux_networks(methods: Sequence[str]) -> set[str]:
    kinds = (METHOD_REQUIREMENTS[method].aux_network for method in validated_methods(methods))
    return {kind for kind in kinds if kind is not None}


@dataclass
class Probes:
    """Held-out eval source models with their own eval samples as probes.

    `x0` concatenates every eval model's samples; `bounds` delimits each
    model's block so a model scores only its own probes.
    """

    models: list[nn.Module]
    x0: Tensor
    bounds: list[tuple[int, int]]
    sidecars: list[JsonDict]


def load_probes(case: Case, refs: dict[int, ModelRef], indices: list[int]) -> Probes:
    paths = [samples_path(case.directory, idx, "eval") for idx in indices]
    batches = [read_array(path) for path in paths]
    ends = np.cumsum([batch.shape[0] for batch in batches])
    return Probes(
        models=[load_model(case, refs[idx]) for idx in indices],
        x0=torch.cat(batches).to(case.device),
        bounds=[(int(end - batch.shape[0]), int(end)) for batch, end in zip(batches, ends, strict=True)],
        sidecars=[read_json(spec_path(path)) for path in paths],
    )


def probe_scores(probes: Probes, case: Case, x_t: Tensor, t: Tensor) -> Tensor:
    """Each eval source model's score on its own probe block, concatenated."""
    return torch.cat(
        [
            model_score(model, case.process, x_t[start:end], t[start:end], case.evals.eval_batch_size)
            for model, (start, end) in zip(probes.models, probes.bounds, strict=True)
        ]
    )


def information_rates(
    eval_scores: Tensor,
    g_squared: float,
    *,
    mixture_scores: Tensor | None = None,
    source_scores: list[Tensor] | None = None,
    weights: Tensor | None = None,
) -> dict[str, float]:
    """One method's rate(s) at one time point.

    Pass `mixture_scores` directly, or `source_scores` with simplex `weights`
    (P, k) to build it as `sum_j W_j s_j`; the weighted form also yields the
    pairwise rate and weight diagnostics.
    """
    stacked = (
        None if source_scores is None else torch.stack([s.flatten(1) for s in source_scores], dim=1)
    )
    if mixture_scores is None:
        assert stacked is not None and weights is not None
        mixture_scores = torch.einsum("pk,pkd->pd", weights, stacked)
    mean, se, _ = information_rate(eval_scores, mixture_scores, g_squared)
    rates = {"rate": mean, "rate_se": se}
    if weights is None or stacked is None:
        return rates
    pairwise = (eval_scores.flatten(1)[:, None, :] - stacked).pow(2).sum(dim=-1)
    per_probe = (g_squared / 4.0) * (weights * pairwise).sum(dim=-1)
    rate_pairwise, rate_pairwise_se = mean_and_se(per_probe)
    ess = 1.0 / weights.pow(2).sum(dim=-1).clamp(min=1e-12)
    return {
        **rates,
        "rate_pairwise": rate_pairwise,
        "rate_pairwise_se": rate_pairwise_se,
        "weight_ess_mean": float(ess.mean()),
        "weight_max_mean": float(weights.max(dim=-1).values.mean()),
    }


def curve_tables(
    rows_by_time: list[dict[str, float]], grid: InformationEvalGrid, case_row: TableRow
) -> tuple[list[TableRow], TableRow]:
    """Curve rows in descending time with cumulative integrals, plus a summary."""
    order = np.argsort(grid.time.numpy())[::-1]
    times = grid.time.numpy()[order]
    log_snr = grid.log_snr.numpy()[order]
    g_squared = grid.g_squared.numpy()[order]
    weight = grid.positive_log_snr_weight.numpy()[order]
    sorted_rows = [rows_by_time[idx] for idx in order]

    rates = np.asarray([row["rate"] for row in sorted_rows])
    cumulative_time = cumulative_trapezoid(rates, times)
    cumulative_log_snr = cumulative_trapezoid(rates * weight, log_snr)
    weighted = bool(sorted_rows) and "rate_pairwise" in sorted_rows[0]
    if weighted:
        rates_pairwise = np.asarray([row["rate_pairwise"] for row in sorted_rows])
        cumulative_pairwise = cumulative_trapezoid(rates_pairwise, times)

    rows: list[TableRow] = []
    for idx, source in enumerate(sorted_rows):
        row: TableRow = {
            **case_row,
            "time": float(times[idx]),
            "log_snr": float(log_snr[idx]),
            "g_squared": float(g_squared[idx]),
            "model_information_rate_time": float(rates[idx]),
            "model_information_rate_se": source["rate_se"],
            "model_information_estimate_time": float(cumulative_time[idx]),
            "model_information_estimate_log_snr": float(cumulative_log_snr[idx]),
        }
        if weighted:
            row |= {
                "model_information_rate_pairwise_time": float(rates_pairwise[idx]),
                "model_information_rate_pairwise_se": source["rate_pairwise_se"],
                "model_information_estimate_pairwise_time": float(cumulative_pairwise[idx]),
                "weight_ess_mean": source["weight_ess_mean"],
                "weight_max_mean": source["weight_max_mean"],
            }
        rows.append(row)

    summary: TableRow = {
        **case_row,
        "model_information_estimate_time": float(cumulative_time[-1]),
        "model_information_estimate_log_snr": float(cumulative_log_snr[-1]),
        "mean_model_information_rate_time": float(rates.mean()),
        "max_model_information_rate_time": float(rates.max()),
        "num_eval_times": int(times.shape[0]),
    }
    if weighted:
        summary["model_information_estimate_pairwise_time"] = float(cumulative_pairwise[-1])
    return rows, summary


def run_information_evals(
    case: Case, refs: dict[int, ModelRef], aux: dict[str, dict[int, ModelRef]]
) -> list[TableRow]:
    evals = case.evals
    methods = validated_methods(evals.information_methods)
    sizes = evals.sizes
    if not methods:
        return []
    train_indices = list(evals.mixture_train_models[: max(sizes)])
    eval_indices = list(evals.mixture_eval_models)
    if not eval_indices:
        raise ValueError("model information evaluation requires mixture_eval_models")

    curve_files = {(m, k): information_curve_path(case.directory, m, k) for m in methods for k in sizes}
    summaries_file = information_summaries_path(case.directory)
    probes = load_probes(case, refs, eval_indices)
    needs_sources = any(METHOD_REQUIREMENTS[m].source_scores for m in methods)

    upstream: JsonDict = {"probes": probes.sidecars}
    for kind in sorted(required_aux_networks(methods)):
        upstream[kind] = {str(k): aux[kind][k].spec() for k in sizes}
    if needs_sources:
        upstream["train_source_models"] = {str(j): refs[j].spec() for j in train_indices}
    spec = normalize_spec(
        {
            "kind": "model_information",
            "methods": sorted(methods),
            "eval_num_times": evals.eval_num_times,
            "t_min": evals.t_min,
            "t_max": evals.t_max,
            "mixture_sizes": sizes,
            "mixture_train_models": train_indices,
            "mixture_eval_models": eval_indices,
            "case_eval_seed": case.seeds.case_eval_seed,
            "immse": (
                {
                    "seed": case.seeds.immse_seed,
                    "num_quadrature_nodes": evals.immse_num_quadrature_nodes,
                    "mc_samples": evals.immse_mc_samples,
                    "num_probes": evals.immse_num_probes,
                    "gamma_min": evals.immse_gamma_min,
                    "gamma_max": evals.immse_gamma_max,
                }
                if "classifier_model_free" in methods
                else None
            ),
            "upstream": upstream,
        }
    )
    if (
        not case.forced("information")
        and summaries_file.exists()
        and all(path.exists() for path in curve_files.values())
        and spec_path(summaries_file).exists()
        and read_json(spec_path(summaries_file)) == spec
    ):
        log_event("information_evals:cache_hit")
        return read_json(summaries_file)["summaries"]

    log_event("information_evals:start", methods=list(methods))
    grid = make_log_snr_eval_grid(
        case.process, num_times=evals.eval_num_times, t_min=evals.t_min, t_max=evals.t_max
    )
    source_models = [load_model(case, refs[j]) for j in train_indices] if needs_sources else []
    mixtures = {
        k: load_model(case, aux["mixture"][k], aux_model_config(case, "mixture", k))
        for k in (sizes if "distilled_mixture" in methods else [])
    }
    classifiers = {
        k: load_model(case, aux["classifier"][k], aux_model_config(case, "classifier", k), k)
        for k in (sizes if "classifier_model_based" in methods else [])
    }

    immse_setup = _immse_setup(case, train_indices, probes) if "classifier_model_free" in methods else None
    rows_by_time: dict[tuple[str, int], list[dict[str, float]]] = {key: [] for key in curve_files}
    batch_size = evals.eval_batch_size

    for time_idx in range(evals.eval_num_times):
        t_value = float(grid.time[time_idx])
        g_squared = float(grid.g_squared[time_idx])
        t = torch.full((probes.x0.shape[0],), t_value, device=case.device)
        # One seed per time point, shared by every method and every k.
        eps = torch.randn(
            probes.x0.shape,
            generator=generator(stable_seed(case.seeds.case_eval_seed, "time", time_idx)),
        ).to(case.device)
        x_t = case.process.forward(probes.x0, eps, t)
        eval_scores = probe_scores(probes, case, x_t, t)
        source_scores = [
            model_score(model, case.process, x_t, t, batch_size) for model in source_models
        ]

        for k in sizes:
            if "distilled_mixture" in methods:
                rows_by_time[("distilled_mixture", k)].append(
                    information_rates(
                        eval_scores,
                        g_squared,
                        mixture_scores=model_score(mixtures[k], case.process, x_t, t, batch_size),
                    )
                )
            if "classifier_model_based" in methods:
                with torch.no_grad():
                    logits = torch.cat(
                        [
                            classifiers[k](x_t[a : a + batch_size], t[a : a + batch_size])
                            for a in range(0, x_t.shape[0], batch_size)
                        ]
                    )
                rows_by_time[("classifier_model_based", k)].append(
                    information_rates(
                        eval_scores,
                        g_squared,
                        source_scores=source_scores[:k],
                        weights=torch.softmax(logits, dim=-1),
                    )
                )

        if immse_setup is not None:
            subset = immse_setup.indices
            x_sub = x_t if subset is None else x_t[subset]
            eval_sub = eval_scores if subset is None else eval_scores[subset]
            source_sub = source_scores if subset is None else [s[subset] for s in source_scores]
            log_densities = torch.stack(
                [
                    immse_log_density(
                        case.process,
                        lambda model, y, s: model_score(model, case.process, y, s, batch_size),
                        source_models[pos],
                        immse_setup.references[pos],
                        x_sub.flatten(1),
                        t_value,
                        gammas=immse_setup.gammas,
                        weights=immse_setup.weights,
                        seed=stable_seed(case.seeds.immse_seed, "time", time_idx),
                        mc_samples=evals.immse_mc_samples,
                        data_shape=case.config.data_shape,
                    )
                    for pos in range(len(train_indices))
                ],
                dim=1,
            )
            for k in sizes:
                rows_by_time[("classifier_model_free", k)].append(
                    information_rates(
                        eval_sub,
                        g_squared,
                        source_scores=source_sub[:k],
                        weights=torch.softmax(log_densities[:, :k], dim=-1),
                    )
                )
        log_event("information_eval_time:finish", time_index=time_idx, t=t_value)

    summaries: list[TableRow] = []
    for method in methods:
        for k in sizes:
            case_row: TableRow = {
                "case": case.name,
                "method": method,
                "mixture_size": k,
                "train_model_indices": ",".join(str(i) for i in train_indices[:k]),
                "eval_model_indices": ",".join(str(i) for i in eval_indices),
            }
            rows, summary = curve_tables(rows_by_time[(method, k)], grid, case_row)
            summaries.append(summary)
            write_csv(curve_files[(method, k)], rows)
    write_json(summaries_file, {"summaries": summaries})
    write_json(spec_path(summaries_file), spec)
    log_event("information_evals:finish")
    return summaries


@dataclass(frozen=True)
class ImmseSetup:
    """Quadrature nodes, per-model Gaussian references, and the probe subset."""

    indices: Tensor | None
    gammas: np.ndarray
    weights: np.ndarray
    references: list[GaussianReference]


def _immse_setup(case: Case, train_indices: list[int], probes: Probes) -> ImmseSetup:
    evals = case.evals
    num_probes = int(probes.x0.shape[0])
    indices = None
    if evals.immse_num_probes is not None and evals.immse_num_probes < num_probes:
        indices = torch.from_numpy(
            draw_indices(case.seeds.immse_seed, pool_size=num_probes, count=evals.immse_num_probes)
        ).to(case.device)
    gammas, weights = quadrature(
        evals.immse_num_quadrature_nodes, evals.immse_gamma_min, evals.immse_gamma_max
    )
    references = [
        fit_gaussian_reference(read_array(samples_path(case.directory, j, "mixture_train")))
        for j in train_indices
    ]
    log_event(
        "information_evals:immse_setup",
        num_probes=num_probes if indices is None else int(indices.shape[0]),
        num_quadrature_nodes=evals.immse_num_quadrature_nodes,
        mc_samples=evals.immse_mc_samples,
    )
    return ImmseSetup(indices, gammas, weights, references)
