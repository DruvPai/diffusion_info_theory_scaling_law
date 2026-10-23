"""Running one case: train the source models, then the evals over them.

Partial runs come from asking for a subset of `STAGES`, not from a stage
machine: every stage is a function of explicit inputs, and each artifact is
cached behind its own spec sidecar.
"""

from __future__ import annotations

import traceback
from collections.abc import Sequence
from dataclasses import asdict

from itg.runner.artifacts import (
    ModelRef,
    TableRow,
    aux_samples_path,
    read_array,
    read_json,
    samples_path,
    spec_path,
    write_csv,
    write_json,
)
from itg.runner.aux_networks import aux_model_config, train_aux_networks
from itg.runner.case import (
    Case,
    generate_samples,
    load_model,
    prepare_dataset,
    train_source_models,
    write_samples,
    training_loader,
)
from itg.runner.events import log_event, log_to, stage
from itg.runner.information_eval import required_aux_networks, run_information_evals
from itg.runner.references import from_arrays, from_dataset, pool_reference
from itg.runner.sample_metrics import (
    write_fid,
    write_memorization,
    write_sample_figure,
)
from itg.seeds import stable_seed

STAGES = ("train", "samples", "per_model_evals", "information")
PER_MODEL_STAGES = ("train", "samples", "per_model_evals")
# What `EvalConfig.force` (and `--force`) may name.
FORCE_STAGES = (
    "train",
    "samples",
    "sample_metrics",
    "train_mixture",
    "train_classifier",
    "information",
)


def per_model_evals(case: Case, refs: dict[int, ModelRef]) -> None:
    dataset = case.dataset()
    for idx in sorted(refs):
        sample_file = samples_path(case.directory, idx, "eval")
        samples = read_array(sample_file)
        sample_spec = read_json(spec_path(sample_file))
        tags = {"model_index": idx}
        write_sample_figure(case, sample_file, samples, sample_spec)
        ids = training_loader(case, dataset, idx).ids
        write_memorization(
            case,
            sample_file,
            samples,
            from_dataset(dataset, ids) if ids is not None else None,
            sample_spec=sample_spec,
            train_spec=None,
            tags=tags,
        )
        log_event("per_model_eval:finish", model_index=idx)


def score_mixture_train_fid(case: Case) -> None:
    """FID on the large per-model sets; the 256-sample eval sets are far too few."""
    reference = pool_reference(
        case,
        count=case.evals.fid_reference_samples,
        seed=stable_seed(case.seeds.base_seed, "fid_reference"),
    )
    for idx in case.evals.mixture_train_models:
        sample_file = samples_path(case.directory, idx, "mixture_train")
        write_fid(
            case,
            sample_file,
            read_array(sample_file),
            reference,
            generated_by="model",
            sample_spec=read_json(spec_path(sample_file)),
            tags={"model_index": idx},
        )


def mixture_samples(case: Case, aux: dict[int, ModelRef]) -> None:
    """Sample each distillation model and score the draws like a source model."""
    evals = case.evals
    # One seed across sizes k on purpose: common random numbers make the
    # k-dependence of the sample metrics directly comparable.
    seed = case.seeds.sample_seed("mixture_eval")
    for k, ref in sorted(aux.items()):
        path = aux_samples_path(case.directory, "mixture", k)
        write_samples(
            case,
            path,
            lambda k=k, ref=ref: load_model(
                case, ref, aux_model_config(case, "mixture", k)
            ),
            seed=seed,
            count=evals.eval_samples_per_model,
            identity={
                "role": "mixture_eval",
                "mixture_size": k,
                "upstream": ref.spec(),
            },
        )
        samples = read_array(path)
        sample_spec = read_json(spec_path(path))
        train_indices = list(evals.mixture_train_models[:k])
        tags = {"mixture_size": k}
        write_sample_figure(case, path, samples, sample_spec)
        write_memorization(
            case,
            path,
            samples,
            from_arrays(
                [
                    read_array(samples_path(case.directory, j, "mixture_train"))
                    for j in train_indices
                ]
            ),
            sample_spec=sample_spec,
            train_spec={"kind": "mixture_train_samples", "mixture_size": k},
            tags=tags,
        )


def run_case(
    case: Case, stages: Sequence[str] = STAGES, only: tuple[int, ...] | None = None
) -> TableRow:
    """Run the requested stages and return the case's headline summary row."""
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; valid: {list(STAGES)}")
    unknown = sorted(case.evals.force - set(FORCE_STAGES))
    if unknown:
        raise ValueError(
            f"unknown force stage(s) {unknown}; valid: {list(FORCE_STAGES)}"
        )
    case.directory.mkdir(parents=True, exist_ok=True)
    with log_to(case.directory / "experiment.log"):
        try:
            log_event("case:start", case=case.name, stages=list(stages))
            write_json(
                case.directory / "config.json",
                {
                    "case": case.fields,
                    "config": asdict(case.config),
                    "evals": {**asdict(case.evals), "force": sorted(case.evals.force)},
                },
            )
            with stage(case.directory, "prepare_dataset"):
                prepare_dataset(case)
            with stage(case.directory, "train"):
                refs, metrics = train_source_models(case, only)
                write_csv(case.directory / "training_metrics.csv", metrics)
            if "samples" in stages:
                with stage(case.directory, "samples"):
                    generate_samples(case, "eval", refs)
            if "per_model_evals" in stages:
                with stage(case.directory, "per_model_evals"):
                    per_model_evals(case, refs)

            summary: TableRow = {"case": case.name, **case.fields}
            if "information" in stages and case.evals.information_methods:
                kinds = sorted(required_aux_networks(case.evals.information_methods))
                with stage(case.directory, "mixture_train_samples"):
                    generate_samples(case, "mixture_train", refs)
                    score_mixture_train_fid(case)
                aux: dict[str, dict[int, ModelRef]] = {}
                for kind in kinds:
                    with stage(case.directory, f"train_{kind}"):
                        aux[kind], aux_metrics = train_aux_networks(case, kind)
                        metrics.extend(aux_metrics)
                write_csv(case.directory / "training_metrics.csv", metrics)
                if "mixture" in aux:
                    with stage(case.directory, "mixture_samples"):
                        mixture_samples(case, aux["mixture"])
                with stage(case.directory, "information"):
                    summaries = run_information_evals(case, refs, aux)
                primary = [
                    row
                    for row in summaries
                    if row["method"] == case.evals.information_methods[0]
                ]
                if primary:
                    summary = max(primary, key=lambda row: int(row["mixture_size"]))
            log_event("case:finish", case=case.name)
            return summary
        except BaseException as error:
            log_event(
                "case:error",
                case=case.name,
                exception=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
            raise
