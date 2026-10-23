"""Training and evaluation configuration.

`Config` is everything that determines a trained checkpoint (given the data
manifest); `EvalConfig` is everything downstream of training. Only `image_root` is
excluded from the weight fingerprint, so moving a dataset on disk does not
invalidate checkpoints.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal


SamplingRegime = Literal["streaming", "iid", "epoch_shuffle"]
InformationMethod = Literal[
    "distilled_mixture", "classifier_model_based", "classifier_model_free"
]


@dataclass(frozen=True)
class EDMConfig:
    """Karras schedule and EDM UNet."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    # SongUNet trunk: base width times these multipliers, one per resolution.
    unet_channel_mult: tuple[int, ...] = (1, 2, 2)
    unet_num_blocks: int = 2
    unet_attn_resolutions: tuple[int, ...] = (16,)


@dataclass(frozen=True)
class Config:
    # --- data ---------------------------------------------------------
    image_root: str
    regime: SamplingRegime = "epoch_shuffle"
    dataset_size: int | None = 4096
    image_size: int = 32
    image_random_horizontal_flip: bool = False

    # --- model --------------------------------------------------------
    hidden_dim: int = 128
    num_layers: int = 4
    dropout_p: float = 0.0
    # 1.3-1.7x on the train step for 4-7s of compile. Compiled runs are
    # bit-identical to each other but not to eager ones (~1e-6), which the
    # weight fingerprint below covers: flipping this invalidates checkpoints
    # rather than resuming across it.
    compile_model: bool = True
    # logSNR ~ N(mean, std).
    time_logsnr_mean: float = 2.4
    time_logsnr_std: float = 2.4

    # --- optimization -------------------------------------------------
    batch_size: int = 128
    # Gradient-accumulation microbatch. Activation memory scales with this;
    # gradients are averaged so the update is unchanged.
    microbatch_size: int | None = None
    num_steps: int = 10_000
    lr: float = 3e-4
    lr_warmup_fraction: float = 0.05
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    optimizer_adam_lr: float | None = None
    weight_decay: float = 0.0
    muon_lr_multiplier: float = 10.0
    muon_beta: float = 0.95
    muon_ns_steps: int = 5
    mup_init_sigma: float = 1.0
    # The table's base block multiplier. Not absorbable into mup_init_sigma and
    # lr: the hidden row multiplies a whole residual branch, and rescaling the
    # branch's weights scales its output by that factor per matrix, not once.
    mup_multiplier: float = 1.0
    # Depth-muP: scale each residual branch by 1 / (branch count).
    mup_residual_depth_scaling: bool = True
    residual_zero_init: bool = True
    ema_decay: float = 0.999

    edm: EDMConfig = field(default_factory=EDMConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout_p must satisfy 0 <= p < 1")
        if self.microbatch_size is not None and (
            self.microbatch_size < 1 or self.batch_size % self.microbatch_size
        ):
            raise ValueError("microbatch_size must be positive and divide batch_size")
        if self.regime != "streaming" and self.dataset_size is None:
            raise ValueError(f"{self.regime} regime requires a finite dataset_size")
        if self.mup_init_sigma <= 0:
            raise ValueError("mup_init_sigma must be positive")
        if self.num_steps <= 0 or self.batch_size <= 0:
            raise ValueError("num_steps and batch_size must be positive")
        if not 0.0 <= self.lr_warmup_fraction < 1.0:
            raise ValueError("lr_warmup_fraction must satisfy 0 <= f < 1")

    @property
    def data_shape(self) -> tuple[int, ...]:
        """Shape of one RGB sample."""
        return (3, self.image_size, self.image_size)

    def weight_fingerprint(self) -> str:
        """sha256 of every field that determines the trained weights."""
        payload = {
            name: value
            for name, value in dataclasses.asdict(self).items()
            if name != "image_root"
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()


def config_from_dict(payload: dict[str, object]) -> Config:
    """Rebuild a `Config` from a written `config.json` (JSON has no tuples)."""

    def fields(values: dict[str, object]) -> dict[str, object]:
        return {
            name: tuple(value) if isinstance(value, list) else value
            for name, value in values.items()
        }

    data = dict(payload)
    edm = data.pop("edm", {})
    # A parse boundary: the splat cannot be statically typed, and Config's
    # own validation is what checks it.
    return Config(**fields(data), edm=EDMConfig(**fields(edm)))  # ty: ignore[invalid-argument-type]


@dataclass(frozen=True)
class EvalConfig:
    """Everything downstream of training, for one case.

    Source models are split by role: a mixture of size k is trained on samples
    of `mixture_train_models[:k]`, and every `mixture_eval_models` entry is
    held out to evaluate it.
    """

    mixture_train_models: tuple[int, ...] = (0, 1, 2, 3, 4)
    mixture_eval_models: tuple[int, ...] = (5, 6, 7, 8, 9)
    seed: int = 0

    sample_steps: int = 200
    # 1 = DDIM, 2 = Heun.
    sample_order: int = 1
    eval_batch_size: int = 32

    eval_num_times: int = 25
    eval_samples_per_model: int = 64
    t_min: float = 0.01
    t_max: float = 0.99

    mixture_sizes: tuple[int, ...] = (2, 3, 4, 5)
    mixture_samples_per_model: int = 512
    mixture_steps: int | None = None
    mixture_batch_size: int | None = None
    mixture_lr: float | None = None
    mixture_num_layers: int | None = None
    # Width per layer for the auxiliary networks; None re-derives it from the
    # case config.
    mixture_model_aspect_ratio: float | None = None

    # Estimators to run, in preference order: the first supplies the case's
    # headline summary row.
    information_methods: tuple[InformationMethod, ...] = (
        "distilled_mixture",
        "classifier_model_based",
        "classifier_model_free",
    )
    # clean-FID reference draw; FID needs far more than the 2048 feature dim.
    fid_reference_samples: int = 50_000
    classifier_steps: int | None = None
    classifier_batch_size: int | None = None
    classifier_lr: float | None = None
    classifier_num_layers: int | None = None

    # I-MMSE knobs: cost per case scales as probes x times x nodes x mc x k.
    immse_num_quadrature_nodes: int = 16
    immse_mc_samples: int = 4
    immse_num_probes: int | None = 64
    immse_gamma_min: float = 1e-4
    immse_gamma_max: float = 1e4


    log_interval: int = 10
    checkpoint_interval: int = 1_000
    # Stage names to recompute unconditionally; see `runner.pipeline.STAGES`.
    force: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        train, evals = self.mixture_train_models, self.mixture_eval_models
        if not train:
            raise ValueError("at least one mixture_train_models index is required")
        if len(set(train)) != len(train) or len(set(evals)) != len(evals):
            raise ValueError("model index lists must not contain duplicates")
        if any(idx < 0 for idx in (*train, *evals)):
            raise ValueError("model indices must be non-negative")
        if set(train) & set(evals):
            raise ValueError("mixture_eval_models must be disjoint from mixture_train_models")
        if not self.mixture_sizes:
            raise ValueError("at least one mixture size is required")
        if any(k < 1 or k > len(train) for k in self.mixture_sizes):
            raise ValueError("mixture_sizes must be in [1, len(mixture_train_models)]")
        if not 0.0 < self.t_min < self.t_max < 1.0:
            raise ValueError("expected 0 < t_min < t_max < 1")
        if self.eval_num_times < 2:
            raise ValueError("eval_num_times must be at least 2")

    @property
    def source_model_indices(self) -> tuple[int, ...]:
        return tuple(sorted({*self.mixture_train_models, *self.mixture_eval_models}))

    @property
    def sizes(self) -> list[int]:
        return sorted(int(k) for k in self.mixture_sizes)
