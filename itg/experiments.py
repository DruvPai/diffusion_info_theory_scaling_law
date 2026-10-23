"""Building blocks for the grid scripts under `scripts/`.

Every experiment is an `(N, depth)` grid with the model width tied to depth.
There is no registry: each script builds its own `Experiment` from these and
calls `itg.runner.cli.run`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from itg.config import Config, EDMConfig, EvalConfig
from itg.optim import half_life_ema_decay

# Prepared by scripts/prepare_image_datasets.py: <root>/<dataset>/{train,val}.
DATA_ROOT = Path(os.environ.get("ITG_DATA_ROOT", "~/data"))


@dataclass(frozen=True)
class Experiment:
    name: str
    output_dir: Path
    base: Config
    evals: EvalConfig
    n_values: tuple[int | None, ...]
    depths: tuple[int, ...] = (2, 4, 8)
    aspect_ratio: float = 32.0
    case_suffix: str = ""
    # Steps = max(min_steps, num_epochs epochs) instead of base.num_steps.
    epoch_scaled: bool = True
    min_steps: int = 10_000
    num_epochs: int = 200
    # Finite size whose step budget the streaming case copies. None follows the
    # largest finite size, which rebudgets and so retrains it -- and moves the
    # bias anchor -- every time the grid grows past its current maximum.
    streaming_steps_from: int | None = None
    max_batch_size: int = 128
    # EMA horizon as a fraction of the run rather than a fixed decay.
    ema_half_life: bool = False
    # UNet grids put the P axis in both width and depth: base channels follow
    # aspect_ratio as usual, residual blocks per level follow depth directly.
    unet_blocks_follow_depth: bool = False

    def hidden_dim(self, depth: int) -> int:
        return max(1, round(depth * self.aspect_ratio))

    def case_name(self, n: int | None, depth: int) -> str:
        label = "n_inf" if n is None else f"n_{n}"
        return f"{label}_h{self.hidden_dim(depth)}_l{depth}{self.case_suffix}"

    def case_config(self, n: int | None, depth: int) -> Config:
        hidden = self.hidden_dim(depth)
        config = replace(
            self.base,
            dataset_size=n,
            regime="streaming" if n is None else self.base.regime,
            hidden_dim=hidden,
            num_layers=depth,
        )
        if self.epoch_scaled:
            # Streaming trains as long as one named finite case, so the two
            # differ in data alone.
            if n is not None:
                finite = n
            elif self.streaming_steps_from is not None:
                finite = self.streaming_steps_from
            else:
                finite = max((v for v in self.n_values if v is not None), default=None)
            if finite is None:
                raise ValueError(f"{self.name} requires one finite dataset size")
            # epoch_shuffle drops the short final batch, so an epoch is
            # N // batch steps (at least 1, since batch <= n).
            batch = min(self.max_batch_size, finite)
            config = replace(
                config,
                batch_size=batch,
                num_steps=max(self.min_steps, self.num_epochs * max(1, finite // batch)),
            )
        if self.unet_blocks_follow_depth:
            config = replace(config, edm=replace(config.edm, unet_num_blocks=depth))
        if self.ema_half_life:
            config = replace(config, ema_decay=half_life_ema_decay(config.num_steps))
        return config

    def case_fields(self, n: int | None, depth: int) -> dict[str, object]:
        config = self.case_config(n, depth)
        return {
            "experiment": self.name,
            "n_samples": "inf" if n is None else str(n),
            "regime": config.regime,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "model_aspect_ratio": self.aspect_ratio,
            "image_size": config.image_size,
            "dropout_p": config.dropout_p,
            "batch_size": config.batch_size,
            "num_steps": config.num_steps,
            "ema_decay": config.ema_decay,
        }

    def cases(self) -> list[tuple[str, int | None, int]]:
        return [
            (self.case_name(n, depth), n, depth)
            for n in self.n_values
            for depth in self.depths
        ]

    def scheduling_order(self) -> list[str]:
        """Longest-first (steps x params) so big cases do not land at the tail."""
        return [
            name
            for name, _n, _depth in sorted(
                self.cases(),
                key=lambda case: -(
                    self.case_config(case[1], case[2]).num_steps
                    * case[2]
                    * self.hidden_dim(case[2]) ** 2
                ),
            )
        ]


def image_root(name: str, split: str) -> str:
    return str((DATA_ROOT / name / split).expanduser())


# sqrt(E[x^2]) per train split; the preconditioner assumes zero-mean data.
SIGMA_DATA = {"fashion_mnist": 0.80, "cifar10": 0.52, "ffhq": 0.55, "imagenet": 0.55}


def edm_base(dataset: str, *, image_size: int) -> Config:
    """Unconditional DDPM++ UNet on the Karras schedule."""
    return Config(
        image_root=image_root(dataset, "train"),
        image_size=image_size,
        edm=EDMConfig(
            sigma_data=SIGMA_DATA[dataset],
            unet_channel_mult=(1, 2, 2),
            unet_num_blocks=1,
            unet_attn_resolutions=(16,),
        ),
    )


# Five bank and five held-out models, all three estimators (Fashion-MNIST).
ALL_ESTIMATORS_EVALS = EvalConfig(
    mixture_train_models=(0, 1, 2, 3, 4),
    mixture_eval_models=(5, 6, 7, 8, 9),
    mixture_sizes=(2, 3, 4, 5),
    eval_samples_per_model=256,
    mixture_samples_per_model=8_192,
    # Fixed-size distillation model (largest grid depth) so the mixture
    # approximation error is uniform across the P sweep.
    mixture_num_layers=8,
    mixture_model_aspect_ratio=32.0,
    eval_batch_size=256,
    t_min=0.05,
)

# At CIFAR/FFHQ scale the model-free I-MMSE estimator carries the information
# numbers on its own and trains no auxiliary network at all.
MODEL_FREE_EVALS = replace(
    ALL_ESTIMATORS_EVALS,
    mixture_train_models=(0, 1),
    mixture_eval_models=(2, 3),
    mixture_sizes=(1, 2),
    information_methods=("classifier_model_free",),
    mixture_num_layers=16,
)
