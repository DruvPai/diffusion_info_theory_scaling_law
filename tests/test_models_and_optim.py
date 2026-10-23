from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from itg.config import Config
from itg.models import build_model, muon_params
from itg.models.common import (
    Conv2d,
    Linear,
    WeightHP,
    Weighted,
    adam_params,
    fan,
    init_std,
    model_width,
    mup_lr_scale,
    mup_residual_scale,
    mup_wd_scale,
    residual_branches,
)
from itg.models.unet import SongUNet
from itg.optim import (
    adam_peak_lr,
    half_life_ema_decay,
    make_optimizer,
    muon_peak_lr,
    orthogonalize,
    schedule_fraction,
)
from itg.processes import make_process
from itg.seeds import generator
from tests.tiny import tiny_config

UNET = tiny_config()


def make(config: Config, seed: int = 0) -> nn.Module:
    return build_model(config, make_process(config), generator(seed))


def test_model_builds_and_runs() -> None:
    x = torch.randn(2, *UNET.data_shape)
    prediction = make(UNET)(x, torch.full((2,), 0.3))
    assert prediction.value.shape == x.shape
    assert prediction.kind == "x_0"


def test_init_depends_only_on_the_seed() -> None:
    same = [dict(make(UNET, seed).named_parameters()) for seed in (3, 3)]
    other = dict(make(UNET, 4).named_parameters())
    for name, param in same[0].items():
        assert torch.equal(param, same[1][name])
    assert any(not torch.equal(param, other[name]) for name, param in same[0].items())


def test_hidden_matrix_init_uses_the_model_width() -> None:
    """The table's row, so it does not follow each matrix's own fan_in."""
    hidden = [m for m in make(UNET).modules() if isinstance(m, Linear) and m.role == "hidden"]
    assert len({fan(m)[1] for m in hidden}) > 1
    for module in hidden:
        assert init_std(module, UNET) == pytest.approx(
            UNET.mup_init_sigma / math.sqrt(UNET.hidden_dim)
        )


def test_hidden_conv_init_uses_its_own_fan_in_including_the_taps() -> None:
    """A conv's RMS operator norm sees channels only, so ||A||_R = Theta(1)
    needs sigma / sqrt(Cin k^2), not the model width."""
    for module in make(UNET).modules():
        if isinstance(module, Conv2d) and module.role == "hidden":
            assert init_std(module, UNET) == pytest.approx(
                UNET.mup_init_sigma / math.sqrt(fan(module)[1])
            )


def test_conv_rms_operator_norm_is_order_one_at_init() -> None:
    """The condition the init row exists to satisfy, measured on the operator."""
    for module in make(UNET).modules():
        if not isinstance(module, Conv2d) or module.zero_init or module.role != "hidden":
            continue
        weight = module.weight.detach()
        c_out, c_in = weight.shape[:2]
        blocks = torch.fft.fft2(weight, s=(8, 8)).permute(2, 3, 0, 1).reshape(-1, c_out, c_in)
        norm = math.sqrt(c_in / c_out) * float(torch.linalg.matrix_norm(blocks, ord=2).max())
        assert 1.0 < norm < 4.0


def test_mup_init_variance_row() -> None:
    """sigma / sqrt(fan_in), except sigma / taps on an output conv."""
    roles = set()
    for module in make(UNET).modules():
        if not isinstance(module, Weighted) or module.zero_init:
            continue
        roles.add(module.role)
        # A sample std of N entries has relative error 1/sqrt(2N); allow 4 sigma.
        tolerance = 4.0 / math.sqrt(2 * module.weight.numel())
        assert float(module.weight.detach().std()) == pytest.approx(
            init_std(module, UNET), rel=tolerance
        )
    assert roles == {"input", "hidden", "output"}


def test_mup_learning_rate_row() -> None:
    """sqrt(n) on input and output weights, 1 on hidden. n is the model width,
    the same for every layer, not the layer's own fan."""
    assert model_width(UNET) == UNET.hidden_dim
    root = math.sqrt(UNET.hidden_dim)
    for module in make(UNET).modules():
        if not isinstance(module, Weighted):
            continue
        expected = {"input": root, "hidden": 1.0, "output": root}[module.role]
        assert mup_lr_scale(module, UNET) == pytest.approx(expected)
        assert mup_wd_scale(module, UNET) == pytest.approx(1.0 / expected)


def test_hyperparameters_are_per_weight() -> None:
    """The table scales each weight's own sigma/eta/lambda, so WeightHP overrides
    compose with the config values."""
    module = Linear(8, 16, role="input", hp=WeightHP(sigma=2.0, lr=3.0, weight_decay=5.0))
    root = math.sqrt(model_width(UNET))
    assert init_std(module, UNET) == pytest.approx(2.0 * UNET.mup_init_sigma / math.sqrt(8))
    assert mup_lr_scale(module, UNET) == pytest.approx(3.0 * root)
    assert mup_wd_scale(module, UNET) == pytest.approx(5.0 / root)


def test_mup_weight_decay_row_is_uniform_in_effect() -> None:
    """lambda / sqrt(n) against a sqrt(n) learning rate leaves every matrix
    decaying by the same lr * weight_decay."""
    config = replace(UNET, weight_decay=0.1, num_steps=100)
    model = make(config)
    optimizer, scheduler = make_optimizer(model, config)
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()  # step 0 has lr = 0 by design, so this is a no-op
    scheduler.step()
    before = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, Weighted)
    }
    optimizer.step()
    expected = 1.0 - optimizer.param_groups[0]["lr"] * config.weight_decay
    for name, module in model.named_modules():
        if isinstance(module, Weighted) and not module.zero_init:
            ratio = float((module.weight.detach() / before[name]).median())
            assert ratio == pytest.approx(expected, rel=1e-5)


def test_mup_block_multipliers() -> None:
    model = make(UNET).net
    assert isinstance(model, SongUNet)
    assert model.input_scale == pytest.approx(UNET.mup_multiplier)
    assert model.output_scale == pytest.approx(UNET.mup_multiplier / UNET.hidden_dim)


def test_base_block_multiplier_is_not_absorbable_into_sigma_and_lr() -> None:
    """alpha scales a whole residual branch, so rescaling the branch's weights
    scales its output once per matrix instead of once."""
    scaled = replace(UNET, mup_multiplier=2.0, residual_zero_init=False)
    base = replace(UNET, residual_zero_init=False)
    assert mup_residual_scale(scaled) == pytest.approx(2 * mup_residual_scale(base))
    x = torch.randn(2, *UNET.data_shape, generator=generator(9))
    t = torch.full((2,), 0.4)
    assert not torch.allclose(make(base)(x, t).value, make(scaled)(x, t).value)


def test_muon_takes_every_matrix_and_adamw_every_vector() -> None:
    model = make(UNET)
    muon = [id(param) for param in muon_params(model)]
    adam = [id(param) for param in adam_params(model)]
    assert len(set(muon) | set(adam)) == len(muon) + len(adam)
    assert set(muon) | set(adam) == {id(param) for param in model.parameters()}
    assert all(param.ndim >= 2 for param in muon_params(model))
    assert all(param.ndim == 1 for param in adam_params(model))


def test_lr_schedule_shape_and_peaks() -> None:
    config = replace(UNET, lr=1.0, num_steps=100, lr_warmup_fraction=0.05)
    assert schedule_fraction(config, 0) == 0.0
    assert schedule_fraction(config, 5) == pytest.approx(1.0)
    assert schedule_fraction(config, 99) == pytest.approx(0.0)
    assert adam_peak_lr(config) == 1.0
    assert muon_peak_lr(config) == pytest.approx(10.0)


def test_orthogonalize_gives_near_unit_singular_values() -> None:
    matrix = torch.randn(8, 5, generator=torch.Generator().manual_seed(0))
    singular = torch.linalg.svdvals(orthogonalize(matrix, ns_steps=5))
    assert float(singular.min()) > 0.5
    assert float(singular.max()) < 1.5


def test_depth_mup_multiplier_is_one_over_the_branch_count() -> None:
    deeper = replace(UNET, edm=replace(UNET.edm, unet_num_blocks=3))
    assert residual_branches(deeper) > residual_branches(UNET)
    assert mup_residual_scale(deeper) == pytest.approx(1 / residual_branches(deeper))
    assert mup_residual_scale(replace(UNET, mup_residual_depth_scaling=False)) == 1.0


def test_depth_mup_reaches_the_forward_pass() -> None:
    assert UNET.mup_residual_depth_scaling
    scales = [
        module.residual_scale
        for module in make(UNET).modules()
        if hasattr(module, "residual_scale")
    ]
    assert scales
    assert scales == pytest.approx([1 / residual_branches(UNET)] * len(scales))


def test_half_life_ema_decay_anchor() -> None:
    assert half_life_ema_decay(78_000) == pytest.approx(0.9999, rel=1e-12)
