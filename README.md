# Information-Theoretic Scaling Laws for Generative Diffusion Training

Code for the paper

> **Information-Theoretic Scaling Laws for Generative Diffusion Training**

> Druv Pai, Dejan Stančević, Yi Ma, Luca Ambrogioni

> [arXiv](https://arxiv.org/abs/2610.33337)

The paper studies *model information*, the mutual information between a
randomly trained diffusion model and a sample it generates, i.e. how much a
generated sample reveals about the particular training set. This repository
(the `itg` package) trains unconditional EDM diffusion models (DDPM++ UNet) on
a grid of training-set sizes N and model sizes P, estimates model information
from a few independently trained models, measures memorization and clean-FID,
and fits the scaling law derived from the linear-Gaussian model.

## Quickstart

```bash
git clone https://github.com/DruvPai/diffusion_info_theory_scaling_law.git
cd diffusion_info_theory_scaling_law
uv sync                     # Python >= 3.13, PyTorch >= 2.12
uv run pytest               # CPU-only smoke tests, about 10 s

# Prepare CIFAR-10 under ~/data and run one small case on one GPU
uv run python scripts/prepare_image_datasets.py --datasets cifar10
uv run python scripts/cifar10_edm.py --case n_710_h32_l1_edm_img32
```

All commands run from the repository root; outputs go to `experiments/`.

## Setup

We use [uv](https://docs.astral.sh/uv/); `uv.lock` pins every dependency.

```bash
uv sync
uv run pytest
```

Training and evaluation need a CUDA GPU; the paper's runs used 80 GB GPUs.
Set `ITG_DATA_ROOT` to put datasets somewhere other than `~/data`.

## Data

Each dataset is an image folder `$ITG_DATA_ROOT/<name>/{train,val}`
(`ITG_DATA_ROOT` defaults to `~/data`). Models draw their training sets from
`train`.

```bash
uv run python scripts/prepare_image_datasets.py --datasets fashion_mnist cifar10 ffhq
uv run python scripts/prepare_image_datasets.py --datasets imagenet \
    --imagenet-source /path/to/ILSVRC2012/train
```

FFHQ is the 70,000 128x128 thumbnails, split 60,000 / 10,000. ImageNet is the
ILSVRC-2012 training set with one directory per class; the last 10 images of
each class are held out, leaving a pool of 1,271,167.

## Running the grids

One script per grid; each runs every (N, depth) case of its grid, one case per
GPU, longest first:

| script | dataset | depths | dataset sizes | models per case |
|---|---|---|---|---|
| `scripts/imagenet_edm.py` | ImageNet 64x64 | 2-5 | 10 to 100,000, and the whole pool | 4 |
| `scripts/cifar10_edm.py` | CIFAR-10 32x32 | 1-5 | 10 to 50,000 | 4 |
| `scripts/fashion_mnist_edm.py` | Fashion-MNIST 32x32 | 1, 2, 4 | 10 to 60,000 | 10 |
| `scripts/ffhq_edm.py` | FFHQ 64x64 | 1, 2, 4 | 10 to 60,000 | 4 |
| `scripts/fashion_mnist_edm_matched_bank.py --k K` | Fashion-MNIST | 1, 2, 4 | as above | 2K (reuses the models) |

```bash
uv run python scripts/cifar10_edm.py                       # every case, across GPUs
uv run python scripts/cifar10_edm.py --case n_710_h160_l5_edm_img32
uv run python scripts/cifar10_edm.py --depths 5 --n-values 710 2900
uv run python scripts/cifar10_edm.py --report-only         # rebuild the report
uv run python scripts/imagenet_edm.py --depths 6 --microbatch-size 64
```

Case names are `n_<N>_h<width>_l<depth>_edm_img<resolution>`. Without
`--case`, a script dispatches its cases as subprocesses, one per visible GPU
(set `CUDA_VISIBLE_DEVICES` to choose them); `--sequential` runs them
in-process instead. `--stages` and `--per-model-only` run a subset of the
pipeline, `--force <stage>` recomputes a stage, and `--model i` trains just
source model `i` of a case (useful for spreading one case over several jobs).
Interrupted runs resume from their last checkpoint.

A run restricted with `--depths` or `--n-values` writes a report over those
cases only; run `--report-only` afterwards to rebuild the full grid's report.
`--microbatch-size` accumulates gradients over microbatches, which leaves the
update unchanged; depth 6 needs it to fit batch 128 into 80 GB.

Compute on 80 GB GPUs: the CIFAR-10, Fashion-MNIST, and FFHQ grids together
take about 530 GPU-hours. On ImageNet the cost is dominated by the cases at
N = 100,000 and on the whole pool, about 48 GPU-hours per model at depth 5.

### Outputs

```
experiments/<grid>/
  <case>/
    config.json                         resolved configuration of the case
    dataset/manifest.json               each source model's training-set ids
    models/model_XXX/                   checkpoints (resumable)
    samples/model_XXX/                  generated samples
    model_information_summaries.json    model-information estimates
    model_information_curve_*.csv       information rate over log-SNR
    training_metrics.csv, logs/
  report/                               grid-level CSVs, figures, and
                                        scaling_law_fit.json
```

## Reproducing the paper

After every grid has finished and has its report:

```bash
uv run python scripts/paper.py <out_dir> > paper_numbers.txt
```

writes every pgfplots CSV of the paper's figures and prints every number in
its text and tables, grouped by dataset. It also reads the depth-5,
N = 250,000 ImageNet run (`scripts/imagenet_edm.py --depths 5 --n-values
250000`), which is outside the grid. The held-out errors of the sigmoid
alternatives in the functional-form comparison depend on the order of float
operations, because their floors are nearly unconstrained; `scripts/paper.py`
evaluates them in the order used for the paper.

## The scaling law

`itg.scaling_law` holds the law of the paper,

    I(P, N) = F(P) + k P~^-b E_z[d_z I_ell(1 / (d_z (1/N - 1/N_pool)))],
    F(P) = c0 + c1 P~^-beta,   d_z = N_c(P) e^{sigma z},   N_c(P) = N_c(P_ref) P~^a,

with `I_ell` the per-dimension model information of the linear-Gaussian model
as the evaluation noise goes to zero, and its fit, `fit_scaling_law`. The fit
uses the grid points with N < N_pool; the runs on the whole pool check the
fitted floor F(P). Each grid's report writes the fit to
`report/scaling_law_fit.json`.

## Package layout

- `itg.config`: `Config`, everything that determines a trained checkpoint, and
  `EvalConfig`, everything downstream of training.
- `itg.processes`: the EDM noise process and conversions between `x_0`,
  `eps`, and score predictions.
- `itg.models`: the DDPM++ UNet under EDM preconditioning, its classifier
  head, and width-depth muP initialization.
- `itg.optim`: Muon on every matrix and AdamW on every vector, the learning-rate
  schedule, and the EMA.
- `itg.data`, `itg.train`, `itg.sampling`: image folders and replayable
  samplers, the training loop with exact resume, and the probability-flow ODE
  sampler (DDIM or Heun).
- `itg.evals`: the information rate on a log-SNR grid, the I-MMSE log-density
  estimator, the nearest-neighbor memorization ratio, and clean-FID.
- `itg.runner`: the per-case pipeline (train, samples, per-model evals,
  information), its artifact cache, multi-GPU dispatch, and the grid report.
- `itg.experiments`: the shared configuration of the grid scripts.
- `itg.scaling_law`: the law above.

## How one case is computed

1. Each source model gets a training set of N ids drawn without replacement
   from the pool, a pure function of its index.
2. Source models are trained (resumable checkpoints under `models/`) and
   sampled.
3. Per model: the memorization ratio against its own training set, clean-FID
   against 50,000 pool images, and a sample grid.
4. Model information: the source models split into a bank of `K_train`
   models, which defines the mixture score, and `K_val` held-out models whose
   scores are compared against it on forward-noised probes. Three estimators
   approximate the mixture score: a distilled mixture model
   (`distilled_mixture`), posterior weights from a trained classifier
   (`classifier_model_based`), and posterior weights from I-MMSE log-densities
   with no extra network (`classifier_model_free`, used in the paper).

Every artifact has a `<name>.spec.json` sidecar with everything that
determines it; an artifact is recomputed if and only if its spec changes, and
specs chain through their inputs, so a retrained model invalidates everything
downstream of it. All randomness comes from explicit generators seeded by
`(seed, case name, purpose)`, so a resumed run replays an uninterrupted one
exactly, and generated samples do not depend on the evaluation batch size.

## Using the scaling law on your own measurements

`itg.scaling_law` needs no GPU and no grid outputs. Given model-information
estimates measured at several (N, P), with P the parameter count,

```python
from itg.scaling_law import fit_scaling_law, scaling_law_prediction, critical_size, floor

fit = fit_scaling_law(df["n"], df["p"], df["information"], pool_size=50_000)
scaling_law_prediction(fit, n, p)   # predicted model information at new (N, P)
critical_size(fit, p)               # N_c(P), the memorization-to-generalization transition
floor(fit, p)                       # F(P), the floor when every model trains on the same data
```

The fit needs at least 4 distinct N and 10 points below the pool size, and
returns `None` otherwise. `scripts/paper.py` shows the full
usage on the grids' reports.

## Citation

```bibtex
@article{pai2026information,
  title   = {Information-Theoretic Scaling Laws for Generative Diffusion Training},
  author  = {Pai, Druv and Stan{\v{c}}evi{\'c}, Dejan and Ma, Yi and Ambrogioni, Luca},
  journal = {arXiv preprint},
  year    = {2026}
}
```
