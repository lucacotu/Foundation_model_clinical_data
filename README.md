# Foundation Models for Survival Analysis on Clinical Data

Master's thesis project studying whether embeddings from tabular **foundation models** — [TabPFN](https://github.com/PriorLabs/TabPFN), [TabICL](https://github.com/soda-inria/tabicl), and [TabDPT](https://github.com/layer6ai-labs/TabDPT) — improve survival analysis on small/medium clinical cohorts compared to classical models trained directly on raw features.

Two clinical datasets are used as case studies:
- **IHD** (internal name `OrmoniTirodei`) — a thyroid-hormones cohort with mortality and cardiovascular follow-up.
- **URRAH** (internal name `HURRAH`) — a second clinical cohort with its own mortality follow-up.

For each dataset, frozen embeddings extracted from a tabular foundation model are fed into several downstream survival models (DeepSurv, Random Survival Forest, Cox proportional hazards) and benchmarked against the same models trained directly on the raw clinical features. The project also studies robustness to missing data, the effect of training-set size, and feature importance via SHAP.

> **Note:** the underlying data (`Dataset Sirbu/`) is private clinical data and is **not** included in this repository.

---

## Table of contents

- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Main foundation-model benchmark](#1-main-foundation-model-benchmark)
  - [2. Missing-data robustness](#2-missing-data-robustness)
  - [3. Sample-size ablation](#3-sample-size-ablation)
  - [4. Aggregating results across seeds](#4-aggregating-results-across-seeds)
  - [5. Plotting](#5-plotting)
- [Generated artifacts](#generated-artifacts)
- [Running on a Slurm cluster](#running-on-a-slurm-cluster)

---

## Repository structure

```
.
├── test_tabular_model.py       # Main benchmark: foundation-model embeddings vs raw-feature baselines
├── test_nan.py                 # Robustness to missing data, embeddings vs classical imputers
├── test_sample_size.py         # Performance vs training-set size
│
├── aggregate_results.py        # Aggregate CV results across seeds (main benchmark)
├── aggregate_nan_results.py    # Aggregate results across seeds (NaN experiment)
├── aggregate_samplesize_results.py  # Aggregate results across seeds (sample-size experiment)
├── aggregate_shap_results.py   # Aggregate SHAP values across seeds
│
├── plot_results.py             # Plots for the main benchmark
├── plot_nan_results.py         # Plots for the missing-data experiment
├── plot_samplesize_results.py  # Plots for the sample-size experiment
├── plot_shap_results.py        # SHAP heatmaps / bar charts
├── plot_survival_curves.py     # Kaplan-Meier vs predicted survival curves
│
├── src/
│   ├── data_loader.py           # Reads and merges the raw Excel files per dataset
│   ├── preprocessing.py         # Cleaning, imputation, train/eval CV splits
│   ├── cox_models.py            # Univariate / multivariate / competing-risks Cox models
│   ├── deep_surv.py             # DeepSurv training helper
│   ├── deep_hit.py              # DeepHit (competing risks) training helper
│   ├── embedding_cox.py         # Cox model on top of foundation-model embeddings
│   ├── plotting.py              # Correlation matrix, feature importance, forest plots
│   ├── other_datasets.py        # Loaders for the URRAH and MIMIC-III datasets
│   ├── tabpfn/                  # TabPFN embedding extraction
│   ├── tabicl/                  # TabICL embedding extraction
│   └── tabdpt/                  # TabDPT embedding extraction
│
└── onedrive_utils/              # Helpers to sync the private dataset from OneDrive/iCloud
```

`results/` and `results_nan/` hold committed, human-readable output; `checkpoints/`, `checkpoints_samplesize/`, `nan/`, `tmp/`, and `survival_predictions/` are local caches (git-ignored, see [Generated artifacts](#generated-artifacts)).

`job.sbatch` (the Slurm job wrapper) and `HOW_TO_USE.md` (a cluster/Slurm/tmux/uv cheat-sheet) are machine-local, git-ignored files kept alongside the code but not part of the versioned repository.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management, pinned to **Python 3.12.12**.

```bash
# Install uv if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment and install dependencies
uv sync
source .venv/bin/activate
```

*(The full `tabpfn` package — not the lightweight `tabpfn-client` — is required, since embeddings can only be extracted from the full installation.)*

Place the private dataset in a `Dataset Sirbu/` folder at the repository root before running any script (see `onedrive_utils/` for scripts that pull it down from OneDrive/iCloud).

TabPFN and TabICL download their pretrained weights automatically from HuggingFace on first use. TabDPT does not: `--model tabdpt` needs a local checkpoint, resolved via (in order) the `checkpoint_path` argument, the `TABDPT_CHECKPOINT` environment variable, or a file at `src/models_diff/tabdpt1_1.pth`.

## Usage

All scripts assume the dataset is already in place and are meant to be run from the repository root, e.g. `python test_tabular_model.py ...` (or `uv run test_tabular_model.py ...`).

### 1. Main foundation-model benchmark

Runs 5-fold CV on both datasets, comparing embedding-based models (DeepSurv simple/vanilla, RSF, Cox) against the same models trained on raw features, under two preprocessing regimes (`-1` fill vs native NaN handling).

```bash
python test_tabular_model.py --model tabpfn --seed 42 [--tuning] [--shap]
```

- `--model {tabpfn,tabicl,tabdpt}` — which foundation model to extract embeddings from.
- `--tuning` — additionally run Optuna-tuned DeepSurv/RSF variants.
- `--shap` — compute and cache SHAP feature-importance values per fold.

Results are written to `results/<dataset>/<model>/results_cv_<dataset>_<model>_seed<seed>.txt`; model checkpoints and survival-prediction caches go to `checkpoints/` and `survival_predictions/` so re-runs resume instead of retraining.

### 2. Missing-data robustness

Compares foundation-model embeddings against classical imputers (mean, median, constant, KNN, MICE/BayesianRidge, MissForest-style RF) as increasing fractions of the raw features are masked with NaN.

```bash
python test_nan.py --model tabpfn --seed 42
```

Results are written to `results_nan/`.

### 3. Sample-size ablation

Repeats the main benchmark while progressively sub-sampling the training set, to study how quickly each model saturates.

```bash
python test_sample_size.py --model tabpfn --seed 42 --sample_sizes 100 500 1000 2500 5000 10000 15000
```

### 4. Aggregating results across seeds

Each experiment above is typically run for several seeds; the corresponding `aggregate_*.py` script merges the per-seed `.txt` files into one aggregated file with mean ± 95% CI:

```bash
python aggregate_results.py            # main benchmark
python aggregate_nan_results.py        # NaN experiment
python aggregate_samplesize_results.py # sample-size experiment
python aggregate_shap_results.py       # SHAP values
```

### 5. Plotting

Each aggregation has a matching plotting script that reads the aggregated `.txt` files under `results/` / `results_nan/` and produces PDF/PNG figures:

```bash
python plot_results.py
python plot_nan_results.py
python plot_samplesize_results.py
python plot_shap_results.py
python plot_survival_curves.py --dataset OrmoniTirodei [--model tabpfn] [--split_feature Age --split_threshold 60]
```

`plot_survival_curves.py` overlays observed Kaplan-Meier curves with mean predicted survival curves from `survival_predictions/`, with optional stratification by a feature.

## Generated artifacts

The following directories are git-ignored and are (re)created locally as scripts run:

| Directory | Produced by | Contents |
|---|---|---|
| `checkpoints/` | `test_tabular_model.py` | Trained model weights / Optuna params, keyed by dataset, preprocessing, seed, fold |
| `checkpoints_samplesize/` | `test_sample_size.py` | Same, for the sample-size ablation |
| `nan/` | `test_nan.py` | Fitted imputers and folds for the missing-data experiment |
| `survival_predictions/` | `test_tabular_model.py` | Cached per-fold survival curves, used by `plot_survival_curves.py` |
| `tmp/splits/` | all experiments | Cached K-fold split indices, keyed by dataset + seed |

## Running on a Slurm cluster

Experiments are typically launched on a GPU cluster via a local `job.sbatch` wrapper (e.g. `sbatch job.sbatch test_tabular_model.py --model tabpfn --seed 42 --shap`); see `HOW_TO_USE.md` (kept locally, not versioned) for the Slurm/tmux/`uv` commands used on that setup.
