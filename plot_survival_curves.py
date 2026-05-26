"""
Load saved survival predictions, average across seeds (mean-of-means), and plot
KM + predicted survival curves with optional feature stratification.

Seeds and preprocess types are discovered automatically from the directory structure.
Both preprocessing variants (-1 and NaN) are plotted in the same PDF on separate pages.

Usage examples:
  # Overall only (no stratification)
  python plot_survival_curves.py --dataset OrmoniTirodei

  # Numeric threshold split
  python plot_survival_curves.py --dataset OrmoniTirodei \\
      --split_feature Age --split_threshold 60

  # Categorical split (e.g. sex)
  python plot_survival_curves.py --dataset OrmoniTirodei --split_feature Sesso

  # Filter to specific tabular models
  python plot_survival_curves.py --dataset OrmoniTirodei \\
      --tabular_model tabpfn --split_feature Age --split_threshold 60
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from joblib import load
from lifelines import KaplanMeierFitter


SURVIVAL_PRED_DIR = Path("survival_predictions")
PREPROCESS_TYPES  = ["-1", "NaN"]

COLORS = [
    "steelblue", "firebrick", "seagreen", "darkorange",
    "purple", "brown", "deeppink", "gray", "olive", "teal",
]


# ── Discovery helpers ─────────────────────────────────────────────────────────

def discover_seeds(tabular_model: str, dataset_name: str, preprocess_type: str) -> list[int]:
    """Return sorted list of seed numbers found on disk for this combination."""
    seed_parent = SURVIVAL_PRED_DIR / tabular_model / dataset_name / preprocess_type
    if not seed_parent.exists():
        return []
    seeds = []
    for p in seed_parent.iterdir():
        if p.is_dir() and p.name.startswith("seed"):
            try:
                seeds.append(int(p.name.replace("seed", "")))
            except ValueError:
                pass
    return sorted(seeds)


def _surv_filename(tabular_model: str, survival_model: str) -> str:
    """Baseline models drop the tabular prefix; embedding models keep it."""
    if survival_model.endswith("_baseline"):
        return survival_model.removesuffix("_baseline") + ".pkl"
    return f"{tabular_model}_{survival_model}.pkl"


def discover_survival_models(tabular_model: str, dataset_name: str, preprocess_type: str, seeds: list[int]) -> list[str]:
    """Return sorted list of logical survival model names found on disk."""
    prefix = f"{tabular_model}_"
    names: set[str] = set()
    for seed in seeds:
        seed_dir = SURVIVAL_PRED_DIR / tabular_model / dataset_name / preprocess_type / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for fold_dir in seed_dir.iterdir():
            for pkl in fold_dir.glob("*.pkl"):
                stem = pkl.stem
                if stem.startswith(prefix):
                    names.add(stem[len(prefix):])          # embedding model
                else:
                    names.add(stem + "_baseline")          # baseline model
    return sorted(names)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_seed_folds(dataset_name: str, tabular_model: str, survival_model: str, preprocess_type: str, seed: int):
    """Return list of fold dicts (sorted by fold number) for one seed, or None."""
    seed_dir = SURVIVAL_PRED_DIR / tabular_model / dataset_name / preprocess_type / f"seed{seed}"
    if not seed_dir.exists():
        return None
    fname = _surv_filename(tabular_model, survival_model)
    fold_dirs = sorted(
        [p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold")],
        key=lambda p: int(p.name.replace("fold", "")),
    )
    folds = [load(fd / fname) for fd in fold_dirs if (fd / fname).exists()]
    return folds or None


# ── Survival curve helpers ────────────────────────────────────────────────────

def interpolate_survival(surv_df: pd.DataFrame, common_times: np.ndarray) -> np.ndarray:
    """Interpolate each patient's curve onto common_times. Returns (n_patients, n_times)."""
    idx = surv_df.index.values.astype(float)
    result = []
    for col in surv_df.columns:
        vals = np.interp(common_times, idx, surv_df[col].values.astype(float))
        vals = np.minimum.accumulate(vals)
        result.append(vals)
    return np.array(result)


def seed_patient_matrix(seed_folds, common_times: np.ndarray) -> np.ndarray:
    """Concatenate all fold survival matrices for one seed → (n_patients, n_times)."""
    return np.concatenate(
        [interpolate_survival(f["surv_df"], common_times) for f in seed_folds],
        axis=0,
    )


def seed_feature_df(seed_folds) -> pd.DataFrame:
    """Concatenate X_test_features across folds for one seed."""
    return pd.concat([f["X_test_features"] for f in seed_folds], ignore_index=True)


# ── Stratification ────────────────────────────────────────────────────────────

def get_groups(X_features: pd.DataFrame, split_feature, split_threshold) -> list[tuple]:
    """
    Return [(mask, label), ...] for the chosen stratification.
    - split_feature is None  → single group "All"
    - split_threshold given  → binary threshold split (< vs >=)
    - split_threshold absent → one group per unique value (categorical or numeric)
    """
    if split_feature is None:
        return [(np.ones(len(X_features), dtype=bool), "All")]

    col = X_features[split_feature]

    if split_threshold is not None:
        thr = float(split_threshold)
        pairs = [
            (col < thr,  f"{split_feature} < {thr}"),
            (col >= thr, f"{split_feature} ≥ {thr}"),
        ]
    else:
        unique_vals = sorted(col.dropna().unique())
        pairs = [(col == v, f"{split_feature} = {v}") for v in unique_vals]

    return [
        (m.values if hasattr(m, "values") else np.asarray(m), lbl)
        for m, lbl in pairs
    ]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_figure(all_seeds_folds, split_feature, split_threshold, title: str, pdf_pages):
    """Build and save one page in the PDF."""
    # Use a single seed for KM/observed data — each seed covers all patients once,
    # so concatenating multiple seeds would duplicate observations.
    ref_folds = all_seeds_folds[0]
    t_all = np.concatenate([fold["t_test"] for fold in ref_folds])
    y_all = np.concatenate([fold["y_test"] for fold in ref_folds])
    X_all = pd.concat(
        [fold["X_test_features"] for fold in ref_folds],
        ignore_index=True,
    )

    t_max = float(t_all.max())
    common_times = np.linspace(0, t_max, 300)

    show_overall = split_feature is None
    n_cols = 1 if show_overall else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=9, wrap=True)

    # ── Overall subplot ───────────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_title("Overall")
    n_all = len(t_all)
    kmf = KaplanMeierFitter()
    kmf.fit(t_all, event_observed=y_all, label=f"KM observed (n={n_all})")
    kmf.plot_survival_function(ax=ax0, ci_show=True, color="steelblue")

    seed_means = [seed_patient_matrix(sf, common_times).mean(axis=0) for sf in all_seeds_folds]
    mean_pred = np.mean(seed_means, axis=0)
    ax0.step(common_times, mean_pred, where="post",
             label=f"Predicted mean (seeds={len(all_seeds_folds)}, n={n_all})",
             color="darkorange", linewidth=2)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Survival probability")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    if show_overall:
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)
        return

    # ── Stratified subplot ────────────────────────────────────────────────────
    ax1 = axes[1]
    ax1.set_title(f"Stratified by {split_feature}")
    groups_all = get_groups(X_all, split_feature, split_threshold)

    for i, (mask_all, lbl) in enumerate(groups_all):
        n_group = int(mask_all.sum())
        if n_group == 0:
            continue
        color = COLORS[i % len(COLORS)]
        full_label = f"{lbl} (n={n_group})"

        kmf_g = KaplanMeierFitter()
        kmf_g.fit(t_all[mask_all], event_observed=y_all[mask_all], label=f"KM {full_label}")
        kmf_g.plot_survival_function(ax=ax1, ci_show=True, color=color)

        seed_group_means = []
        for sf in all_seeds_folds:
            X_seed = seed_feature_df(sf)
            surv_seed = seed_patient_matrix(sf, common_times)
            for s_mask, s_lbl in get_groups(X_seed, split_feature, split_threshold):
                if s_lbl == lbl and s_mask.sum() > 0:
                    seed_group_means.append(surv_seed[s_mask].mean(axis=0))
                    break

        if not seed_group_means:
            continue
        ax1.step(common_times, np.mean(seed_group_means, axis=0), where="post",
                 label=f"Predicted {full_label}", color=color, linewidth=2, linestyle="--")

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Survival probability")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot survival curves from saved predictions.")
    parser.add_argument("--dataset", required=True,
                        choices=["OrmoniTirodei", "HURRAH"],
                        help="Dataset name")
    parser.add_argument("--model", default=None,
                        choices=["tabpfn", "tabicl", "tabdpt"],
                        help="Tabular model to include; default: all found")
    parser.add_argument("--split_feature",  default=None,
                        help="Feature column to stratify by (e.g. Age, Sesso)")
    parser.add_argument("--split_threshold", type=float, default=None,
                        help="Threshold for numeric split")
    args = parser.parse_args()

    if not SURVIVAL_PRED_DIR.exists():
        print(f"[ERROR] Directory not found: {SURVIVAL_PRED_DIR}")
        return

    tab_models = [args.model] if args.model else [
        p.name for p in sorted(SURVIVAL_PRED_DIR.iterdir()) if p.is_dir()
    ]

    split_tag = args.split_feature if args.split_feature else "overall"
    tab_tag   = tab_models[0] if len(tab_models) == 1 else "all"
    output    = Path("results") / f"curves_{tab_tag}_{args.dataset}_{split_tag}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:
        for preprocess_type in PREPROCESS_TYPES:
            for tab_model in tab_models:
                tab_dir = SURVIVAL_PRED_DIR / tab_model / args.dataset / preprocess_type
                if not tab_dir.exists():
                    print(f"[SKIP] Not found: {tab_dir}")
                    continue

                seeds = discover_seeds(tab_model, args.dataset, preprocess_type)
                if not seeds:
                    print(f"[SKIP] No seeds found: {tab_model}/{args.dataset}/{preprocess_type}")
                    continue

                surv_models = discover_survival_models(
                    tab_model, args.dataset, preprocess_type, seeds
                )
                if not surv_models:
                    print(f"[SKIP] No survival models: {tab_model}/{preprocess_type}")
                    continue

                for surv_model in surv_models:
                    all_seeds_folds = []
                    for seed in seeds:
                        folds = load_seed_folds(
                            args.dataset, tab_model, surv_model, preprocess_type, seed
                        )
                        if folds:
                            all_seeds_folds.append(folds)

                    if not all_seeds_folds:
                        print(f"[SKIP] No data: {tab_model}/{surv_model} "
                              f"preprocess={preprocess_type}")
                        continue

                    total_obs = sum(len(f["t_test"]) for sf in all_seeds_folds for f in sf)
                    print(f"[PLOT] preprocess={preprocess_type} | {tab_model}/{surv_model} | "
                          f"{len(all_seeds_folds)} seed(s) {seeds} | {total_obs} patient-obs")

                    if surv_model.endswith("_baseline"):
                        model_label = surv_model.removesuffix("_baseline")
                    else:
                        model_label = f"{tab_model.upper()} → {surv_model}"
                    title = (
                        f"{args.dataset}  |  {model_label}"
                        f"  |  preprocess: {preprocess_type}"
                        f"  |  seeds: {seeds}"
                    )
                    plot_figure(all_seeds_folds, args.split_feature, args.split_threshold, title, pdf)

    print(f"[DONE] Saved → {output}")


if __name__ == "__main__":
    main()
