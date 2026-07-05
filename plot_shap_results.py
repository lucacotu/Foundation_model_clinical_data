#!/usr/bin/env python3
"""
Plot SHAP importance results from aggregated SHAP files.

For each preprocessing condition generates:
  1. Heatmap (per-model normalized SHAP): union of top-N features × models
  2. Faceted horizontal bar charts: top-N features per model with error bars

Usage:
  python plot_shap_results.py [results_dir]
"""

import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
TOP_N = 20

DATASET_DISPLAY_NAMES = {
    "OrmoniTirodei": "IHD",
    "HURRAH": "URRAH",
}

MODEL_DISPLAY = {
    "deepsurv_simple": "DeepSurv Simple",
    "deepsurv_vanilla": "DeepSurv Vanilla",
    "deepsurv_tuned": "DeepSurv Tuned",
    "rsf": "RSF",
    "rsf_tuned": "RSF tuned",
    "cox": "Cox",
    "deepsurv_simple_baseline": "DeepSurv Simple\n(Baseline)",
    "deepsurv_vanilla_baseline": "DeepSurv Vanilla\n(Baseline)",
    "rsf_baseline": "RSF (Baseline)",
    "cox_baseline": "Cox (Baseline)",
}

# Foundation model variant colors match plot_results.py's model_color() palette
MODEL_COLOR = {
    "deepsurv_simple": "#e377c2",
    "deepsurv_vanilla": "#17becf",
    "deepsurv_tuned": "#1f77b4",
    "rsf": "#8c564b",
    "rsf_tuned": "#9467bd",
    "cox": "#bcbd22",
    "deepsurv_simple_baseline": "#e6550d",
    "deepsurv_vanilla_baseline": "#fd8d3c",
    "rsf_baseline": "#31a354",
    "cox_baseline": "#74c476",
}


def clean_name(name: str) -> str:
    return name.replace("\n", " ")


def parse_aggregated(filepath: Path) -> dict:
    """
    Parse shap_*_aggregated.txt.

    Returns:
        {preprocess: {model: [(feature, mean_shap, std_shap), ...]}}
    """
    data: dict = {}
    current_pre = None
    current_model = None
    pending: list = []

    value_re = re.compile(r"^(.*?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s*$")
    pre_re = re.compile(r"Preprocessing:\s*(\S+)\s")
    model_re = re.compile(r"^---\s+(.+?)\s+---$")

    with open(filepath) as fh:
        for line in fh:
            s = line.rstrip("\n")
            stripped = s.strip()

            pm = pre_re.search(stripped)
            if pm:
                current_pre = pm.group(1)
                data.setdefault(current_pre, {})
                current_model = None
                pending = []
                continue

            mm = model_re.match(stripped)
            if mm:
                current_model = mm.group(1)
                if current_pre:
                    data[current_pre].setdefault(current_model, [])
                pending = []
                continue

            if not stripped or re.match(r"^[-=]+$", stripped) or "Mean|SHAP|" in stripped:
                continue

            if current_pre is None or current_model is None:
                continue

            vm = value_re.match(s)
            if vm:
                name_part = vm.group(1).strip()
                pending.append(name_part)
                feature = clean_name("\n".join(pending))
                pending = []
                data[current_pre][current_model].append(
                    (feature, float(vm.group(2)), float(vm.group(3)))
                )
            else:
                pending.append(stripped)

    return data


def _title(text: str, ax, fontsize: int = 10) -> None:
    ax.set_title(text, fontsize=fontsize, fontweight="bold")


def plot_heatmap(model_data: dict, preprocess: str, dataset: str, model: str, out_dir: Path) -> None:
    """Heatmap: union of top-N features (rows) × models (cols), normalized per model."""
    models = list(model_data.keys())

    # Union of top-N features preserving order of first appearance
    seen: set = set()
    union_features: list = []
    for m in models:
        for feat, _, _ in model_data[m][:TOP_N]:
            if feat not in seen:
                union_features.append(feat)
                seen.add(feat)

    # Build value dict per model
    val_map = {m: {feat: mv for feat, mv, _ in model_data[m]} for m in models}

    # Matrix: rows=features, cols=models, values normalized 0-1 per model
    matrix = np.zeros((len(union_features), len(models)))
    for j, m in enumerate(models):
        col = np.array([val_map[m].get(f, 0.0) for f in union_features])
        if col.max() > 0:
            matrix[:, j] = col / col.max()

    fig_h = max(8, len(union_features) * 0.33)
    fig_w = max(10, len(models) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in models], rotation=45, ha="right", fontsize=9
    )
    ax.set_yticks(range(len(union_features)))
    ax.set_yticklabels(union_features, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Normalized Mean |SHAP|", fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(union_features), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)

    display_dataset = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    ax.set_title(
        f"SHAP Feature Importance — {display_dataset} | {model} | Preprocessing: {preprocess}\n"
        f"Top {TOP_N} features per model, normalized per model",
        fontsize=11, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    stem = f"shap_{dataset}_{model}_heatmap_preprocess{preprocess}"
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else 72)
    print(f"    Saved heatmap → {stem}.pdf/png")
    plt.close(fig)


def plot_bar_charts(model_data: dict, preprocess: str, dataset: str, model: str, out_dir: Path) -> None:
    """Faceted horizontal bar charts: one panel per model, top-N features."""
    models = list(model_data.keys())
    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 5.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, m in enumerate(models):
        ax = axes_flat[idx]
        entries = list(reversed(model_data[m][:TOP_N]))  # lowest importance at bottom

        features = [e[0] for e in entries]
        means = np.array([e[1] for e in entries])
        stds = np.array([e[2] for e in entries])

        color = MODEL_COLOR.get(m, "#4393c3")
        hatch = "" if m.endswith("_baseline") else "///"
        y = np.arange(len(features))
        ax.barh(
            y, means, xerr=stds, height=0.65,
            color=color, alpha=0.85, edgecolor="black", hatch=hatch,
            error_kw={"elinewidth": 0.9, "ecolor": "#555555", "capsize": 2.5},
        )

        ax.set_yticks(y)
        ax.set_yticklabels(features, fontsize=7.5)
        ax.set_xlabel("Mean |SHAP|", fontsize=9)
        ax.set_title(MODEL_DISPLAY.get(m, m), fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    display_dataset = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    fig.suptitle(
        f"SHAP Feature Importance (Top {TOP_N}) — {display_dataset} | {model} | Preprocessing: {preprocess}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    stem = f"shap_{dataset}_{model}_bars_preprocess{preprocess}"
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else 72)
    print(f"    Saved bar chart → {stem}.pdf/png")
    plt.close(fig)


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR

    files = sorted(results_dir.rglob("shap_*_aggregated.txt"))
    if not files:
        print(f"No shap_*_aggregated.txt files found in {results_dir}")
        return

    for filepath in files:
        # filename: shap_{dataset}_{model}_aggregated.txt
        # model = second-to-last token, dataset = everything between 'shap_' and '_{model}_aggregated'
        stem_parts = filepath.stem.split("_")  # ['shap', ...dataset_parts..., model, 'aggregated']
        if len(stem_parts) < 4 or stem_parts[0] != "shap" or stem_parts[-1] != "aggregated":
            print(f"Skipping unrecognized filename: {filepath.name}")
            continue
        model_name = stem_parts[-2]
        dataset_name = "_".join(stem_parts[1:-2])
        print(f"\nProcessing: {filepath.name}  (dataset={dataset_name}, model={model_name})")

        all_data = parse_aggregated(filepath)

        out_dir = results_dir / dataset_name / model_name / "aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)
        for preprocess, model_data in all_data.items():
            print(f"  Preprocessing: {preprocess}  ({len(model_data)} models)")
            plot_heatmap(model_data, preprocess, dataset_name, model_name, out_dir)
            plot_bar_charts(model_data, preprocess, dataset_name, model_name, out_dir)


if __name__ == "__main__":
    main()
