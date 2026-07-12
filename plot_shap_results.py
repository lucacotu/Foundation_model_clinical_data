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

BACKBONE_DISPLAY = {
    "tabicl": "TabICL",
    "tabpfn": "TabPFN",
}

FOUNDATION_MODEL_KEYS = [
    "deepsurv_simple", "deepsurv_vanilla", "rsf", "cox", "deepsurv_tuned", "rsf_tuned",
]
BASELINE_MODEL_KEYS = [
    "deepsurv_simple_baseline", "deepsurv_vanilla_baseline", "rsf_baseline", "cox_baseline",
]

MODEL_DISPLAY = {
    "deepsurv_simple": "DeepSurv Linear",
    "deepsurv_vanilla": "DeepSurv MLP",
    "deepsurv_tuned": "DeepSurv Tuned",
    "rsf": "RSF",
    "rsf_tuned": "RSF tuned",
    "cox": "Cox",
    "deepsurv_simple_baseline": "DeepSurv Linear\n(Baseline)",
    "deepsurv_vanilla_baseline": "DeepSurv MLP\n(Baseline)",
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

# "Preprocessing" is kept untranslated in both languages.
BAR_SUPTITLE_TEMPLATES = {
    "en": "SHAP Feature Importance (Top {top_n}) — {dataset} | {model} | Preprocessing: {preprocess}",
    "it": "Importanza delle Feature SHAP (Prime {top_n}) — {dataset} | {model} | Preprocessing: {preprocess}",
}

HEATMAP_TITLE_TEMPLATES = {
    "en": (
        "SHAP Feature Importance — {dataset} | TabICL + TabPFN | Preprocessing: {preprocess}\n"
        "Top {top_n} features per model, normalized per model"
    ),
    "it": (
        "Importanza delle Feature SHAP — {dataset} | TabICL + TabPFN | Preprocessing: {preprocess}\n"
        "Prime {top_n} feature per modello, normalizzate per modello"
    ),
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


def plot_heatmap(
    model_data: dict, models: list, xtick_labels: list, title: str, stem: str, out_dir: Path
) -> None:
    """Heatmap: union of top-N features (rows) × models (cols), normalized per model."""
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
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=17)
    ax.set_yticks(range(len(union_features)))
    ax.set_yticklabels(union_features, fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Normalized Mean |SHAP|", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(union_features), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else 72)
    print(f"    Saved heatmap → {stem}.pdf/png")
    plt.close(fig)


def plot_bar_charts(model_data: dict, preprocess: str, dataset: str, model: str, out_dir: Path, lang: str = "en") -> None:
    """Faceted horizontal bar charts: one panel per model, top-N features."""
    models = list(model_data.keys())
    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 5.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    backbone_display = BACKBONE_DISPLAY.get(model, model)

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
        ax.set_yticklabels(features, fontsize=10)
        ax.tick_params(axis="x", labelsize=11)
        ax.set_xlabel("Mean |SHAP|", fontsize=13)
        model_title = MODEL_DISPLAY.get(m, m)
        if not m.endswith("_baseline"):
            model_title = f"{backbone_display} - {model_title}"
        ax.set_title(model_title, fontsize=15, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    display_dataset = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    fig.suptitle(
        BAR_SUPTITLE_TEMPLATES[lang].format(top_n=TOP_N, dataset=display_dataset, model=model, preprocess=preprocess),
        fontsize=19, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    suffix = "" if lang == "en" else "_ita"
    stem = f"shap_{dataset}_{model}_bars_preprocess{preprocess}{suffix}"
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else 72)
    print(f"    Saved bar chart → {stem}.pdf/png")
    plt.close(fig)


def plot_combined_heatmap(
    tabicl_data: dict, tabpfn_data: dict, preprocess: str, dataset: str, out_dir: Path, lang: str = "en"
) -> None:
    """Heatmap merging TabICL and TabPFN foundation-model columns, sharing one baseline group."""
    merged_model_data: dict = {}
    xtick_labels: list = []

    for src_prefix, src_data in (("TabICL", tabicl_data), ("TabPFN", tabpfn_data)):
        for key in FOUNDATION_MODEL_KEYS:
            if key not in src_data:
                continue
            merged_key = f"{src_prefix.lower()}::{key}"
            merged_model_data[merged_key] = src_data[key]
            xtick_labels.append(f"{src_prefix} - {MODEL_DISPLAY.get(key, key)}")

    # Baselines don't depend on the embedding model, so TabPFN's copy is used for both.
    for key in BASELINE_MODEL_KEYS:
        if key not in tabpfn_data:
            continue
        merged_model_data[key] = tabpfn_data[key]
        xtick_labels.append(MODEL_DISPLAY.get(key, key))

    display_dataset = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    title = HEATMAP_TITLE_TEMPLATES[lang].format(dataset=display_dataset, preprocess=preprocess, top_n=TOP_N)
    suffix = "" if lang == "en" else "_ita"
    stem = f"shap_{dataset}_combined_heatmap_preprocess{preprocess}{suffix}"
    plot_heatmap(merged_model_data, list(merged_model_data.keys()), xtick_labels, title, stem, out_dir)


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR

    files = sorted(results_dir.rglob("shap_*_aggregated.txt"))
    if not files:
        print(f"No shap_*_aggregated.txt files found in {results_dir}")
        return

    dataset_data: dict = {}  # dataset -> {model: {preprocess: model_data}}

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
        dataset_data.setdefault(dataset_name, {})[model_name] = all_data

        out_dir = results_dir / dataset_name / model_name / "aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)
        for preprocess, model_data in all_data.items():
            print(f"  Preprocessing: {preprocess}  ({len(model_data)} models)")
            for lang in ("en", "it"):
                plot_bar_charts(model_data, preprocess, dataset_name, model_name, out_dir, lang=lang)

    for dataset_name, per_model in dataset_data.items():
        if "tabicl" not in per_model or "tabpfn" not in per_model:
            print(f"\nSkipping combined heatmap for {dataset_name}: need both tabicl and tabpfn")
            continue
        print(f"\nBuilding combined heatmap: {dataset_name}")
        out_dir = results_dir / dataset_name / "aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)
        preprocesses = sorted(set(per_model["tabicl"]) | set(per_model["tabpfn"]))
        for preprocess in preprocesses:
            for lang in ("en", "it"):
                plot_combined_heatmap(
                    per_model["tabicl"].get(preprocess, {}),
                    per_model["tabpfn"].get(preprocess, {}),
                    preprocess, dataset_name, out_dir, lang=lang,
                )


if __name__ == "__main__":
    main()
