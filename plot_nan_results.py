import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_NAN_DIR = Path("results_nan")
RESULTS_DIR = Path("results")

FILE_RE = re.compile(r"^results_cv_(.+)_([^_]+)_nan_aggregated\.txt$")

ROW_KEY_RE = re.compile(r"\(([\d.]+),\s*'([^']+)'\)")
CELL_RE = re.compile(r"([\d.]+)\s*±\s*([\d.]+)")

DATASET_DISPLAY_NAME = {
    "HURRAH": "URRAH",
    "OrmoniTirodei": "IHD",
}

SURVIVAL_MODEL_ORDER = ["Cox", "Cox Simple", "Cox Vanilla", "RSF"]
SURVIVAL_MODEL_DISPLAY_NAME = {
    "Cox Simple": "DeepSurv Simple",
    "Cox Vanilla": "DeepSurv Vanilla",
}

METHOD_ORDER = ["constant", "mean", "median", "knn", "imputer_bayesian", "embeddings"]
METHOD_COLORS = {
    "embeddings": "#1f77b4",
    "constant": "#7f7f7f",
    "mean": "#ffbb78",
    "median": "#ff7f0e",
    "knn": "#2ca02c",
    "imputer_bayesian": "#9467bd",
}

COMBO_COLORS = {
    ("HURRAH", "tabicl"): "#6baed6",
    ("HURRAH", "tabpfn"): "#08519c",
    ("OrmoniTirodei", "tabicl"): "#fd8d3c",
    ("OrmoniTirodei", "tabpfn"): "#a63603",
}


def parse_markdown_table(lines):
    """Parse a markdown table (header, separator, data rows...).

    Returns dict keyed by (nan_ratio, survival_model) -> {method: (mean, std)}.
    """
    header_cells = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    columns = header_cells[1:]

    data = {}
    for line in lines[2:]:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = ROW_KEY_RE.search(cells[0])
        if not m:
            continue
        nan_ratio = float(m.group(1))
        survival_model = m.group(2)

        row = {}
        for col, cell in zip(columns, cells[1:]):
            cm = CELL_RE.search(cell)
            if cm:
                row[col] = (float(cm.group(1)), float(cm.group(2)))
        data[(nan_ratio, survival_model)] = row

    return data


def parse_nan_file(filepath):
    """Parse a results_cv_{dataset}_{model}_nan_aggregated.txt file.

    Returns {"test": {...}, "train": {...}} where each is shaped:
      {survival_model: {nan_ratio: {method: (mean, std)}}}
    """
    text = Path(filepath).read_text()

    blocks = []
    current = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    table_blocks = [b for b in blocks if b[0].strip().startswith("|")]
    if len(table_blocks) < 2:
        raise ValueError(f"Expected 2 markdown tables, found {len(table_blocks)}")

    def reshape(table):
        out = {}
        for (nan_ratio, survival_model), methods in table.items():
            out.setdefault(survival_model, {})[nan_ratio] = methods
        return out

    test_table = parse_markdown_table(table_blocks[0])
    train_table = parse_markdown_table(table_blocks[1])

    return {"test": reshape(test_table), "train": reshape(train_table)}


def plot_method_lines(ax, section_data, methods, color_of, label_of):
    """Plot one line per method in `methods` present in section_data.

    section_data: {nan_ratio: {method: (mean, std)}}
    """
    nan_ratios = sorted(section_data.keys())
    x_positions = np.arange(len(nan_ratios))

    all_means, all_stds = [], []

    for method in methods:
        x_vals, y_vals, s_vals = [], [], []
        for i, nr in enumerate(nan_ratios):
            entry = section_data[nr].get(method)
            if entry is None:
                continue
            x_vals.append(x_positions[i])
            y_vals.append(entry[0])
            s_vals.append(entry[1])
            all_means.append(entry[0])
            all_stds.append(entry[1])

        if not x_vals:
            continue

        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)
        s_arr = np.array(s_vals)
        color = color_of(method)
        linewidth = 2.6 if method == "embeddings" else 1.6

        ax.plot(x_arr, y_arr, marker="o", linewidth=linewidth, markersize=5,
                 color=color, label=label_of(method))
        ax.fill_between(x_arr, y_arr - s_arr, y_arr + s_arr, alpha=0.15, color=color)

    tick_labels = [f"{nr:g}" for nr in nan_ratios]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("NaN ratio", fontsize=10)
    ax.set_ylabel("C-index (Test)", fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)

    if all_means:
        pad = 0.03
        y_min = max(0.0, min(m - s for m, s in zip(all_means, all_stds)) - pad)
        y_max = min(1.0, max(m + s for m, s in zip(all_means, all_stds)) + pad)
        ax.set_ylim(y_min, y_max)


def plot_per_file(dataset, model, data):
    test_data = data["test"]
    survival_models = [sm for sm in SURVIVAL_MODEL_ORDER if sm in test_data]
    if not survival_models:
        print(f"  No survival models found, skipped: {dataset}/{model}")
        return

    n_cols = 2
    n_rows = (len(survival_models) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 6 * n_rows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, survival_model in zip(flat_axes, survival_models):
        plot_method_lines(
            ax, test_data[survival_model], METHOD_ORDER,
            color_of=lambda m: METHOD_COLORS[m],
            label_of=lambda m: m,
        )
        ax.set_title(SURVIVAL_MODEL_DISPLAY_NAME.get(survival_model, survival_model), fontsize=11, fontweight="bold")

    for ax in flat_axes[len(survival_models):]:
        ax.set_visible(False)

    display_name = DATASET_DISPLAY_NAME.get(dataset, dataset)
    fig.suptitle(
        f"C-index Test — Robustness to Missingness  |  {display_name} / {model.upper()}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_dir = RESULTS_DIR / dataset / model / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results_nan_{dataset}_{model}_aggregated"

    out_pdf = out_dir / f"{stem}.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_pdf}")

    out_png = out_dir / f"{stem}.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_png}")

    plt.close(fig)


def plot_embeddings_comparison(all_data):
    """Compare only the 'embeddings' method's Test C-index across all
    (dataset, model) combinations, one subplot per survival model.
    """
    combos = sorted(all_data.keys())

    survival_models = []
    for data in all_data.values():
        for sm in SURVIVAL_MODEL_ORDER:
            if sm in data["test"] and sm not in survival_models:
                survival_models.append(sm)

    if not survival_models:
        print("  No data available for embeddings comparison plot")
        return

    n_cols = 2
    n_rows = (len(survival_models) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 6 * n_rows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, survival_model in zip(flat_axes, survival_models):
        section_data = {}
        color_map = {}
        label_map = {}
        for dataset, model in combos:
            sm_data = all_data[(dataset, model)]["test"].get(survival_model)
            if sm_data is None:
                continue
            combo_key = f"{dataset}|{model}"
            display_name = DATASET_DISPLAY_NAME.get(dataset, dataset)
            color_map[combo_key] = COMBO_COLORS.get((dataset, model), "#333333")
            label_map[combo_key] = f"{display_name} / {model.upper()}"

            for nr, methods in sm_data.items():
                entry = methods.get("embeddings")
                if entry is None:
                    continue
                section_data.setdefault(nr, {})[combo_key] = entry

        plot_method_lines(
            ax, section_data, list(label_map.keys()),
            color_of=lambda k: color_map[k],
            label_of=lambda k: label_map[k],
        )
        ax.set_title(SURVIVAL_MODEL_DISPLAY_NAME.get(survival_model, survival_model), fontsize=11, fontweight="bold")

    for ax in flat_axes[len(survival_models):]:
        ax.set_visible(False)

    fig.suptitle(
        "C-index Test — 'embeddings' Method Comparison Across Dataset/Model",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_pdf = RESULTS_DIR / "nan_embeddings_comparison.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_pdf}")

    out_png = RESULTS_DIR / "nan_embeddings_comparison.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_png}")

    plt.close(fig)


def main():
    if not RESULTS_NAN_DIR.exists():
        print(f"Directory '{RESULTS_NAN_DIR}' non trovata.")
        return

    all_data = {}
    found = False
    for txt_file in sorted(RESULTS_NAN_DIR.glob("results_cv_*_nan_aggregated.txt")):
        m = FILE_RE.match(txt_file.name)
        if not m:
            continue
        found = True
        dataset, model = m.group(1), m.group(2)
        print(f"Processing: {txt_file.name}  (dataset={dataset}, model={model})")
        try:
            data = parse_nan_file(txt_file)
        except Exception as e:
            print(f"  Error: {e}")
            continue
        all_data[(dataset, model)] = data
        plot_per_file(dataset, model, data)

    if not found:
        print("Nessun file results_cv_{dataset}_{model}_nan_aggregated.txt trovato.")
        return

    if all_data:
        print("Building embeddings comparison plot...")
        plot_embeddings_comparison(all_data)


if __name__ == "__main__":
    main()
