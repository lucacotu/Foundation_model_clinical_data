import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")

FILE_RE = re.compile(r"^results_samplesize_(.+)_([^_]+)_aggregated\.txt$")

PREPROCESS_RE = re.compile(r"^Preprocess:\s*(.+)$")
SAMPLE_RE = re.compile(r"Sample size \(requested\):\s+\d+\s+\|\s+actual mean:\s+([\d.]+)")
TEST_RE = re.compile(r"TEST\s+(.+?)\s+→ mean:\s+([\d.]+)\s+\|\s+std:\s+([\d.]+)")


def parse_samplesize_file(filepath):
    """
    Returns nested dict:
      { preprocess_key: { actual_mean: { model_name: {"mean": float, "std": float} } } }
    """
    data = {}
    current_prep = None
    current_sample = None

    with open(filepath) as f:
        for line in f:
            m = PREPROCESS_RE.match(line.strip())
            if m:
                current_prep = m.group(1).strip()
                data[current_prep] = {}
                current_sample = None
                continue

            if current_prep is None:
                continue

            m = SAMPLE_RE.search(line)
            if m:
                current_sample = float(m.group(1))
                data[current_prep][current_sample] = {}
                continue

            if current_sample is None:
                continue

            m = TEST_RE.search(line)
            if m:
                model_name = m.group(1).strip()
                data[current_prep][current_sample][model_name] = {
                    "mean": float(m.group(2)),
                    "std": float(m.group(3)),
                }

    return data


DATASET_DISPLAY_NAMES = {
    "OrmoniTirodei": "IHD",
    "HURRAH": "URRAH",
}


def display_dataset_name(dataset):
    return DATASET_DISPLAY_NAMES.get(dataset, dataset)


def model_marker(name, model):
    name_lower = name.lower()
    if "vanilla" in name_lower:
        return "o"
    if "simple" in name_lower:
        return "s"
    if "rsf" in name_lower:
        return "^"
    if "cox" in name_lower:
        return "D"
    return "*"


def model_color(name, model):
    name_lower = name.lower()
    if name_lower.startswith(model.lower()):
        if "vanilla" in name_lower:
            return "#6baed6"
        if "simple" in name_lower:
            return "#3182bd"
        if "rsf" in name_lower:
            return "#08519c"
        if "cox" in name_lower:
            return "#9ecae1"
        return "#1f77b4"
    else:
        if "vanilla" in name_lower:
            return "#ff7f0e"
        if "simple" in name_lower:
            return "#ffbb78"
        if "rsf" in name_lower:
            return "#2ca02c"
        if "cox" in name_lower:
            return "#98df8a"
        return "#d62728"


def legend_label(name, model):
    m_up = model.upper()
    name_lower = name.lower()
    if name_lower.startswith(model.lower()):
        if "vanilla" in name_lower:
            return f"{m_up} + DS-Vanilla"
        if "simple" in name_lower:
            return f"{m_up} + DS-Simple"
        if "rsf" in name_lower:
            return f"{m_up} + RSF"
        if "cox" in name_lower:
            return f"{m_up} + Cox"
        return name
    else:
        if "vanilla" in name_lower:
            return "DS-Vanilla (base)"
        if "simple" in name_lower:
            return "DS-Simple (base)"
        if "rsf" in name_lower:
            return "RSF (base)"
        if "cox" in name_lower:
            return "Cox (base)"
        return name


def plot_section(prep_data, section_label, ax, model):
    sample_sizes = sorted(prep_data.keys())
    # Use evenly-spaced indices so that close values (e.g. 5000 vs 5161) don't overlap
    x_positions = np.arange(len(sample_sizes))
    ss_to_idx = {ss: i for i, ss in enumerate(sample_sizes)}

    all_models = []
    seen = set()
    for ss in sample_sizes:
        for name in prep_data[ss]:
            if name not in seen:
                all_models.append(name)
                seen.add(name)

    for name in all_models:
        x_vals, y_vals, s_vals = [], [], []
        for ss in sample_sizes:
            entry = prep_data[ss].get(name)
            if entry is not None:
                x_vals.append(ss_to_idx[ss])
                y_vals.append(entry["mean"])
                s_vals.append(entry["std"])

        if not x_vals:
            continue

        color = model_color(name, model)
        marker = model_marker(name, model)
        label = legend_label(name, model)
        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)
        s_arr = np.array(s_vals)

        ax.plot(x_arr, y_arr, marker=marker, linewidth=1.8, markersize=6,
                color=color, label=label)

        if s_arr.any():
            ax.fill_between(x_arr, y_arr - s_arr, y_arr + s_arr,
                            alpha=0.15, color=color)

    all_means = [
        entry["mean"]
        for ss in sample_sizes
        for entry in prep_data[ss].values()
    ]
    all_stds = [
        entry["std"]
        for ss in sample_sizes
        for entry in prep_data[ss].values()
    ]
    pad = 0.03
    y_min = max(0.0, min(m - s for m, s in zip(all_means, all_stds)) - pad)
    y_max = min(1.0, max(m + s for m, s in zip(all_means, all_stds)) + pad)

    tick_labels = [str(int(ss)) if ss == int(ss) else f"{ss:.0f}" for ss in sample_sizes]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Sample size (actual mean)", fontsize=10)
    ax.set_ylabel("C-index (Test)", fontsize=10)
    ax.set_title(f"Preprocessing: {section_label}", fontsize=11, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.8)


def plot_file(filepath: Path, dataset: str, model: str):
    data = parse_samplesize_file(filepath)
    if not data:
        print(f"  No data found, skipped: {filepath.name}")
        return

    n_sections = len(data)
    fig, axes = plt.subplots(1, n_sections, figsize=(9 * n_sections, 6), squeeze=False)

    for ax, (prep_key, prep_data) in zip(axes[0], data.items()):
        plot_section(prep_data, prep_key, ax, model)

    fig.suptitle(
        f"C-index Test — Sample Size Effect  |  {display_dataset_name(dataset)} / {model.upper()}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_dir = RESULTS_DIR / dataset / model / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"results_samplesize_{dataset}_{model}_aggregated.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def main():
    if not RESULTS_DIR.exists():
        print(f"Directory '{RESULTS_DIR}' not found.")
        return

    found = False
    for txt_file in sorted(RESULTS_DIR.rglob("results_samplesize_*_aggregated.txt")):
        m = FILE_RE.match(txt_file.name)
        if not m:
            continue
        found = True
        dataset, model = m.group(1), m.group(2)
        print(f"Processing: {txt_file.name}  (dataset={dataset}, model={model})")
        try:
            plot_file(txt_file, dataset, model)
        except Exception as e:
            print(f"  Error: {e}")

    if not found:
        print("No results_samplesize_*_aggregated.txt files found.")


if __name__ == "__main__":
    main()
