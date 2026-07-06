import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")

# matches: results_aggregated_{dataset}_{model}.txt
# dataset may contain underscores; model is the last token before .txt
FILE_RE = re.compile(r"^results_aggregated_(.+)_([^_]+)\.txt$")

DATASET_DISPLAY_NAMES = {
    "OrmoniTirodei": "IHD",
    "HURRAH": "URRAH",
}


def parse_results_file(filepath):
    """Parse aggregated results file, returning dict keyed by section name (-1, NaN)."""
    sections = {}
    current_section = None

    section_re = re.compile(r"TOTAL \[(.+?)\]:")
    entry_re = re.compile(
        r"^\s+(.+?)\s+→\s+mean:\s+([\d.]+)\s+\|\s+std:\s+([\d.]+)"
    )

    with open(filepath) as f:
        for line in f:
            m = section_re.search(line)
            if m:
                current_section = m.group(1)
                sections[current_section] = []
                continue

            if current_section is None:
                continue

            m = entry_re.match(line)
            if m:
                name, mean, std = m.group(1).strip(), float(m.group(2)), float(m.group(3))
                sections[current_section].append({"name": name, "mean": mean, "std": std})

    return sections


def is_test_entry(name):
    name_lower = name.lower()
    # Exclude train entries
    return "train" not in name_lower


def model_color(name, model, hatch="///"):
    """Assign a consistent (color, hatch) pair based on whether the entry belongs to the file's model."""
    name_lower = name.lower()
    if name_lower.startswith(model.lower()):
        # Distinct hues per embeddings-model variant, hatch marks the group
        if "tuned deepsurv" in name_lower:
            return "#1f77b4", hatch
        if "tuned rsf" in name_lower:
            return "#9467bd", hatch
        if "vanilla" in name_lower:
            return "#17becf", hatch
        if "simple" in name_lower:
            return "#e377c2", hatch
        if "rsf" in name_lower:
            return "#8c564b", hatch
        if "cox" in name_lower:
            return "#bcbd22", hatch
        return "#7f7f7f", hatch
    else:
        # Shades of orange/green for baseline models, no hatch
        if "deepsurv vanilla" in name_lower:
            return "#ff7f0e", ""
        if "deepsurv simple" in name_lower:
            return "#ffbb78", ""
        if "rsf" in name_lower:
            return "#2ca02c", ""
        if "cox" in name_lower:
            return "#98df8a", ""
        return "#d62728", ""


def legend_label(name, model):
    """Return the same label used in the legend for a given raw entry name."""
    m_up = model.upper()
    name_lower = name.lower()
    if name_lower.startswith(model.lower()):
        if "tuned deepsurv" in name_lower:
            return f"{m_up} - Tuned DeepSurv"
        if "tuned rsf" in name_lower:
            return f"{m_up} - Tuned RSF"
        if "vanilla" in name_lower:
            return f"{m_up} - Deepsurv Vanilla"
        if "simple" in name_lower:
            return f"{m_up} - Deepsurv Simple"
        if "rsf" in name_lower:
            return f"{m_up} - RSF"
        if "cox" in name_lower:
            return f"{m_up} - Cox"
    else:
        if "deepsurv vanilla" in name_lower or "vanilla" in name_lower:
            return "DeepSurv Vanilla"
        if "deepsurv simple" in name_lower or "simple" in name_lower:
            return "DeepSurv Simple"
        if "rsf" in name_lower:
            return "RSF"
        if "cox" in name_lower:
            return "Cox"
    return name


def render_bars(ax, entries_data, section_label, legend_kwargs=None):
    """Draw one axis of bars from pre-built per-bar data.

    entries_data: list of dicts with keys mean, std, color, hatch, label.
    legend_kwargs: overrides merged into the default legend placement
      (loc="lower right", framealpha=0.8), e.g. to move a crowded legend
      outside the axes.
    """
    means = np.array([e["mean"] for e in entries_data])
    stds = np.array([e["std"] for e in entries_data])
    colors = [e["color"] for e in entries_data]
    hatches = [e["hatch"] for e in entries_data]
    labels = [e["label"] for e in entries_data]

    x = np.arange(len(entries_data))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black",
                  linewidth=0.6, error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Annotate values on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=7, rotation=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("C-index (Test)", fontsize=15)
    ax.set_title(f"Preprocessing: {section_label}", fontsize=17, fontweight="bold")
    ax.set_ylim(max(0, means.min() - stds.max() - 0.05), min(1.0, means.max() + stds.max() + 0.07))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Build legend from the actual entries present in the chart
    seen = {}
    for e in entries_data:
        if e["label"] not in seen:
            seen[e["label"]] = (e["color"], e["hatch"])
    patches = [
        mpatches.Patch(facecolor=clr, hatch=h, edgecolor="black", label=lbl)
        for lbl, (clr, h) in seen.items()
    ]
    legend_options = {"fontsize": 8, "loc": "lower right", "framealpha": 0.8}
    if legend_kwargs:
        legend_options.update(legend_kwargs)
    ax.legend(handles=patches, **legend_options)


def plot_section(entries, section_label, ax, model):
    test_entries = [e for e in entries if is_test_entry(e["name"])]
    if not test_entries:
        return

    entries_data = []
    for e in test_entries:
        color, hatch = model_color(e["name"], model)
        label = legend_label(e["name"], model)
        entries_data.append({**e, "color": color, "hatch": hatch, "label": label})

    render_bars(ax, entries_data, section_label)


def plot_combined_section(tabpfn_entries, tabicl_entries, section_label, ax):
    tabpfn_test = [e for e in tabpfn_entries if is_test_entry(e["name"])]
    tabicl_test = [e for e in tabicl_entries if is_test_entry(e["name"])]

    tabpfn_model_entries = [e for e in tabpfn_test if e["name"].lower().startswith("tabpfn")]
    tabicl_model_entries = [e for e in tabicl_test if e["name"].lower().startswith("tabicl")]
    baseline_entries = [e for e in tabpfn_test if not e["name"].lower().startswith("tabpfn")]

    if not (tabpfn_model_entries or tabicl_model_entries or baseline_entries):
        return

    entries_data = []
    for e in tabpfn_model_entries:
        color, hatch = model_color(e["name"], "tabpfn", hatch="///")
        label = legend_label(e["name"], "tabpfn")
        entries_data.append({**e, "color": color, "hatch": hatch, "label": label})
    for e in tabicl_model_entries:
        color, hatch = model_color(e["name"], "tabicl", hatch="xxx")
        label = legend_label(e["name"], "tabicl")
        entries_data.append({**e, "color": color, "hatch": hatch, "label": label})
    for e in baseline_entries:
        color, hatch = model_color(e["name"], "tabpfn")
        label = legend_label(e["name"], "tabpfn")
        entries_data.append({**e, "color": color, "hatch": hatch, "label": label})

    render_bars(ax, entries_data, section_label, legend_kwargs={
        "loc": "upper left", "bbox_to_anchor": (1.01, 1.0), "borderaxespad": 0,
    })


def plot_file(filepath: Path, dataset: str, model: str):
    sections = parse_results_file(filepath)

    if not sections:
        print(f"  No sections found, skipped: {filepath.name}")
        return

    n_sections = len(sections)
    fig, axes = plt.subplots(1, n_sections, figsize=(10 * n_sections, 6), squeeze=False)

    for ax, (section_key, entries) in zip(axes[0], sections.items()):
        plot_section(entries, section_key, ax, model)

    display_dataset = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    fig.suptitle(
        f"C-index Test Results — {display_dataset} / {model}",
        fontsize=19, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_dir = RESULTS_DIR / dataset / model / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = filepath.stem  # e.g. results_aggregated_{dataset}_{model}

    out_pdf = out_dir / f"{stem}.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_pdf}")

    out_png = out_dir / f"{stem}.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_png}")

    plt.close(fig)


def plot_combined_file(dataset: str, tabpfn_path: Path, tabicl_path: Path):
    tabpfn_sections = parse_results_file(tabpfn_path)
    tabicl_sections = parse_results_file(tabicl_path)

    common_sections = [k for k in tabpfn_sections if k in tabicl_sections]
    for key in tabpfn_sections:
        if key not in tabicl_sections:
            print(f"  Skipping section [{key}]: not present in both tabpfn and tabicl files")
    for key in tabicl_sections:
        if key not in tabpfn_sections:
            print(f"  Skipping section [{key}]: not present in both tabpfn and tabicl files")

    if not common_sections:
        print(f"  No common sections between tabpfn and tabicl for dataset {dataset}, skipped combined plot")
        return

    n_sections = len(common_sections)
    fig, axes = plt.subplots(1, n_sections, figsize=(10 * n_sections, 6), squeeze=False)

    for ax, section_key in zip(axes[0], common_sections):
        plot_combined_section(tabpfn_sections[section_key], tabicl_sections[section_key], section_key, ax)

    display_dataset = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    fig.suptitle(
        f"C-index Test Results — {display_dataset} / TABPFN + TABICL",
        fontsize=19, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_dir = RESULTS_DIR / dataset / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results_aggregated_{dataset}_combined"

    out_pdf = out_dir / f"{stem}.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_pdf}")

    out_png = out_dir / f"{stem}.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"  Saved: {out_png}")

    plt.close(fig)


def main():
    if not RESULTS_DIR.exists():
        print(f"Directory '{RESULTS_DIR}' non trovata.")
        return

    found = False
    files_by_dataset = {}
    for txt_file in sorted(RESULTS_DIR.rglob("results_aggregated_*.txt")):
        m = FILE_RE.match(txt_file.name)
        if not m:
            continue
        found = True
        dataset, model = m.group(1), m.group(2)
        files_by_dataset.setdefault(dataset, {})[model] = txt_file
        print(f"Processing: {txt_file.name}  (dataset={dataset}, model={model})")
        try:
            plot_file(txt_file, dataset, model)
        except Exception as e:
            print(f"  Error: {e}")

    for dataset, models in sorted(files_by_dataset.items()):
        if "tabpfn" in models and "tabicl" in models:
            print(f"Processing combined plot: dataset={dataset} (tabpfn + tabicl)")
            try:
                plot_combined_file(dataset, models["tabpfn"], models["tabicl"])
            except Exception as e:
                print(f"  Error: {e}")

    if not found:
        print("Nessun file results_aggregated_{dataset}_{model}.txt trovato.")


if __name__ == "__main__":
    main()
