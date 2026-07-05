# Plot Results Labels & Colors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In `plot_results.py`, show display-friendly dataset names in chart titles (`OrmoniTirodei`→`IHD`, `HURRAH`→`URRAH`) and replace the all-blue embeddings-model color palette with a distinguishable qualitative palette + hatch pattern, without touching baseline colors or output file/directory naming.

**Architecture:** Single-file change to `plot_results.py`. No new modules. Verification is done by running `python plot_results.py` against the existing `results/OrmoniTirodei/tabpfn/aggregated/results_aggregated_OrmoniTirodei_tabpfn.txt` fixture and visually inspecting the regenerated PNG (there is no pytest suite covering this script).

**Tech Stack:** Python, matplotlib (already a dependency).

## Global Constraints

- Only `fig.suptitle` text changes for dataset display names — file names, output directories (`results/{dataset}/{model}/aggregated/...`), and console `print()` messages keep using the raw `dataset` value (`OrmoniTirodei`, `HURRAH`).
- Baseline model colors (the `else` branch of `model_color()`) are unchanged: `deepsurv vanilla` → `#ff7f0e`, `deepsurv simple` → `#ffbb78`, `rsf` → `#2ca02c`, `cox` → `#98df8a`, fallback → `#d62728`. No hatch on baseline bars.
- Embeddings model colors (the `if` branch of `model_color()`) use this exact mapping:
  - `tuned deepsurv` → `#1f77b4`
  - `tuned rsf` → `#9467bd`
  - `vanilla` → `#17becf`
  - `simple` → `#e377c2`
  - `rsf` → `#8c564b`
  - `cox` → `#bcbd22`
  - fallback (no match) → `#7f7f7f`
  - All embeddings bars get hatch `"///"`.
- `model_color()` changes its return type from `str` (color) to `tuple[str, str]` (`(color, hatch)`). Every call site must be updated in the same task.

---

### Task 1: Dataset display names in chart title

**Files:**
- Modify: `plot_results.py:151-167` (function `plot_file`, plus a new module-level constant near the top)

**Interfaces:**
- Produces: `DATASET_DISPLAY_NAMES: dict[str, str]` module-level constant, used only inside `plot_file()`.

- [ ] **Step 1: Add the display-name mapping constant**

In `plot_results.py`, add this right after the `FILE_RE` definition (currently ends at line 11):

```python
DATASET_DISPLAY_NAMES = {
    "OrmoniTirodei": "IHD",
    "HURRAH": "URRAH",
}
```

- [ ] **Step 2: Use the mapping in the suptitle**

In `plot_file()`, replace:

```python
    fig.suptitle(
        f"C-index Test Results — {dataset} / {model}",
        fontsize=13, fontweight="bold", y=1.01,
    )
```

with:

```python
    display_dataset = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    fig.suptitle(
        f"C-index Test Results — {display_dataset} / {model}",
        fontsize=13, fontweight="bold", y=1.01,
    )
```

Do not touch the `out_dir` line (`RESULTS_DIR / dataset / model / "aggregated"`) or the `print(f"  Saved: ...")` lines below it — they must keep using the raw `dataset`.

- [ ] **Step 3: Run the script and verify the title**

Run: `python plot_results.py`

Expected console output includes lines like:
```
Processing: results_aggregated_OrmoniTirodei_tabpfn.txt  (dataset=OrmoniTirodei, model=tabpfn)
  Saved: results/OrmoniTirodei/tabpfn/aggregated/results_aggregated_OrmoniTirodei_tabpfn.pdf
  Saved: results/OrmoniTirodei/tabpfn/aggregated/results_aggregated_OrmoniTirodei_tabpfn.png
```
(paths unchanged — only the in-image title differs). Open `results/OrmoniTirodei/tabpfn/aggregated/results_aggregated_OrmoniTirodei_tabpfn.png` and confirm the title reads `C-index Test Results — IHD / tabpfn`. Open the corresponding HURRAH png and confirm it reads `... — URRAH / ...`.

- [ ] **Step 4: Commit**

```bash
git add plot_results.py
git commit -m "plot_results: show IHD/URRAH display names in chart titles"
```

---

### Task 2: Distinguishable colors + hatch for embeddings models

**Files:**
- Modify: `plot_results.py:49-77` (`model_color`), `plot_results.py:109-148` (`plot_section`)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `model_color(name, model) -> tuple[str, str]` (color, hatch) — used by `plot_section()`.

- [ ] **Step 1: Rewrite `model_color` to return `(color, hatch)`**

Replace the whole function (current lines 49-77):

```python
def model_color(name, model):
    """Assign a consistent (color, hatch) pair based on whether the entry belongs to the file's model."""
    name_lower = name.lower()
    if name_lower.startswith(model.lower()):
        # Distinct hues per embeddings-model variant, hatched to mark the group
        hatch = "///"
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
```

- [ ] **Step 2: Update `plot_section` to unpack colors/hatches and apply per-bar hatch**

In `plot_section()`, replace:

```python
    colors = [model_color(n, model) for n in names]
    labels = [legend_label(n, model) for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black",
                  linewidth=0.6, error_kw={"elinewidth": 1.5, "ecolor": "black"})
```

with:

```python
    colors_hatches = [model_color(n, model) for n in names]
    colors = [c for c, _ in colors_hatches]
    hatches = [h for _, h in colors_hatches]
    labels = [legend_label(n, model) for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black",
                  linewidth=0.6, error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
```

- [ ] **Step 3: Update the legend to carry color + hatch**

In `plot_section()`, replace:

```python
    # Build legend from the actual entries present in the chart
    seen = {}
    for n, lbl, clr in zip(names, labels, colors):
        if lbl not in seen:
            seen[lbl] = clr
    patches = [mpatches.Patch(color=clr, label=lbl) for lbl, clr in seen.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right", framealpha=0.8)
```

with:

```python
    # Build legend from the actual entries present in the chart
    seen = {}
    for n, lbl, clr, h in zip(names, labels, colors, hatches):
        if lbl not in seen:
            seen[lbl] = (clr, h)
    patches = [
        mpatches.Patch(facecolor=clr, hatch=h, edgecolor="black", label=lbl)
        for lbl, (clr, h) in seen.items()
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right", framealpha=0.8)
```

- [ ] **Step 4: Run the script and verify colors/hatch visually**

Run: `python plot_results.py`

Open `results/OrmoniTirodei/tabpfn/aggregated/results_aggregated_OrmoniTirodei_tabpfn.png` and confirm:
- Bars whose legend label starts with `TABPFN -` are hatched with diagonal lines and use distinct hues (blue/purple/cyan/pink/brown/olive) rather than all-blue shades.
- Baseline bars (`DeepSurv Vanilla`, `DeepSurv Simple`, `RSF`, `Cox`) are solid (no hatch) and keep their existing orange/green/red shades.
- The legend swatches show the hatch pattern for embeddings entries.

Repeat the check on one `tabicl` and one `HURRAH` output to confirm the mapping generalizes.

- [ ] **Step 5: Commit**

```bash
git add plot_results.py
git commit -m "plot_results: distinct hues + hatch for embeddings-model bars"
```
