# Combined TABPFN+TABICL Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add, for each dataset, one combined bar chart showing TABPFN + TABICL results plus their shared baselines, in addition to (not replacing) the existing per-model plots produced by `plot_results.py`.

**Architecture:** Extract the bar-drawing logic of `plot_section` into a reusable `render_bars` helper that operates on a pre-built list of per-bar dicts (mean/std/color/hatch/label). `plot_section` becomes a thin wrapper that builds that list for one file's entries; a new `plot_combined_section` builds the same shape of list by merging TABPFN model-specific entries, TABICL model-specific entries, and de-duplicated baseline entries (taken only from the TABPFN file). A new `plot_combined_file` mirrors `plot_file` but takes two source files and writes to a dataset-level output directory. `main()` is extended to detect, per dataset, whether both a `tabpfn` and `tabicl` file exist and if so invoke the combined path in addition to the existing per-file loop.

**Tech Stack:** Python 3.12, matplotlib, numpy — all already in use in `plot_results.py`. No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-06-plot-combined-tabpfn-tabicl-design.md`
- Existing per-model plots (`tabpfn`, `tabicl`, `tabdpt`) must remain byte-for-byte unchanged — this is additive only.
- `tabdpt` is never included in the combined plot.
- Combined plot only generated for datasets that have both a `tabpfn` and `tabicl` aggregated results file.
- Baseline bars come only from the TABPFN file's entries; the TABICL file's baseline entries are ignored.
- TABPFN model-specific bars use hatch `"///"`; TABICL model-specific bars use hatch `"xxx"`; baseline bars keep no hatch.
- Bar order in the combined chart: all TABPFN model-specific bars, then all TABICL model-specific bars, then baseline bars.
- Combined output path: `results/{dataset}/aggregated/results_aggregated_{dataset}_combined.{pdf,png}`.
- Use `.venv/bin/python` to run the script (project venv), not bare `python`/`python3`.

---

### Task 1: Extract `render_bars` helper from `plot_section` (pure refactor, no behavior change)

**Files:**
- Modify: `plot_results.py:115-162` (the `plot_section` function)

**Interfaces:**
- Produces: `render_bars(ax, entries_data, section_label)` where `entries_data` is a list of dicts, each with keys `name`, `mean`, `std`, `color`, `hatch`, `label`. Later tasks (2, 3) call this function directly.
- Produces: `plot_section(entries, section_label, ax, model)` keeps its exact existing signature and behavior — callers in `plot_file` are unaffected.

- [ ] **Step 1: Record baseline output hashes before touching any code**

Run:
```bash
cd "/Users/luca/Library/Mobile Documents/com~apple~CloudDocs/Università/Magistrale/Tesi/Progetto"
.venv/bin/python plot_results.py
shasum -a 256 results/HURRAH/tabpfn/aggregated/results_aggregated_HURRAH_tabpfn.png
shasum -a 256 results/HURRAH/tabicl/aggregated/results_aggregated_HURRAH_tabicl.png
shasum -a 256 results/OrmoniTirodei/tabdpt/aggregated/results_aggregated_OrmoniTirodei_tabdpt.png
```
Expected: the script runs to completion printing `Processing: ...` lines and `Saved: ...` lines for every existing file, no `Error:` lines. Note down the three hash values printed — they are the "before" baseline you'll diff against in Task 2's Step 3 and Task 5's Step 4.

- [ ] **Step 2: Replace `plot_section` with `render_bars` + thin wrapper**

Replace the current `plot_section` function (`plot_results.py:115-162`) with:

```python
def render_bars(ax, entries_data, section_label):
    """Draw one axis of bars from pre-built per-bar data.

    entries_data: list of dicts with keys mean, std, color, hatch, label.
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
    ax.legend(handles=patches, fontsize=8, loc="lower right", framealpha=0.8)


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
```

- [ ] **Step 3: Re-run the script and confirm outputs are byte-identical to baseline**

Run:
```bash
.venv/bin/python plot_results.py
shasum -a 256 results/HURRAH/tabpfn/aggregated/results_aggregated_HURRAH_tabpfn.png
shasum -a 256 results/HURRAH/tabicl/aggregated/results_aggregated_HURRAH_tabicl.png
shasum -a 256 results/OrmoniTirodei/tabdpt/aggregated/results_aggregated_OrmoniTirodei_tabdpt.png
```
Expected: all three hashes are identical to the ones recorded in Step 1, and the script prints no `Error:` lines.

- [ ] **Step 4: Commit**

```bash
git add plot_results.py
git commit -m "Refactor plot_section into render_bars helper (no behavior change)"
```

---

### Task 2: Add a `hatch` parameter to `model_color`

**Files:**
- Modify: `plot_results.py:54-83` (the `model_color` function)

**Interfaces:**
- Consumes: nothing new.
- Produces: `model_color(name, model, hatch="///")` — same return shape `(color, hatch)` as before. Existing callers (`plot_section`, via Task 1's rewrite) don't pass `hatch`, so behavior is unchanged. Task 3's `plot_combined_section` will pass `hatch="///"` or `hatch="xxx"` explicitly.

- [ ] **Step 1: Change the function signature and drop the local `hatch = "///"` assignment**

Replace `plot_results.py:54-83`:

```python
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
```

- [ ] **Step 2: Re-run the script and confirm outputs are still byte-identical to Task 1's baseline**

Run:
```bash
.venv/bin/python plot_results.py
shasum -a 256 results/HURRAH/tabpfn/aggregated/results_aggregated_HURRAH_tabpfn.png
shasum -a 256 results/HURRAH/tabicl/aggregated/results_aggregated_HURRAH_tabicl.png
```
Expected: hashes match Task 1 Step 1's recorded values; no `Error:` lines.

- [ ] **Step 3: Commit**

```bash
git add plot_results.py
git commit -m "Add optional hatch parameter to model_color"
```

---

### Task 3: Add `plot_combined_section`

**Files:**
- Modify: `plot_results.py` — add the new function directly after `plot_section` (i.e., after the block added in Task 1, before `plot_file`)

**Interfaces:**
- Consumes: `render_bars(ax, entries_data, section_label)` (Task 1), `model_color(name, model, hatch="///")` (Task 2), `legend_label(name, model)` (unchanged, `plot_results.py:86-112`), `is_test_entry(name)` (unchanged, `plot_results.py:48-51`).
- Produces: `plot_combined_section(tabpfn_entries, tabicl_entries, section_label, ax)` — takes the raw entry lists for one section from each of the two parsed files (as returned by `parse_results_file`) and draws the merged bars onto `ax`. Task 4 calls this once per common section.

- [ ] **Step 1: Add the function**

```python
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

    render_bars(ax, entries_data, section_label)
```

- [ ] **Step 2: Verify bar counts and hatches against the real HURRAH data**

Run:
```bash
.venv/bin/python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from plot_results import parse_results_file, plot_combined_section

tabpfn = parse_results_file(Path('results/HURRAH/tabpfn/aggregated/results_aggregated_HURRAH_tabpfn.txt'))
tabicl = parse_results_file(Path('results/HURRAH/tabicl/aggregated/results_aggregated_HURRAH_tabicl.txt'))

fig, ax = plt.subplots()
plot_combined_section(tabpfn['-1'], tabicl['-1'], '-1', ax)
hatches = [b.get_hatch() for b in ax.patches]
assert len(ax.patches) == 16, f'expected 16 bars for section -1, got {len(ax.patches)}'
assert hatches[:6] == ['///'] * 6, hatches[:6]
assert hatches[6:12] == ['xxx'] * 6, hatches[6:12]
assert hatches[12:] == [None] * 4 or hatches[12:] == [''] * 4, hatches[12:]
plt.close(fig)

fig, ax = plt.subplots()
plot_combined_section(tabpfn['NaN'], tabicl['NaN'], 'NaN', ax)
assert len(ax.patches) == 12, f'expected 12 bars for section NaN, got {len(ax.patches)}'
plt.close(fig)

print('OK')
"
```
Expected: prints `OK` with no assertion errors.

- [ ] **Step 3: Commit**

```bash
git add plot_results.py
git commit -m "Add plot_combined_section for merged TABPFN+TABICL bars"
```

---

### Task 4: Add `plot_combined_file`

**Files:**
- Modify: `plot_results.py` — add the new function directly after `plot_file` (`plot_results.py:165-197`), before `main`

**Interfaces:**
- Consumes: `parse_results_file(filepath)` (unchanged, `plot_results.py:19-45`), `plot_combined_section(...)` (Task 3), `DATASET_DISPLAY_NAMES` (module-level dict, unchanged), `RESULTS_DIR` (module-level `Path`, unchanged).
- Produces: `plot_combined_file(dataset: str, tabpfn_path: Path, tabicl_path: Path)`. Task 5's `main()` calls this once per dataset that has both files.

- [ ] **Step 1: Add the function**

```python
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
```

- [ ] **Step 2: Verify it produces the expected output files**

Run:
```bash
rm -rf results/HURRAH/aggregated
.venv/bin/python -c "
from pathlib import Path
from plot_results import plot_combined_file

plot_combined_file(
    'HURRAH',
    Path('results/HURRAH/tabpfn/aggregated/results_aggregated_HURRAH_tabpfn.txt'),
    Path('results/HURRAH/tabicl/aggregated/results_aggregated_HURRAH_tabicl.txt'),
)
"
ls -la results/HURRAH/aggregated/
```
Expected: two `Saved: ...` lines printed, and `ls` shows both `results_aggregated_HURRAH_combined.pdf` and `results_aggregated_HURRAH_combined.png` with non-zero size.

- [ ] **Step 3: Commit**

```bash
git add plot_results.py
git commit -m "Add plot_combined_file to render the merged dataset-level chart"
```

---

### Task 5: Wire the combined plot into `main()`

**Files:**
- Modify: `plot_results.py:200-223` (the `main` function)

**Interfaces:**
- Consumes: `FILE_RE` (unchanged module-level regex), `plot_file(filepath, dataset, model)` (unchanged), `plot_combined_file(dataset, tabpfn_path, tabicl_path)` (Task 4).

- [ ] **Step 1: Replace `main()`**

Replace `plot_results.py:200-223`:

```python
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
```

- [ ] **Step 2: Run the full script**

Run:
```bash
.venv/bin/python plot_results.py
```
Expected: prints `Processing: ...` for every `tabpfn`/`tabicl`/`tabdpt` file as before, then `Processing combined plot: dataset=HURRAH (tabpfn + tabicl)` and `Processing combined plot: dataset=OrmoniTirodei (tabpfn + tabicl)`, each followed by two `Saved: ...` lines. No `Error:` lines anywhere.

- [ ] **Step 3: Check the combined output files exist for both datasets**

Run:
```bash
ls -la results/HURRAH/aggregated/ results/OrmoniTirodei/aggregated/
```
Expected: both directories contain `results_aggregated_{dataset}_combined.pdf` and `.png`, non-zero size.

- [ ] **Step 4: Confirm single-model outputs are still byte-identical to the Task 1 baseline**

Run:
```bash
shasum -a 256 results/HURRAH/tabpfn/aggregated/results_aggregated_HURRAH_tabpfn.png
shasum -a 256 results/HURRAH/tabicl/aggregated/results_aggregated_HURRAH_tabicl.png
```
Expected: hashes match the values recorded in Task 1 Step 1.

- [ ] **Step 5: Commit**

```bash
git add plot_results.py
git commit -m "Generate combined TABPFN+TABICL plot per dataset in main()"
```

---

### Task 6: Full run and visual sanity check

**Files:**
- None (verification only)

**Interfaces:**
- None — this task only exercises the code from Tasks 1–5 end to end.

- [ ] **Step 1: Clean run from scratch**

Run:
```bash
rm -rf results/HURRAH/aggregated results/OrmoniTirodei/aggregated
.venv/bin/python plot_results.py
```
Expected: no `Error:` lines; `Saved:` lines for every per-model file plus the two combined datasets.

- [ ] **Step 2: Visually inspect one combined chart**

Read the file `results/HURRAH/aggregated/results_aggregated_HURRAH_combined.png` (e.g. via the IDE or an image viewer/the Read tool) and confirm:
- The `-1` subplot has 16 bars: 6 with `///` hatch labeled `TABPFN - ...`, 6 with `xxx` hatch labeled `TABICL - ...`, and 4 unhatched baseline bars with plain labels (no `TABPFN -`/`TABICL -` prefix).
- The `NaN` subplot has 12 bars (6 `///` + 6 `xxx`, no baseline).
- The legend lists both TABPFN and TABICL entries plus baseline entries without duplicates.
- The figure title reads `C-index Test Results — URRAH / TABPFN + TABICL`.

- [ ] **Step 3: Report result to the user**

No commit needed for this task — it's verification only. Summarize the visual check result in the conversation.
