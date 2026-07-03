# Design: `plot_nan_results.py`

## Purpose

Visualize the NaN-ratio robustness experiments stored in
`results_nan/results_cv_{dataset}_{model}_nan_aggregated.txt`. These files
compare 6 imputation methods (`constant`, `embeddings`, `imputer_bayesian`,
`knn`, `mean`, `median`) across 5 NaN ratios (0.1, 0.3, 0.5, 0.7, 0.9) and 4
survival models (Cox, Cox Simple, Cox Vanilla, RSF), reporting Train/Test
C-index ± std averaged over seeds.

The goal is to show how Test C-index degrades as missingness increases, for
each imputation method, and to highlight the `embeddings` method (foundation
model based imputation, tied to the file's `model` — tabicl/tabpfn) against
the classical imputation baselines.

## Input format

Each aggregated file has two markdown tables at the top:

- Lines matching `\| \(([\d.]+), '([^']+)'\)\s*\| ... \|` — first table
  (rows 3-24 in the example) is the **Test** C-index table; second table
  (rows 26-47) is the **Train** C-index table. Both share the same row keys
  `(nan_ratio, survival_model)` and the same 6 imputation-method columns.
- The verbose `=== NaN Ratio: X ===` / `-- {model} --` text block below is
  redundant with the two tables and will NOT be parsed — parsing the two
  markdown tables directly is simpler and sufficient.
- Cell format: `0.733 ± 0.013`.

Filenames matched via regex: `results_cv_(.+)_([^_]+)_nan_aggregated\.txt`
→ `(dataset, model)`, e.g. `(HURRAH, tabpfn)`.

## Parsed data shape

```
data[dataset][model]["test"][survival_model][nan_ratio][imputation_method] = (mean, std)
data[dataset][model]["train"][survival_model][nan_ratio][imputation_method] = (mean, std)
```

## Output 1 — per-file robustness plot (one figure per input file)

- 4 subplots side by side, one per survival model (Cox, Cox Simple, Cox
  Vanilla, RSF), matching the existing horizontal-subplot convention used in
  `plot_samplesize_results.py` / `plot_results.py`.
- X axis: NaN ratio (0.1 → 0.9, evenly spaced categorical positions).
- Y axis: Test C-index.
- One line per imputation method with a shaded ±std band and markers,
  mirroring `plot_section` in `plot_samplesize_results.py`.
- Fixed color palette, consistent across all figures produced by this
  script (not model-dependent, since here we're comparing imputation
  methods, not survival models):
  - `embeddings` → strong blue (`#1f77b4`), thicker line (highlighted —
    this is the foundation-model-based method of interest)
  - `constant` → gray (`#7f7f7f`)
  - `mean` → light orange (`#ffbb78`)
  - `median` → orange (`#ff7f0e`)
  - `knn` → green (`#2ca02c`)
  - `imputer_bayesian` → purple (`#9467bd`)
- Figure title uses the display-name mapping (see below), e.g.
  `"C-index Test — Robustezza al Missing | URRAH / TABPFN"`.
- Output path (unchanged from existing convention — display names are for
  labels only, not paths):
  `results/{dataset}/{model}/aggregated/results_nan_{dataset}_{model}_aggregated.pdf`
  and `.png`.

## Output 2 — cross-dataset/model comparison plot (one figure total)

- Isolates only the `embeddings` method's Test C-index and compares it
  across all 4 (dataset, model) combinations found.
- Same 4-subplot-per-survival-model layout; one line per
  `(dataset, model)` combination (up to 4 lines per subplot).
- Legend labels use the display-name mapping, e.g. `"URRAH / TABPFN"`,
  `"IHD / TABICL"`.
- Output path: `results/nan_embeddings_comparison.pdf` and `.png`.

## Dataset display-name mapping (labels only, not paths)

```python
DATASET_DISPLAY_NAME = {
    "HURRAH": "URRAH",
    "OrmoniTirodei": "IHD",
}
```

Applied only to figure titles / legend text. File and directory paths keep
using the raw dataset name (`HURRAH`, `OrmoniTirodei`) to stay consistent
with the existing `results/{dataset}/{model}/...` directory layout.

## Script structure

Single new file `plot_nan_results.py` at the project root, following the
existing scripts' structure (`RESULTS_NAN_DIR = Path("results_nan")`,
`RESULTS_DIR = Path("results")`):

1. `parse_nan_file(filepath) -> dict` — parses both markdown tables into the
   shape above for one file.
2. `plot_per_file(dataset, model, data)` — Output 1.
3. `plot_embeddings_comparison(all_data)` — Output 2, called once after all
   files are parsed.
4. `main()` — globs `results_nan/results_cv_*_nan_aggregated.txt`, parses
   each, calls `plot_per_file` for each, then `plot_embeddings_comparison`
   once with everything collected.

No changes to existing scripts. No new dependencies (matplotlib/numpy
already used elsewhere in the project).
