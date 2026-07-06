# Design: Combined TABPFN+TABICL plot in plot_results.py

## Context

`plot_results.py` currently generates one bar chart per
`results_aggregated_{dataset}_{model}.txt` file, i.e. one chart per
(dataset, model) pair. For `tabpfn` and `tabicl`, this means two separate
charts per dataset, even though both files share the same 4 baseline
entries (DeepSurv Vanilla, DeepSurv Simple, RSF, Cox — verified identical
values across tabpfn/tabicl files for the same dataset).

The request: for each dataset, also produce a single combined chart with
TABPFN results, TABICL results, and the (deduplicated) baselines together,
so the two embeddings models can be compared directly.

## Scope

- Additive only: existing per-model plots (`plot_file`, used for `tabpfn`,
  `tabicl`, and `tabdpt`) are unchanged and keep being generated.
- `tabdpt` is not included in the combined plot — it stays as its own
  separate chart, untouched.
- Applies only to datasets that have both a `tabpfn` and a `tabicl`
  aggregated results file. If a dataset is missing one of the two, the
  combined plot is skipped for that dataset with a console message.

## Behavior

### Grouping in `main()`

After discovering result files via the existing `FILE_RE`-matching loop,
group the found `(dataset, model, filepath)` entries by `dataset`. For each
dataset that has both a `tabpfn` and a `tabicl` file, call a new
`plot_combined_file(dataset, tabpfn_path, tabicl_path)`. This runs in
addition to the existing per-file `plot_file(...)` calls, which are
unaffected.

### `plot_combined_file(dataset, tabpfn_path, tabicl_path)`

1. Parse both files with the existing `parse_results_file`.
2. Determine sections to plot: the intersection of section keys present in
   both parsed results (expected: `-1` and `NaN`). Sections present in only
   one file are skipped with a console warning.
3. For each common section, build one subplot (same `1 x n_sections`
   layout as `plot_file`) via a new `plot_combined_section(...)`:
   - Test entries only (reuse `is_test_entry`).
   - Bar order: 6 TABPFN model-specific entries, then 6 TABICL
     model-specific entries, then 4 baseline entries. Baseline entries are
     taken from the tabpfn section only; the matching baseline entries in
     the tabicl section are ignored (not plotted, not deduplicated by
     value-checking — assumed identical per current data).
   - Model-specific entries identified the same way as today: name starts
     with the file's model string (`model_color`'s existing check).
4. Figure title: `f"C-index Test Results — {display_dataset} / TABPFN + TABICL"`
   using the existing `DATASET_DISPLAY_NAMES` lookup.
5. Output path: `results/{dataset}/aggregated/results_aggregated_{dataset}_combined.pdf`
   and matching `.png`, mirroring the save logic in `plot_file` (creating
   the `aggregated` dir if needed).

### Color / hatch scheme

Reuse the existing 6-color palette for model-specific entry types (Tuned
DeepSurv, Tuned RSF, Vanilla, Simple, RSF, Cox) — same hue regardless of
which embeddings model produced the entry. To distinguish TABPFN bars from
TABICL bars sharing the same type, vary the hatch instead of the color:
- TABPFN model-specific bars: hatch `"///"` (same as today's single-model
  plots).
- TABICL model-specific bars: hatch `"xxx"`.
- Baseline bars: no hatch, existing baseline colors (unchanged).

This requires a small generalization of `model_color(name, model)` (or a
sibling helper) so the hatch depends on which model string matched, not a
single fixed `"///"`.

### Labels

Reuse `legend_label(name, model)` unchanged — it already prefixes
model-specific entries with `TABPFN -` / `TABICL -` (via `model.upper()`)
and leaves baseline names unprefixed. Calling it once per file's entries
with the correct `model` argument produces the right combined legend.

## Out of scope

- No change to `tabdpt` handling.
- No change to existing single-model plot output (files, paths, content).
- No cross-checking that tabpfn/tabicl baseline values actually match —
  baselines are trusted to be identical and only the tabpfn copy is used.
