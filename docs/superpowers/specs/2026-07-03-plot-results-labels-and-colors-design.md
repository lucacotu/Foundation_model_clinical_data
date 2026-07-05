# Design: Dataset display labels and embeddings color palette in plot_results.py

## Context

`plot_results.py` generates per-dataset/per-model bar charts of C-index test
results from `results_aggregated_{dataset}_{model}.txt` files. Two issues
were raised:

1. The dataset names used internally (`OrmoniTirodei`, `HURRAH`) are not the
   names that should appear on the final plots.
2. In `model_color()`, all "embeddings" model variants (entries whose name
   starts with the file's model, e.g. TabPFN/TabICL) are colored with shades
   of blue, which are hard to tell apart from one another.

## 1. Dataset display labels

Only the chart title (`fig.suptitle` in `plot_file()`) should show the new
names. File names, output directories, and console log messages keep using
the original dataset identifiers (`OrmoniTirodei`, `HURRAH`), since those
drive `RESULTS_DIR.rglob(...)` matching and output paths elsewhere.

Mapping:
- `OrmoniTirodei` → `IHD`
- `HURRAH` → `URRAH`

Implementation: a module-level dict `DATASET_DISPLAY_NAMES`, looked up with
`.get(dataset, dataset)` (falls back to the raw name for any other dataset)
when building the suptitle string.

## 2. Embeddings color palette

Baseline model colors (the `else` branch of `model_color()` — entries that
don't start with the file's model) are unchanged: they already use
distinguishable warm colors (orange/green/red shades).

Embeddings model colors (the `if` branch — entries starting with the file's
model, e.g. `tabpfn tuned deepsurv`) switch from blue shades to a qualitative
palette with distinct hues, avoiding hues already used by the baseline
group:

| Variant           | Color     | Hex       |
|--------------------|-----------|-----------|
| tuned deepsurv      | blue      | `#1f77b4` |
| tuned rsf           | purple    | `#9467bd` |
| vanilla             | cyan      | `#17becf` |
| simple              | pink      | `#e377c2` |
| rsf                 | brown     | `#8c564b` |
| cox                 | olive     | `#bcbd22` |
| fallback (no match) | dark gray | `#7f7f7f` |

Additionally, bars belonging to the embeddings group get a hatch pattern
(`"///"`) so the group stays visually identifiable as a set even in
grayscale/print, independent of color. Baseline bars keep no hatch (empty
string).

`model_color()` currently returns only a color string. It will be changed to
return a `(color, hatch)` tuple, and `plot_section()` will apply the hatch
per-bar via `bar.set_hatch(...)` after `ax.bar(...)` (matplotlib's `ax.bar`
does not accept a list of different hatch strings via a single `hatch=`
kwarg, so hatches must be set per-bar).

## Out of scope

- No change to baseline colors.
- No change to output file/directory naming.
- No change to legend rendering logic beyond picking up the new
  color/hatch per entry.
