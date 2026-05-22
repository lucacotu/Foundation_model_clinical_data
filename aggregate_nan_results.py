#!/usr/bin/env python3
"""Aggregate tabpfn CV results across multiple seeds.

Reads all results_cv_tabpfn_nan_seed*.txt files and produces
results_cv_tabpfn_nan_aggregated.txt with mean ± std across seeds.
The ± is the mean of the per-seed standard deviations (CV variability).
"""

import re
import glob
import numpy as np
from collections import defaultdict
import os


def parse_result_file(filepath):
    """Parse a single seed result file into {(nan_ratio, model, imputer): {train, test}}."""
    results = {}
    key_order = []
    current_nan_ratio = None
    current_model = None
    imp_line_re = re.compile(
        r'^\s*([\w]+)\s*\|\s*Train:\s*([\d.]+)\s*±\s*([\d.]+)\s*\|\s*Test:\s*([\d.]+)\s*±\s*([\d.]+)'
    )

    with open(filepath) as f:
        for line in f:
            m = re.match(r'=== NaN Ratio: ([\d.]+) ===', line.strip())
            if m:
                current_nan_ratio = float(m.group(1))
                current_model = None
                continue

            m = re.match(r'-- (.+?) --', line.strip())
            if m:
                current_model = m.group(1)
                continue

            if current_nan_ratio is not None and current_model is not None:
                m = imp_line_re.match(line)
                if m:
                    key = (current_nan_ratio, current_model, m.group(1))
                    results[key] = {
                        'train': (float(m.group(2)), float(m.group(3))),
                        'test':  (float(m.group(4)), float(m.group(5))),
                    }
                    key_order.append(key)

    return results, key_order


def _ordered_unique(iterable):
    seen = set()
    out = []
    for x in iterable:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def aggregate_files(file_pattern):
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {file_pattern}")
    print(f"Aggregating {len(files)} file(s): {', '.join(os.path.basename(f) for f in files)}")

    global_key_order = []
    collected = defaultdict(lambda: {'train': [], 'test': []})

    for fp in files:
        parsed, key_order = parse_result_file(fp)
        global_key_order.extend(key_order)
        for key, vals in parsed.items():
            collected[key]['train'].append(vals['train'])
            collected[key]['test'].append(vals['test'])

    aggregated = {}
    for key in _ordered_unique(global_key_order):
        train = np.array(collected[key]['train'])  # shape (n_seeds, 2): col0=mean, col1=std
        test  = np.array(collected[key]['test'])
        aggregated[key] = {
            'train': (float(np.mean(train[:, 0])), float(np.mean(train[:, 1]))),
            'test':  (float(np.mean(test[:, 0])),  float(np.mean(test[:, 1]))),
        }

    return aggregated


def write_output(aggregated, output_path, n_seeds):
    all_keys    = list(aggregated.keys())
    nan_ratios  = _ordered_unique(sorted(set(k[0] for k in all_keys)))
    models      = _ordered_unique(k[1] for k in all_keys)
    imp_methods = _ordered_unique(k[2] for k in all_keys)

    def fmt(mean, std):
        return f"{mean:.3f} ± {std:.3f}"

    col_w = {imp: max(len(imp), 13) for imp in imp_methods}

    def table_header():
        cols = ' | '.join(imp.ljust(col_w[imp]) for imp in imp_methods)
        return f"|{'':22}| {cols} |"

    def table_sep():
        cols = '|'.join(':' + '-' * col_w[imp] for imp in imp_methods)
        return f"|:{'-' * 21}|{cols}|"

    def table_row(nan_ratio, model, score_type):
        label = f"({nan_ratio}, '{model}')"
        cells = []
        for imp in imp_methods:
            key = (nan_ratio, model, imp)
            v = fmt(*aggregated[key][score_type]) if key in aggregated else 'N/A'
            cells.append(v.ljust(col_w[imp]))
        return f"| {label:<21}| {' | '.join(cells)} |"

    lines = [
        f"# Aggregated over {n_seeds} seed(s) — ± is mean of per-seed CV std",
        '',
    ]

    # Markdown tables: test scores first, then train scores
    for score_type in ('test', 'train'):
        lines += [table_header(), table_sep()]
        for nan_ratio in nan_ratios:
            for model in models:
                lines.append(table_row(nan_ratio, model, score_type))
        lines.append('')

    # Detailed sections per NaN ratio
    imp_col_w = max(len(imp) for imp in imp_methods)
    for nan_ratio in nan_ratios:
        lines += [f"=== NaN Ratio: {nan_ratio} ===", '']
        for model in models:
            lines.append(f"-- {model} --")
            for imp in imp_methods:
                key = (nan_ratio, model, imp)
                if key in aggregated:
                    tr = fmt(*aggregated[key]['train'])
                    te = fmt(*aggregated[key]['test'])
                    lines.append(f"  {imp:>{imp_col_w}} | Train: {tr} | Test: {te}")
            lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Written → {output_path}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, 'results_nan')

    all_files = glob.glob(os.path.join(results_dir, 'results_cv_*_nan_seed*.txt'))
    model_re = re.compile(r'results_cv_(.+?)_nan_seed[\d\w]+\.txt$')

    models_found = _ordered_unique(
        m.group(1)
        for f in sorted(all_files)
        for m in [model_re.search(os.path.basename(f))]
        if m
    )

    if not models_found:
        raise FileNotFoundError(f"No seed result files found in {results_dir}")

    for model in models_found:
        pattern = os.path.join(results_dir, f'results_cv_{model}_nan_seed*.txt')
        aggregated = aggregate_files(pattern)
        n_seeds = len(glob.glob(pattern))
        out = os.path.join(results_dir, f'results_cv_{model}_nan_aggregated.txt')
        write_output(aggregated, out, n_seeds)
