#!/usr/bin/env python3
"""
Aggregates SHAP result files across seeds for each (dataset, model) directory.

Reads all results/{dataset}/{model}/shap_{dataset}_{model}_{preprocessing}_seed{seed}.txt files,
computes mean Mean|SHAP| and mean Std|SHAP| per feature per section, grouped by preprocessing,
and writes results/shap_{dataset}_{model}_aggregated.txt.

Usage: python aggregate_shap_results.py [results_dir]
"""

import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_shap_file(filepath):
    """
    Parse a single SHAP result file.

    Returns:
        preprocess (str): preprocessing identifier from the file header
        ordered_sections (list[str]): section names in order of appearance
        section_data (dict): {section_name: (ordered_feature_names, {feature_name: (mean_shap, std_shap)})}

    Feature names may contain '\\n' for names that span multiple lines in the source file.
    """
    with open(filepath) as f:
        lines = f.readlines()

    header = lines[0].strip()
    m = re.search(r'preprocess:\s*(\S+)', header)
    preprocess = m.group(1) if m else "unknown"

    ordered_sections = []
    section_data = {}
    current_section = None
    current_features = {}
    current_ordered = []
    pending = []

    # Matches a line ending with two floats separated by whitespace (the data values).
    # The feature name precedes at least two spaces before the first float.
    value_re = re.compile(r'^(.*?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s*$')

    for line in lines[2:]:  # skip title and ====== separator
        s = line.rstrip('\n')
        stripped = s.strip()

        section_m = re.match(r'^---\s+(.+?)\s+---$', stripped)
        if section_m:
            if current_section is not None:
                section_data[current_section] = (current_ordered, current_features)
                ordered_sections.append(current_section)
            current_section = section_m.group(1)
            current_features = {}
            current_ordered = []
            pending = []
            continue

        if not stripped or re.match(r'^[-=]+$', stripped) or 'Mean|SHAP|' in stripped:
            continue

        vm = value_re.match(s)
        if vm:
            name_part = vm.group(1).strip()
            mean_shap = float(vm.group(2))
            std_shap = float(vm.group(3))
            pending.append(name_part)
            feature_name = '\n'.join(pending)
            pending = []
            current_ordered.append(feature_name)
            current_features[feature_name] = (mean_shap, std_shap)
        else:
            # First line of a multi-line feature name
            pending.append(stripped)

    if current_section is not None:
        section_data[current_section] = (current_ordered, current_features)
        ordered_sections.append(current_section)

    return preprocess, ordered_sections, section_data


def aggregate(files_by_preprocess):
    """
    Compute per-feature averages across seeds for each (preprocessing, section).

    Args:
        files_by_preprocess: {preprocess: [(ordered_sections, section_data), ...]}

    Returns:
        {preprocess: (ordered_section_names, {section: (sorted_feature_names, {feat: (avg_mean, avg_std)})})}
    """
    result = {}

    for preprocess, file_list in files_by_preprocess.items():
        # Preserve section order from first file, append any extras from later files
        all_section_order = []
        seen_sections = set()
        all_feature_values = defaultdict(lambda: defaultdict(list))  # section -> feat -> [(mean, std)]

        for ordered_sections, section_data in file_list:
            for sec in ordered_sections:
                if sec not in seen_sections:
                    all_section_order.append(sec)
                    seen_sections.add(sec)
            for sec, (_, feat_data) in section_data.items():
                for feat, vals in feat_data.items():
                    all_feature_values[sec][feat].append(vals)

        agg_sections = {}
        for sec in all_section_order:
            feat_vals = all_feature_values[sec]
            agg_features = {
                feat: (
                    sum(v[0] for v in val_list) / len(val_list),
                    sum(v[1] for v in val_list) / len(val_list),
                )
                for feat, val_list in feat_vals.items()
            }
            sorted_feats = sorted(agg_features, key=lambda f: agg_features[f][0], reverse=True)
            agg_sections[sec] = (sorted_feats, agg_features)

        result[preprocess] = (all_section_order, agg_sections)

    return result


def format_data_line(feature_name, mean_shap, std_shap, col_width=42):
    """
    Format a feature row. For multi-line feature names, prefix lines are
    written without values, matching the original file style.
    """
    parts = feature_name.split('\n')
    last = parts[-1]
    value_line = f"{last:<{col_width}}{mean_shap:>12.6f}{std_shap:>14.6f}"
    if len(parts) == 1:
        return value_line
    prefix = '\n'.join(parts[:-1])
    return f"{prefix}\n{value_line}"


def write_aggregated(output_path, dataset, model, aggregated, n_seeds_by_preprocess):
    col_width = 42
    header_line = f"{'Feature':<{col_width}}{'Mean|SHAP|':>12}{'Std|SHAP|':>14}"
    sep_line = "-" * (col_width + 26)

    def preprocess_sort_key(p):
        # Put numeric preprocessing (-1) before NaN, rest alphabetically
        if p == '-1':
            return (0, p)
        if p == 'NaN':
            return (1, p)
        return (2, p)

    out = []
    out.append(f"SHAP Aggregated — {dataset} | {model}")
    out.append("=" * 70)
    out.append("")

    for preprocess in sorted(aggregated, key=preprocess_sort_key):
        n_seeds = n_seeds_by_preprocess[preprocess]
        ordered_sections, agg_sections = aggregated[preprocess]

        out.append("=" * 70)
        out.append(f"  Preprocessing: {preprocess}  (aggregated over {n_seeds} seed(s))")
        out.append("=" * 70)
        out.append("")

        for sec in ordered_sections:
            if sec not in agg_sections:
                continue
            sorted_feats, feat_data = agg_sections[sec]

            out.append(f"--- {sec} ---")
            out.append(header_line)
            out.append(sep_line)

            for feat in sorted_feats:
                mean_shap, std_shap = feat_data[feat]
                out.append(format_data_line(feat, mean_shap, std_shap, col_width))

            out.append("")

    output_path.write_text('\n'.join(out) + '\n', encoding='utf-8')


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

    if not results_dir.exists():
        print(f"Error: directory '{results_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    processed = set()

    for shap_file in sorted(results_dir.rglob("shap_*.txt")):
        # Skip already-generated aggregated files
        if 'aggregated' in shap_file.stem:
            continue

        rel = shap_file.relative_to(results_dir)
        parts = rel.parts
        if len(parts) < 3:
            continue
        dataset, model = parts[0], parts[1]

        key = (dataset, model)
        if key in processed:
            continue
        processed.add(key)

        model_dir = results_dir / dataset / model
        print(f"Processing: {dataset}/{model}")

        files_by_preprocess = defaultdict(list)
        n_seeds_by_preprocess = defaultdict(int)

        for f in sorted(model_dir.glob("shap_*.txt")):
            if 'aggregated' in f.stem:
                continue
            preprocess, ordered_sections, section_data = parse_shap_file(f)
            files_by_preprocess[preprocess].append((ordered_sections, section_data))
            n_seeds_by_preprocess[preprocess] += 1
            print(f"  Parsed: {f.name}  (preprocess={preprocess})")

        if not files_by_preprocess:
            print("  No files found, skipping.")
            continue

        aggregated = aggregate(files_by_preprocess)
        out_dir = results_dir / dataset / model / "aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"shap_{dataset}_{model}_aggregated.txt"
        write_aggregated(output_path, dataset, model, aggregated, n_seeds_by_preprocess)
        print(f"  -> Written: {output_path}\n")


if __name__ == "__main__":
    main()
