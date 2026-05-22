import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")

# matches: results_cv_{dataset}_{model}_seed{seed}.txt
# dataset may contain underscores; model is the last token before _seed
FILE_RE = re.compile(r"^results_cv_(.+)_([^_]+)_seed(\d+)\.txt$")

LINE_RE = re.compile(
    r"^(.+?)\s*→\s*mean:\s*([\d.]+)\s*\|\s*std:\s*([\d.]+)\s*\|\s*min:\s*([\d.]+)\s*\|\s*max:\s*([\d.]+)"
)


SECTION_RE = re.compile(r"Preprocess:\s*([-\d]+|NaN)\s*\|")


def parse_file(path: Path) -> dict:
    """Legge un file e restituisce {preprocess_key: {metric_name: {mean, std, min, max}}}"""
    with open(path, "r") as f:
        lines = f.readlines()

    results = {}
    current_key = None
    for line in lines:
        stripped = line.strip()
        sec_match = SECTION_RE.search(stripped)
        if sec_match:
            current_key = sec_match.group(1)
            if current_key not in results:
                results[current_key] = {}
            continue
        if current_key is None:
            continue
        m = LINE_RE.match(stripped)
        if m:
            name = m.group(1).strip()
            results[current_key][name] = {
                "mean": float(m.group(2)),
                "std":  float(m.group(3)),
                "min":  float(m.group(4)),
                "max":  float(m.group(5)),
            }
    return results


def aggregate(all_parsed: list[dict]) -> dict:
    """Media su tutti i file per ogni (key, metric)."""
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for parsed in all_parsed:
        for key, metrics in parsed.items():
            for name, vals in metrics.items():
                for stat, value in vals.items():
                    agg[key][name][stat].append(value)

    final = {}
    for key, metrics in agg.items():
        final[key] = {}
        for name, stats in metrics.items():
            final[key][name] = {
                "mean": sum(stats["mean"]) / len(stats["mean"]),
                "std":  sum(stats["std"])  / len(stats["std"]),
                "min":  min(stats["min"]),
                "max":  max(stats["max"]),
                "n":    len(stats["mean"]),
            }
    return final


def format_results(final: dict, n_files: int, dataset: str, model: str) -> str:
    lines = []
    lines.append("═" * 52)
    lines.append(f"AGGREGATED RESULTS — {dataset} / {model} — {n_files} files")
    lines.append("═" * 52)

    for key, metrics in final.items():
        lines.append("\n" + "─" * 52)
        lines.append(f"TOTAL [{key}]:")
        for name, v in metrics.items():
            lines.append(
                f"  {name} → "
                f"mean: {v['mean']:.6f} | "
                f"std: {v['std']:.6f} | "
                f"min: {v['min']:.6f} | "
                f"max: {v['max']:.6f}  "
                f"(n={v['n']})"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    if not RESULTS_DIR.exists():
        print(f"Directory '{RESULTS_DIR}' non trovata.")
        return

    # Raggruppa i file per (dataset, model) filtrando per pattern results_cv_{dataset}_{model}_seed{seed}.txt
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for txt_file in sorted(RESULTS_DIR.rglob("*.txt")):
        m = FILE_RE.match(txt_file.name)
        if not m:
            continue
        dataset, model = m.group(1), m.group(2)
        groups[(dataset, model)].append(txt_file)

    if not groups:
        print("Nessun file results_cv_{dataset}_{model}_seed{seed}.txt trovato.")
        return

    for (dataset, model), files in sorted(groups.items()):
        print(f"\n{'─' * 52}")
        print(f"Dataset: {dataset}  |  Model: {model}  |  {len(files)} file")

        all_parsed = []
        for path in files:
            try:
                parsed = parse_file(path)
                keys_found = list(parsed.keys())
                print(f"  ✓ {path.name}  →  sezioni: {keys_found}")
                all_parsed.append(parsed)
            except Exception as e:
                print(f"  ✗ {path.name}  →  errore: {e}")

        if not all_parsed:
            print("  Nessun file valido, skipped.")
            continue

        final = aggregate(all_parsed)
        output = format_results(final, len(all_parsed), dataset, model)

        out_dir = RESULTS_DIR / dataset / model / "aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"results_aggregated_{dataset}_{model}.txt"
        with open(out_path, "w") as f:
            f.write(output)
        print(f"  → Salvato: {out_path}")


if __name__ == "__main__":
    main()
