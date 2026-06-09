import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")

# matches: results_samplesize_{dataset}_{model}_seed{seed}.txt  (no _data suffix)
FILE_RE = re.compile(r"^results_samplesize_(.+)_([^_]+)_seed(\d+)\.txt$")

PREPROCESS_RE = re.compile(r"^Preprocess:\s*(.+)$")
SAMPLESIZE_RE = re.compile(r"^\s+Sample size \(requested\):\s+(\d+)\s+\|\s+actual mean:\s+([\d.]+)")
METRIC_RE = re.compile(
    r"^\s+(TRAIN|TEST)\s+(.+?)\s+→\s+mean:\s*([\d.]+)\s*\|\s*std:\s*([\d.]+)\s*\|\s*min:\s*([\d.]+)\s*\|\s*max:\s*([\d.]+)"
)
NO_VALID_RE = re.compile(r"^\s+(TRAIN|TEST)\s+(.+?)\s+→\s+no valid scores")


def parse_file(path: Path) -> dict:
    """
    Returns {preprocess: [{'requested': int, 'actual_mean': float,
                            'metrics': [(tt, name, values_or_None), ...]}]}
    """
    data: dict = {}
    current_pp = None
    current_block = None

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = PREPROCESS_RE.match(line)
            if m:
                current_pp = m.group(1).strip()
                data.setdefault(current_pp, [])
                current_block = None
                continue

            if current_pp is None:
                continue

            m = SAMPLESIZE_RE.match(line)
            if m:
                current_block = {
                    "requested": int(m.group(1)),
                    "actual_mean": float(m.group(2)),
                    "metrics": [],
                }
                data[current_pp].append(current_block)
                continue

            if current_block is None:
                continue

            m = METRIC_RE.match(line)
            if m:
                current_block["metrics"].append(
                    (
                        m.group(1),
                        m.group(2).strip(),
                        {
                            "mean": float(m.group(3)),
                            "std": float(m.group(4)),
                            "min": float(m.group(5)),
                            "max": float(m.group(6)),
                        },
                    )
                )
                continue

            m = NO_VALID_RE.match(line)
            if m:
                current_block["metrics"].append((m.group(1), m.group(2).strip(), None))

    return data


def _ordered_unique(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def aggregate(all_parsed: list[dict]) -> dict:
    """Average metrics across seeds, preserving structure."""
    all_pp = _ordered_unique(pp for fd in all_parsed for pp in fd)

    result = {}
    for pp in all_pp:
        pp_files = [fd[pp] for fd in all_parsed if pp in fd]

        all_sizes = _ordered_unique(
            b["requested"] for fd in pp_files for b in fd
        )

        result_blocks = []
        for sz in all_sizes:
            blocks = [
                next((b for b in fd if b["requested"] == sz), None)
                for fd in pp_files
            ]
            blocks = [b for b in blocks if b is not None]
            if not blocks:
                continue

            actual_mean = sum(b["actual_mean"] for b in blocks) / len(blocks)

            all_keys = _ordered_unique(
                (tt, mn) for b in blocks for tt, mn, _ in b["metrics"]
            )

            agg_metrics = []
            for tt, mn in all_keys:
                valid = [
                    vals
                    for b in blocks
                    for btt, bmn, vals in b["metrics"]
                    if btt == tt and bmn == mn and vals is not None
                ]
                if not valid:
                    agg_metrics.append((tt, mn, None))
                    continue

                means = [v["mean"] for v in valid]
                n = len(means)
                avg = sum(means) / n
                seed_std = (sum((x - avg) ** 2 for x in means) / n) ** 0.5

                agg_metrics.append(
                    (
                        tt,
                        mn,
                        {
                            "mean": avg,
                            "std": seed_std,
                            "min": min(v["min"] for v in valid),
                            "max": max(v["max"] for v in valid),
                            "n": n,
                        },
                    )
                )

            result_blocks.append(
                {"requested": sz, "actual_mean": actual_mean, "metrics": agg_metrics}
            )

        result[pp] = result_blocks

    return result


def write_output(
    dataset: str,
    model: str,
    seeds: list[int],
    agg_data: dict,
    output_path: Path,
) -> None:
    n_seeds = len(seeds)
    seeds_str = ", ".join(str(s) for s in sorted(seeds))

    lines = [
        "",
        f"Dataset: {dataset} | Model: {model} | Seeds: {seeds_str} (aggregated, n={n_seeds})",
        "=" * 70,
    ]

    for pp, blocks in agg_data.items():
        lines += ["", f"Preprocess: {pp}", "-" * 60]

        for block in blocks:
            req = block["requested"]
            actual = block["actual_mean"]
            lines.append("")
            lines.append(f"  Sample size (requested): {req:>6}  |  actual mean: {actual:.0f}")

            for tt, mn, vals in block["metrics"]:
                label = f"{tt:<5} {mn}"
                pad = f"    {label:<40}"
                if vals is None:
                    lines.append(f"{pad}→ no valid scores")
                else:
                    note = f"  (n={vals['n']})" if vals["n"] < n_seeds else ""
                    lines.append(
                        f"{pad}→ mean: {vals['mean']:.4f} | std: {vals['std']:.4f} | "
                        f"min: {vals['min']:.4f} | max: {vals['max']:.4f}{note}"
                    )

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"  Written → {output_path}")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR

    groups: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)
    for path in root.rglob("results_samplesize_*.txt"):
        m = FILE_RE.match(path.name)
        if m:
            dataset, model, seed = m.group(1), m.group(2), int(m.group(3))
            groups[(dataset, model)].append((seed, path))

    if not groups:
        print(f"No samplesize seed files found under {root}")
        return

    for (dataset, model), seed_files in sorted(groups.items()):
        seed_files.sort()
        seeds = [s for s, _ in seed_files]
        print(f"\n{dataset}/{model}  ({len(seeds)} seed{'s' if len(seeds) != 1 else ''}): {seeds}")

        all_parsed = [parse_file(p) for _, p in seed_files]
        agg_data = aggregate(all_parsed)

        out_dir = seed_files[0][1].parent / "aggregated"
        out_name = f"results_samplesize_{dataset}_{model}_aggregated.txt"
        write_output(dataset, model, seeds, agg_data, out_dir / out_name)


if __name__ == "__main__":
    main()
