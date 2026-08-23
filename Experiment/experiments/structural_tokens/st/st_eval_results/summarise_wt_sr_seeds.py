#!/usr/bin/env python3
"""
 mean ± SD for WT and SR results, structural tokens experiments.

expected filenames:
    shared_seed42.txt
    shared_seed43.txt
    shared_seed44.txt
    none_seed42.txt
    ...
    disjoint_seed44.txt

expects contents to include lines such as:
    ── Test 1: Word Translation Precision ──
      top-1 precision : 0.3685  (36.9%)

    ── Test 2: Sentence Retrieval Precision ──
      top-1 precision : 0.3860  (38.6%)

usage:
    python summarise_wt_sr_seeds.py . --output st_wt_sr_summary.csv

"""

import argparse
import csv
import re
import statistics
from pathlib import Path


FILENAME_RE = re.compile(
    r"(?P<setting>shared|none|disjoint)_seed(?P<seed>\d+)\.txt$",
    re.IGNORECASE,
)

WT_SECTION_RE = re.compile(
    r"Test\s*1:\s*Word Translation Precision(?P<body>.*?)(?=Test\s*2:|\Z)",
    re.IGNORECASE | re.DOTALL,
)

SR_SECTION_RE = re.compile(
    r"Test\s*2:\s*Sentence Retrieval Precision(?P<body>.*?)(?=\Z)",
    re.IGNORECASE | re.DOTALL,
)

TOP1_RE = re.compile(
    r"top-1\s+precision\s*:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def extract_top1(text, section_re, label, path):
    section = section_re.search(text)
    if not section:
        raise ValueError(f"{path}: could not find {label} section")

    match = TOP1_RE.search(section.group("body"))
    if not match:
        raise ValueError(f"{path}: could not find top-1 precision in {label}")

    return float(match.group(1))


def parse_file(path):
    m = FILENAME_RE.search(path.name)
    if not m:
        return None

    text = path.read_text(encoding="utf-8", errors="replace")

    wt = extract_top1(text, WT_SECTION_RE, "Word Translation", path)
    sr = extract_top1(text, SR_SECTION_RE, "Sentence Retrieval", path)

    return {
        "setting": m.group("setting").lower(),
        "seed": int(m.group("seed")),
        "word_translation": wt,
        "sentence_retrieval": sr,
        "file": path.name,
    }


def sample_sd(values):
    # Sample SD across training runs (n-1 denominator).
    return statistics.stdev(values) if len(values) >= 2 else float("nan")


def fmt(x):
    return f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="Average WT/SR top-1 precision across seeds."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="directory containing *_seed*.txt files (default: current directory)",
    )
    parser.add_argument(
        "--output",
        default="wt_sr_seed_summary.csv",
        help="summary CSV filename (default: wt_sr_seed_summary.csv)",
    )
    parser.add_argument(
        "--raw-output",
        default="wt_sr_seed_values.csv",
        help="per-seed CSV filename (default: wt_sr_seed_values.csv)",
    )
    args = parser.parse_args()

    directory = Path(args.directory)

    if not directory.is_dir():
        raise SystemExit(f"ERROR: not a directory: {directory}")

    rows = []
    for path in sorted(directory.glob("*.txt")):
        parsed = parse_file(path)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        raise SystemExit(
            "ERROR: no matching files found. Expected names such as "
            "shared_seed42.txt, none_seed43.txt, disjoint_seed44.txt"
        )

    # Save the extracted per-seed values first.
    raw_path = directory / args.raw_output
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "seed",
                "word_translation",
                "sentence_retrieval",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    settings_order = ["shared", "none", "disjoint"]
    metrics = ["word_translation", "sentence_retrieval"]

    summary_rows = []

    print("\nPer-seed values")
    print("=" * 72)
    print(f"{'setting':<12} {'seed':<6} {'WT P@1':>10} {'SR P@1':>10}")
    print("-" * 72)

    for row in sorted(
        rows,
        key=lambda r: (
            settings_order.index(r["setting"])
            if r["setting"] in settings_order
            else 999,
            r["seed"],
        ),
    ):
        print(
            f"{row['setting']:<12} {row['seed']:<6} "
            f"{row['word_translation']:>10.4f} "
            f"{row['sentence_retrieval']:>10.4f}"
        )

    print("\nMean ± SD across seeds")
    print("=" * 72)
    print(f"{'setting':<12} {'metric':<22} {'n':>4} {'mean':>10} {'SD':>10}")
    print("-" * 72)

    for setting in settings_order:
        setting_rows = [r for r in rows if r["setting"] == setting]
        if not setting_rows:
            continue

        for metric in metrics:
            values = [r[metric] for r in setting_rows]
            mean = statistics.mean(values)
            sd = sample_sd(values)

            summary_rows.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": mean,
                    "sd": sd,
                }
            )

            sd_str = fmt(sd) if len(values) >= 2 else "NA"
            print(
                f"{setting:<12} {metric:<22} {len(values):>4} "
                f"{mean:>10.4f} {sd_str:>10}"
            )

    summary_path = directory / args.output
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["setting", "metric", "n_seeds", "mean", "sd"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nReport-ready values")
    print("=" * 72)
    for setting in settings_order:
        vals = [r for r in summary_rows if r["setting"] == setting]
        if not vals:
            continue
        print(setting)
        for r in vals:
            if r["n_seeds"] >= 2:
                print(
                    f"  {r['metric']:<22}: "
                    f"{r['mean']:.4f} ± {r['sd']:.4f}"
                )
            else:
                print(
                    f"  {r['metric']:<22}: "
                    f"{r['mean']:.4f} (SD unavailable; n=1)"
                )

    print(f"\nSaved per-seed values to: {raw_path}")
    print(f"Saved summary to:         {summary_path}")


if __name__ == "__main__":
    main()
