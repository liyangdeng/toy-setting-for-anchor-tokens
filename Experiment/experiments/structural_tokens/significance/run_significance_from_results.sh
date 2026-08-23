#!/usr/bin/env bash
# =============================================================================
# Significance tests for PUNCTUATION + SPECIAL TOKENS
#
# Reads the result files from the project tree and calls compare_significance.py
#
# Expected result locations:
#
# WT / SR
#   punctuation/evaluation/punct_wt_sr_results/{shared,none,disjoint}_seed{42,43,44}.txt
#   special_tokens/evaluation/st_wt_sr_results/{shared,none,disjoint}_seed{42,43,44}.txt
#
# Linear probe (seed 42)
#   punctuation/evaluation/probing_lmhead/probe_results/res_probe_punct_{shared,none,disjoint}_seed42/layerwise_accuracy.csv
#   special_tokens/evaluation/probing_lmhead/probe_results/res_probe_st_{shared,none,disjoint}_seed42/layerwise_accuracy.csv
#
# LM head (seed 42)
#   punctuation/evaluation/probing_lmhead/lm_head_results/lm_head_punct_{shared,none,disjoint}_seed42/lm_head_accuracy.csv
#   special_tokens/evaluation/probing_lmhead/lm_head_results/lm_head_st_{shared,none,disjoint}_seed42/lm_head_accuracy.csv
#
# Linear probe uses fixed layer 3.
# LM head uses strict_token, top1.
#
# Compares:
# shared   > disjoint
# none     > disjoint
# shared   vs none
#
# Output: significance_results.txt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ge 1 ]]; then
    ROOT="$(cd "$1" && pwd)"
else
    ROOT="$SCRIPT_DIR"
    while [[ "$ROOT" != "/" && ! ( -d "$ROOT/punctuation" && -d "$ROOT/special_tokens" ) ]]; do
        ROOT="$(dirname "$ROOT")"
    done

    if [[ ! ( -d "$ROOT/punctuation" && -d "$ROOT/special_tokens" ) ]]; then
        echo "Could not locate structural_tokens root from $SCRIPT_DIR" >&2
        exit 1
    fi
fi

LOG="${ROOT}/significance_results.txt"

COMPARE=""
for candidate in \
    "${ROOT}/compare_significance.py" \
    "${ROOT}/probing/compare_significance.py" \
    "${ROOT}/significance/compare_significance.py"
do
    if [[ -f "$candidate" ]]; then
        COMPARE="$candidate"
        break
    fi
done

if [[ -z "$COMPARE" ]]; then
    COMPARE="$(find "$ROOT" -maxdepth 4 -type f -name 'compare_significance.py' -print -quit 2>/dev/null || true)"
fi

echo "Project root : $ROOT"
echo "Comparison   : $COMPARE"
echo "Output log   : $LOG"
echo

python -u - "$ROOT" "$COMPARE" <<'PY' | tee "$LOG"
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
COMPARE = Path(sys.argv[2]).resolve()

ARMS = ("shared", "none", "disjoint")
SEEDS = (42, 43, 44)

COMPARISONS = (
    ("shared", "disjoint", "greater"),
    ("none", "disjoint", "greater"),
    ("shared", "none", "two-sided"),
)

PUNCT_ROOT = ROOT / "punctuation"
ST_ROOT = ROOT / "special_tokens"

if not PUNCT_ROOT.is_dir():
    raise FileNotFoundError(f"Could not find {PUNCT_ROOT}")
if not ST_ROOT.is_dir():
    raise FileNotFoundError(f"Could not find {ST_ROOT}")

EXPERIMENTS = {
    "punctuation": {
        "root": PUNCT_ROOT,
        "eval_dir": PUNCT_ROOT / "evaluation" / "punct_wt_sr_results",
        "probe_prefix": "punct",
        "lm_prefix": "punct",
    },
    "special_tokens": {
        "root": ST_ROOT,
        "eval_dir": ST_ROOT / "evaluation" / "st_wt_sr_results",
        "probe_prefix": "st",
        "lm_prefix": "st",
    },
}

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def find_file_starting_with(directory, stem):
    """Accept .csv or a file whose extension is hidden/absent."""
    if not directory.is_dir():
        return None

    exact = directory / f"{stem}.csv"
    if exact.is_file():
        return exact

    exact_no_ext = directory / stem
    if exact_no_ext.is_file():
        return exact_no_ext

    matches = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.name.lower().startswith(stem.lower())
    )
    return matches[0] if matches else None

def find_probe_dir(exp, arm):
    base = exp["root"] / "evaluation" / "probing_lmhead" / "probe_results"
    prefix = exp["probe_prefix"]

    directory = base / f"res_probe_{prefix}_{arm}_seed42"
    if directory.is_dir():
        return directory

    raise FileNotFoundError(
        f"Could not find linear-probe directory for {prefix}/{arm}: {directory}"
    )

def find_lm_dir(exp, arm):
    base = exp["root"] / "evaluation" / "probing_lmhead" / "lm_head_results"
    prefix = exp["lm_prefix"]

    directory = base / f"lm_head_{prefix}_{arm}_seed42"
    if directory.is_dir():
        return directory

    raise FileNotFoundError(
        f"Could not find LM-head directory for {prefix}/{arm}: {directory}"
    )

WT_N_RE = re.compile(
    r"Synset content-token pairs in joint vocab\s*:\s*(\d+)",
    re.IGNORECASE,
)
SR_N_RE = re.compile(
    r"Sentence pairs evaluated\s*:\s*(\d+)",
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

def parse_eval_txt(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing WT/SR result file: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    wt_section = WT_SECTION_RE.search(text)
    sr_section = SR_SECTION_RE.search(text)
    wt_n = WT_N_RE.search(text)
    sr_n = SR_N_RE.search(text)

    if not wt_section or not sr_section:
        raise ValueError(f"Could not find WT/SR sections in {path}")
    if not wt_n or not sr_n:
        raise ValueError(f"Could not find WT/SR n values in {path}")

    wt_acc_match = TOP1_RE.search(wt_section.group("body"))
    sr_acc_match = TOP1_RE.search(sr_section.group("body"))
    if not wt_acc_match or not sr_acc_match:
        raise ValueError(f"Could not find top-1 precision in {path}")

    return {
        "word_translation": {
            "acc": float(wt_acc_match.group(1)),
            "n": int(wt_n.group(1)),
            "source": path,
        },
        "sentence_retrieval": {
            "acc": float(sr_acc_match.group(1)),
            "n": int(sr_n.group(1)),
            "source": path,
        },
    }

def normalise_row(row):
    return {
        str(k).strip().lower(): (str(v).strip() if v is not None else "")
        for k, v in row.items()
    }

def parse_probe(exp, arm):
    directory = find_probe_dir(exp, arm)
    path = find_file_starting_with(directory, "layerwise_accuracy")

    if path is None:
        raise FileNotFoundError(
            f"No layerwise_accuracy(.csv) in {directory}"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [normalise_row(r) for r in csv.DictReader(f)]

    for row in rows:
        layer = row.get("layer", "")
        try:
            is_layer3 = int(float(layer)) == 3
        except ValueError:
            is_layer3 = False

        if is_layer3:
            if "accuracy" not in row or "n" not in row:
                raise ValueError(
                    f"{path}: expected columns layer,accuracy,n; "
                    f"found {list(row)}"
                )
            return {
                "acc": float(row["accuracy"]),
                "n": int(float(row["n"])),
                "source": path,
            }

    raise ValueError(f"{path}: no layer 3 row found")

def parse_lm_head(exp, arm):
    directory = find_lm_dir(exp, arm)
    path = find_file_starting_with(directory, "lm_head_accuracy")

    if path is None:
        raise FileNotFoundError(
            f"No lm_head_accuracy(.csv) in {directory}"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [normalise_row(r) for r in csv.DictReader(f)]

    for row in rows:
        if row.get("metric", "").lower() == "strict_token":
            if "top1" not in row or "n" not in row:
                raise ValueError(
                    f"{path}: expected columns metric,top1,top3,n; "
                    f"found {list(row)}"
                )
            return {
                "acc": float(row["top1"]),
                "n": int(float(row["n"])),
                "source": path,
            }

    raise ValueError(f"{path}: no strict_token row found")

# collect everything

data = {}

for exp_name, exp in EXPERIMENTS.items():
    data[exp_name] = {
        "word_translation": {},
        "sentence_retrieval": {},
        "linear_probe": {},
        "lm_head": {},
    }

    # WT/SR: three seeds per arm.
    for arm in ARMS:
        data[exp_name]["word_translation"][arm] = {}
        data[exp_name]["sentence_retrieval"][arm] = {}

        for seed in SEEDS:
            path = exp["eval_dir"] / f"{arm}_seed{seed}.txt"
            parsed = parse_eval_txt(path)

            data[exp_name]["word_translation"][arm][seed] = \
                parsed["word_translation"]
            data[exp_name]["sentence_retrieval"][arm][seed] = \
                parsed["sentence_retrieval"]

    # Linear probe and LM head: seed 42 only.
    for arm in ARMS:
        data[exp_name]["linear_probe"][arm] = parse_probe(exp, arm)
        data[exp_name]["lm_head"][arm] = parse_lm_head(exp, arm)

# check files read

print("#" * 94, flush=True)
print("# INPUT CHECK", flush=True)
print("#" * 94, flush=True)

for exp_name in ("punctuation", "special_tokens"):
    print(f"\n[{exp_name.upper()}]", flush=True)

    for metric in ("word_translation", "sentence_retrieval"):
        print(f"\n  {metric}:", flush=True)
        for arm in ARMS:
            for seed in SEEDS:
                d = data[exp_name][metric][arm][seed]
                print(
                    f"    {arm:<8} seed {seed}: "
                    f"acc={d['acc']:.4f}, n={d['n']}  "
                    f"[{d['source'].relative_to(ROOT)}]",
                    flush=True,
                )

    for metric in ("linear_probe", "lm_head"):
        print(f"\n  {metric} (seed 42):", flush=True)
        for arm in ARMS:
            d = data[exp_name][metric][arm]
            print(
                f"    {arm:<8}: acc={d['acc']:.4f}, n={d['n']}  "
                f"[{d['source'].relative_to(ROOT)}]",
                flush=True,
            )

# testss

def hits(acc, n):
    return int(round(acc * n))

def run_original_test(
    exp_name, metric, seed, name_a, a, name_b, b, alternative
):
    ha = hits(a["acc"], a["n"])
    hb = hits(b["acc"], b["n"])

    if alternative == "greater":
        comparison = f"{name_a} > {name_b}"
        sidedness = "one-sided"
    elif alternative == "less":
        comparison = f"{name_a} < {name_b}"
        sidedness = "one-sided"
    else:
        comparison = f"{name_a} vs {name_b}"
        sidedness = "two-sided"

    header = [
        "",
        "=" * 94,
        f"EXPERIMENT  : {exp_name.upper().replace('_', ' ')}",
        f"METRIC      : {metric.upper().replace('_', ' ')}",
        f"SEED        : {seed}",
        f"COMPARISON  : {comparison}",
        f"ALTERNATIVE : {alternative} ({sidedness})",
        f"SOURCE A    : {a['source'].relative_to(ROOT)}",
        f"SOURCE B    : {b['source'].relative_to(ROOT)}",
        "-" * 94,
    ]
    print("\n".join(header), flush=True)

    cmd = [
        sys.executable,
        str(COMPARE),
        "--metric", metric,
        "--name_a", name_a,
        "--hits_a", str(ha),
        "--n_a", str(a["n"]),
        "--name_b", name_b,
        "--hits_b", str(hb),
        "--n_b", str(b["n"]),
        "--alternative", alternative,
    ]

    # Capture first, then print: this guarantees that the header and its result
    # can never be separated by stdout buffering.
    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout.rstrip(), flush=True)
    print("=" * 94, flush=True)

# run testsss

for exp_name in ("punctuation", "special_tokens"):
    print("\n\n" + "#" * 94, flush=True)
    print(f"# {exp_name.upper().replace('_', ' ')}", flush=True)
    print("#" * 94, flush=True)

    # WT/SR: per seed.
    for metric in ("word_translation", "sentence_retrieval"):
        for name_a, name_b, alternative in COMPARISONS:
            for seed in SEEDS:
                run_original_test(
                    exp_name,
                    metric,
                    seed,
                    name_a,
                    data[exp_name][metric][name_a][seed],
                    name_b,
                    data[exp_name][metric][name_b][seed],
                    alternative,
                )

    # Linear probe / LM head: seed 42 only.
    for metric in ("linear_probe", "lm_head"):
        for name_a, name_b, alternative in COMPARISONS:
            run_original_test(
                exp_name,
                metric,
                42,
                name_a,
                data[exp_name][metric][name_a],
                name_b,
                data[exp_name][metric][name_b],
                alternative,
            )

print("\n\n" + "#" * 94, flush=True)
print("# WT/SR: seeds 42, 43, 44 tested separately.", flush=True)
print("# Linear probe: seed 42, fixed layer 3.", flush=True)
print("# LM head: seed 42, strict_token top1.", flush=True)
print("#" * 94, flush=True)
PY

echo
echo "Output saved to:"
echo "  $LOG"