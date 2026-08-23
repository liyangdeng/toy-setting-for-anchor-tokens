"""
Trend-type significance test
============================

Spearman rank correlation between an ORDERED experimental variable and a metric
score, across conditions.

Use this when the claim is "as X increases, alignment improves", e.g.

    graph density   : 40% -> 60% -> 80% -> 100%
    semantic overlap: 0%  -> 25% -> 50% -> 75%
    lexical overlap : 0%  -> 2.5% -> 5% -> 7.5% -> 10%   (per strategy)

Design decision (see README): the unit of the trend test is the CONDITION, not
the individual seed. For each condition we AVERAGE the seeds into one score,
then run Spearman across the (few) condition points. Seeds stabilise each point;
they are NOT inflated into pseudo-independent data (that would understate p).

With only 4-5 ordered conditions the Spearman p has a floor (~0.04 for 4
points, ~0.017 for 5). The effect size (rho, -1..+1) carries the trend; a clean
monotone rise gives rho near +1.

Grouping
--------
For a second categorical factor (lexical overlap: high / mid / low frequency
strategy), pass --group. One Spearman is run PER group and reported separately.
Never pool differently-behaving groups into one trend, since that dilutes a real
signal in one group with noise from another.

Score per (condition, seed)
---------------------------
    linear_probe        : mean accuracy over layers 2-4 (fixed plateau window,
                          the SAME for every run; never the per-run best layer).
    word_translation    : the single P@1 (static embedding).
    sentence_retrieval  : the single P@1 (static embedding).
    lm_head             : the chosen accuracy (strict or lenient), fixed.

You compute these scores upstream and hand this script a long-format table.

Input
-----
A CSV with columns (header required):

    group      optional. categorical factor, e.g. strategy=high/mid/low.
    condition  the condition label, e.g. overlap_050.
    x          the ordered numeric variable, e.g. 50 (density/overlap %).
    seed       optional. if present, rows sharing (group, condition) are averaged.
    score      the metric score for that run.

Usage
-----
    python trend_significance.py --table graph_density_probe.csv --metric linear_probe
    python trend_significance.py --table lexical_probe.csv --metric linear_probe --group

Output: per group, the number of conditions, Spearman rho, and a permutation
p-value (exact when the number of conditions is small).
"""

import argparse
import csv
import itertools
import math
from collections import defaultdict

import numpy as np


def read_table(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for r in reader:
            rows.append(r)
    return rows, cols


def rankdata(a):
    """Average ranks, ties handled (like scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def spearman_rho(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _extreme(val, observed, alternative):
    """Is a permuted rho as extreme as observed, given the alternative?"""
    if alternative == "greater":
        return val >= observed - 1e-12
    if alternative == "less":
        return val <= observed + 1e-12
    return abs(val) >= abs(observed) - 1e-12


def permutation_p(x, y, observed, alternative="greater", max_exact=40320):
    """Permutation p-value for Spearman rho. One-sided by default ('greater' =
    positive trend, declared in advance). Exact over all permutations of y when
    n! <= max_exact, else sampled."""
    n = len(x)
    rx = rankdata(x)
    ry = rankdata(y)
    if math.factorial(n) <= max_exact:
        count = tot = 0
        for perm in itertools.permutations(ry):
            tot += 1
            if _extreme(spearman_rho_ranks(rx, np.asarray(perm)), observed, alternative):
                count += 1
        return count / tot
    rng = np.random.default_rng(0)
    n_samp = 20000
    count = 0
    ry_arr = np.asarray(ry)
    for _ in range(n_samp):
        if _extreme(spearman_rho_ranks(rx, rng.permutation(ry_arr)), observed, alternative):
            count += 1
    return count / n_samp


def spearman_rho_ranks(rx, ry):
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    denom = np.sqrt((rxc * rxc).sum() * (ryc * ryc).sum())
    if denom == 0:
        return 0.0
    return float((rxc * ryc).sum() / denom)


def run_group(label, rows, metric, alternative):
    # average seeds within each condition
    by_cond = defaultdict(list)
    x_of = {}
    for r in rows:
        cond = r["condition"]
        by_cond[cond].append(float(r["score"]))
        x_of[cond] = float(r["x"])
    conds = sorted(by_cond, key=lambda c: x_of[c])
    x = np.array([x_of[c] for c in conds])
    y = np.array([np.mean(by_cond[c]) for c in conds])
    n_seed = max(len(v) for v in by_cond.values())

    print(f"\n=== group: {label}  (metric: {metric}) ===")
    print(f"{'condition':<16}{'x':>8}{'score(mean)':>14}{'n_seed':>8}")
    for c in conds:
        print(f"{c:<16}{x_of[c]:>8.3g}{np.mean(by_cond[c]):>14.4f}{len(by_cond[c]):>8}")

    if len(conds) < 3:
        print("  [skip] need >= 3 conditions for a trend test")
        return
    rho = spearman_rho(x, y)
    p = permutation_p(x, y, rho, alternative)
    print(f"  conditions     : {len(conds)}  (seeds/condition up to {n_seed})")
    print(f"  Spearman rho   : {rho:+.4f}")
    print(f"  permutation p  : {p:.4g}  ({alternative})")
    print(f"  trend          : {'monotone, ' if abs(rho) > 0.5 else ''}"
          f"{'significant' if p < 0.05 else 'NOT significant'} at 0.05")


def main():
    p = argparse.ArgumentParser(description="Trend-type Spearman significance test")
    p.add_argument("--table", required=True, help="long-format CSV (see header docs)")
    p.add_argument("--metric", required=True, help="metric label")
    p.add_argument("--group", action="store_true",
                   help="run one Spearman per value of the 'group' column")
    p.add_argument("--alternative", choices=["greater", "less", "two-sided"], default="greater",
                   help="directional hypothesis; default 'greater' = score rises with x")
    args = p.parse_args()

    rows, cols = read_table(args.table)
    for req in ("condition", "x", "score"):
        if req not in cols:
            raise ValueError(f"table needs a '{req}' column; got {cols}")

    if args.group:
        if "group" not in cols:
            raise ValueError("--group given but table has no 'group' column")
        groups = defaultdict(list)
        for r in rows:
            groups[r["group"]].append(r)
        for g in sorted(groups):
            run_group(g, groups[g], args.metric, args.alternative)
    else:
        run_group("all", rows, args.metric, args.alternative)


if __name__ == "__main__":
    main()
