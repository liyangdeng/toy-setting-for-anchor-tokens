"""
Comparison-type significance test
=================================

Is the accuracy of condition A higher than condition B, or could that gap be
sampling noise?

    anchor necessity : 111  vs  011
    punctuation      : shared  vs  disjoint
    special tokens   : shared  vs  disjoint

The metrics here are all BINARY per item (a triple / word-pair / sentence-pair
either hits or misses), and the two conditions use DIFFERENT evaluation items
(different corpora / different omitted-triple sets), so the comparison is
UNPAIRED. For two independent binomials the textbook tool is a two-proportion
test, so no bootstrap, no per-item file, no re-run needed. You only need four
numbers, all already in the committed result CSVs:

    hits_a, n_a, hits_b, n_b        (hits = accuracy * n, rounded)

We report Fisher's exact test (exact, safe for small n) and, for reference, the
two-proportion z-test.

Direction
---------
The hypotheses are directional (we predict A > B in advance), so the test is
ONE-SIDED by default (`--alternative greater`). Declare the direction in the
write-up; do not pick it after seeing which way the gap goes.

Layer note (linear probe)
-------------------------
Fisher needs a clean binomial (integer hits at ONE layer). So for the probe,
feed the counts at a single FIXED layer. We use layer 3 (a mid plateau layer,
not the per-condition best layer). The L2-4 average is for the TREND test only,
where a scalar is all that is needed. For necessity the effect is so large the
choice is immaterial.

Which model produced the counts (per metric, decided upstream):
    word_translation / sentence_retrieval : NORMAL model (no facts omitted)
    linear_probe / lm_head                : OMITTED model (a fact held out)

Usage
-----
    python compare_significance.py --metric linear_probe \
        --name_a cfg_111 --hits_a 132 --n_a 173 \
        --name_b cfg_011 --hits_b 5   --n_b 167
"""

import argparse
import math

from scipy.stats import fisher_exact, norm


def two_proportion_z(hits_a, n_a, hits_b, n_b, alternative):
    pa, pb = hits_a / n_a, hits_b / n_b
    p_pool = (hits_a + hits_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return pa - pb, float("nan"), float("nan")
    z = (pa - pb) / se
    if alternative == "greater":
        p = norm.sf(z)
    elif alternative == "less":
        p = norm.cdf(z)
    else:
        p = 2 * norm.sf(abs(z))
    return pa - pb, z, p


def main():
    p = argparse.ArgumentParser(description="Comparison-type significance test (two independent proportions)")
    p.add_argument("--metric", required=True,
                   help="label: word_translation / sentence_retrieval / linear_probe / lm_head")
    p.add_argument("--name_a", default="A")
    p.add_argument("--name_b", default="B")
    p.add_argument("--hits_a", type=int, required=True, help="number of hits in A (accuracy*n, rounded)")
    p.add_argument("--n_a", type=int, required=True, help="number of items in A")
    p.add_argument("--hits_b", type=int, required=True)
    p.add_argument("--n_b", type=int, required=True)
    p.add_argument("--alternative", choices=["greater", "less", "two-sided"], default="greater",
                   help="directional hypothesis; default 'greater' = A > B (declare it in advance)")
    args = p.parse_args()

    ha, na, hb, nb = args.hits_a, args.n_a, args.hits_b, args.n_b
    acc_a, acc_b = ha / na, hb / nb

    table = [[ha, na - ha], [hb, nb - hb]]
    _, p_fisher = fisher_exact(table, alternative=args.alternative)
    diff, z, p_z = two_proportion_z(ha, na, hb, nb, args.alternative)

    print(f"metric         : {args.metric}")
    print(f"{args.name_a:<14} : {ha}/{na}  = {acc_a:.4f}")
    print(f"{args.name_b:<14} : {hb}/{nb}  = {acc_b:.4f}")
    print(f"difference     : {diff:+.4f}  ({args.name_a} - {args.name_b})")
    print(f"alternative    : {args.alternative}")
    print(f"Fisher exact p : {p_fisher:.4g}")
    print(f"two-prop z     : z = {z:+.3f},  p = {p_z:.4g}")
    print(f"significant    : {'YES' if p_fisher < 0.05 else 'no'}  (Fisher, 0.05)")


if __name__ == "__main__":
    main()
