# Significance testing

Two small tools to check whether a difference in a metric is real or could be
sampling noise. Both run on the **aggregate numbers already in the committed
result files**. No retraining, no re-running evaluation, no per-item dumps.

| Script | Question it answers | Test | When to use |
| --- | --- | --- | --- |
| `compare_significance.py` | Is condition A better than condition B? | two-proportion (Fisher exact) | two conditions, e.g. necessity `111` vs `011`, punctuation `shared` vs `disjoint` |
| `trend_significance.py` | Does the score rise as an ordered variable increases? | Spearman rank correlation + permutation | many ordered conditions, e.g. graph density, semantic overlap, lexical overlap |

Each teammate decides which test fits their experiment.

---

## Method 1: comparison (`compare_significance.py`)

Our metrics are **binary per item** (a triple / word-pair / sentence-pair
either hits or misses), and two conditions use **different** evaluation items
(different corpora / different omitted-triple sets), so the comparison is
**unpaired**. Comparing two accuracies from two independent groups of Bernoulli
trials is exactly what a **two-proportion test** is for.

You need four numbers, all recoverable from the result CSVs:

```
hits_a, n_a, hits_b, n_b        (hits = accuracy * n, rounded to an integer)
```

The script reports **Fisher's exact test** (exact, safe for small n) plus a
two-proportion z-test for reference. No bootstrap and no per-item file are
needed here, since the sufficient statistics are just the counts.

```bash
python compare_significance.py --metric linear_probe \
    --name_a cfg_111 --hits_a 132 --n_a 173 \
    --name_b cfg_011 --hits_b 5   --n_b 167
```

## Method 2: trend (`trend_significance.py`)

For experiments that sweep an **ordered** variable (density %, overlap %), the
claim is a monotone trend ("more ⇒ better"), not a pairwise gap. The right tool
is **Spearman's rank correlation**: it turns both the ordering variable and the
score into ranks and measures whether they rise together. `rho` (−1..+1) is the
trend strength; the **permutation** p-value is its significance (it does *not*
use Fisher, the two tests are independent).

The unit of the trend test is the **condition**, not the seed. For each
condition we average its seeds into one score, then run Spearman across the few
condition points. Seeds stabilise each point; they are not inflated into
pseudo-independent data (that would understate the p-value).

Input is a long-format CSV (`group` optional; `seed` optional and averaged):

```
group,condition,x,seed,score
low,low_P0,0,42,0.42
low,low_P0,0,43,0.44
...
```

```bash
python trend_significance.py --table graph_density_probe.csv --metric linear_probe
python trend_significance.py --table lexical_probe.csv --metric linear_probe --group
```

For a second categorical factor (lexical overlap: high/mid/low frequency),
pass `--group`, and one Spearman is run **per group** and reported separately.
Never pool differently-behaving groups into one trend.

---

## Linear probing: which layer value goes into the test

The probe produces a **per-layer** accuracy curve. Layer 0 is the embedding
floor, layer 1 is the rising edge, and the plateau (where every condition is
near its peak) is **layers 2–4**. So the significance input is taken from that
plateau, but the two tests take it differently:

- **Trend test (Spearman):** score per run = **mean over layers 2–4**. Spearman
  only needs a scalar, so the average is fine.
- **Comparison test (Fisher):** use the counts at a **single fixed layer
  (layer 3)**. Fisher needs a clean binomial: integer hits at one layer. A
  layer-2–4 average is a non-integer and mixes three correlated measurements of
  the same triples, so it cannot go into Fisher. For necessity the effect is so
  large that layer-3 vs plateau-mean gives the same verdict anyway.

In all cases the layer choice is **fixed and identical across conditions**,
never the per-condition best layer (that is post-hoc selection and inflates the
result).

## Which model produces the numbers (per metric)

Decided upstream, documented here so the counts come from the right run:

- **word translation / sentence retrieval** → the **normal** model (full corpus,
  no facts omitted); these measure embedding-space alignment.
- **linear probe / LM head** → the **omitted** model (a fact held out from one
  language); these measure knowledge transfer.

For retrieval metrics, keep the **gallery size identical** across the two
conditions being compared (e.g. both sampled at 500), because P@1 depends on it.

## Direction (one-sided)

Both tests default to **one-sided** (`--alternative greater`): our hypotheses
are directional and stated in advance (more anchors / more structure ⇒ better
alignment). With only 4–5 ordered conditions this matters: a perfect monotone
trend on 4 points is p = 0.083 two-sided but **0.042 one-sided**. One-sided is
legitimate *only* because the direction is pre-registered; do not pick the
direction after seeing which way the data went.

---

## Limitations (state this in the write-up)

Because of time and compute limits we did **not** run 30+ seeds per condition.
Our significance results therefore **cannot be generalised to arbitrary
conditions or a new training run**. They show only that, **for the specific
models and conditions we trained**, the observed differences (and trends) are
statistically significant. With so few ordered conditions the p-values are also
floored by the permutation combinatorics, so the **effect size** (the size of
the gap, or `rho`) carries as much of the conclusion as the p-value does.
