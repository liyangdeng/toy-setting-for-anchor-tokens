"""
KG statistics for designing the masked-token probing experiments.

Computes the three things we need to decide what is probe-able:

  (1) Multiple edges between the same nodes
        - ordered  (source, target) carrying >1 relation
              -> kills mask-RELATION well-posedness ("s [MASK] t" is ambiguous)
        - unordered {a, b} carrying >1 edge (either direction)
              -> a co-occurrence confounder: the model can learn that two nodes
                 simply go together, independent of the masked relation/entity.

  (2) Entity uniqueness per relation (mask-ENTITY well-posedness)
        forward  : given (source, relation), how many distinct targets?
                   unique == exactly 1 target  ->  "s r [MASK]" has one answer
        backward : given (relation, target), how many distinct sources?
                   unique == exactly 1 source  ->  "[MASK] r t" has one answer

  (3) Triple count per relation (the Zipf head/tail -> training sufficiency).

And the two final usability counts the probing experiments select on:

  (4) MASK-RELATION usable (mask the relation in "a [MASK] b"): TWO rules --
        the ordered pair (a,b) carries a unique relation, and both nodes have
        degree >= 2.

  (5) MASK-ENTITY usable (always mask the 2nd entity: "a r [MASK]" -> b): TWO
        rules -- (a,r) maps to a unique target b, and b has degree >= 2.

Co-occurrence / reverse edges are NOT part of selection; they are handled at
hold-out time (selected probe triples are simply omitted from a language's
corpus). These stats therefore match the usable pools in select_probe_triples.py.

Input: the v3 generated-sentences JSON that actually feeds the corpus
       (default v3_generated_sentences_adj.json), OR a raw edges.json (list of
       {source, relation, target}). Auto-detected. HasQuality / virtual-adjective
       edges are dropped (they feed adjectives, they are not triples).

Usage:
    python analyze_kg.py
    python analyze_kg.py --input /path/to/edges.json
    python analyze_kg.py --examples 15
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


# This file lives at <repo>/evaluation/masked_language_masking/analyze_kg.py,
# so parents[2] is the repo root. Reads the v3 generated corpus -- the ACTUAL
# rendered/trained triples (edges_adj.json is an older, out-of-sync edge set).
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "data/generate_sentences/v3_generated_sentences_adj.json"
)


def load_triples(path):
    """
    Return a list of (source, relation, target).
    Accepts either the raw edges list (edges_adj.json: {source, relation, target,
    source_type, ...}) or a generated-sentences JSON ({"results": [...]}).
    HasQuality edges and virtual-adjective sources are dropped -- they feed the
    adjective slot, they are not triples (matches the sentence generator).
    """
    data = json.load(open(path))
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    out = []
    for r in rows:
        if r.get("relation") == "HasQuality":
            continue
        if r.get("source_type") == "virtual_adjective":
            continue
        out.append((r["source"], r["relation"], r["target"]))
    return out


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# (1) ---------------------------------------------------------------------------
def multi_edges(triples, n_examples):
    section("(1) MULTIPLE EDGES BETWEEN THE SAME NODES")

    ordered = defaultdict(set)          # (s,t) -> {relations}
    for s, r, t in triples:
        ordered[(s, t)].add(r)
    multi_ordered = {p: rs for p, rs in ordered.items() if len(rs) > 1}
    by_count = Counter(len(rs) for rs in ordered.values())

    print(f"distinct ordered (source,target) pairs : {len(ordered)}")
    print(f"  carrying >1 relation (multi-edge)    : {len(multi_ordered)}"
          f"  ({100*len(multi_ordered)/len(ordered):.1f}%)")
    print(f"  distribution #relations-per-pair     : "
          f"{dict(sorted(by_count.items()))}")
    print("  --> mask-RELATION must exclude these pairs (or accept any of them).")
    print(f"  examples:")
    for p, rs in list(multi_ordered.items())[:n_examples]:
        print(f"     {p[0]:24s} {p[1]:24s} -> {sorted(rs)}")

    # unordered: any edge in either direction between {a,b}
    unordered = defaultdict(set)        # frozenset{a,b} -> {(s,r,t)}
    for s, r, t in triples:
        if s == t:
            continue
        unordered[frozenset((s, t))].add((s, r, t))
    multi_unordered = {k: v for k, v in unordered.items() if len(v) > 1}
    # of those, how many span both directions (s,t) and (t,s)
    reverse_pairs = sum(
        1 for v in multi_unordered.values()
        if len({(a, c) for a, _, c in v}) > 1
    )

    print(f"\ndistinct unordered node-pairs          : {len(unordered)}")
    print(f"  carrying >1 edge (any direction)     : {len(multi_unordered)}"
          f"  ({100*len(multi_unordered)/len(unordered):.1f}%)")
    print(f"    of which span both directions      : {reverse_pairs}")


# (2) ---------------------------------------------------------------------------
def entity_uniqueness(triples):
    section("(2) ENTITY UNIQUENESS PER RELATION  (triple-based, mask-ENTITY well-posedness)")

    fwd = defaultdict(set)   # (s,r) -> {t}
    bwd = defaultdict(set)   # (r,t) -> {s}
    for s, r, t in triples:
        fwd[(s, r)].add(t)
        bwd[(r, t)].add(s)

    # Triple-based: walk every triple, denominator = total triples of the relation
    # (consistent with sections 4 and 5).
    relc = Counter(r for _, r, _ in triples)
    fwd_uniq = defaultdict(int)   # triples whose (s,r) maps to a unique target
    bwd_uniq = defaultdict(int)   # triples whose (r,t) maps to a unique source
    for s, r, t in triples:
        if len(fwd[(s, r)]) == 1:
            fwd_uniq[r] += 1
        if len(bwd[(r, t)]) == 1:
            bwd_uniq[r] += 1

    print(f"{'relation':22s} {'n_triples':>9s}  "
          f"{'fwd unique tgt':>16s}  {'bwd unique src':>16s}")
    print("-" * 72)
    for rel, n in relc.most_common():
        fu, bu = fwd_uniq[rel], bwd_uniq[rel]
        fpct = f"{fu}/{n} ({100*fu/n:.0f}%)"
        bpct = f"{bu}/{n} ({100*bu/n:.0f}%)"
        print(f"{rel:22s} {n:9d}  {fpct:>16s}  {bpct:>16s}")
    print("\nRead (denominator = total triples of the relation): 'fwd 99%' = 99% of")
    print("this relation's triples have a unique target given (source, relation), so")
    print("masking the TARGET is well-posed. hypernym/HasProperty high; hyponym low.")


# (3) ---------------------------------------------------------------------------
def relation_counts(triples):
    section("(3) TRIPLES PER RELATION  (Zipf -> training sufficiency)")
    relc = Counter(r for _, r, _ in triples)
    total = len(triples)
    cum = 0
    print(f"{'relation':22s} {'count':>6s} {'share':>7s} {'cum%':>6s}")
    print("-" * 48)
    for rel, c in relc.most_common():
        cum += c
        print(f"{rel:22s} {c:6d} {100*c/total:6.1f}% {100*cum/total:5.0f}%")
    print(f"\ntotal triples: {total} | distinct relations: {len(relc)}")
    thin = [r for r, c in relc.items() if c < 50]
    print(f"relations with <50 triples (too thin to hold-out & probe): "
          f"{len(thin)}")
    print(f"  {sorted(thin)}")


def entity_degree(triples):
    section("(extra) ENTITY DEGREE  (grounding: masked entity needs degree>=2)")
    deg = Counter()
    for s, _, t in triples:
        deg[s] += 1
        deg[t] += 1
    ds = sorted(deg.values())
    d1 = sum(1 for v in ds if v == 1)
    print(f"distinct entities: {len(deg)} | "
          f"min/median/max degree: {ds[0]}/{median(ds)}/{ds[-1]}")
    print(f"entities with degree==1 (cannot be the masked entity): "
          f"{d1} ({100*d1/len(deg):.0f}%)")


# (4) ---------------------------------------------------------------------------
def mask_relation_feasibility(triples, min_n=50, min_degree=2):
    """
    mask-RELATION: mask the relation in "a [MASK] b" -> predict r.
    Final two rules (co-occurrence / reverse edges are NOT considered here --
    they are dealt with at hold-out, not selection):
      1. the ordered pair (a, b) carries a UNIQUE relation
         (no (a, r2, b) with r2 != r  ->  "a [MASK] b" is unambiguous), AND
      2. deg(a) >= min_degree and deg(b) >= min_degree (both ends grounded).
    Restricted to relations with >= min_n triples (training sufficiency).
    Drops reported sequentially: -multi (cond1), -deg (cond2).
    """
    section("(4) MASK-RELATION USABLE  (unique relation + both nodes degree>=2)")

    deg = Counter()
    for s, _, t in triples:
        deg[s] += 1
        deg[t] += 1

    pair_rel = defaultdict(set)            # (s,t) -> {relations}
    for s, r, t in triples:
        pair_rel[(s, t)].add(r)

    relc = Counter(r for _, r, _ in triples)
    usable = defaultdict(int)
    lost_multi = defaultdict(int)
    lost_deg = defaultdict(int)
    for s, r, t in triples:
        if len(pair_rel[(s, t)]) > 1:                    # cond1: multi-relation pair
            lost_multi[r] += 1
            continue
        if deg[s] < min_degree or deg[t] < min_degree:   # cond2: grounding
            lost_deg[r] += 1
            continue
        usable[r] += 1

    print(f"{'relation':18s} {'total':>6s} {'-multi':>7s} {'-deg<2':>7s} "
          f"{'USABLE':>7s} {'USABLE%':>8s}")
    print("-" * 56)
    tot = lm = ld = tu = 0
    for r, n in relc.most_common():
        if n < min_n:
            continue
        print(f"{r:18s} {n:6d} {lost_multi[r]:7d} {lost_deg[r]:7d} "
              f"{usable[r]:7d} {100*usable[r]/n:7.0f}%")
        tot += n; lm += lost_multi[r]; ld += lost_deg[r]; tu += usable[r]
    print("-" * 56)
    print(f"{'TOTAL (n>=' + str(min_n) + ')':18s} {tot:6d} {lm:7d} {ld:7d} "
          f"{tu:7d} {100*tu/tot:7.0f}%")
    print(f"\n  --> {tu} mask-relation-usable triples ({100*tu/tot:.0f}% of n>={min_n}).")


def mask_entity_feasibility(triples, min_n=50, min_degree=2):
    """
    mask-ENTITY: always mask the SECOND entity. For (a, r, b) mask b -> "a r [MASK]".
    Final two rules (co-occurrence / reverse / role are NOT considered here --
    dealt with at hold-out, not selection):
      1. (a, r) maps to a unique target  (no (a, r, X) with X != b), AND
      2. the masked entity b has degree >= min_degree (grounding).
    Restricted to relations with >= min_n triples.
    Drops reported sequentially: -nonuniq (cond1), -deg (cond2).
    """
    section("(5) MASK-ENTITY USABLE  (mask 2nd entity: unique target + degree>=2)")

    deg = Counter()
    for s, _, t in triples:
        deg[s] += 1
        deg[t] += 1

    fwd = defaultdict(set)        # (s,r) -> {t}
    for s, r, t in triples:
        fwd[(s, r)].add(t)

    relc = Counter(r for _, r, _ in triples)
    usable = defaultdict(int)
    lost_nonuniq = defaultdict(int)
    lost_deg = defaultdict(int)
    for a, r, b in triples:
        if len(fwd[(a, r)]) > 1:          # cond1: (a,r) not unique
            lost_nonuniq[r] += 1
            continue
        if deg[b] < min_degree:           # cond2: masked entity grounding
            lost_deg[r] += 1
            continue
        usable[r] += 1

    print(f"{'relation':18s} {'total':>6s} {'-nonuniq':>9s} {'-deg<2':>7s} "
          f"{'USABLE':>7s} {'USABLE%':>8s}")
    print("-" * 58)
    tot = ln = ld = tu = 0
    for r, n in relc.most_common():
        if n < min_n:
            continue
        print(f"{r:18s} {n:6d} {lost_nonuniq[r]:9d} {lost_deg[r]:7d} "
              f"{usable[r]:7d} {100*usable[r]/n:7.0f}%")
        tot += n; ln += lost_nonuniq[r]; ld += lost_deg[r]; tu += usable[r]
    print("-" * 58)
    print(f"{'TOTAL (n>=' + str(min_n) + ')':18s} {tot:6d} {ln:9d} {ld:7d} "
          f"{tu:7d} {100*tu/tot:7.0f}%")
    print(f"\n  --> {tu} mask-entity-usable triples "
          f"({100*tu/tot:.0f}% of n>={min_n}, always masking the 2nd entity).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--examples", type=int, default=10)
    p.add_argument("--min_n", type=int, default=50,
                   help="min triples per relation to count it as probe-able")
    args = p.parse_args()

    triples = load_triples(args.input)
    print(f"loaded {len(triples)} triples ({len(set(triples))} distinct) "
          f"from {args.input}")

    multi_edges(triples, args.examples)
    entity_uniqueness(triples)
    relation_counts(triples)
    entity_degree(triples)
    mask_relation_feasibility(triples, min_n=args.min_n)
    mask_entity_feasibility(triples, min_n=args.min_n)


if __name__ == "__main__":
    main()
