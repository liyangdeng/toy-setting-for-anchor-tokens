"""
Select probe triples to omit for the anchor-token transfer experiment.

Two independent tracks, each with exactly TWO filter rules. The rules ONLY
select which triples are suitable probes; nothing here deletes edges. Every
selected triple is simply omitted from a language's corpus at hold-out time.

  mask-ENTITY  (always mask the 2nd entity b in "a r [MASK]" -> predict b):
      1. (a, r) maps to a unique target   (no (a, r, X) with X != b)
      2. deg(b) >= 2                       (masked entity stays grounded)

  mask-RELATION (mask the relation in "a [MASK] b" -> predict r):
      1. (a, b) carries a unique relation  (no (a, r2, b) with r2 != r)
      2. deg(a) >= 2 and deg(b) >= 2       (both ends grounded)

Sampling: per relation, take a fraction of that relation's usable pool
(proportional to the pool), skipping relations whose pool is below --min_usable.
Each selected probe is tagged with a probe direction for the two-model design:
  probe_in = "B"  -> A-only fact, probed in B  (tests A->B transfer)
  probe_in = "A"  -> B-only fact, probed in A  (tests B->A transfer)
The same selected set is what model-2 (floor) omits from BOTH languages.

Usage:
    python select_probe_triples.py
    python select_probe_triples.py --frac 0.2 --min_usable 30 --seed 42
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# This file lives at <repo>/Experiment/evaluation/masked_language_probing/select_probe_triples.py,
# so parents[3] is the repo root. Reads the v3 generated corpus (actual triples).
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[3]
    / "data/generate_sentences/v3_generated_sentences_adj.json"
)


def load_triples(path):
    """(source, relation, target) from the v3 generated JSON or a raw edges list.
    HasQuality / virtual-adjective edges are dropped (they feed adjectives)."""
    data = json.load(open(path))
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    out = []
    for r in rows:
        if r.get("relation") == "HasQuality" or r.get("source_type") == "virtual_adjective":
            continue
        out.append((r["source"], r["relation"], r["target"]))
    return out


def build_indexes(triples):
    deg = Counter()
    fwd = defaultdict(set)          # (s, r) -> {targets}
    pair_rel = defaultdict(set)     # (s, t) -> {relations}
    for s, r, t in triples:
        deg[s] += 1
        deg[t] += 1
        fwd[(s, r)].add(t)
        pair_rel[(s, t)].add(r)
    return deg, fwd, pair_rel


def usable_entity(triple, deg, fwd, min_degree):
    a, r, b = triple
    return len(fwd[(a, r)]) == 1 and deg[b] >= min_degree


def usable_relation(triple, deg, pair_rel, min_degree):
    a, r, b = triple
    return len(pair_rel[(a, b)]) == 1 and deg[a] >= min_degree and deg[b] >= min_degree


def select(triples, track, deg, fwd, pair_rel, total_by_rel,
           n_per_rel, min_total, min_degree, direction, rng):
    """
    Keep a relation only if it has >= min_total triples IN TOTAL (so omitting
    n_per_rel leaves enough of the relation to still be learned) AND its usable
    pool has >= n_per_rel triples (so we can actually draw n_per_rel probes).
    The n_per_rel probes are sampled from the USABLE pool.
    A single transfer direction is tested: probe_in = direction for every probe.
    Returns (selected list, per-relation usable-pool sizes).
    """
    pool = defaultdict(list)
    for tr in triples:
        ok = (usable_entity(tr, deg, fwd, min_degree) if track == "entity"
              else usable_relation(tr, deg, pair_rel, min_degree))
        if ok:
            pool[tr[1]].append(tr)

    selected = []
    pool_sizes = {}
    for rel, items in pool.items():
        pool_sizes[rel] = len(items)
        if total_by_rel[rel] < min_total:   # relation too small overall
            continue
        if len(items) < n_per_rel:          # not enough usable to sample n
            continue
        for a, r, b in rng.sample(items, n_per_rel):
            selected.append({
                "track": track,
                "source": a, "relation": r, "target": b,
                "mask": "target" if track == "entity" else "relation",
                "gold": b if track == "entity" else r,   # concept (synset) vs relation label
                "probe_in": direction,
            })
    return selected, pool_sizes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--n_per_relation", type=int, default=50,
                   help="probes to omit per kept relation (drawn from usable pool)")
    p.add_argument("--min_total", type=int, default=100,
                   help="only keep relations with >= this many TOTAL triples "
                        "(keeps the omitted fraction of each relation small)")
    p.add_argument("--min_degree", type=int, default=2)
    p.add_argument("--direction", choices=["A", "B"], default="B",
                   help="single transfer direction: probe_in language "
                        "(fact kept in the other language, omitted from this one)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "probe_manifest.json"))
    args = p.parse_args()

    rng = random.Random(args.seed)
    triples = load_triples(args.input)
    deg, fwd, pair_rel = build_indexes(triples)
    total_by_rel = Counter(r for _, r, _ in triples)
    print(f"loaded {len(triples)} triples from {args.input}\n")

    manifest = {"seed": args.seed, "n_per_relation": args.n_per_relation,
                "min_total": args.min_total, "min_degree": args.min_degree,
                "direction": args.direction, "probes": []}

    n, min_total = args.n_per_relation, args.min_total
    for track in ("entity", "relation"):
        sel, pools = select(triples, track, deg, fwd, pair_rel, total_by_rel,
                            n, min_total, args.min_degree, args.direction, rng)
        manifest["probes"].extend(sel)

        kept = [r for r in pools if total_by_rel[r] >= min_total and pools[r] >= n]
        dropped = [r for r in pools if r not in kept]
        print(f"=== mask-{track.upper()} ===")
        print(f"{'relation':18s} {'total':>6s} {'usable':>7s} {'omit':>6s} {'%oftotal':>9s}")
        print("-" * 50)
        for rel in sorted(kept, key=lambda r: -total_by_rel[r]):
            print(f"{rel:18s} {total_by_rel[rel]:6d} {pools[rel]:7d} {n:6d} "
                  f"{100*n/total_by_rel[rel]:8.0f}%")
        print(f"{'TOTAL':18s} {sum(total_by_rel[r] for r in kept):6d} "
              f"{sum(pools[r] for r in kept):7d} {len(sel):6d}")
        print(f"kept {len(kept)} relations | dropped "
              f"(total<{min_total} or usable<{n}): "
              f"{sorted(dropped, key=lambda r: -total_by_rel[r])}\n")

    json.dump(manifest, open(args.out, "w"), indent=2, ensure_ascii=False)
    n_e = sum(1 for x in manifest["probes"] if x["track"] == "entity")
    n_r = sum(1 for x in manifest["probes"] if x["track"] == "relation")
    print(f"omit set: {n_e} mask-entity + {n_r} mask-relation = {len(manifest['probes'])} triples")
    print(f"saved manifest -> {args.out}")


if __name__ == "__main__":
    main()
