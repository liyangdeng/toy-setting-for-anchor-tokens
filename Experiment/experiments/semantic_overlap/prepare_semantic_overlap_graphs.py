#!/usr/bin/env python3
"""Prepare paired graph samples for semantic-overlap experiments."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def edge_identity(edge):
    return (
        edge["source"],
        edge["relation"],
        edge["target"],
        edge.get("source_type", ""),
    )


def edge_group_key(edge):
    return edge.get("relation", ""), edge.get("source_type", "")


def nodes_from_edges(edges):
    return sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})


def unique_edges(edges):
    seen = set()
    deduped = []
    for edge in edges:
        identity = edge_identity(edge)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(edge)
    return deduped


def target_counts_by_group(indices, edges, target_total):
    groups = defaultdict(list)
    for index in indices:
        groups[edge_group_key(edges[index])].append(index)

    if not indices:
        return groups, {}

    raw_targets = {
        key: len(group_indices) * target_total / len(indices)
        for key, group_indices in groups.items()
    }
    floors = {key: int(value) for key, value in raw_targets.items()}
    remainder = target_total - sum(floors.values())
    ranked = sorted(
        raw_targets,
        key=lambda key: (raw_targets[key] - floors[key], len(groups[key]), key),
        reverse=True,
    )

    counts = dict(floors)
    for key in ranked[:remainder]:
        counts[key] += 1
    return groups, counts


def stratified_sample_indices(indices, edges, target_total, rng):
    if target_total < 0:
        raise ValueError("target_total must be non-negative")
    if target_total > len(indices):
        raise ValueError(f"Cannot sample {target_total} edges from pool of {len(indices)}")

    groups, counts = target_counts_by_group(indices, edges, target_total)
    selected = []
    for key in sorted(groups):
        group_indices = groups[key][:]
        rng.shuffle(group_indices)
        selected.extend(group_indices[:counts.get(key, 0)])

    if len(selected) != target_total:
        raise RuntimeError(f"Expected {target_total} sampled edges, got {len(selected)}")
    return set(selected)


def graph_stats(nodes, edges):
    node_set = set(nodes)
    active = set()
    degree = Counter()
    missing_endpoint_edges = 0
    self_loops = 0

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source == target:
            self_loops += 1
        if source not in node_set or target not in node_set:
            missing_endpoint_edges += 1
            continue
        active.add(source)
        active.add(target)
        degree[source] += 1
        degree[target] += 1

    degrees = [degree[node] for node in nodes]
    active_degrees = [degree[node] for node in active]
    return {
        "edge_count": len(edges),
        "node_count": len(nodes),
        "active_node_count": len(active),
        "isolated_node_count": len(nodes) - len(active),
        "average_degree_all_nodes": sum(degrees) / len(degrees) if degrees else 0.0,
        "average_degree_active_nodes": (
            sum(active_degrees) / len(active_degrees) if active_degrees else 0.0
        ),
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "degree_histogram": dict(sorted(Counter(degrees).items())),
        "relation_counts": dict(sorted(Counter(edge["relation"] for edge in edges).items())),
        "source_type_counts": dict(
            sorted(Counter(edge.get("source_type", "") for edge in edges).items())
        ),
        "missing_endpoint_edge_count": missing_endpoint_edges,
        "self_loop_count": self_loops,
    }


def active_nodes_for_indices(edges, indices):
    active = set()
    for index in indices:
        active.add(edges[index]["source"])
        active.add(edges[index]["target"])
    return active


def degrees_for_indices(edges, indices):
    degree = Counter()
    for index in indices:
        degree[edges[index]["source"]] += 1
        degree[edges[index]["target"]] += 1
    return degree


def relation_distance(counts, target_counts):
    keys = set(counts) | set(target_counts)
    return sum(abs(counts.get(key, 0) - target_counts.get(key, 0)) for key in keys)


def repair_variable_coverage(
    edges,
    nodes,
    variable,
    fixed,
    available,
    target_counts,
    seed,
    max_rounds=10_000,
):
    """Swap variable edges to cover more nodes while preserving edge count.

    The fixed set is kept untouched. New edges are drawn from available, and
    removals are restricted to variable edges so overlap sizes remain fixed.
    """
    rng = random.Random(seed)
    variable = set(variable)
    fixed = set(fixed)
    available = set(available)

    selected = fixed | variable
    selected_counts = Counter(edge_group_key(edges[index]) for index in variable)
    current_distance = relation_distance(selected_counts, target_counts)
    node_set = set(nodes)

    for _ in range(max_rounds):
        active = active_nodes_for_indices(edges, selected)
        uncovered = node_set - active
        if not uncovered:
            break

        candidates = [
            index for index in available
            if index not in selected
            and (edges[index]["source"] in uncovered or edges[index]["target"] in uncovered)
        ]
        if not candidates:
            break

        rng.shuffle(candidates)
        candidates.sort(
            key=lambda index: (
                edges[index]["source"] in uncovered,
                edges[index]["target"] in uncovered,
                target_counts.get(edge_group_key(edges[index]), 0)
                - selected_counts[edge_group_key(edges[index])],
                rng.random(),
            ),
            reverse=True,
        )

        improved = False
        for add_index in candidates[:300]:
            add_group = edge_group_key(edges[add_index])
            after_add = selected | {add_index}
            degree_after_add = degrees_for_indices(edges, after_add)
            active_after_add = active_nodes_for_indices(edges, after_add)
            add_gain = len(active_after_add - active)

            removable = [
                index for index in variable
                if degree_after_add[edges[index]["source"]] > 1
                and degree_after_add[edges[index]["target"]] > 1
            ]
            if not removable:
                continue

            rng.shuffle(removable)
            best_remove = None
            best_score = None
            for remove_index in removable[:300]:
                remove_group = edge_group_key(edges[remove_index])
                trial_counts = selected_counts.copy()
                trial_counts[add_group] += 1
                trial_counts[remove_group] -= 1
                trial_distance = relation_distance(trial_counts, target_counts)
                score = (
                    add_gain,
                    -trial_distance,
                    current_distance - trial_distance,
                    rng.random(),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_remove = remove_index

            if best_remove is None:
                continue

            selected.add(add_index)
            selected.remove(best_remove)
            variable.add(add_index)
            variable.remove(best_remove)
            selected_counts[add_group] += 1
            selected_counts[edge_group_key(edges[best_remove])] -= 1
            current_distance = relation_distance(selected_counts, target_counts)
            improved = True
            break

        if not improved:
            break

    return variable


def relation_share(stats):
    total = stats["edge_count"]
    if not total:
        return {}
    return {
        relation: count / total
        for relation, count in sorted(stats["relation_counts"].items())
    }


def max_relation_share_delta(reference_stats, sample_stats):
    reference = relation_share(reference_stats)
    sample = relation_share(sample_stats)
    keys = set(reference) | set(sample)
    if not keys:
        return 0.0
    return max(abs(reference.get(key, 0.0) - sample.get(key, 0.0)) for key in keys)


def sample_pair(edges, nodes, per_language_edges, overlap_ratio, seed, coverage_repair=False):
    shared_count = round(per_language_edges * overlap_ratio)
    exclusive_count = per_language_edges - shared_count
    total_needed = shared_count + 2 * exclusive_count
    if total_needed > len(edges):
        raise ValueError(
            f"Need {total_needed} unique edges for overlap={overlap_ratio}, "
            f"but full graph has only {len(edges)}"
        )

    all_indices = list(range(len(edges)))
    rng = random.Random(seed)
    shared = stratified_sample_indices(all_indices, edges, shared_count, rng)
    if coverage_repair and shared_count:
        _, shared_target_counts = target_counts_by_group(all_indices, edges, shared_count)
        shared = repair_variable_coverage(
            edges=edges,
            nodes=nodes,
            variable=shared,
            fixed=set(),
            available=all_indices,
            target_counts=shared_target_counts,
            seed=seed + 30_000,
        )

    remaining = [index for index in all_indices if index not in shared]
    l1_rng = random.Random(seed + 10_000)
    l1_only = stratified_sample_indices(remaining, edges, exclusive_count, l1_rng)
    if coverage_repair and exclusive_count:
        _, l1_target_counts = target_counts_by_group(remaining, edges, exclusive_count)
        l1_only = repair_variable_coverage(
            edges=edges,
            nodes=nodes,
            variable=l1_only,
            fixed=shared,
            available=remaining,
            target_counts=l1_target_counts,
            seed=seed + 40_000,
        )

    remaining = [index for index in remaining if index not in l1_only]
    l2_rng = random.Random(seed + 20_000)
    l2_only = stratified_sample_indices(remaining, edges, exclusive_count, l2_rng)
    if coverage_repair and exclusive_count:
        _, l2_target_counts = target_counts_by_group(remaining, edges, exclusive_count)
        l2_only = repair_variable_coverage(
            edges=edges,
            nodes=nodes,
            variable=l2_only,
            fixed=shared,
            available=remaining,
            target_counts=l2_target_counts,
            seed=seed + 50_000,
        )

    l1_indices = shared | l1_only
    l2_indices = shared | l2_only

    return {
        "shared": [edges[index] for index in sorted(shared)],
        "l1_only": [edges[index] for index in sorted(l1_only)],
        "l2_only": [edges[index] for index in sorted(l2_only)],
        "l1": [edges[index] for index in sorted(l1_indices)],
        "l2": [edges[index] for index in sorted(l2_indices)],
    }


def overlap_stats(l1_edges, l2_edges):
    l1 = {edge_identity(edge) for edge in l1_edges}
    l2 = {edge_identity(edge) for edge in l2_edges}
    shared = l1 & l2
    return {
        "l1_edge_count": len(l1),
        "l2_edge_count": len(l2),
        "shared_edge_count": len(shared),
        "overlap_l1": len(shared) / len(l1) if l1 else 0.0,
        "overlap_l2": len(shared) / len(l2) if l2 else 0.0,
        "jaccard": len(shared) / len(l1 | l2) if l1 or l2 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, default=Path("data/semantic_backbones/edges_adj.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/semantic_overlap"))
    parser.add_argument("--condition-set", default="v1_seed_42")
    parser.add_argument("--per-language-fraction", type=float, default=0.4)
    parser.add_argument("--overlap-ratios", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage-repair", action="store_true")
    args = parser.parse_args()

    if not 0 < args.per_language_fraction <= 0.5:
        raise SystemExit("--per-language-fraction must be in the interval (0, 0.5].")

    raw_edges = read_json(args.edges)
    edges = unique_edges(raw_edges)
    nodes = nodes_from_edges(edges)
    full_stats = graph_stats(nodes, edges)
    per_language_edges = round(len(edges) * args.per_language_fraction)

    output_root = args.output_root / args.condition_set
    summary = {
        "input_edges": str(args.edges),
        "raw_edge_count": len(raw_edges),
        "deduplicated_edge_count": len(edges),
        "duplicate_edge_identity_count": len(raw_edges) - len(edges),
        "seed": args.seed,
        "per_language_fraction": args.per_language_fraction,
        "per_language_edges": per_language_edges,
        "full_graph": full_stats,
        "conditions": [],
    }

    for overlap_ratio in args.overlap_ratios:
        if not 0 <= overlap_ratio <= 1:
            raise SystemExit("All overlap ratios must be in the interval [0, 1].")

        pair = sample_pair(
            edges=edges,
            nodes=nodes,
            per_language_edges=per_language_edges,
            overlap_ratio=overlap_ratio,
            seed=args.seed + round(overlap_ratio * 1_000),
            coverage_repair=args.coverage_repair,
        )
        condition = f"overlap_{int(round(overlap_ratio * 100)):03d}"
        condition_dir = output_root / condition

        write_json(condition_dir / "edges_l1.json", pair["l1"])
        write_json(condition_dir / "edges_l2.json", pair["l2"])
        write_json(condition_dir / "shared_edges.json", pair["shared"])
        write_json(condition_dir / "l1_only_edges.json", pair["l1_only"])
        write_json(condition_dir / "l2_only_edges.json", pair["l2_only"])

        l1_stats = graph_stats(nodes, pair["l1"])
        l2_stats = graph_stats(nodes, pair["l2"])
        condition_metadata = {
            "condition": condition,
            "target_overlap_ratio": overlap_ratio,
            "seed": args.seed,
            "coverage_repair": args.coverage_repair,
            "shared_edge_count": len(pair["shared"]),
            "l1_only_edge_count": len(pair["l1_only"]),
            "l2_only_edge_count": len(pair["l2_only"]),
            "overlap": overlap_stats(pair["l1"], pair["l2"]),
            "l1_graph": l1_stats,
            "l2_graph": l2_stats,
            "max_relation_share_delta_vs_full": {
                "l1": max_relation_share_delta(full_stats, l1_stats),
                "l2": max_relation_share_delta(full_stats, l2_stats),
            },
        }
        write_json(condition_dir / "metadata.json", condition_metadata)
        summary["conditions"].append(condition_metadata)

    write_json(output_root / "summary.json", summary)
    print(f"Wrote semantic-overlap graph samples to {output_root}")
    print(f"Per-language edges: {per_language_edges} / {len(edges)}")
    for condition in summary["conditions"]:
        overlap = condition["overlap"]
        print(
            f"{condition['condition']}: shared={overlap['shared_edge_count']} "
            f"overlap={overlap['overlap_l1']:.3f} "
            f"isolated_l1={condition['l1_graph']['isolated_node_count']} "
            f"isolated_l2={condition['l2_graph']['isolated_node_count']} "
            f"max_rel_delta_l1={condition['max_relation_share_delta_vs_full']['l1']:.4f} "
            f"max_rel_delta_l2={condition['max_relation_share_delta_vs_full']['l2']:.4f}"
        )


if __name__ == "__main__":
    main()
