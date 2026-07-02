#!/usr/bin/env python3
"""Prepare pruned graph-density conditions from the noun-only full graph."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def nodes_from_edges(edges):
    endpoints = sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})
    return [{"id": endpoint} for endpoint in endpoints]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def edge_group_key(edge):
    return edge.get("relation", ""), edge.get("source_type", "")


def edge_identity(edge):
    return edge["source"], edge["relation"], edge["target"], edge.get("source_type", "")


def target_counts_by_group(edges, fraction):
    groups = defaultdict(list)
    for index, edge in enumerate(edges):
        groups[edge_group_key(edge)].append(index)

    raw_targets = {
        key: len(indices) * fraction
        for key, indices in groups.items()
    }
    floors = {
        key: int(value)
        for key, value in raw_targets.items()
    }

    target_total = round(len(edges) * fraction)
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


def stratified_sample_edges(edges, fraction, seed):
    rng = random.Random(seed)
    groups, counts = target_counts_by_group(edges, fraction)
    selected_indices = []

    for key in sorted(groups):
        indices = groups[key][:]
        rng.shuffle(indices)
        selected_indices.extend(indices[:counts[key]])

    return [edges[index] for index in sorted(selected_indices)]


def stratified_sample_indices(edges, fraction, seed):
    rng = random.Random(seed)
    groups, counts = target_counts_by_group(edges, fraction)
    selected = set()
    for key in sorted(groups):
        indices = groups[key][:]
        rng.shuffle(indices)
        selected.update(indices[:counts[key]])
    return selected


def edge_cover_indices(nodes, edges, seed):
    rng = random.Random(seed)
    node_ids = [node["id"] for node in nodes]
    node_set = set(node_ids)
    degree = Counter()
    incident = defaultdict(list)
    for index, edge in enumerate(edges):
        source = edge["source"]
        target = edge["target"]
        if source not in node_set or target not in node_set:
            continue
        degree[source] += 1
        degree[target] += 1
        incident[source].append(index)
        incident[target].append(index)

    edge_order = list(range(len(edges)))
    rng.shuffle(edge_order)
    edge_order.sort(
        key=lambda index: (
            min(degree[edges[index]["source"]], degree[edges[index]["target"]]),
            degree[edges[index]["source"]] + degree[edges[index]["target"]],
        )
    )

    selected = set()
    matched_nodes = set()
    for index in edge_order:
        source = edges[index]["source"]
        target = edges[index]["target"]
        if source in matched_nodes or target in matched_nodes:
            continue
        selected.add(index)
        matched_nodes.add(source)
        matched_nodes.add(target)

    covered = set()
    for index in selected:
        covered.add(edges[index]["source"])
        covered.add(edges[index]["target"])

    for node_id in sorted(node_ids, key=lambda item: (degree[item], item)):
        if node_id in covered:
            continue
        candidates = [index for index in incident[node_id] if index not in selected]
        if not candidates:
            continue
        candidates.sort(
            key=lambda index: (
                edges[index]["source"] in covered,
                edges[index]["target"] in covered,
                degree[edges[index]["source"]] + degree[edges[index]["target"]],
                rng.random(),
            )
        )
        chosen = candidates[0]
        selected.add(chosen)
        covered.add(edges[chosen]["source"])
        covered.add(edges[chosen]["target"])

    return selected


def selected_degrees(edges, selected):
    degree = Counter()
    for index in selected:
        degree[edges[index]["source"]] += 1
        degree[edges[index]["target"]] += 1
    return degree


def uncovered_nodes(nodes, edges, selected):
    covered = set()
    for index in selected:
        covered.add(edges[index]["source"])
        covered.add(edges[index]["target"])
    return [node["id"] for node in nodes if node["id"] not in covered]


def relation_first_sample_edges(nodes, edges, fraction, seed):
    """Prioritize original relation proportions, then repair isolated nodes."""
    target_total = round(len(edges) * fraction)
    selected = stratified_sample_indices(edges, fraction, seed)
    if len(selected) != target_total:
        raise SystemExit("Internal error: stratified sample did not hit target edge count.")

    _, target_counts = target_counts_by_group(edges, fraction)
    selected_counts = Counter(edge_group_key(edges[index]) for index in selected)
    incident = defaultdict(list)
    for index, edge in enumerate(edges):
        incident[edge["source"]].append(index)
        incident[edge["target"]].append(index)

    rng = random.Random(seed + 50_000)
    for node_id in uncovered_nodes(nodes, edges, selected):
        add_candidates = [index for index in incident[node_id] if index not in selected]
        if not add_candidates:
            continue
        rng.shuffle(add_candidates)
        add_candidates.sort(
            key=lambda index: (
                selected_counts[edge_group_key(edges[index])]
                - target_counts.get(edge_group_key(edges[index]), 0),
                edge_group_key(edges[index]),
            )
        )

        added = False
        for add_index in add_candidates:
            add_group = edge_group_key(edges[add_index])
            trial_selected = set(selected)
            trial_selected.add(add_index)
            degree_after_add = selected_degrees(edges, trial_selected)

            remove_candidates = [
                index for index in selected
                if degree_after_add[edges[index]["source"]] > 1
                and degree_after_add[edges[index]["target"]] > 1
            ]
            rng.shuffle(remove_candidates)
            remove_candidates.sort(
                key=lambda index: (
                    -(
                        selected_counts[edge_group_key(edges[index])]
                        - target_counts.get(edge_group_key(edges[index]), 0)
                    ),
                    edge_group_key(edges[index]),
                )
            )

            best_remove = None
            best_distance = None
            for remove_index in remove_candidates[:500]:
                trial_counts = selected_counts.copy()
                trial_counts[add_group] += 1
                trial_counts[edge_group_key(edges[remove_index])] -= 1
                distance = relation_distance(trial_counts, target_counts)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_remove = remove_index

            if best_remove is None:
                continue

            selected.add(add_index)
            selected.remove(best_remove)
            selected_counts[add_group] += 1
            selected_counts[edge_group_key(edges[best_remove])] -= 1
            added = True
            break

        if not added:
            raise SystemExit(f"Could not repair isolated node {node_id!r}.")

    remaining_uncovered = uncovered_nodes(nodes, edges, selected)
    if remaining_uncovered:
        raise SystemExit(
            f"Could not cover all nodes; remaining isolated nodes: {len(remaining_uncovered)}"
        )

    return [edges[index] for index in sorted(selected)]


def coverage_first_sample_edges(nodes, edges, fraction, seed):
    target_total = round(len(edges) * fraction)
    selected = edge_cover_indices(nodes, edges, seed)
    if len(selected) > target_total:
        raise SystemExit(
            f"Cannot cover all nodes with {target_total} edges; "
            f"edge-cover pass needed {len(selected)} edges. "
            "Increase --fraction."
        )

    _, target_counts = target_counts_by_group(edges, fraction)
    selected_counts = Counter(edge_group_key(edges[index]) for index in selected)
    remaining = [index for index in range(len(edges)) if index not in selected]
    rng = random.Random(seed + 1_000)
    rng.shuffle(remaining)
    remaining.sort(
        key=lambda index: (
            target_counts.get(edge_group_key(edges[index]), 0)
            - selected_counts[edge_group_key(edges[index])],
            edge_group_key(edges[index]),
        ),
        reverse=True,
    )

    for index in remaining:
        if len(selected) >= target_total:
            break
        selected.add(index)
        selected_counts[edge_group_key(edges[index])] += 1

    return [edges[index] for index in sorted(selected)]


def dense_control_sample_edges(nodes, full_edges, candidate_edges, training_fraction, seed):
    """Select a low-sized edge set from a denser candidate graph.

    Node coverage is mandatory. Remaining edges are selected from the candidate
    graph by preferring endpoints with higher candidate-graph degree. Counts are
    nudged toward the low-density relation/source-type proportions.
    """
    rng = random.Random(seed + 10_000)
    target_total = round(len(full_edges) * training_fraction)
    selected_edges = coverage_first_sample_edges(
        nodes=nodes,
        edges=candidate_edges,
        fraction=target_total / len(candidate_edges),
        seed=seed + 20_000,
    )
    selected_keys = {
        (edge["source"], edge["relation"], edge["target"], edge.get("source_type", ""))
        for edge in selected_edges
    }
    candidate_degrees = Counter()
    for edge in candidate_edges:
        candidate_degrees[edge["source"]] += 1
        candidate_degrees[edge["target"]] += 1

    _, target_counts = target_counts_by_group(full_edges, training_fraction)
    selected_counts = Counter(edge_group_key(edge) for edge in selected_edges)
    remaining = [
        edge for edge in candidate_edges
        if (edge["source"], edge["relation"], edge["target"], edge.get("source_type", ""))
        not in selected_keys
    ]
    rng.shuffle(remaining)
    remaining.sort(
        key=lambda edge: (
            target_counts.get(edge_group_key(edge), 0) - selected_counts[edge_group_key(edge)],
            candidate_degrees[edge["source"]] + candidate_degrees[edge["target"]],
            max(candidate_degrees[edge["source"]], candidate_degrees[edge["target"]]),
            edge["source"],
            edge["target"],
            edge["relation"],
        ),
        reverse=True,
    )

    selected = list(selected_edges)
    for edge in remaining:
        if len(selected) >= target_total:
            break
        selected.append(edge)
        selected_counts[edge_group_key(edge)] += 1

    return sorted(selected, key=lambda edge: (edge["source"], edge["relation"], edge["target"]))


def relation_distance(counts, target_counts):
    keys = set(counts) | set(target_counts)
    return sum(abs(counts.get(key, 0) - target_counts.get(key, 0)) for key in keys)


def relation_control_sample_edges(nodes, full_edges, candidate_edges, training_fraction, seed):
    """Select a low-sized control graph while prioritizing relation proportions."""
    target_total = round(len(full_edges) * training_fraction)
    selected_edges = coverage_first_sample_edges(
        nodes=nodes,
        edges=candidate_edges,
        fraction=target_total / len(candidate_edges),
        seed=seed + 30_000,
    )
    _, target_counts = target_counts_by_group(full_edges, training_fraction)
    selected = {edge_identity(edge): edge for edge in selected_edges}
    unselected = {
        edge_identity(edge): edge for edge in candidate_edges
        if edge_identity(edge) not in selected
    }

    node_degree = Counter()
    for edge in selected.values():
        node_degree[edge["source"]] += 1
        node_degree[edge["target"]] += 1

    selected_counts = Counter(edge_group_key(edge) for edge in selected.values())
    current_distance = relation_distance(selected_counts, target_counts)

    rng = random.Random(seed + 40_000)
    for _ in range(target_total * 4):
        over_edges = [
            edge for edge in selected.values()
            if selected_counts[edge_group_key(edge)] > target_counts.get(edge_group_key(edge), 0)
            and node_degree[edge["source"]] > 1
            and node_degree[edge["target"]] > 1
        ]
        under_edges = [
            edge for edge in unselected.values()
            if selected_counts[edge_group_key(edge)] < target_counts.get(edge_group_key(edge), 0)
        ]
        if not over_edges or not under_edges:
            break

        rng.shuffle(over_edges)
        rng.shuffle(under_edges)
        improved = False
        for remove_edge in over_edges[:200]:
            remove_group = edge_group_key(remove_edge)
            for add_edge in under_edges[:200]:
                add_group = edge_group_key(add_edge)
                trial_counts = selected_counts.copy()
                trial_counts[remove_group] -= 1
                trial_counts[add_group] += 1
                trial_distance = relation_distance(trial_counts, target_counts)
                if trial_distance >= current_distance:
                    continue

                selected.pop(edge_identity(remove_edge))
                unselected[edge_identity(remove_edge)] = remove_edge
                unselected.pop(edge_identity(add_edge))
                selected[edge_identity(add_edge)] = add_edge
                node_degree[remove_edge["source"]] -= 1
                node_degree[remove_edge["target"]] -= 1
                node_degree[add_edge["source"]] += 1
                node_degree[add_edge["target"]] += 1
                selected_counts = trial_counts
                current_distance = trial_distance
                improved = True
                break
            if improved:
                break
        if not improved:
            break

    return sorted(selected.values(), key=lambda edge: (edge["source"], edge["relation"], edge["target"]))


def graph_stats(nodes, edges):
    node_ids = [node["id"] for node in nodes]
    node_set = set(node_ids)
    neighbors = {node_id: set() for node_id in node_ids}
    active = set()
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
        neighbors[source].add(target)
        neighbors[target].add(source)
        active.add(source)
        active.add(target)

    degrees = [len(neighbors[node_id]) for node_id in node_ids]
    active_degrees = [len(neighbors[node_id]) for node_id in active]

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


def main():
    parser = argparse.ArgumentParser(
        description="Create a pruned graph-density condition from the full KG."
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/semantic_backbones/kg_noun_only/edges.json"),
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("data/semantic_backbones/kg_noun_only/synsets.json"),
    )
    parser.add_argument(
        "--nodes-from-edges",
        action="store_true",
        help="Use all unique edge endpoints as the node universe.",
    )
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument(
        "--sampling-strategy",
        choices=["coverage", "relation"],
        default="coverage",
        help="Base graph sampling strategy before any control downsampling.",
    )
    parser.add_argument(
        "--training-fraction",
        type=float,
        help=(
            "If set below --fraction, write a control graph with this "
            "fraction of full-graph training edges selected from the candidate graph."
        ),
    )
    parser.add_argument(
        "--control-strategy",
        choices=["dense", "relation"],
        default="dense",
        help="Control sampling strategy when --training-fraction is below --fraction.",
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/graph_density"),
    )
    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        raise SystemExit("--fraction must be in the interval (0, 1].")

    edges = read_json(args.edges)
    nodes = nodes_from_edges(edges) if args.nodes_from_edges else read_json(args.nodes)
    if args.sampling_strategy == "relation":
        candidate_edges = relation_first_sample_edges(
            nodes=nodes,
            edges=edges,
            fraction=args.fraction,
            seed=args.seed,
        )
    else:
        candidate_edges = coverage_first_sample_edges(
            nodes=nodes,
            edges=edges,
            fraction=args.fraction,
            seed=args.seed,
        )
    training_fraction = args.training_fraction or args.fraction
    if not 0 < training_fraction <= args.fraction:
        raise SystemExit("--training-fraction must be in the interval (0, --fraction].")

    if training_fraction < args.fraction:
        if args.control_strategy == "relation":
            sampled_edges = relation_control_sample_edges(
                nodes=nodes,
                full_edges=edges,
                candidate_edges=candidate_edges,
                training_fraction=training_fraction,
                seed=args.seed,
            )
        else:
            sampled_edges = dense_control_sample_edges(
                nodes=nodes,
                full_edges=edges,
                candidate_edges=candidate_edges,
                training_fraction=training_fraction,
                seed=args.seed,
            )
    else:
        sampled_edges = candidate_edges

    output_dir = args.output_root / f"{args.condition}_seed_{args.seed}"
    if training_fraction < args.fraction:
        write_json(output_dir / "candidate_edges.json", candidate_edges)
    write_json(output_dir / "edges.json", sampled_edges)
    write_json(output_dir / "metadata.json", {
        "condition": args.condition,
        "seed": args.seed,
        "candidate_fraction": args.fraction,
        "training_fraction": training_fraction,
        "sampling_strategy": args.sampling_strategy,
        "control_strategy": args.control_strategy if training_fraction < args.fraction else None,
        "is_control": training_fraction < args.fraction,
        "inputs": {
            "edges": str(args.edges),
            "nodes": "edge_endpoints" if args.nodes_from_edges else str(args.nodes),
        },
        "full_graph": graph_stats(nodes, edges),
        "candidate_graph": graph_stats(nodes, candidate_edges),
        "sampled_graph": graph_stats(nodes, sampled_edges),
    })

    stats = graph_stats(nodes, sampled_edges)
    print(f"Wrote {output_dir / 'edges.json'}")
    print(f"Edges: {stats['edge_count']} / {len(edges)}")
    print(f"Active nodes: {stats['active_node_count']} / {len(nodes)}")
    print(f"Isolated nodes: {stats['isolated_node_count']}")
    print(f"Average degree: {stats['average_degree_all_nodes']:.3f}")
    print(f"Max degree: {stats['max_degree']}")


if __name__ == "__main__":
    main()
