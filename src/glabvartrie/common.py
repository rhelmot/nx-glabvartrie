from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Generic, Hashable, TypeVar

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

N = TypeVar("N", bound=Hashable)
L = TypeVar("L", bound=Hashable)
V = TypeVar("V", bound=Hashable)
I = TypeVar("I", bound=Hashable)


def _default_node_order_key(node: Hashable) -> Any:
    return node


def _default_label_matches(left: Any, right: Any) -> bool:
    return left == right


def _stable_value_key(value: Any) -> tuple[str, str, str]:
    return (type(value).__module__, type(value).__qualname__, repr(value))


def graphs_are_isomorphic(
    g1: nx.DiGraph[N],
    g2: nx.DiGraph[N],
    node_label: Callable[[dict[str, Any]], L],
    node_vars: Callable[[dict[str, Any]], tuple[V, ...]],
    label_matches: Callable[[L, L], bool] | None = None,
) -> bool:
    if g1.number_of_nodes() != g2.number_of_nodes():
        return False
    if g1.number_of_edges() != g2.number_of_edges():
        return False

    match_labels = label_matches or _default_label_matches
    g1_labels = {node: node_label(g1.nodes[node]) for node in g1.nodes}
    g2_labels = {node: node_label(g2.nodes[node]) for node in g2.nodes}
    g1_vars = {node: node_vars(g1.nodes[node]) for node in g1.nodes}
    g2_vars = {node: node_vars(g2.nodes[node]) for node in g2.nodes}

    class _Matcher(DiGraphMatcher):
        def semantic_feasibility(self, G1_node: N, G2_node: N) -> bool:
            if not match_labels(g1_labels[G1_node], g2_labels[G2_node]):
                return False

            source_vars = g1_vars[G1_node]
            target_vars = g2_vars[G2_node]
            if len(source_vars) != len(target_vars):
                return False

            source_to_target: defaultdict[int, dict[V, V]] = defaultdict(dict)
            target_to_source: defaultdict[int, dict[V, V]] = defaultdict(dict)
            for matched_g1, matched_g2 in self.core_1.items():
                matched_source_vars = g1_vars[matched_g1]
                matched_target_vars = g2_vars[matched_g2]
                if len(matched_source_vars) != len(matched_target_vars):
                    return False
                for variable_class, source_identifier in enumerate(matched_source_vars):
                    target_identifier = matched_target_vars[variable_class]
                    existing_target = source_to_target[variable_class].get(source_identifier)
                    existing_source = target_to_source[variable_class].get(target_identifier)
                    if existing_target is not None and existing_target != target_identifier:
                        return False
                    if existing_source is not None and existing_source != source_identifier:
                        return False
                    source_to_target[variable_class][source_identifier] = target_identifier
                    target_to_source[variable_class][target_identifier] = source_identifier

            for variable_class, source_identifier in enumerate(source_vars):
                target_identifier = target_vars[variable_class]
                existing_target = source_to_target[variable_class].get(source_identifier)
                existing_source = target_to_source[variable_class].get(target_identifier)
                if existing_target is not None and existing_target != target_identifier:
                    return False
                if existing_source is not None and existing_source != source_identifier:
                    return False

            return True

    return _Matcher(g1, g2).is_isomorphic()


def graph_isomorphism_mapping(
    g1: nx.DiGraph[N],
    g2: nx.DiGraph[N],
    node_label: Callable[[dict[str, Any]], L],
    node_vars: Callable[[dict[str, Any]], tuple[V, ...]],
    label_matches: Callable[[L, L], bool] | None = None,
) -> dict[N, N] | None:
    if g1.number_of_nodes() != g2.number_of_nodes():
        return None
    if g1.number_of_edges() != g2.number_of_edges():
        return None

    match_labels = label_matches or _default_label_matches
    g1_labels = {node: node_label(g1.nodes[node]) for node in g1.nodes}
    g2_labels = {node: node_label(g2.nodes[node]) for node in g2.nodes}
    g1_vars = {node: node_vars(g1.nodes[node]) for node in g1.nodes}
    g2_vars = {node: node_vars(g2.nodes[node]) for node in g2.nodes}

    class _Matcher(DiGraphMatcher):
        def semantic_feasibility(self, G1_node: N, G2_node: N) -> bool:
            if not match_labels(g1_labels[G1_node], g2_labels[G2_node]):
                return False

            source_vars = g1_vars[G1_node]
            target_vars = g2_vars[G2_node]
            if len(source_vars) != len(target_vars):
                return False

            source_to_target: defaultdict[int, dict[V, V]] = defaultdict(dict)
            target_to_source: defaultdict[int, dict[V, V]] = defaultdict(dict)
            for matched_g1, matched_g2 in self.core_1.items():
                matched_source_vars = g1_vars[matched_g1]
                matched_target_vars = g2_vars[matched_g2]
                if len(matched_source_vars) != len(matched_target_vars):
                    return False
                for variable_class, source_identifier in enumerate(matched_source_vars):
                    target_identifier = matched_target_vars[variable_class]
                    existing_target = source_to_target[variable_class].get(source_identifier)
                    existing_source = target_to_source[variable_class].get(target_identifier)
                    if existing_target is not None and existing_target != target_identifier:
                        return False
                    if existing_source is not None and existing_source != source_identifier:
                        return False
                    source_to_target[variable_class][source_identifier] = target_identifier
                    target_to_source[variable_class][target_identifier] = source_identifier

            for variable_class, source_identifier in enumerate(source_vars):
                target_identifier = target_vars[variable_class]
                existing_target = source_to_target[variable_class].get(source_identifier)
                existing_source = target_to_source[variable_class].get(target_identifier)
                if existing_target is not None and existing_target != target_identifier:
                    return False
                if existing_source is not None and existing_source != source_identifier:
                    return False

            return True

    matcher = _Matcher(g1, g2)
    return next(matcher.isomorphisms_iter(), None)


def topologies_are_isomorphic(g1: nx.DiGraph[N], g2: nx.DiGraph[N]) -> bool:
    if g1.number_of_nodes() != g2.number_of_nodes():
        return False
    if g1.number_of_edges() != g2.number_of_edges():
        return False
    return DiGraphMatcher(g1, g2).is_isomorphic()


def _position_graph(
    graph: nx.DiGraph[N],
    order: tuple[N, ...],
) -> tuple[tuple[tuple[bool, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    adjacency = tuple(
        tuple(graph.has_edge(source, target) for target in order)
        for source in order
    )
    neighbours = tuple(
        tuple(
            index
            for index, target in enumerate(order)
            if adjacency[position][index] or adjacency[index][position]
        )
        for position, _ in enumerate(order)
    )
    neighbour_counts = tuple(len(node_neighbours) for node_neighbours in neighbours)
    return adjacency, neighbours, neighbour_counts


def _articulation_points(neighbours: tuple[tuple[int, ...], ...], remaining: frozenset[int]) -> frozenset[int]:
    if len(remaining) <= 4:
        return frozenset()

    size = len(neighbours)
    remaining_mask = [False] * size
    for position in remaining:
        remaining_mask[position] = True

    minimum_remaining_degree = min(
        sum(1 for neighbour in neighbours[position] if remaining_mask[neighbour])
        for position in remaining
    )
    if minimum_remaining_degree * 2 >= len(remaining):
        return frozenset()

    remaining_count = len(remaining)
    root = next(iter(remaining))
    discovery = [-1] * size
    low = [0] * size
    parent = [-1] * size
    articulation: set[int] = set()
    time_counter = 0
    visited_count = 0

    def dfs(position: int) -> None:
        nonlocal time_counter, visited_count
        discovery[position] = time_counter
        low[position] = time_counter
        time_counter += 1
        visited_count += 1

        child_count = 0
        for neighbour in neighbours[position]:
            if not remaining_mask[neighbour]:
                continue
            if discovery[neighbour] == -1:
                parent[neighbour] = position
                child_count += 1
                dfs(neighbour)
                low[position] = min(low[position], low[neighbour])
                if parent[position] == -1:
                    if child_count > 1:
                        articulation.add(position)
                elif low[neighbour] >= discovery[position]:
                    articulation.add(position)
            elif neighbour != parent[position]:
                low[position] = min(low[position], discovery[neighbour])

    dfs(root)
    if visited_count != remaining_count:
        return remaining
    return frozenset(articulation)


def topology_order(graph: nx.DiGraph[N]) -> tuple[N, ...]:
    nodes = tuple(graph.nodes)
    size = len(nodes)
    if size == 0:
        return ()
    adjacency, _, _ = _position_graph(graph, nodes)
    color_classes = _refined_color_classes(adjacency)
    ordered_classes = tuple(tuple(sorted(color_class)) for color_class in color_classes)
    best_order: tuple[int, ...] | None = None
    best_key: tuple[int, ...] | None = None

    def recurse(class_index: int, chosen: list[int]) -> None:
        nonlocal best_order, best_key
        if class_index == len(ordered_classes):
            order_key = tuple(
                int(adjacency[source][target])
                for source in chosen
                for target in chosen
            )
            if best_key is None or order_key < best_key:
                best_key = order_key
                best_order = tuple(chosen)
            return

        color_class = ordered_classes[class_index]
        for permutation in _permutations(color_class):
            chosen.extend(permutation)
            recurse(class_index + 1, chosen)
            del chosen[-len(permutation) :]

    recurse(0, [])
    assert best_order is not None
    return tuple(nodes[position] for position in best_order)


@dataclass(frozen=True, slots=True)
class TopologyPattern:
    prev_to_new: tuple[int, ...]
    new_to_prev: tuple[int, ...]
    self_loop: bool


@dataclass(frozen=True, slots=True)
class LabeledNodePattern(Generic[L]):
    node_label: Any
    repeated_from: tuple[int | None, ...]
    arity: int


def topology_patterns_for_order(graph: nx.DiGraph[N], order: tuple[N, ...]) -> tuple[TopologyPattern, ...]:
    patterns: list[TopologyPattern] = []
    for position, node in enumerate(order):
        patterns.append(
            TopologyPattern(
                prev_to_new=tuple(
                    ancestor_position
                    for ancestor_position, ancestor in enumerate(order[:position])
                    if graph.has_edge(ancestor, node)
                ),
                new_to_prev=tuple(
                    ancestor_position
                    for ancestor_position, ancestor in enumerate(order[:position])
                    if graph.has_edge(node, ancestor)
                ),
                self_loop=graph.has_edge(node, node),
            )
        )
    return tuple(patterns)


def label_patterns_for_order(
    graph: nx.DiGraph[N],
    order: tuple[N, ...],
    node_label: Callable[[dict[str, Any]], L],
    node_vars: Callable[[dict[str, Any]], tuple[V, ...]],
    include_labels: bool,
) -> tuple[LabeledNodePattern[L], ...]:
    label_patterns: list[LabeledNodePattern[L]] = []
    first_variable_position: dict[tuple[int, V], int] = {}

    for node in order:
        variables = node_vars(graph.nodes[node])
        repeated_from: list[int | None] = []
        for variable_class, identifier in enumerate(variables):
            key = (variable_class, identifier)
            first_position = first_variable_position.get(key)
            repeated_from.append(first_position)
            if first_position is None:
                first_variable_position[key] = len(label_patterns)
        label_patterns.append(
            LabeledNodePattern(
                node_label=node_label(graph.nodes[node]) if include_labels else None,
                repeated_from=tuple(repeated_from),
                arity=len(variables),
            )
        )
    return tuple(label_patterns)


def canonicalize_graph(
    graph: nx.DiGraph[N],
    node_label: Callable[[dict[str, Any]], L],
    node_vars: Callable[[dict[str, Any]], tuple[V, ...]],
) -> tuple[nx.DiGraph[int], tuple[N, ...]]:
    order = topology_order(graph)
    positions = {node: position for position, node in enumerate(order)}
    canonical = nx.DiGraph()
    canonical.add_nodes_from(range(len(order)))
    for source_position, source in enumerate(order):
        canonical.nodes[source_position]["label"] = node_label(graph.nodes[source])
        canonical.nodes[source_position]["vars"] = node_vars(graph.nodes[source])
    for source, target in graph.edges:
        canonical.add_edge(positions[source], positions[target])
    return canonical, order


def _support_matrix(neighbours: tuple[tuple[int, ...], ...], neighbour_counts: tuple[int, ...]) -> tuple[tuple[bool, ...], ...]:
    size = len(neighbours)
    sequences = tuple(
        tuple(sorted([neighbour_counts[other] if other in node_neighbours else 0 for other in range(size)], reverse=True))
        for node_neighbours in neighbours
    )
    return tuple(
        tuple(sequences[left] == sequences[right] for right in range(size))
        for left in range(size)
    )


def _refined_color_classes(
    adjacency: tuple[tuple[bool, ...], ...],
    initial_keys: tuple[Hashable, ...] | None = None,
) -> tuple[tuple[int, ...], ...]:
    size = len(adjacency)
    if initial_keys is None:
        colors = tuple(
            hash(
                (
                    sum(adjacency[position]),
                    sum(adjacency[other][position] for other in range(size)),
                    adjacency[position][position],
                )
            )
            for position in range(size)
        )
    else:
        color_ids: dict[Hashable, int] = {}
        colors = tuple(color_ids.setdefault(key, len(color_ids)) for key in initial_keys)

    while True:
        signatures = [
            (
                colors[position],
                tuple(sorted(colors[other] for other in range(size) if adjacency[position][other])),
                tuple(sorted(colors[other] for other in range(size) if adjacency[other][position])),
            )
            for position in range(size)
        ]
        signature_ids = {
            signature: index
            for index, signature in enumerate(sorted(set(signatures)))
        }
        next_colors = tuple(signature_ids[signature] for signature in signatures)
        if next_colors == colors:
            break
        colors = next_colors

    classes: defaultdict[int, list[int]] = defaultdict(list)
    for position, color in enumerate(colors):
        classes[color].append(position)
    return tuple(tuple(class_members) for _, class_members in sorted(classes.items(), key=lambda item: item[0]))


def _permutations(values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not values:
        return ((),)
    permutations: list[tuple[int, ...]] = []
    for index, value in enumerate(values):
        rest = values[:index] + values[index + 1 :]
        for suffix in _permutations(rest):
            permutations.append((value, *suffix))
    return tuple(permutations)


def _automorphisms_for_position_graph(
    adjacency: tuple[tuple[bool, ...], ...],
    neighbours: tuple[tuple[int, ...], ...],
    neighbour_counts: tuple[int, ...],
    color_classes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    size = len(adjacency)
    if size == 0:
        return ((),)

    support = _support_matrix(neighbours, neighbour_counts)
    color_by_position = [-1] * size
    for color_index, color_class in enumerate(color_classes):
        for position in color_class:
            color_by_position[position] = color_index

    mapping = [-1] * size
    reverse_mapping = [-1] * size
    automorphisms: list[tuple[int, ...]] = []

    def recurse(mapped_count: int) -> None:
        if mapped_count == size:
            automorphisms.append(tuple(mapping))
            return

        counts = [0] * size
        frontier: list[int] = []
        for source in range(size):
            if mapping[source] == -1:
                continue
            for neighbour in neighbours[source]:
                if mapping[neighbour] != -1:
                    continue
                if counts[neighbour] == 0:
                    frontier.append(neighbour)
                counts[neighbour] += 1

        if frontier:
            target_position = max(frontier, key=lambda position: (counts[position], -position))
        else:
            target_position = next(position for position, image in enumerate(mapping) if image == -1)

        image_candidates: list[int] = []
        already_seen = [False] * size
        if frontier:
            for source in range(size):
                mapped_source = mapping[source]
                if mapped_source == -1:
                    continue
                for neighbour in neighbours[mapped_source]:
                    if (
                        already_seen[neighbour]
                        or reverse_mapping[neighbour] != -1
                        or color_by_position[target_position] != color_by_position[neighbour]
                        or not support[target_position][neighbour]
                    ):
                        continue
                    image_candidates.append(neighbour)
                    already_seen[neighbour] = True
        else:
            image_candidates.extend(
                position
                for position in range(size)
                if (
                    reverse_mapping[position] == -1
                    and color_by_position[target_position] == color_by_position[position]
                    and support[target_position][position]
                )
            )

        for image_position in image_candidates:
            consistent = True
            for source in range(size):
                mapped_source = mapping[source]
                if mapped_source == -1:
                    continue
                if adjacency[target_position][source] != adjacency[image_position][mapped_source]:
                    consistent = False
                    break
                if adjacency[source][target_position] != adjacency[mapped_source][image_position]:
                    consistent = False
                    break
            if not consistent:
                continue
            mapping[target_position] = image_position
            reverse_mapping[image_position] = target_position
            recurse(mapped_count + 1)
            reverse_mapping[image_position] = -1
            mapping[target_position] = -1

    for image_position in range(size):
        if color_by_position[0] != color_by_position[image_position] or not support[0][image_position]:
            continue
        mapping[0] = image_position
        reverse_mapping[image_position] = 0
        recurse(1)
        reverse_mapping[image_position] = -1
        mapping[0] = -1

    if automorphisms:
        return tuple(automorphisms)
    return (tuple(range(size)),)


def graph_automorphisms(
    graph: nx.DiGraph[int],
    *,
    node_label: Callable[[dict[str, Any]], Any] | None = None,
    node_vars: Callable[[dict[str, Any]], tuple[Any, ...]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    order = tuple(graph.nodes)
    adjacency, neighbours, neighbour_counts = _position_graph(graph, order)
    initial_keys: tuple[Hashable, ...] | None = None
    if node_label is not None and node_vars is not None:
        initial_keys = tuple(
            (
                node_label(graph.nodes[node]),
                len(node_vars(graph.nodes[node])),
            )
            for node in order
        )
    color_classes = _refined_color_classes(adjacency, initial_keys)
    return _automorphisms_for_position_graph(adjacency, neighbours, neighbour_counts, color_classes)


def canonical_labeled_signature(
    graph: nx.DiGraph[int],
    node_label: Callable[[dict[str, Any]], L],
    node_vars: Callable[[dict[str, Any]], tuple[V, ...]],
    *,
    include_labels: bool,
) -> tuple[LabeledNodePattern[L], ...]:
    automorphisms = graph_automorphisms(graph)
    best_signature: tuple[LabeledNodePattern[L], ...] | None = None
    best_key: tuple[tuple[Any, ...], ...] | None = None

    for automorphism in automorphisms:
        order = tuple(automorphism[position] for position in range(len(automorphism)))
        signature = label_patterns_for_order(
            graph,
            order,
            node_label,
            node_vars,
            include_labels=include_labels,
        )
        signature_key = tuple(
            (
                _stable_value_key(pattern.node_label),
                tuple(-1 if position is None else position for position in pattern.repeated_from),
                pattern.arity,
            )
            for pattern in signature
        )
        if best_key is None or signature_key < best_key:
            best_key = signature_key
            best_signature = signature

    assert best_signature is not None
    return best_signature


__all__ = [
    "I",
    "L",
    "N",
    "V",
    "LabeledNodePattern",
    "TopologyPattern",
    "_default_label_matches",
    "_default_node_order_key",
    "_stable_value_key",
    "canonical_labeled_signature",
    "canonicalize_graph",
    "graph_automorphisms",
    "graph_isomorphism_mapping",
    "graphs_are_isomorphic",
    "label_patterns_for_order",
    "topologies_are_isomorphic",
    "topology_order",
    "topology_patterns_for_order",
]
