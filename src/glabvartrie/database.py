from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
import os
import time
from typing import Any, Callable, Generic, Hashable, TypeVar

import networkx as nx
try:
    from ortools.sat.python import cp_model
except Exception:
    cp_model = None
try:
    import z3
except Exception:
    z3 = None

N = TypeVar("N", bound=Hashable)
V = TypeVar("V", bound=Hashable)
VariableClass = Hashable
Label = tuple[VariableClass | None, V]
NodeMapping = dict[N, N]
VariableMapping = dict[VariableClass, dict[V, V]]


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _iter_mask_indices(mask: int) -> list[int]:
    indices: list[int] = []
    while mask:
        low_bit = mask & -mask
        indices.append(low_bit.bit_length() - 1)
        mask ^= low_bit
    return indices


class _SearchTimeout(Exception):
    pass


def _label_stats(labels: dict[N, Label[V]]) -> tuple[Counter[V], dict[VariableClass, tuple[int, ...]]]:
    constant_counts: Counter[V] = Counter()
    variable_groups: defaultdict[VariableClass, Counter[V]] = defaultdict(Counter)

    for variable_class, identifier in labels.values():
        if variable_class is None:
            constant_counts[identifier] += 1
        else:
            variable_groups[variable_class][identifier] += 1

    variable_group_sizes = {
        variable_class: tuple(sorted(identifier_counts.values(), reverse=True))
        for variable_class, identifier_counts in variable_groups.items()
    }
    return constant_counts, variable_group_sizes


def _group_sizes_fit(target_sizes: tuple[int, ...], query_sizes: tuple[int, ...]) -> bool:
    if len(target_sizes) > len(query_sizes):
        return False
    return all(target_size <= query_size for target_size, query_size in zip(target_sizes, query_sizes, strict=False))


def _position_graph(graph: nx.DiGraph[N], order: tuple[N, ...]) -> tuple[tuple[tuple[bool, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
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

    remaining_set = set(remaining)
    minimum_remaining_degree = min(
        sum(1 for neighbour in neighbours[position] if neighbour in remaining_set)
        for position in remaining_set
    )
    if minimum_remaining_degree * 2 >= len(remaining_set):
        return frozenset()

    articulation_points: set[int] = set()
    for candidate in remaining:
        start = next(iter(remaining_set - {candidate}))
        visited = {candidate, start}
        frontier: deque[int] = deque((start,))

        while frontier:
            current = frontier.popleft()
            for neighbour in neighbours[current]:
                if neighbour in visited or neighbour not in remaining_set:
                    continue
                visited.add(neighbour)
                frontier.append(neighbour)

        if len(visited) != len(remaining_set):
            articulation_points.add(candidate)

    return frozenset(articulation_points)


def _topology_order(graph: nx.DiGraph[N], labels: dict[N, Label[V]]) -> tuple[N, ...]:
    del labels

    nodes = tuple(graph.nodes)
    size = len(nodes)
    if size == 0:
        return ()

    adjacency, neighbours, _ = _position_graph(graph, nodes)
    degree = [
        sum(int(adjacency[position][other]) + int(adjacency[other][position]) for other in range(size))
        for position in range(size)
    ]
    last_degree = degree.copy()
    total_degree = degree.copy()
    used = [False] * size
    canonical_positions = [-1] * size

    for suffix_position in range(size - 1, -1, -1):
        remaining = frozenset(position for position, is_used in enumerate(used) if not is_used)
        articulation_points = _articulation_points(neighbours, remaining)

        best_position: int | None = None
        best_key: tuple[int, int, int, tuple[bool, ...], tuple[bool, ...], int] | None = None
        for position in range(size):
            if used[position] or position in articulation_points:
                continue

            candidate_key = (
                degree[position],
                last_degree[position],
                total_degree[position],
                tuple(adjacency[position][canonical_positions[index]] for index in range(suffix_position + 1, size)),
                tuple(adjacency[canonical_positions[index]][position] for index in range(suffix_position + 1, size)),
                position,
            )
            if best_position is None:
                best_position = position
                best_key = candidate_key
                continue
            if best_key is not None and candidate_key < best_key:
                best_position = position
                best_key = candidate_key

        if best_position is None:
            best_position = min(remaining)

        for position in range(size):
            last_degree[position] = degree[position]
            if adjacency[position][best_position]:
                degree[position] -= 1
            if adjacency[best_position][position]:
                degree[position] -= 1

        canonical_positions[suffix_position] = best_position
        used[best_position] = True

    return tuple(nodes[position] for position in canonical_positions)


@dataclass(frozen=True, slots=True)
class _TopologyPattern:
    prev_to_new: tuple[int, ...]
    new_to_prev: tuple[int, ...]
    self_loop: bool

    @property
    def anchor_positions(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.prev_to_new) | set(self.new_to_prev)))


@dataclass(frozen=True, slots=True)
class _LabelPattern(Generic[V]):
    kind: str
    constant_identifier: V | None = None
    variable_class: VariableClass | None = None
    repeated_from: int | None = None
    same_class_previous: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _ConditionOption:
    prefix_pairs: tuple[tuple[int, int], ...]
    current_left_positions: tuple[int, ...]


def _topology_patterns_for_order(graph: nx.DiGraph[N], order: tuple[N, ...]) -> tuple[_TopologyPattern, ...]:
    patterns: list[_TopologyPattern] = []

    for position, node in enumerate(order):
        patterns.append(
            _TopologyPattern(
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


def _label_patterns_for_order(labels: dict[N, Label[V]], order: tuple[N, ...]) -> tuple[_LabelPattern[V], ...]:
    label_patterns: list[_LabelPattern[V]] = []
    first_variable_position: dict[tuple[VariableClass, V], int] = {}
    previous_by_class: defaultdict[VariableClass, list[int]] = defaultdict(list)

    for position, node in enumerate(order):
        variable_class, identifier = labels[node]
        if variable_class is None:
            label_patterns.append(_LabelPattern(kind="const", constant_identifier=identifier))
            continue

        key = (variable_class, identifier)
        repeated_from = first_variable_position.get(key)
        if repeated_from is not None:
            label_patterns.append(
                _LabelPattern(
                    kind="var-repeat",
                    variable_class=variable_class,
                    repeated_from=repeated_from,
                )
            )
            previous_by_class[variable_class].append(position)
            continue

        first_variable_position[key] = position
        label_patterns.append(
            _LabelPattern(
                kind="var-new",
                variable_class=variable_class,
                same_class_previous=tuple(previous_by_class[variable_class]),
            )
        )
        previous_by_class[variable_class].append(position)

    return tuple(label_patterns)


def _conditions_hold(conditions: tuple[tuple[int, int], ...], matched_query_nodes: list[N], node_ranks: dict[N, int]) -> bool:
    depth = len(matched_query_nodes)
    for left_position, right_position in conditions:
        if right_position >= depth:
            continue
        if node_ranks[matched_query_nodes[left_position]] >= node_ranks[matched_query_nodes[right_position]]:
            return False
    return True


def _option_for_depth(conditions: tuple[tuple[int, int], ...], current_position: int) -> _ConditionOption:
    return _ConditionOption(
        prefix_pairs=tuple((left, right) for left, right in conditions if right < current_position),
        current_left_positions=tuple(sorted(left for left, right in conditions if right == current_position)),
    )


def _option_prefix_holds(option: _ConditionOption, matched_query_nodes: list[N], node_ranks: dict[N, int]) -> bool:
    for left_position, right_position in option.prefix_pairs:
        if node_ranks[matched_query_nodes[left_position]] >= node_ranks[matched_query_nodes[right_position]]:
            return False
    return True


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


def _refined_color_classes(adjacency: tuple[tuple[bool, ...], ...]) -> tuple[tuple[int, ...], ...]:
    size = len(adjacency)
    colors: tuple[int, ...] = tuple(
        hash(
            (
                sum(adjacency[position]),
                sum(adjacency[other][position] for other in range(size)),
                adjacency[position][position],
            )
        )
        for position in range(size)
    )

    while True:
        signatures = [
            (
                colors[position],
                tuple(sorted(colors[other] for other in range(size) if adjacency[position][other])),
                tuple(sorted(colors[other] for other in range(size) if adjacency[other][position])),
            )
            for position in range(size)
        ]
        signature_ids: dict[tuple[Any, ...], int] = {}
        next_colors_list: list[int] = []
        for signature in signatures:
            next_colors_list.append(signature_ids.setdefault(signature, len(signature_ids)))
        next_colors = tuple(next_colors_list)
        if next_colors == colors:
            break
        colors = next_colors

    classes: defaultdict[int, list[int]] = defaultdict(list)
    for position, color in enumerate(colors):
        classes[color].append(position)
    return tuple(tuple(class_members) for _, class_members in sorted(classes.items(), key=lambda item: item[0]))


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


def _reduce_conditions(conditions: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    edges = sorted(set(conditions))
    if len(edges) < 2:
        return tuple(edges)

    reduced: list[tuple[int, int]] = []
    for edge in edges:
        start, end = edge
        adjacency: defaultdict[int, set[int]] = defaultdict(set)
        for current in edges:
            if current == edge:
                continue
            adjacency[current[0]].add(current[1])

        frontier = [start]
        visited = {start}
        reachable = False
        while frontier and not reachable:
            current = frontier.pop()
            for neighbour in adjacency[current]:
                if neighbour == end:
                    reachable = True
                    break
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                frontier.append(neighbour)

        if not reachable:
            reduced.append(edge)

    return tuple(reduced)


def _symmetry_conditions(graph: nx.DiGraph[N], order: tuple[N, ...]) -> tuple[tuple[int, int], ...]:
    adjacency, neighbours, neighbour_counts = _position_graph(graph, order)
    color_classes = _refined_color_classes(adjacency)
    if all(len(color_class) == 1 for color_class in color_classes):
        return ()

    ambiguous_nodes = sum(len(color_class) for color_class in color_classes if len(color_class) > 1)
    if len(order) > 18 or ambiguous_nodes > 12:
        return ()

    automorphisms = _automorphisms_for_position_graph(adjacency, neighbours, neighbour_counts, color_classes)
    if len(automorphisms) <= 1:
        return ()

    conditions: list[tuple[int, int]] = []
    broken = [False] * len(automorphisms)
    size = len(order)

    for position in range(size):
        if any(not broken[index] and automorphism[position] != position for index, automorphism in enumerate(automorphisms)):
            for other_position in range(position + 1, size):
                if any(not broken[index] and automorphism[position] == other_position for index, automorphism in enumerate(automorphisms)):
                    conditions.append((position, other_position))

        for index, automorphism in enumerate(automorphisms):
            if automorphism[position] != position:
                broken[index] = True

    return _reduce_conditions(tuple(conditions))


@dataclass(slots=True)
class _StoredGraph(Generic[N, V]):
    graph: nx.DiGraph[N]
    labels: dict[N, Label[V]]
    constant_counts: Counter[V]
    variable_group_sizes: dict[VariableClass, tuple[int, ...]]
    in_degrees: dict[N, int]
    out_degrees: dict[N, int]
    predecessors: dict[N, frozenset[N]]
    successors: dict[N, frozenset[N]]
    neighbours: dict[N, frozenset[N]]
    order: tuple[N, ...]
    order_positions: dict[N, int]
    variable_key_nodes: dict[tuple[VariableClass, V], tuple[N, ...]]
    variable_class_nodes: dict[VariableClass, tuple[N, ...]]
    topology_patterns: tuple[_TopologyPattern, ...]
    label_patterns: tuple[_LabelPattern[V], ...]
    condition_options: tuple[_ConditionOption, ...]
    full_conditions: tuple[tuple[int, int], ...]
    future_out_positions: tuple[tuple[int, ...], ...]
    future_in_positions: tuple[tuple[int, ...], ...]


@dataclass(slots=True)
class _TrieNode:
    depth: int
    topology_pattern: _TopologyPattern | None
    children: dict[_TopologyPattern, _TrieNode] = field(default_factory=dict)
    terminal_graph_indices: list[int] = field(default_factory=list)
    descendant_graph_indices: set[int] = field(default_factory=set)
    label_pattern_graph_indices: dict[_LabelPattern[Any], set[int]] = field(default_factory=dict)
    condition_options: set[_ConditionOption] = field(default_factory=set)
    full_conditions: tuple[tuple[int, int], ...] | None = None


@dataclass(slots=True)
class _QueryData(Generic[N, V]):
    graph: nx.DiGraph[N]
    labels: dict[N, Label[V]]
    constant_nodes: dict[V, frozenset[N]]
    constant_masks: dict[V, int]
    variable_nodes: dict[VariableClass, frozenset[N]]
    variable_masks: dict[VariableClass, int]
    variable_identifier_nodes: dict[tuple[VariableClass, V], frozenset[N]]
    variable_identifier_masks: dict[tuple[VariableClass, V], int]
    predecessors: dict[N, frozenset[N]]
    predecessor_masks: tuple[int, ...]
    successors: dict[N, frozenset[N]]
    successor_masks: tuple[int, ...]
    constant_counts: Counter[V]
    variable_group_sizes: dict[VariableClass, tuple[int, ...]]
    in_degrees: dict[N, int]
    out_degrees: dict[N, int]
    node_ranks: dict[N, int]
    node_indices: dict[N, int]
    index_to_node: tuple[N, ...]
    all_nodes_mask: int


@dataclass(slots=True)
class _DynamicSearchState(Generic[N, V]):
    target_to_query: dict[N, N]
    used_query_nodes: set[N]
    target_var_to_query_identifier: dict[VariableClass, dict[V, V]]
    used_query_identifiers: dict[VariableClass, set[V]]


def _build_query_data(graph: nx.DiGraph[N], labels: dict[N, Label[V]]) -> _QueryData[N, V]:
    constant_nodes_mut: defaultdict[V, set[N]] = defaultdict(set)
    variable_nodes_mut: defaultdict[VariableClass, set[N]] = defaultdict(set)
    variable_identifier_nodes_mut: defaultdict[tuple[VariableClass, V], set[N]] = defaultdict(set)
    index_to_node = tuple(graph.nodes)
    node_indices = {node: index for index, node in enumerate(index_to_node)}
    all_nodes_mask = (1 << len(index_to_node)) - 1

    for node, (variable_class, identifier) in labels.items():
        if variable_class is None:
            constant_nodes_mut[identifier].add(node)
        else:
            variable_nodes_mut[variable_class].add(node)
            variable_identifier_nodes_mut[(variable_class, identifier)].add(node)

    def mask_for_nodes(nodes: set[N]) -> int:
        mask = 0
        for node in nodes:
            mask |= 1 << node_indices[node]
        return mask

    constant_counts, variable_group_sizes = _label_stats(labels)
    return _QueryData(
        graph=graph,
        labels=labels,
        constant_nodes={identifier: frozenset(nodes) for identifier, nodes in constant_nodes_mut.items()},
        constant_masks={identifier: mask_for_nodes(nodes) for identifier, nodes in constant_nodes_mut.items()},
        variable_nodes={variable_class: frozenset(nodes) for variable_class, nodes in variable_nodes_mut.items()},
        variable_masks={variable_class: mask_for_nodes(nodes) for variable_class, nodes in variable_nodes_mut.items()},
        variable_identifier_nodes={
            key: frozenset(nodes)
            for key, nodes in variable_identifier_nodes_mut.items()
        },
        variable_identifier_masks={
            key: mask_for_nodes(nodes)
            for key, nodes in variable_identifier_nodes_mut.items()
        },
        predecessors={node: frozenset(graph.pred[node]) for node in graph.nodes},
        predecessor_masks=tuple(
            sum(1 << node_indices[predecessor] for predecessor in graph.pred[node])
            for node in index_to_node
        ),
        successors={node: frozenset(graph.succ[node]) for node in graph.nodes},
        successor_masks=tuple(
            sum(1 << node_indices[successor] for successor in graph.succ[node])
            for node in index_to_node
        ),
        constant_counts=constant_counts,
        variable_group_sizes=variable_group_sizes,
        in_degrees=dict(graph.in_degree()),
        out_degrees=dict(graph.out_degree()),
        node_ranks={node: rank for rank, node in enumerate(sorted(graph.nodes, key=repr))},
        node_indices=node_indices,
        index_to_node=index_to_node,
        all_nodes_mask=all_nodes_mask,
    )


class Database(Generic[N, V]):
    def __init__(self, node_label: Callable[[dict[str, Any]], Label[V]]):
        self._node_label = node_label
        self._graphs: list[_StoredGraph[N, V]] = []
        self._root = _TrieNode(depth=0, topology_pattern=None)
        self._z3_enabled = _env_enabled("GLABVARTRIE_ENABLE_Z3", True) and z3 is not None

    def index(self, g: nx.DiGraph[N]) -> None:
        labels = {node: self._node_label(g.nodes[node]) for node in g.nodes}
        order = _topology_order(g, labels)
        topology_patterns = _topology_patterns_for_order(g, order)
        label_patterns = _label_patterns_for_order(labels, order)
        conditions = _symmetry_conditions(g, order)
        constant_counts, variable_group_sizes = _label_stats(labels)
        order_positions = {node: position for position, node in enumerate(order)}
        variable_key_nodes_mut: defaultdict[tuple[VariableClass, V], list[N]] = defaultdict(list)
        variable_class_nodes_mut: defaultdict[VariableClass, list[N]] = defaultdict(list)
        for node in order:
            variable_class, identifier = labels[node]
            if variable_class is None:
                continue
            variable_key_nodes_mut[(variable_class, identifier)].append(node)
            variable_class_nodes_mut[variable_class].append(node)

        graph_index = len(self._graphs)
        future_out_positions_mut: list[list[int]] = [[] for _ in order]
        future_in_positions_mut: list[list[int]] = [[] for _ in order]
        for future_position, topology_pattern in enumerate(topology_patterns):
            for previous_position in topology_pattern.prev_to_new:
                future_out_positions_mut[previous_position].append(future_position)
            for previous_position in topology_pattern.new_to_prev:
                future_in_positions_mut[previous_position].append(future_position)
        self._graphs.append(
            _StoredGraph(
                graph=g,
                labels=labels,
                constant_counts=constant_counts,
                variable_group_sizes=variable_group_sizes,
                in_degrees=dict(g.in_degree()),
                out_degrees=dict(g.out_degree()),
                predecessors={node: frozenset(g.pred[node]) for node in g.nodes},
                successors={node: frozenset(g.succ[node]) for node in g.nodes},
                neighbours={
                    node: frozenset(set(g.pred[node]) | set(g.succ[node]))
                    for node in g.nodes
                },
                order=order,
                order_positions=order_positions,
                variable_key_nodes={
                    key: tuple(nodes)
                    for key, nodes in variable_key_nodes_mut.items()
                },
                variable_class_nodes={
                    variable_class: tuple(nodes)
                    for variable_class, nodes in variable_class_nodes_mut.items()
                },
                topology_patterns=topology_patterns,
                label_patterns=label_patterns,
                condition_options=tuple(_option_for_depth(conditions, position) for position in range(len(order))),
                full_conditions=conditions,
                future_out_positions=tuple(tuple(positions) for positions in future_out_positions_mut),
                future_in_positions=tuple(tuple(positions) for positions in future_in_positions_mut),
            )
        )

        node = self._root
        node.descendant_graph_indices.add(graph_index)
        for depth, topology_pattern in enumerate(topology_patterns, start=1):
            child = node.children.get(topology_pattern)
            if child is None:
                child = _TrieNode(depth=depth, topology_pattern=topology_pattern)
                node.children[topology_pattern] = child
            child.descendant_graph_indices.add(graph_index)
            child.label_pattern_graph_indices.setdefault(label_patterns[depth - 1], set()).add(graph_index)
            child.condition_options.add(_option_for_depth(conditions, depth - 1))
            node = child

        node.terminal_graph_indices.append(graph_index)
        if node.full_conditions is None:
            node.full_conditions = conditions

    def update(self, g: nx.DiGraph[N]) -> None:
        self.index(g)

    def query(self, g: nx.DiGraph[N]) -> list[tuple[nx.DiGraph[N], NodeMapping[N], VariableMapping[V]]]:
        query_labels = {node: self._node_label(g.nodes[node]) for node in g.nodes}
        query_data = _build_query_data(g, query_labels)

        eligible = {
            index
            for index, stored in enumerate(self._graphs)
            if self._can_match(stored, query_data)
        }
        if not eligible:
            return []

        if len(eligible) <= 16:
            return self._query_direct(query_data, eligible)

        matches: list[tuple[nx.DiGraph[N], NodeMapping[N], VariableMapping[V]]] = []
        used_query_nodes: set[N] = set()
        matched_query_nodes: list[N] = []

        for child in sorted(self._root.children.values(), key=self._node_priority, reverse=True):
            if child.descendant_graph_indices.isdisjoint(eligible):
                continue
            self._search(child, query_data, eligible, matched_query_nodes, used_query_nodes, matches)

        return matches

    def _query_direct(
        self,
        query_data: _QueryData[N, V],
        eligible: set[int],
    ) -> list[tuple[nx.DiGraph[N], NodeMapping[N], VariableMapping[V]]]:
        matches: list[tuple[nx.DiGraph[N], NodeMapping[N], VariableMapping[V]]] = []
        remaining = set(eligible)

        graph_order = sorted(
            remaining,
            key=lambda graph_index: (
                self._direct_graph_priority(self._graphs[graph_index], query_data),
                graph_index,
            ),
        )
        for graph_index in graph_order:
            if graph_index not in remaining:
                continue
            self._collect_single_graph_matches(
                graph_index,
                query_data,
                remaining,
                matches,
                1,
                use_lookahead=True,
            )

        return matches

    def _direct_graph_priority(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
    ) -> tuple[int, int, str]:
        root_target = min(
            stored.order,
            key=lambda target_node: (
                len(
                    self._dynamic_label_candidates(
                        stored,
                        target_node,
                        query_data,
                        {},
                        {},
                    )
                ),
                -(stored.in_degrees[target_node] + stored.out_degrees[target_node]),
                repr(target_node),
            ),
        )
        return (
            len(self._dynamic_label_candidates(stored, root_target, query_data, {}, {})),
            -(stored.in_degrees[root_target] + stored.out_degrees[root_target]),
            repr(root_target),
        )

    def _collect_single_graph_matches(
        self,
        graph_index: int,
        query_data: _QueryData[N, V],
        eligible: set[int],
        matches: list[tuple[nx.DiGraph[N], NodeMapping[N], VariableMapping[V]]],
        match_limit: int,
        use_lookahead: bool,
    ) -> int:
        if graph_index not in eligible or match_limit <= 0:
            return 0

        stored = self._graphs[graph_index]
        local_matches: list[tuple[NodeMapping[N], VariableMapping[V]]] = []
        if match_limit > 1:
            self._enumerate_single_graph_best_first(
                stored,
                query_data,
                local_matches,
                match_limit,
                use_lookahead,
            )
        else:
            match: tuple[NodeMapping[N], VariableMapping[V]] | None
            try:
                deadline = (
                    time.monotonic() + 1.0
                    if self._z3_enabled and stored.graph.number_of_nodes() >= 50
                    else None
                )
                match = self._find_single_graph_match(
                    stored,
                    query_data,
                    {},
                    set(),
                    {},
                    {},
                    deadline=deadline,
                )
            except _SearchTimeout:
                match = self._timeout_fallback_match(stored, query_data)
            if match is not None:
                local_matches.append(match)
        for node_mapping, variable_mapping in local_matches:
            matches.append((stored.graph, node_mapping, variable_mapping))

        if local_matches:
            eligible.discard(graph_index)
        return len(local_matches)

    def _copy_variable_state(
        self,
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> tuple[dict[VariableClass, dict[V, V]], dict[VariableClass, set[V]]]:
        return (
            {
                variable_class: dict(identifier_mapping)
                for variable_class, identifier_mapping in target_var_to_query_identifier.items()
            },
            {
                variable_class: set(identifiers)
                for variable_class, identifiers in used_query_identifiers.items()
            },
        )

    def _timeout_fallback_match(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
    ) -> tuple[NodeMapping[N], VariableMapping[V]] | None:
        if stored.graph.number_of_nodes() < 60:
            return None
        if self._direct_graph_priority(stored, query_data)[0] > 16:
            return None
        match = self._find_single_graph_match_ortools(stored, query_data)
        if match is not None:
            return match
        if self._z3_enabled and cp_model is None:
            return self._find_single_graph_match_z3(stored, query_data)
        return None

    def _enumerate_single_graph_best_first(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        results: list[tuple[NodeMapping[N], VariableMapping[V]]],
        match_limit: int,
        use_lookahead: bool,
    ) -> None:
        queue: list[tuple[int, int, int, _DynamicSearchState[N, V]]] = []
        serial = 0
        heappush(
            queue,
            (
                0,
                0,
                serial,
                _DynamicSearchState(
                    target_to_query={},
                    used_query_nodes=set(),
                    target_var_to_query_identifier={},
                    used_query_identifiers={},
                ),
            ),
        )

        while queue and len(results) < match_limit:
            penalty, negative_depth, _, state = heappop(queue)
            del negative_depth

            if len(state.target_to_query) == len(stored.order):
                ordered_query_nodes = [state.target_to_query[target_node] for target_node in stored.order]
                if _conditions_hold(stored.full_conditions, ordered_query_nodes, query_data.node_ranks):
                    results.append(
                        (
                            state.target_to_query.copy(),
                            self._variable_mapping(stored, query_data, ordered_query_nodes),
                        )
                    )
                continue

            best_target, ordered_candidates = self._dynamic_choice(
                stored,
                state.target_to_query,
                query_data,
                state.used_query_nodes,
                state.target_var_to_query_identifier,
                state.used_query_identifiers,
                use_lookahead,
            )
            if best_target is None or not ordered_candidates:
                continue

            target_class, target_identifier = stored.labels[best_target]
            for penalty_delta, query_node in enumerate(ordered_candidates):
                new_target_to_query = dict(state.target_to_query)
                new_target_to_query[best_target] = query_node
                new_used_query_nodes = set(state.used_query_nodes)
                new_used_query_nodes.add(query_node)
                new_target_var_to_query_identifier, new_used_query_identifiers = self._copy_variable_state(
                    state.target_var_to_query_identifier,
                    state.used_query_identifiers,
                )

                if target_class is not None:
                    query_identifier = query_data.labels[query_node][1]
                    variable_mapping = new_target_var_to_query_identifier.setdefault(target_class, {})
                    if target_identifier not in variable_mapping:
                        variable_mapping[target_identifier] = query_identifier

                serial += 1
                heappush(
                    queue,
                    (
                        penalty + penalty_delta,
                        -len(new_target_to_query),
                        serial,
                        _DynamicSearchState(
                            target_to_query=new_target_to_query,
                            used_query_nodes=new_used_query_nodes,
                            target_var_to_query_identifier=new_target_var_to_query_identifier,
                            used_query_identifiers=new_used_query_identifiers,
                        ),
                    ),
                )

    def _search(
        self,
        node: _TrieNode,
        query_data: _QueryData[N, V],
        eligible: set[int],
        matched_query_nodes: list[N],
        used_query_nodes: set[N],
        matches: list[tuple[nx.DiGraph[N], NodeMapping[N], VariableMapping[V]]],
        candidate_plan: list[tuple[N, frozenset[int]]] | None = None,
    ) -> None:
        active_graphs = node.descendant_graph_indices & eligible
        if not active_graphs or node.topology_pattern is None:
            return

        if len(active_graphs) == 1:
            graph_index = next(iter(active_graphs))
            self._search_single_graph(
                graph_index,
                node.depth - 1,
                query_data,
                eligible,
                matched_query_nodes,
                used_query_nodes,
                matches,
                candidate_plan,
            )
            return

        candidates = candidate_plan
        if candidates is None:
            candidates = self._candidate_nodes(
                node,
                query_data,
                active_graphs,
                matched_query_nodes,
                used_query_nodes,
            )

        for candidate, next_active_graphs in candidates:
            used_query_nodes.add(candidate)
            matched_query_nodes.append(candidate)

            if next_active_graphs:
                if node.terminal_graph_indices and node.full_conditions is not None and _conditions_hold(node.full_conditions, matched_query_nodes, query_data.node_ranks):
                    for graph_index in node.terminal_graph_indices:
                        if graph_index not in next_active_graphs:
                            continue
                        stored = self._graphs[graph_index]
                        target_to_query = {
                            target_node: matched_query_nodes[position]
                            for position, target_node in enumerate(stored.order)
                        }
                        matches.append((stored.graph, target_to_query, self._variable_mapping(stored, query_data, matched_query_nodes)))
                        eligible.discard(graph_index)

                remaining_active = set(next_active_graphs & eligible)
                child_plans: list[tuple[int, int, _TrieNode, list[tuple[N, frozenset[int]]]]] = []
                for child in node.children.values():
                    if child.descendant_graph_indices.isdisjoint(remaining_active):
                        continue
                    child_candidates = self._candidate_nodes(
                        child,
                        query_data,
                        remaining_active,
                        matched_query_nodes,
                        used_query_nodes,
                    )
                    if not child_candidates:
                        continue
                    child_plans.append(
                        (
                            len(child_candidates),
                            -max(len(graph_indices) for _, graph_indices in child_candidates),
                            child,
                            child_candidates,
                        )
                    )

                child_plans.sort(key=lambda item: (item[0], item[1], self._node_priority(item[2])), reverse=False)
                for _, _, child, child_candidates in child_plans:
                    self._search(
                        child,
                        query_data,
                        remaining_active,
                        matched_query_nodes,
                        used_query_nodes,
                        matches,
                        child_candidates,
                    )

            matched_query_nodes.pop()
            used_query_nodes.remove(candidate)

    def _search_single_graph(
        self,
        graph_index: int,
        position: int,
        query_data: _QueryData[N, V],
        eligible: set[int],
        matched_query_nodes: list[N],
        used_query_nodes: set[N],
        matches: list[tuple[nx.DiGraph[N], NodeMapping[N], VariableMapping[V]]],
        candidate_plan: list[tuple[N, frozenset[int]]] | None = None,
    ) -> bool:
        del position
        del candidate_plan

        if graph_index not in eligible:
            return False

        stored = self._graphs[graph_index]
        target_to_query = {
            target_node: matched_query_nodes[current_position]
            for current_position, target_node in enumerate(stored.order[: len(matched_query_nodes)])
        }
        target_var_to_query_identifier, used_query_identifiers = self._variable_state(stored, target_to_query, query_data)
        try:
            match = self._find_single_graph_match(
                stored,
                query_data,
                target_to_query,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
                deadline=(
                    time.monotonic() + 1.0
                    if self._z3_enabled and stored.graph.number_of_nodes() >= 50
                    else None
                ),
            )
        except _SearchTimeout:
            match = self._timeout_fallback_match(stored, query_data)
        if match is not None:
            node_mapping, variable_mapping = match
            matches.append((stored.graph, node_mapping, variable_mapping))
            eligible.discard(graph_index)
            return True

        return False

    def _variable_state(
        self,
        stored: _StoredGraph[N, V],
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
    ) -> tuple[dict[VariableClass, dict[V, V]], dict[VariableClass, set[V]]]:
        target_var_to_query_identifier: defaultdict[VariableClass, dict[V, V]] = defaultdict(dict)
        used_query_identifiers: defaultdict[VariableClass, set[V]] = defaultdict(set)

        for target_node, query_node in target_to_query.items():
            target_class, target_identifier = stored.labels[target_node]
            if target_class is None:
                continue

            query_class, query_identifier = query_data.labels[query_node]
            if query_class != target_class:
                raise ValueError("Matched variable classes disagree")

            existing = target_var_to_query_identifier[target_class].get(target_identifier)
            if existing is not None and existing != query_identifier:
                raise ValueError("Matched target variable maps to multiple query identifiers")
            target_var_to_query_identifier[target_class][target_identifier] = query_identifier

        return dict(target_var_to_query_identifier), {}

    def _partial_conditions_hold(
        self,
        stored: _StoredGraph[N, V],
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
    ) -> bool:
        for left_position, right_position in stored.full_conditions:
            left_target = stored.order[left_position]
            right_target = stored.order[right_position]
            if left_target not in target_to_query or right_target not in target_to_query:
                continue
            if query_data.node_ranks[target_to_query[left_target]] >= query_data.node_ranks[target_to_query[right_target]]:
                return False
        return True

    def _initial_single_graph_domains(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        target_to_query: dict[N, N],
        used_query_nodes: set[N],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> dict[N, frozenset[N]] | None:
        domains: dict[N, frozenset[N]] = {}

        for target_node in stored.order:
            assigned_query = target_to_query.get(target_node)
            target_in_degree = stored.in_degrees[target_node]
            target_out_degree = stored.out_degrees[target_node]
            requires_self_loop = target_node in stored.successors[target_node]

            if assigned_query is not None:
                if assigned_query in used_query_nodes - {assigned_query}:
                    return None
                if query_data.in_degrees[assigned_query] < target_in_degree or query_data.out_degrees[assigned_query] < target_out_degree:
                    return None
                if requires_self_loop and not query_data.graph.has_edge(assigned_query, assigned_query):
                    return None
                if not self._dynamic_label_compatible(
                    stored,
                    target_node,
                    assigned_query,
                    query_data,
                    target_var_to_query_identifier,
                    used_query_identifiers,
                ):
                    return None
                domains[target_node] = frozenset((assigned_query,))
                continue

            label_candidates = self._dynamic_label_candidates(
                stored,
                target_node,
                query_data,
                target_var_to_query_identifier,
                used_query_identifiers,
            )
            if not label_candidates:
                return None

            accepted = frozenset(
                query_node
                for query_node in label_candidates
                if query_node not in used_query_nodes
                and query_data.in_degrees[query_node] >= target_in_degree
                and query_data.out_degrees[query_node] >= target_out_degree
                and (not requires_self_loop or query_data.graph.has_edge(query_node, query_node))
                and all(
                    query_node in query_data.successors[target_to_query[predecessor]]
                    for predecessor in stored.predecessors[target_node]
                    if predecessor in target_to_query
                )
                and all(
                    query_node in query_data.predecessors[target_to_query[successor]]
                    for successor in stored.successors[target_node]
                    if successor in target_to_query
                )
            )
            if not accepted:
                return None
            domains[target_node] = accepted

        return domains

    def _target_candidate_supported(
        self,
        stored: _StoredGraph[N, V],
        target_node: N,
        query_node: N,
        domains: dict[N, frozenset[N]],
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
    ) -> bool:
        if target_node in stored.successors[target_node] and not query_data.graph.has_edge(query_node, query_node):
            return False

        for predecessor in stored.predecessors[target_node]:
            assigned_predecessor = target_to_query.get(predecessor)
            if assigned_predecessor is not None:
                if query_node not in query_data.successors[assigned_predecessor]:
                    return False
            else:
                predecessor_domain = domains[predecessor]
                if not any(query_node in query_data.successors[candidate] for candidate in predecessor_domain):
                    return False

        for successor in stored.successors[target_node]:
            assigned_successor = target_to_query.get(successor)
            if assigned_successor is not None:
                if query_node not in query_data.predecessors[assigned_successor]:
                    return False
            else:
                successor_domain = domains[successor]
                if not any(query_node in query_data.predecessors[candidate] for candidate in successor_domain):
                    return False

        return True

    def _refine_single_graph_domains(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        target_to_query: dict[N, N],
        domains: dict[N, frozenset[N]],
    ) -> dict[N, frozenset[N]] | None:
        refined = dict(domains)

        changed = True
        while changed:
            changed = False
            for target_node in stored.order:
                current_domain = refined[target_node]
                filtered = frozenset(
                    query_node
                    for query_node in current_domain
                    if self._target_candidate_supported(
                        stored,
                        target_node,
                        query_node,
                        refined,
                        target_to_query,
                        query_data,
                    )
                )
                if not filtered:
                    return None
                if filtered != current_domain:
                    refined[target_node] = filtered
                    changed = True

        return refined

    def _candidate_order_key(
        self,
        stored: _StoredGraph[N, V],
        target_node: N,
        query_node: N,
        domains: dict[N, frozenset[N]],
        query_data: _QueryData[N, V],
    ) -> tuple[int, int, str]:
        impact = 0
        for successor in stored.successors[target_node]:
            if successor in domains and len(domains[successor]) > 1:
                impact += len(domains[successor] & query_data.successors[query_node])
        for predecessor in stored.predecessors[target_node]:
            if predecessor in domains and len(domains[predecessor]) > 1:
                impact += len(domains[predecessor] & query_data.predecessors[query_node])
        return (
            impact,
            -(query_data.in_degrees[query_node] + query_data.out_degrees[query_node]),
            repr(query_node),
        )

    def _propagate_assignment_domains(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        target_to_query: dict[N, N],
        domains: dict[N, frozenset[N]],
        assigned_target: N,
        assigned_query: N,
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> tuple[dict[N, frozenset[N]], dict[VariableClass, dict[V, V]], dict[VariableClass, set[V]]] | None:
        next_domains = dict(domains)
        next_domains[assigned_target] = frozenset((assigned_query,))

        for other_target, other_domain in tuple(next_domains.items()):
            if other_target == assigned_target:
                continue
            if assigned_query not in other_domain:
                continue
            reduced = other_domain - {assigned_query}
            if not reduced:
                return None
            next_domains[other_target] = reduced

        next_target_var_to_query_identifier, next_used_query_identifiers = self._copy_variable_state(
            target_var_to_query_identifier,
            used_query_identifiers,
        )

        target_class, target_identifier = stored.labels[assigned_target]
        if target_class is not None:
            query_identifier = query_data.labels[assigned_query][1]
            variable_mapping = next_target_var_to_query_identifier.setdefault(target_class, {})
            existing_query_identifier = variable_mapping.get(target_identifier)
            if existing_query_identifier is not None:
                if existing_query_identifier != query_identifier:
                    return None
            else:
                variable_mapping[target_identifier] = query_identifier

            same_identifier_domain = query_data.variable_identifier_nodes.get((target_class, query_identifier), frozenset())
            for variable_target in stored.variable_class_nodes.get(target_class, ()):
                if variable_target == assigned_target:
                    continue
                current_domain = next_domains[variable_target]
                variable_target_identifier = stored.labels[variable_target][1]
                if variable_target_identifier == target_identifier:
                    reduced = current_domain & same_identifier_domain
                else:
                    reduced = current_domain
                if not reduced:
                    return None
                next_domains[variable_target] = reduced

        for successor in stored.successors[assigned_target]:
            if successor == assigned_target:
                continue
            current_domain = next_domains[successor]
            reduced = current_domain & query_data.successors[assigned_query]
            if not reduced:
                return None
            next_domains[successor] = reduced
        for predecessor in stored.predecessors[assigned_target]:
            if predecessor == assigned_target:
                continue
            current_domain = next_domains[predecessor]
            reduced = current_domain & query_data.predecessors[assigned_query]
            if not reduced:
                return None
            next_domains[predecessor] = reduced

        refined = self._refine_single_graph_domains(
            stored,
            query_data,
            target_to_query,
            next_domains,
        )
        if refined is None:
            return None

        return refined, next_target_var_to_query_identifier, next_used_query_identifiers

    def _bitmask_label_candidates(
        self,
        stored: _StoredGraph[N, V],
        target_node: N,
        query_data: _QueryData[N, V],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> int:
        target_class, target_identifier = stored.labels[target_node]
        if target_class is None:
            return query_data.constant_masks.get(target_identifier, 0)

        existing = target_var_to_query_identifier.get(target_class, {}).get(target_identifier)
        if existing is not None:
            return query_data.variable_identifier_masks.get((target_class, existing), 0)

        return query_data.variable_masks.get(target_class, 0)

    def _initial_single_graph_masks(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        target_to_query: dict[N, N],
        used_query_nodes: set[N],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> dict[N, int] | None:
        used_mask = 0
        for query_node in used_query_nodes:
            used_mask |= 1 << query_data.node_indices[query_node]

        masks: dict[N, int] = {}
        for target_node in stored.order:
            target_in_degree = stored.in_degrees[target_node]
            target_out_degree = stored.out_degrees[target_node]
            requires_self_loop = target_node in stored.successors[target_node]
            assigned_query = target_to_query.get(target_node)

            if assigned_query is not None:
                if not self._dynamic_label_compatible(
                    stored,
                    target_node,
                    assigned_query,
                    query_data,
                    target_var_to_query_identifier,
                    used_query_identifiers,
                ):
                    return None
                if query_data.in_degrees[assigned_query] < target_in_degree or query_data.out_degrees[assigned_query] < target_out_degree:
                    return None
                if requires_self_loop and not query_data.graph.has_edge(assigned_query, assigned_query):
                    return None
                masks[target_node] = 1 << query_data.node_indices[assigned_query]
                continue

            mask = self._bitmask_label_candidates(
                stored,
                target_node,
                query_data,
                target_var_to_query_identifier,
                used_query_identifiers,
            )
            if not mask:
                return None
            mask &= query_data.all_nodes_mask ^ used_mask

            for predecessor in stored.predecessors[target_node]:
                assigned_predecessor = target_to_query.get(predecessor)
                if assigned_predecessor is None:
                    continue
                mask &= query_data.successor_masks[query_data.node_indices[assigned_predecessor]]
            for successor in stored.successors[target_node]:
                assigned_successor = target_to_query.get(successor)
                if assigned_successor is None:
                    continue
                mask &= query_data.predecessor_masks[query_data.node_indices[assigned_successor]]

            filtered_mask = 0
            for query_index in _iter_mask_indices(mask):
                query_node = query_data.index_to_node[query_index]
                if query_data.in_degrees[query_node] < target_in_degree or query_data.out_degrees[query_node] < target_out_degree:
                    continue
                if requires_self_loop and not query_data.graph.has_edge(query_node, query_node):
                    continue
                filtered_mask |= 1 << query_index

            if not filtered_mask:
                return None
            masks[target_node] = filtered_mask

        return masks

    def _mask_candidate_supported(
        self,
        stored: _StoredGraph[N, V],
        target_node: N,
        query_index: int,
        masks: dict[N, int],
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
    ) -> bool:
        query_node = query_data.index_to_node[query_index]
        if target_node in stored.successors[target_node] and not query_data.graph.has_edge(query_node, query_node):
            return False

        for predecessor in stored.predecessors[target_node]:
            assigned_predecessor = target_to_query.get(predecessor)
            if assigned_predecessor is not None:
                predecessor_index = query_data.node_indices[assigned_predecessor]
                if not (query_data.successor_masks[predecessor_index] & (1 << query_index)):
                    return False
            elif not (query_data.predecessor_masks[query_index] & masks[predecessor]):
                return False

        for successor in stored.successors[target_node]:
            assigned_successor = target_to_query.get(successor)
            if assigned_successor is not None:
                successor_index = query_data.node_indices[assigned_successor]
                if not (query_data.predecessor_masks[successor_index] & (1 << query_index)):
                    return False
            elif not (query_data.successor_masks[query_index] & masks[successor]):
                return False

        return True

    def _refine_single_graph_masks(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        target_to_query: dict[N, N],
        masks: dict[N, int],
        deadline: float | None = None,
    ) -> dict[N, int] | None:
        refined = dict(masks)

        changed = True
        while changed:
            if deadline is not None and time.monotonic() >= deadline:
                raise _SearchTimeout
            changed = False
            for target_node in stored.order:
                current_mask = refined[target_node]
                filtered_mask = 0
                for query_index in _iter_mask_indices(current_mask):
                    if self._mask_candidate_supported(
                        stored,
                        target_node,
                        query_index,
                        refined,
                        target_to_query,
                        query_data,
                    ):
                        filtered_mask |= 1 << query_index
                if not filtered_mask:
                    return None
                if filtered_mask != current_mask:
                    refined[target_node] = filtered_mask
                    changed = True

        return refined

    def _mask_candidate_order_key(
        self,
        stored: _StoredGraph[N, V],
        target_node: N,
        query_index: int,
        masks: dict[N, int],
        query_data: _QueryData[N, V],
    ) -> tuple[int, int, str]:
        impact = 0
        successor_mask = query_data.successor_masks[query_index]
        predecessor_mask = query_data.predecessor_masks[query_index]
        for successor in stored.successors[target_node]:
            if successor in masks and (masks[successor] & (masks[successor] - 1)):
                impact += (masks[successor] & successor_mask).bit_count()
        for predecessor in stored.predecessors[target_node]:
            if predecessor in masks and (masks[predecessor] & (masks[predecessor] - 1)):
                impact += (masks[predecessor] & predecessor_mask).bit_count()
        query_node = query_data.index_to_node[query_index]
        return (
            impact,
            -(query_data.in_degrees[query_node] + query_data.out_degrees[query_node]),
            repr(query_node),
        )

    def _propagate_assignment_masks(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        target_to_query: dict[N, N],
        masks: dict[N, int],
        assigned_target: N,
        assigned_query_index: int,
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
        deadline: float | None = None,
    ) -> tuple[dict[N, int], dict[VariableClass, dict[V, V]], dict[VariableClass, set[V]]] | None:
        assigned_bit = 1 << assigned_query_index
        next_masks = {target_node: (assigned_bit if target_node == assigned_target else mask & (query_data.all_nodes_mask ^ assigned_bit)) for target_node, mask in masks.items()}
        if any(mask == 0 for mask in next_masks.values()):
            return None

        next_target_var_to_query_identifier, next_used_query_identifiers = self._copy_variable_state(
            target_var_to_query_identifier,
            used_query_identifiers,
        )

        target_class, target_identifier = stored.labels[assigned_target]
        if target_class is not None:
            query_identifier = query_data.labels[query_data.index_to_node[assigned_query_index]][1]
            variable_mapping = next_target_var_to_query_identifier.setdefault(target_class, {})
            existing_query_identifier = variable_mapping.get(target_identifier)
            if existing_query_identifier is not None:
                if existing_query_identifier != query_identifier:
                    return None
            else:
                variable_mapping[target_identifier] = query_identifier

            same_identifier_mask = query_data.variable_identifier_masks.get((target_class, query_identifier), 0)
            for variable_target in stored.variable_class_nodes.get(target_class, ()):
                if variable_target == assigned_target:
                    continue
                current_mask = next_masks[variable_target]
                variable_target_identifier = stored.labels[variable_target][1]
                if variable_target_identifier == target_identifier:
                    reduced_mask = current_mask & same_identifier_mask
                else:
                    reduced_mask = current_mask
                if not reduced_mask:
                    return None
                next_masks[variable_target] = reduced_mask

        successor_mask = query_data.successor_masks[assigned_query_index]
        predecessor_mask = query_data.predecessor_masks[assigned_query_index]
        for successor in stored.successors[assigned_target]:
            if successor == assigned_target:
                continue
            reduced_mask = next_masks[successor] & successor_mask
            if not reduced_mask:
                return None
            next_masks[successor] = reduced_mask
        for predecessor in stored.predecessors[assigned_target]:
            if predecessor == assigned_target:
                continue
            reduced_mask = next_masks[predecessor] & predecessor_mask
            if not reduced_mask:
                return None
            next_masks[predecessor] = reduced_mask

        refined = self._refine_single_graph_masks(
            stored,
            query_data,
            target_to_query,
            next_masks,
            deadline,
        )
        if refined is None:
            return None

        return refined, next_target_var_to_query_identifier, next_used_query_identifiers

    def _find_single_graph_match(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        target_to_query: dict[N, N],
        used_query_nodes: set[N],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
        deadline: float | None = None,
    ) -> tuple[NodeMapping[N], VariableMapping[V]] | None:
        if not self._partial_conditions_hold(stored, target_to_query, query_data):
            return None

        if deadline is not None and time.monotonic() >= deadline:
            raise _SearchTimeout
        masks = self._initial_single_graph_masks(
            stored,
            query_data,
            target_to_query,
            used_query_nodes,
            target_var_to_query_identifier,
            used_query_identifiers,
        )
        if masks is None:
            return None

        refined = self._refine_single_graph_masks(
            stored,
            query_data,
            target_to_query,
            masks,
            deadline,
        )
        if refined is None:
            return None

        variable_state = self._copy_variable_state(target_var_to_query_identifier, used_query_identifiers)
        return self._search_single_graph_masks(
            stored,
            query_data,
            dict(target_to_query),
            dict(refined),
            variable_state[0],
            variable_state[1],
            deadline,
        )

    def _find_single_graph_match_z3(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
    ) -> tuple[NodeMapping[N], VariableMapping[V]] | None:
        if not self._z3_enabled or z3 is None:
            return None

        order = list(stored.order)
        query_nodes = list(query_data.graph.nodes)
        query_position = {node: position for position, node in enumerate(query_nodes)}
        target_position = {node: position for position, node in enumerate(order)}

        candidates = [
            [query_position[query_node] for query_node in self._dynamic_candidates_for_target(stored, target_node, {}, query_data, set(), {}, {})]
            for target_node in order
        ]
        if any(not target_candidates for target_candidates in candidates):
            return None

        solver = z3.Solver()
        solver.set(timeout=10_000)

        assignment_variables: dict[tuple[int, int], Any] = {
            (target_index, query_index): z3.Bool(f"x_{target_index}_{query_index}")
            for target_index, target_candidates in enumerate(candidates)
            for query_index in target_candidates
        }

        for target_index, target_candidates in enumerate(candidates):
            solver.add(z3.PbEq([(assignment_variables[(target_index, query_index)], 1) for query_index in target_candidates], 1))

        for query_index in range(len(query_nodes)):
            uses = [
                assignment_variables[(target_index, query_index)]
                for target_index, target_candidates in enumerate(candidates)
                if query_index in target_candidates
            ]
            if uses:
                solver.add(z3.PbLe([(query_use, 1) for query_use in uses], 1))

        query_successors = [set(query_data.graph.succ[node]) for node in query_nodes]
        for source, target in stored.graph.edges:
            source_index = target_position[source]
            target_index = target_position[target]
            for source_query_index in candidates[source_index]:
                invalid_targets = [
                    target_query_index
                    for target_query_index in candidates[target_index]
                    if query_nodes[target_query_index] not in query_successors[source_query_index]
                ]
                for target_query_index in invalid_targets:
                    solver.add(
                        z3.Or(
                            z3.Not(assignment_variables[(source_index, source_query_index)]),
                            z3.Not(assignment_variables[(target_index, target_query_index)]),
                        )
                    )

        variables_by_class: dict[VariableClass, dict[V, list[int]]] = {}
        for target_index, target_node in enumerate(order):
            variable_class, identifier = stored.labels[target_node]
            if variable_class is None:
                continue
            variables_by_class.setdefault(variable_class, {}).setdefault(identifier, []).append(target_index)

        for variable_class, target_groups in variables_by_class.items():
            query_identifiers = sorted(
                {
                    query_data.labels[query_nodes[query_index]][1]
                    for target_indices in target_groups.values()
                    for target_index in target_indices
                    for query_index in candidates[target_index]
                    if query_data.labels[query_nodes[query_index]][0] == variable_class
                },
                key=repr,
            )
            identifier_variables = {
                (identifier, query_identifier): z3.Bool(
                    f"y_{repr(variable_class)}_{repr(identifier)}_{repr(query_identifier)}"
                )
                for identifier in target_groups
                for query_identifier in query_identifiers
            }
            for identifier in target_groups:
                solver.add(
                    z3.PbEq(
                        [
                            (identifier_variables[(identifier, query_identifier)], 1)
                            for query_identifier in query_identifiers
                        ],
                        1,
                    )
                )
            for identifier, target_indices in target_groups.items():
                for target_index in target_indices:
                    for query_index in candidates[target_index]:
                        query_identifier = query_data.labels[query_nodes[query_index]][1]
                        solver.add(
                            z3.Implies(
                                assignment_variables[(target_index, query_index)],
                                identifier_variables[(identifier, query_identifier)],
                            )
                        )

        if solver.check() != z3.sat:
            return None
        model = solver.model()

        node_mapping = {
            target_node: query_nodes[query_index]
            for target_index, target_node in enumerate(order)
            for query_index in candidates[target_index]
            if z3.is_true(model.evaluate(assignment_variables[(target_index, query_index)], model_completion=True))
        }
        if len(node_mapping) != len(order):
            return None

        ordered_query_nodes = [node_mapping[target_node] for target_node in order]
        return node_mapping, self._variable_mapping(stored, query_data, ordered_query_nodes)

    def _find_single_graph_match_ortools(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
    ) -> tuple[NodeMapping[N], VariableMapping[V]] | None:
        if cp_model is None:
            return None
        cp: Any = cp_model

        order = list(stored.order)
        query_nodes = list(query_data.graph.nodes)
        query_position = {node: position for position, node in enumerate(query_nodes)}
        target_position = {node: position for position, node in enumerate(order)}

        candidates = [
            [query_position[query_node] for query_node in self._dynamic_candidates_for_target(stored, target_node, {}, query_data, set(), {}, {})]
            for target_node in order
        ]
        if any(not target_candidates for target_candidates in candidates):
            return None

        model: Any = cp.CpModel()
        assignment_variables: dict[tuple[int, int], Any] = {
            (target_index, query_index): model.NewBoolVar(f"x_{target_index}_{query_index}")
            for target_index, target_candidates in enumerate(candidates)
            for query_index in target_candidates
        }

        for target_index, target_candidates in enumerate(candidates):
            model.AddExactlyOne(assignment_variables[(target_index, query_index)] for query_index in target_candidates)

        uses_by_query: defaultdict[int, list[Any]] = defaultdict(list)
        for (target_index, query_index), variable in assignment_variables.items():
            del target_index
            uses_by_query[query_index].append(variable)
        for uses in uses_by_query.values():
            model.AddAtMostOne(uses)

        query_successors = [set(query_data.graph.succ[node]) for node in query_nodes]
        for source, target in stored.graph.edges:
            source_index = target_position[source]
            target_index = target_position[target]
            for source_query_index in candidates[source_index]:
                for target_query_index in candidates[target_index]:
                    if query_nodes[target_query_index] in query_successors[source_query_index]:
                        continue
                    model.AddBoolOr(
                        [
                            assignment_variables[(source_index, source_query_index)].Not(),
                            assignment_variables[(target_index, target_query_index)].Not(),
                        ]
                    )

        variables_by_class: dict[VariableClass, dict[V, list[int]]] = {}
        for target_index, target_node in enumerate(order):
            variable_class, identifier = stored.labels[target_node]
            if variable_class is None:
                continue
            variables_by_class.setdefault(variable_class, {}).setdefault(identifier, []).append(target_index)

        for variable_class, target_groups in variables_by_class.items():
            query_identifiers = sorted(
                {
                    query_data.labels[query_nodes[query_index]][1]
                    for target_indices in target_groups.values()
                    for target_index in target_indices
                    for query_index in candidates[target_index]
                    if query_data.labels[query_nodes[query_index]][0] == variable_class
                },
                key=repr,
            )
            identifier_variables = {
                (identifier, query_identifier): model.NewBoolVar(
                    f"y_{repr(variable_class)}_{repr(identifier)}_{repr(query_identifier)}"
                )
                for identifier in target_groups
                for query_identifier in query_identifiers
            }
            for identifier in target_groups:
                model.AddExactlyOne(identifier_variables[(identifier, query_identifier)] for query_identifier in query_identifiers)
            for identifier, target_indices in target_groups.items():
                for target_index in target_indices:
                    for query_index in candidates[target_index]:
                        query_identifier = query_data.labels[query_nodes[query_index]][1]
                        model.AddImplication(
                            assignment_variables[(target_index, query_index)],
                            identifier_variables[(identifier, query_identifier)],
                        )

        solver: Any = cp.CpSolver()
        solver.parameters.max_time_in_seconds = 3.0
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status not in (cp.OPTIMAL, cp.FEASIBLE):
            return None

        node_mapping = {
            target_node: query_nodes[query_index]
            for target_index, target_node in enumerate(order)
            for query_index in candidates[target_index]
            if solver.BooleanValue(assignment_variables[(target_index, query_index)])
        }
        if len(node_mapping) != len(order):
            return None

        ordered_query_nodes = [node_mapping[target_node] for target_node in order]
        return node_mapping, self._variable_mapping(stored, query_data, ordered_query_nodes)

    def _search_single_graph_masks(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        target_to_query: dict[N, N],
        masks: dict[N, int],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
        deadline: float | None = None,
    ) -> tuple[NodeMapping[N], VariableMapping[V]] | None:
        if deadline is not None and time.monotonic() >= deadline:
            raise _SearchTimeout
        if len(target_to_query) == len(stored.order):
            ordered_query_nodes = [target_to_query[target_node] for target_node in stored.order]
            if not _conditions_hold(stored.full_conditions, ordered_query_nodes, query_data.node_ranks):
                return None
            return target_to_query.copy(), self._variable_mapping(stored, query_data, ordered_query_nodes)

        frontier_targets = [
            target_node
            for target_node in stored.order
            if target_node not in target_to_query
            and any(neighbour in target_to_query for neighbour in stored.neighbours[target_node])
        ]
        candidate_targets = frontier_targets or [target_node for target_node in stored.order if target_node not in target_to_query]
        target_node = min(
            candidate_targets,
            key=lambda current_target: (
                masks[current_target].bit_count(),
                -(stored.in_degrees[current_target] + stored.out_degrees[current_target]),
                repr(current_target),
            ),
        )

        ordered_candidates = sorted(
            _iter_mask_indices(masks[target_node]),
            key=lambda query_index: self._mask_candidate_order_key(stored, target_node, query_index, masks, query_data),
        )

        for query_index in ordered_candidates:
            query_node = query_data.index_to_node[query_index]
            next_target_to_query = dict(target_to_query)
            next_target_to_query[target_node] = query_node
            if not self._partial_conditions_hold(stored, next_target_to_query, query_data):
                continue

            propagated = self._propagate_assignment_masks(
                stored,
                query_data,
                next_target_to_query,
                masks,
                target_node,
                query_index,
                target_var_to_query_identifier,
                used_query_identifiers,
                deadline,
            )
            if propagated is None:
                continue

            next_masks, next_target_var_to_query_identifier, next_used_query_identifiers = propagated
            match = self._search_single_graph_masks(
                stored,
                query_data,
                next_target_to_query,
                next_masks,
                next_target_var_to_query_identifier,
                next_used_query_identifiers,
                deadline,
            )
            if match is not None:
                return match

        return None

    def _dynamic_label_candidates(
        self,
        stored: _StoredGraph[N, V],
        target_node: N,
        query_data: _QueryData[N, V],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> frozenset[N]:
        target_class, target_identifier = stored.labels[target_node]
        if target_class is None:
            return query_data.constant_nodes.get(target_identifier, frozenset())

        existing = target_var_to_query_identifier.get(target_class, {}).get(target_identifier)
        if existing is not None:
            return query_data.variable_identifier_nodes.get((target_class, existing), frozenset())

        return query_data.variable_nodes.get(target_class, frozenset())

    def _dynamic_label_compatible(
        self,
        stored: _StoredGraph[N, V],
        target_node: N,
        query_node: N,
        query_data: _QueryData[N, V],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> bool:
        target_class, target_identifier = stored.labels[target_node]
        query_class, query_identifier = query_data.labels[query_node]

        if target_class is None:
            return query_class is None and query_identifier == target_identifier

        if query_class != target_class:
            return False

        existing = target_var_to_query_identifier.get(target_class, {}).get(target_identifier)
        if existing is not None:
            return query_identifier == existing

        return True

    def _dynamic_candidates_for_target(
        self,
        stored: _StoredGraph[N, V],
        target_node: N,
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> list[N]:
        label_candidates = self._dynamic_label_candidates(
            stored,
            target_node,
            query_data,
            target_var_to_query_identifier,
            used_query_identifiers,
        )
        if not label_candidates:
            return []

        candidate_source: frozenset[N] = label_candidates

        predecessor_anchors = [target_to_query[predecessor] for predecessor in stored.predecessors[target_node] if predecessor in target_to_query]
        successor_anchors = [target_to_query[successor] for successor in stored.successors[target_node] if successor in target_to_query]

        for predecessor_query in predecessor_anchors:
            candidate_source = candidate_source & query_data.successors[predecessor_query]
        for successor_query in successor_anchors:
            candidate_source = candidate_source & query_data.predecessors[successor_query]

        accepted: list[N] = []
        target_in_degree = stored.in_degrees[target_node]
        target_out_degree = stored.out_degrees[target_node]
        for query_node in candidate_source:
            if query_node in used_query_nodes:
                continue
            if query_data.in_degrees[query_node] < target_in_degree or query_data.out_degrees[query_node] < target_out_degree:
                continue
            if not self._dynamic_label_compatible(
                stored,
                target_node,
                query_node,
                query_data,
                target_var_to_query_identifier,
                used_query_identifiers,
            ):
                continue
            if target_node in stored.successors[target_node] and not query_data.graph.has_edge(query_node, query_node):
                continue
            accepted.append(query_node)

        accepted.sort(key=lambda node: repr(node))
        return accepted

    def _dynamic_choice(
        self,
        stored: _StoredGraph[N, V],
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
        use_lookahead: bool,
    ) -> tuple[N | None, list[N]]:
        unmatched_targets = [target_node for target_node in stored.order if target_node not in target_to_query]
        frontier_targets = [
            target_node
            for target_node in unmatched_targets
            if any(
                neighbor in target_to_query
                for neighbor in stored.neighbours[target_node]
            )
        ]
        candidate_targets = frontier_targets or unmatched_targets

        best_target: N | None = None
        best_candidates: list[N] | None = None
        best_key: tuple[int, int, str] | None = None

        for target_node in candidate_targets:
            candidates = self._dynamic_candidates_for_target(
                stored,
                target_node,
                target_to_query,
                query_data,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
            )
            if not candidates:
                return None, []

            key = (
                len(candidates),
                -(stored.in_degrees[target_node] + stored.out_degrees[target_node]),
                repr(target_node),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_target = target_node
                best_candidates = candidates

        if best_target is None or best_candidates is None:
            return None, []

        ordered_candidates = best_candidates
        if use_lookahead and 1 < len(best_candidates) <= 16:
            candidate_scores: list[tuple[tuple[float, float, str], N]] = []
            target_class, target_identifier = stored.labels[best_target]
            for query_node in best_candidates:
                target_to_query[best_target] = query_node
                used_query_nodes.add(query_node)

                added_variable_mapping = False
                query_identifier: V | None = None
                if target_class is not None:
                    query_identifier = query_data.labels[query_node][1]
                    variable_mapping = target_var_to_query_identifier.setdefault(target_class, {})
                    if target_identifier not in variable_mapping:
                        variable_mapping[target_identifier] = query_identifier
                        added_variable_mapping = True

                next_key = self._dynamic_next_choice_key(
                    stored,
                    target_to_query,
                    query_data,
                    used_query_nodes,
                    target_var_to_query_identifier,
                    used_query_identifiers,
                )
                candidate_scores.append((next_key, query_node))

                if target_class is not None and added_variable_mapping and query_identifier is not None:
                    del target_var_to_query_identifier[target_class][target_identifier]
                    if not target_var_to_query_identifier[target_class]:
                        del target_var_to_query_identifier[target_class]

                used_query_nodes.remove(query_node)
                del target_to_query[best_target]

            candidate_scores.sort(key=lambda item: (item[0], repr(item[1])))
            ordered_candidates = [query_node for _, query_node in candidate_scores]

        return best_target, ordered_candidates

    def _complete_single_graph_dynamic(
        self,
        stored: _StoredGraph[N, V],
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> bool:
        if len(target_to_query) == len(stored.order):
            return True

        best_target, ordered_candidates = self._dynamic_choice(
            stored,
            target_to_query,
            query_data,
            used_query_nodes,
            target_var_to_query_identifier,
            used_query_identifiers,
            True,
        )
        if best_target is None or not ordered_candidates:
            return False

        target_class, target_identifier = stored.labels[best_target]
        for query_node in ordered_candidates:
            target_to_query[best_target] = query_node
            used_query_nodes.add(query_node)

            added_variable_mapping = False
            query_identifier: V | None = None
            if target_class is not None:
                query_identifier = query_data.labels[query_node][1]
                variable_mapping = target_var_to_query_identifier.setdefault(target_class, {})
                if target_identifier not in variable_mapping:
                    variable_mapping[target_identifier] = query_identifier
                    added_variable_mapping = True

            if self._complete_single_graph_dynamic(
                stored,
                target_to_query,
                query_data,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
            ):
                return True

            if target_class is not None and added_variable_mapping and query_identifier is not None:
                del target_var_to_query_identifier[target_class][target_identifier]
                if not target_var_to_query_identifier[target_class]:
                    del target_var_to_query_identifier[target_class]

            used_query_nodes.remove(query_node)
            del target_to_query[best_target]

        return False

    def _enumerate_single_graph_dynamic(
        self,
        stored: _StoredGraph[N, V],
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
        results: list[tuple[NodeMapping[N], VariableMapping[V]]],
        match_limit: int,
        use_lookahead: bool,
    ) -> bool:
        if len(results) >= match_limit:
            return True

        if len(target_to_query) == len(stored.order):
            ordered_query_nodes = [target_to_query[target_node] for target_node in stored.order]
            if _conditions_hold(stored.full_conditions, ordered_query_nodes, query_data.node_ranks):
                results.append((target_to_query.copy(), self._variable_mapping(stored, query_data, ordered_query_nodes)))
            return len(results) >= match_limit

        best_target, ordered_candidates = self._dynamic_choice(
            stored,
            target_to_query,
            query_data,
            used_query_nodes,
            target_var_to_query_identifier,
            used_query_identifiers,
            use_lookahead,
        )
        if best_target is None or not ordered_candidates:
            return False

        target_class, target_identifier = stored.labels[best_target]
        for query_node in ordered_candidates:
            target_to_query[best_target] = query_node
            used_query_nodes.add(query_node)

            added_variable_mapping = False
            query_identifier: V | None = None
            if target_class is not None:
                query_identifier = query_data.labels[query_node][1]
                variable_mapping = target_var_to_query_identifier.setdefault(target_class, {})
                if target_identifier not in variable_mapping:
                    variable_mapping[target_identifier] = query_identifier
                    added_variable_mapping = True

            stop = self._enumerate_single_graph_dynamic(
                stored,
                target_to_query,
                query_data,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
                results,
                match_limit,
                use_lookahead,
            )

            if target_class is not None and added_variable_mapping and query_identifier is not None:
                del target_var_to_query_identifier[target_class][target_identifier]
                if not target_var_to_query_identifier[target_class]:
                    del target_var_to_query_identifier[target_class]

            used_query_nodes.remove(query_node)
            del target_to_query[best_target]

            if stop:
                return True

        return False

    def _dynamic_next_choice_key(
        self,
        stored: _StoredGraph[N, V],
        target_to_query: dict[N, N],
        query_data: _QueryData[N, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: dict[VariableClass, dict[V, V]],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> tuple[float, float, str]:
        if len(target_to_query) == len(stored.order):
            return (-1.0, 0.0, "")

        unmatched_targets = [target_node for target_node in stored.order if target_node not in target_to_query]
        frontier_targets = [
            target_node
            for target_node in unmatched_targets
            if any(
                neighbor in target_to_query
                for neighbor in stored.neighbours[target_node]
            )
        ]
        candidate_targets = frontier_targets or unmatched_targets

        best_key: tuple[float, float, str] | None = None
        for target_node in candidate_targets:
            candidates = self._dynamic_candidates_for_target(
                stored,
                target_node,
                target_to_query,
                query_data,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
            )
            if not candidates:
                return (float("inf"), float("inf"), repr(target_node))

            key = (
                float(len(candidates)),
                float(-(stored.in_degrees[target_node] + stored.out_degrees[target_node])),
                repr(target_node),
            )
            if best_key is None or key < best_key:
                best_key = key

        return best_key or (-1.0, 0.0, "")

    def _single_graph_candidates(
        self,
        graph_index: int,
        stored: _StoredGraph[N, V],
        position: int,
        query_data: _QueryData[N, V],
        matched_query_nodes: list[N],
        used_query_nodes: set[N],
    ) -> list[tuple[N, frozenset[int]]]:
        topology_pattern = stored.topology_patterns[position]
        condition_option = stored.condition_options[position]
        target_node = stored.order[position]
        target_in_degree = stored.in_degrees[target_node]
        target_out_degree = stored.out_degrees[target_node]

        if not _option_prefix_holds(condition_option, matched_query_nodes, query_data.node_ranks):
            return []

        labelmin_rank = (
            max(query_data.node_ranks[matched_query_nodes[current_position]] for current_position in condition_option.current_left_positions)
            if condition_option.current_left_positions else -1
        )

        label_candidate_source = self._label_candidate_nodes(stored.label_patterns[position], matched_query_nodes, query_data)
        if not label_candidate_source:
            return []

        if topology_pattern.anchor_positions:
            anchor_position = min(
                topology_pattern.anchor_positions,
                key=lambda current_position: self._anchor_size(topology_pattern, current_position, matched_query_nodes, query_data),
            )
            candidate_source = self._anchor_candidates(topology_pattern, anchor_position, matched_query_nodes, query_data)
            candidate_source = candidate_source & label_candidate_source
        else:
            candidate_source = label_candidate_source

        accepted: list[tuple[N, frozenset[int]]] = []
        for candidate in candidate_source:
            if query_data.node_ranks[candidate] <= labelmin_rank:
                continue
            if candidate in used_query_nodes:
                continue
            if query_data.in_degrees[candidate] < target_in_degree or query_data.out_degrees[candidate] < target_out_degree:
                continue
            if topology_pattern.self_loop and not query_data.graph.has_edge(candidate, candidate):
                continue
            if any(candidate not in query_data.successors[matched_query_nodes[current_position]] for current_position in topology_pattern.prev_to_new):
                continue
            if any(candidate not in query_data.predecessors[matched_query_nodes[current_position]] for current_position in topology_pattern.new_to_prev):
                continue
            if not self._label_pattern_compatible(stored.label_patterns[position], matched_query_nodes, query_data, candidate):
                continue
            if not self._future_neighbor_feasible(stored, position, candidate, query_data, matched_query_nodes, used_query_nodes):
                continue
            accepted.append((candidate, frozenset({graph_index})))

        accepted.sort(key=lambda item: repr(item[0]))
        return accepted

    def _candidate_nodes(
        self,
        node: _TrieNode,
        query_data: _QueryData[N, V],
        active_graphs: set[int],
        matched_query_nodes: list[N],
        used_query_nodes: set[N],
    ) -> list[tuple[N, frozenset[int]]]:
        topology_pattern = node.topology_pattern
        if topology_pattern is None:
            return []

        active_label_patterns_list: list[tuple[_LabelPattern[Any], set[int]]] = []
        for label_pattern, graph_indices in node.label_pattern_graph_indices.items():
            active_label_graphs = graph_indices & active_graphs
            if active_label_graphs:
                active_label_patterns_list.append((label_pattern, active_label_graphs))
        active_label_patterns = tuple(active_label_patterns_list)
        if not active_label_patterns:
            return []

        respected_options = [
            option
            for option in node.condition_options
            if _option_prefix_holds(option, matched_query_nodes, query_data.node_ranks)
        ]
        if not respected_options:
            return []

        labelmin_rank = min(
            max(query_data.node_ranks[matched_query_nodes[position]] for position in option.current_left_positions)
            if option.current_left_positions else -1
            for option in respected_options
        )

        label_candidates: set[N] = set()
        for label_pattern, _ in active_label_patterns:
            label_candidates.update(self._label_candidate_nodes(label_pattern, matched_query_nodes, query_data))
        if not label_candidates:
            return []

        label_candidate_source = frozenset(label_candidates)
        if topology_pattern.anchor_positions:
            anchor_position = min(
                topology_pattern.anchor_positions,
                key=lambda position: self._anchor_size(topology_pattern, position, matched_query_nodes, query_data),
            )
            candidate_source = self._anchor_candidates(topology_pattern, anchor_position, matched_query_nodes, query_data)
            candidate_source = candidate_source & label_candidate_source
        else:
            candidate_source = label_candidate_source

        accepted: list[tuple[N, frozenset[int]]] = []
        for candidate in candidate_source:
            if query_data.node_ranks[candidate] <= labelmin_rank:
                continue
            if candidate in used_query_nodes:
                continue
            if topology_pattern.self_loop and not query_data.graph.has_edge(candidate, candidate):
                continue
            if any(candidate not in query_data.successors[matched_query_nodes[position]] for position in topology_pattern.prev_to_new):
                continue
            if any(candidate not in query_data.predecessors[matched_query_nodes[position]] for position in topology_pattern.new_to_prev):
                continue
            matching_graphs = frozenset(
                graph_index
                for label_pattern, graph_indices in active_label_patterns
                if self._label_pattern_compatible(label_pattern, matched_query_nodes, query_data, candidate)
                for graph_index in graph_indices
            )
            if not matching_graphs:
                continue
            accepted.append((candidate, matching_graphs))

        accepted.sort(key=lambda item: (-len(item[1]), repr(item[0])))
        return accepted

    def _anchor_candidates(
        self,
        topology_pattern: _TopologyPattern,
        anchor_position: int,
        matched_query_nodes: list[N],
        query_data: _QueryData[N, V],
    ) -> frozenset[N]:
        anchored: frozenset[N] | None = None

        if anchor_position in topology_pattern.prev_to_new:
            anchored = query_data.successors[matched_query_nodes[anchor_position]]
        if anchor_position in topology_pattern.new_to_prev:
            predecessors = query_data.predecessors[matched_query_nodes[anchor_position]]
            anchored = predecessors if anchored is None else anchored & predecessors

        return anchored or frozenset()

    def _anchor_size(
        self,
        topology_pattern: _TopologyPattern,
        anchor_position: int,
        matched_query_nodes: list[N],
        query_data: _QueryData[N, V],
    ) -> int:
        size: int | None = None

        if anchor_position in topology_pattern.prev_to_new:
            size = len(query_data.successors[matched_query_nodes[anchor_position]])
        if anchor_position in topology_pattern.new_to_prev:
            predecessor_size = len(query_data.predecessors[matched_query_nodes[anchor_position]])
            size = predecessor_size if size is None else min(size, predecessor_size)

        return size if size is not None else 0

    def _label_candidate_nodes(
        self,
        label_pattern: _LabelPattern[V],
        matched_query_nodes: list[N],
        query_data: _QueryData[N, V],
    ) -> frozenset[N]:
        if label_pattern.kind == "const":
            if label_pattern.constant_identifier is None:
                return frozenset()
            return query_data.constant_nodes.get(label_pattern.constant_identifier, frozenset())

        if label_pattern.variable_class is None:
            return frozenset()

        if label_pattern.kind == "var-repeat":
            if label_pattern.repeated_from is None:
                return frozenset()
            repeated_query_node = matched_query_nodes[label_pattern.repeated_from]
            _, repeated_identifier = query_data.labels[repeated_query_node]
            return query_data.variable_identifier_nodes.get(
                (label_pattern.variable_class, repeated_identifier),
                frozenset(),
            )

        return query_data.variable_nodes.get(label_pattern.variable_class, frozenset())

    def _label_pattern_compatible(
        self,
        label_pattern: _LabelPattern[V],
        matched_query_nodes: list[N],
        query_data: _QueryData[N, V],
        candidate: N,
    ) -> bool:
        query_class, query_identifier = query_data.labels[candidate]

        if label_pattern.kind == "const":
            return query_class is None and query_identifier == label_pattern.constant_identifier

        if query_class != label_pattern.variable_class:
            return False

        if label_pattern.kind == "var-repeat":
            if label_pattern.repeated_from is None:
                return False
            _, repeated_identifier = query_data.labels[matched_query_nodes[label_pattern.repeated_from]]
            return query_identifier == repeated_identifier

        return all(
            query_identifier != query_data.labels[matched_query_nodes[previous_position]][1]
            for previous_position in label_pattern.same_class_previous
        )

    def _future_requirement_key(
        self,
        label_pattern: _LabelPattern[V],
        prefix_query_nodes: list[N],
        query_data: _QueryData[N, V],
    ) -> tuple[Any, ...]:
        if label_pattern.kind == "const":
            return ("const", label_pattern.constant_identifier)

        if label_pattern.variable_class is None:
            return ("invalid",)

        if label_pattern.kind == "var-repeat":
            if label_pattern.repeated_from is not None and label_pattern.repeated_from < len(prefix_query_nodes):
                _, identifier = query_data.labels[prefix_query_nodes[label_pattern.repeated_from]]
                return ("var-id", label_pattern.variable_class, identifier)
            return ("var-class", label_pattern.variable_class)

        forbidden_identifiers = tuple(
            sorted(
                (
                    query_data.labels[prefix_query_nodes[previous_position]][1]
                    for previous_position in label_pattern.same_class_previous
                    if previous_position < len(prefix_query_nodes)
                ),
                key=repr,
            )
        )
        return ("var-new", label_pattern.variable_class, forbidden_identifiers)

    def _neighbor_matches_requirement(
        self,
        requirement: tuple[Any, ...],
        query_node: N,
        query_data: _QueryData[N, V],
    ) -> bool:
        query_class, query_identifier = query_data.labels[query_node]
        kind = requirement[0]

        if kind == "const":
            return query_class is None and query_identifier == requirement[1]
        if kind == "var-id":
            return query_class == requirement[1] and query_identifier == requirement[2]
        if kind == "var-class":
            return query_class == requirement[1]
        if kind == "var-new":
            return query_class == requirement[1] and query_identifier not in requirement[2]
        return False

    def _future_neighbor_feasible(
        self,
        stored: _StoredGraph[N, V],
        position: int,
        candidate: N,
        query_data: _QueryData[N, V],
        matched_query_nodes: list[N],
        used_query_nodes: set[N],
    ) -> bool:
        prefix_query_nodes = [*matched_query_nodes, candidate]
        used_or_current = used_query_nodes | {candidate}

        future_out = set(stored.future_out_positions[position])
        future_in = set(stored.future_in_positions[position])
        future_both = future_out & future_in

        requirements: Counter[tuple[Any, ...]] = Counter()

        for future_position in future_both:
            requirements[("both",) + self._future_requirement_key(stored.label_patterns[future_position], prefix_query_nodes, query_data)] += 1
        for future_position in future_out - future_both:
            requirements[("out",) + self._future_requirement_key(stored.label_patterns[future_position], prefix_query_nodes, query_data)] += 1
        for future_position in future_in - future_both:
            requirements[("in",) + self._future_requirement_key(stored.label_patterns[future_position], prefix_query_nodes, query_data)] += 1

        if not requirements:
            return True

        successors = query_data.successors[candidate]
        predecessors = query_data.predecessors[candidate]
        both_directions = successors & predecessors

        for requirement, count in requirements.items():
            direction = requirement[0]
            label_requirement = requirement[1:]
            if direction == "both":
                candidate_nodes = both_directions
            elif direction == "out":
                candidate_nodes = successors
            else:
                candidate_nodes = predecessors

            available = sum(
                1
                for query_node in candidate_nodes
                if query_node not in used_or_current
                and self._neighbor_matches_requirement(label_requirement, query_node, query_data)
            )
            if available < count:
                return False

        return True

    def _node_priority(self, node: _TrieNode) -> tuple[int, int, int]:
        topology_pattern = node.topology_pattern
        if topology_pattern is None:
            return (0, 0, 0)
        return (
            len(topology_pattern.anchor_positions),
            int(topology_pattern.self_loop),
            -len(node.descendant_graph_indices),
        )

    def _can_match(self, stored: _StoredGraph[N, V], query_data: _QueryData[N, V]) -> bool:
        if stored.graph.number_of_nodes() > query_data.graph.number_of_nodes():
            return False
        if stored.graph.number_of_edges() > query_data.graph.number_of_edges():
            return False

        for constant_identifier, count in stored.constant_counts.items():
            if count > query_data.constant_counts[constant_identifier]:
                return False

        for variable_class, target_sizes in stored.variable_group_sizes.items():
            query_sizes = query_data.variable_group_sizes.get(variable_class, ())
            if not _group_sizes_fit(target_sizes, query_sizes):
                return False

        for target_node, target_label in stored.labels.items():
            target_in_degree = stored.in_degrees[target_node]
            target_out_degree = stored.out_degrees[target_node]
            if target_label[0] is None:
                candidates = query_data.constant_nodes.get(target_label[1], frozenset())
            else:
                candidates = query_data.variable_nodes.get(target_label[0], frozenset())

            if not any(
                query_data.in_degrees[query_node] >= target_in_degree
                and query_data.out_degrees[query_node] >= target_out_degree
                for query_node in candidates
            ):
                return False

        return True

    def _variable_mapping(
        self,
        stored: _StoredGraph[N, V],
        query_data: _QueryData[N, V],
        matched_query_nodes: list[N],
    ) -> VariableMapping[V]:
        variable_mapping: defaultdict[VariableClass, dict[V, V]] = defaultdict(dict)

        for target_node, query_node in zip(stored.order, matched_query_nodes, strict=True):
            target_class, target_identifier = stored.labels[target_node]
            if target_class is None:
                continue

            query_class, query_identifier = query_data.labels[query_node]
            if query_class != target_class:
                raise ValueError("Matched variable classes disagree")

            existing = variable_mapping[target_class].get(target_identifier)
            if existing is not None and existing != query_identifier:
                raise ValueError("Matched target variable maps to multiple query identifiers")
            variable_mapping[target_class][target_identifier] = query_identifier

        return dict(variable_mapping)
