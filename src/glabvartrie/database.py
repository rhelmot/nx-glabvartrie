from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass, field
from heapq import heappop, heappush
import math
import os
from typing import Any, Callable, Generic, Hashable, Iterator, TypeAlias, TypeVar

import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx.algorithms.isomorphism import DiGraphMatcher
try:
    from ortools.sat.python import cp_model
except Exception:
    cp_model = None
try:
    import z3
except Exception:
    z3 = None

N = TypeVar("N", bound=Hashable)
L = TypeVar("L", bound=Hashable)
V = TypeVar("V", bound=Hashable)
I = TypeVar("I", bound=Hashable)
VariableClass: TypeAlias = int
CanonicalNode: TypeAlias = int
CanonicalVariable: TypeAlias = int
Label: TypeAlias = L
Variables: TypeAlias = tuple[V, ...]
CanonicalVariables: TypeAlias = tuple[CanonicalVariable, ...]
NodeMapping: TypeAlias = dict[N, N]
CanonicalNodeMapping: TypeAlias = dict[CanonicalNode, N]
VariableMapping: TypeAlias = dict[VariableClass, dict[V, V]]
CanonicalVariableMapping: TypeAlias = dict[VariableClass, dict[CanonicalVariable, V]]
MatchResult: TypeAlias = tuple[NodeMapping[N], VariableMapping[V], I]
MatchSignature: TypeAlias = Hashable
InternalMatchSignature: TypeAlias = tuple[int, tuple[Any, ...]]


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _iter_mask_indices(mask: int) -> Iterator[int]:
    while mask:
        low_bit = mask & -mask
        yield low_bit.bit_length() - 1
        mask ^= low_bit


class _SearchTimeout(Exception):
    pass


@dataclass(slots=True)
class _OperationBudget:
    remaining: int

    def spend(self, cost: int = 1) -> None:
        self.remaining -= cost
        if self.remaining < 0:
            raise _SearchTimeout


def _spend(budget: _OperationBudget | None, cost: int = 1) -> None:
    if budget is not None:
        budget.spend(cost)


def _label_stats(labels: dict[N, L], variables: dict[N, Variables[V]]) -> tuple[Counter[L], dict[VariableClass, tuple[int, ...]]]:
    constant_counts: Counter[L] = Counter()
    variable_groups: defaultdict[VariableClass, Counter[V]] = defaultdict(Counter)

    for node_label in labels.values():
        constant_counts[node_label] += 1
    for node_variables in variables.values():
        for variable_class, identifier in enumerate(node_variables):
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
    articulation_points: set[int] = set()
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
                        articulation_points.add(position)
                elif low[neighbour] >= discovery[position]:
                    articulation_points.add(position)
            elif neighbour != parent[position]:
                low[position] = min(low[position], discovery[neighbour])

    dfs(root)
    if visited_count != remaining_count:
        return remaining

    return frozenset(articulation_points)


def _topology_order(graph: nx.DiGraph[N], labels: dict[N, L]) -> tuple[N, ...]:
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
class _LabelPattern(Generic[L]):
    node_label: L
    repeated_from: tuple[int | None, ...]


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


def _label_patterns_for_order(
    labels: dict[N, L],
    variables: dict[N, Variables[V]],
    order: tuple[N, ...],
) -> tuple[_LabelPattern[L], ...]:
    label_patterns: list[_LabelPattern[L]] = []
    first_variable_position: dict[tuple[VariableClass, V], int] = {}

    for node in order:
        node_variables = variables[node]
        actual_repeated_from: list[int | None] = []
        for variable_class, identifier in enumerate(node_variables):
            key = (variable_class, identifier)
            first_position = first_variable_position.get(key)
            actual_repeated_from.append(first_position)
            if first_position is None:
                first_variable_position[key] = len(label_patterns)
        label_patterns.append(
            _LabelPattern(
                node_label=labels[node],
                repeated_from=tuple(actual_repeated_from),
            )
        )

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


def _symmetry_conditions(
    graph: nx.DiGraph[N],
    order: tuple[N, ...],
    labels: dict[N, L],
    variables: dict[N, Variables[V]],
) -> tuple[tuple[int, int], ...]:
    adjacency, neighbours, neighbour_counts = _position_graph(graph, order)
    color_classes = _refined_color_classes(
        adjacency,
        tuple((labels[node], variables[node]) for node in order),
    )
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
class _StoredGraph(Generic[N, L, V]):
    graph: nx.DiGraph[CanonicalNode]
    labels: dict[CanonicalNode, L]
    variables: dict[CanonicalNode, CanonicalVariables]
    witness_graph: nx.DiGraph[N]
    canonical_to_witness_nodes: dict[CanonicalNode, N]
    canonical_to_witness_variables: dict[VariableClass, dict[CanonicalVariable, V]]
    constant_counts: Counter[L]
    variable_group_sizes: dict[VariableClass, tuple[int, ...]]
    in_degrees: dict[CanonicalNode, int]
    out_degrees: dict[CanonicalNode, int]
    predecessors: dict[CanonicalNode, frozenset[CanonicalNode]]
    successors: dict[CanonicalNode, frozenset[CanonicalNode]]
    neighbours: dict[CanonicalNode, frozenset[CanonicalNode]]
    order: tuple[CanonicalNode, ...]
    order_positions: dict[CanonicalNode, int]
    variable_key_nodes: dict[tuple[VariableClass, CanonicalVariable], tuple[CanonicalNode, ...]]
    variable_class_nodes: dict[VariableClass, tuple[CanonicalNode, ...]]
    topology_patterns: tuple[_TopologyPattern, ...]
    label_patterns: tuple[_LabelPattern[L], ...]
    condition_options: tuple[_ConditionOption, ...]
    full_conditions: tuple[tuple[int, int], ...]
    future_out_positions: tuple[tuple[int, ...], ...]
    future_in_positions: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class _IdentifierWitness(Generic[N, V]):
    canonical_to_witness_nodes: dict[CanonicalNode, N]
    canonical_to_witness_variables: dict[VariableClass, dict[CanonicalVariable, V]]


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
class _QueryData(Generic[N, L, V]):
    graph: nx.DiGraph[N]
    labels: dict[N, L]
    variables: dict[N, Variables[V]]
    constant_nodes: dict[L, frozenset[N]]
    constant_masks: dict[L, int]
    variable_identifier_nodes: dict[tuple[VariableClass, V], frozenset[N]]
    variable_identifier_masks: dict[tuple[VariableClass, V], int]
    predecessors: dict[N, AbstractSet[N]]
    predecessor_masks: tuple[int, ...]
    successors: dict[N, AbstractSet[N]]
    successor_masks: tuple[int, ...]
    constant_counts: Counter[L]
    variable_group_sizes: dict[VariableClass, tuple[int, ...]]
    in_degrees: dict[N, int]
    out_degrees: dict[N, int]
    node_ranks: dict[N, int]
    node_indices: dict[N, int]
    index_to_node: tuple[N, ...]
    all_nodes_mask: int


@dataclass(slots=True)
class _DynamicSearchState(Generic[N, V]):
    target_to_query: CanonicalNodeMapping[N]
    used_query_nodes: set[N]
    target_var_to_query_identifier: CanonicalVariableMapping[V]
    used_query_identifiers: dict[VariableClass, set[V]]


@dataclass(slots=True)
class _SearchCaches(Generic[N, V]):
    refined_masks: dict[tuple[Any, ...], dict[CanonicalNode, int] | None] = field(default_factory=dict)
    propagated_masks: dict[
        tuple[Any, ...],
        tuple[dict[CanonicalNode, int], CanonicalVariableMapping[V], dict[VariableClass, set[V]]] | None,
    ] = field(default_factory=dict)
    dead_single_states: set[tuple[Any, ...]] = field(default_factory=set)
    dead_collect_states: set[tuple[Any, ...]] = field(default_factory=set)


def _build_query_data(
    graph: nx.DiGraph[N],
    labels: dict[N, L],
    variables: dict[N, Variables[V]],
) -> _QueryData[N, L, V]:
    constant_nodes_mut: defaultdict[L, set[N]] = defaultdict(set)
    variable_identifier_nodes_mut: defaultdict[tuple[VariableClass, V], set[N]] = defaultdict(set)
    index_to_node = tuple(graph.nodes)
    node_indices = {node: index for index, node in enumerate(index_to_node)}
    all_nodes_mask = (1 << len(index_to_node)) - 1

    for node, node_label in labels.items():
        constant_nodes_mut[node_label].add(node)
    for node, node_variables in variables.items():
        for variable_class, identifier in enumerate(node_variables):
            variable_identifier_nodes_mut[(variable_class, identifier)].add(node)

    def mask_for_nodes(nodes: set[N]) -> int:
        mask = 0
        for node in nodes:
            mask |= 1 << node_indices[node]
        return mask

    constant_counts, variable_group_sizes = _label_stats(labels, variables)
    return _QueryData(
        graph=graph,
        labels=labels,
        variables=variables,
        constant_nodes={identifier: frozenset(nodes) for identifier, nodes in constant_nodes_mut.items()},
        constant_masks={identifier: mask_for_nodes(nodes) for identifier, nodes in constant_nodes_mut.items()},
        variable_identifier_nodes={
            key: frozenset(nodes)
            for key, nodes in variable_identifier_nodes_mut.items()
        },
        variable_identifier_masks={
            key: mask_for_nodes(nodes)
            for key, nodes in variable_identifier_nodes_mut.items()
        },
        predecessors={node: graph.pred[node].keys() for node in graph.nodes},
        predecessor_masks=tuple(
            sum(1 << node_indices[predecessor] for predecessor in graph.pred[node])
            for node in index_to_node
        ),
        successors={node: graph.succ[node].keys() for node in graph.nodes},
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


class Database(Generic[N, L, V, I]):
    def __init__(self, node_label: Callable[[dict[str, Any]], L], node_vars: Callable[[dict[str, Any]], Variables[V]]):
        self._node_label = node_label
        self._node_vars = node_vars
        self._graphs: list[_StoredGraph[N, L, V]] = []
        self._idents: list[dict[I, _IdentifierWitness[N, V]]] = []
        self._canonical_buckets: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
        self._root = _TrieNode(depth=0, topology_pattern=None)
        self._heuristic_fallbacks_enabled = _env_enabled("GLABVARTRIE_ENABLE_HEURISTIC_FALLBACKS", True)
        self._ortools_enabled = _env_enabled("GLABVARTRIE_ENABLE_ORTOOLS", True) and cp_model is not None
        self._z3_enabled = _env_enabled("GLABVARTRIE_ENABLE_Z3", True) and z3 is not None
        self._native_ops = _env_int("GLABVARTRIE_NATIVE_OPS", 1_000_000)
        self._scc_ops = _env_int("GLABVARTRIE_SCC_OPS", 20_000_000)
        self._anchored_ops = _env_int("GLABVARTRIE_ANCHORED_OPS", 6_000_000)
        self._z3_rlimit = _env_int("GLABVARTRIE_Z3_RLIMIT", 5_000_000)
        self._ortools_deterministic_time = _env_float("GLABVARTRIE_ORTOOLS_DETERMINISTIC_TIME", 0.2)

    def _identifier_witness(
        self,
        canonical_to_witness_nodes: dict[CanonicalNode, N],
        canonical_to_witness_variables: dict[VariableClass, dict[CanonicalVariable, V]],
    ) -> _IdentifierWitness[N, V]:
        return _IdentifierWitness(
            canonical_to_witness_nodes=dict(canonical_to_witness_nodes),
            canonical_to_witness_variables={
                variable_class: dict(identifier_mapping)
                for variable_class, identifier_mapping in canonical_to_witness_variables.items()
            },
        )

    def _native_budget(self, stored: _StoredGraph[N, L, V]) -> _OperationBudget | None:
        if stored.graph.number_of_nodes() < 50:
            return None
        if not (self._heuristic_fallbacks_enabled or self._ortools_enabled or self._z3_enabled):
            return None
        return _OperationBudget(self._native_ops)

    def _build_stored_graph(
        self,
        g: nx.DiGraph[CanonicalNode],
        labels: dict[CanonicalNode, L],
        variables: dict[CanonicalNode, CanonicalVariables],
        witness_graph: nx.DiGraph[N],
        canonical_to_witness_nodes: dict[CanonicalNode, N],
        canonical_to_witness_variables: dict[VariableClass, dict[CanonicalVariable, V]],
    ) -> _StoredGraph[N, L, V]:
        order = _topology_order(g, labels)
        topology_patterns = _topology_patterns_for_order(g, order)
        label_patterns = _label_patterns_for_order(labels, variables, order)
        conditions = _symmetry_conditions(g, order, labels, variables)
        constant_counts, variable_group_sizes = _label_stats(labels, variables)
        order_positions = {node: position for position, node in enumerate(order)}
        variable_key_nodes_mut: defaultdict[tuple[VariableClass, CanonicalVariable], list[CanonicalNode]] = defaultdict(list)
        variable_class_nodes_mut: defaultdict[VariableClass, list[CanonicalNode]] = defaultdict(list)
        for node in order:
            for variable_class, identifier in enumerate(variables[node]):
                variable_key_nodes_mut[(variable_class, identifier)].append(node)
                variable_class_nodes_mut[variable_class].append(node)

        future_out_positions_mut: list[list[int]] = [[] for _ in order]
        future_in_positions_mut: list[list[int]] = [[] for _ in order]
        for future_position, topology_pattern in enumerate(topology_patterns):
            for previous_position in topology_pattern.prev_to_new:
                future_out_positions_mut[previous_position].append(future_position)
            for previous_position in topology_pattern.new_to_prev:
                future_in_positions_mut[previous_position].append(future_position)

        return _StoredGraph(
            graph=g,
            labels=labels,
            variables=variables,
            witness_graph=witness_graph,
            canonical_to_witness_nodes=canonical_to_witness_nodes,
            canonical_to_witness_variables=canonical_to_witness_variables,
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

    def _canonicalize_index_variables(
        self,
        order: tuple[N, ...],
        variables: dict[N, Variables[V]],
    ) -> dict[N, CanonicalVariables]:
        canonical_identifiers_by_class: defaultdict[VariableClass, dict[V, int]] = defaultdict(dict)
        canonical_variables_mut: dict[N, CanonicalVariables] = {}

        for node in order:
            canonical_variables_mut[node] = tuple(
                canonical_identifiers_by_class[variable_class].setdefault(identifier, len(canonical_identifiers_by_class[variable_class]))
                for variable_class, identifier in enumerate(variables[node])
            )

        return canonical_variables_mut

    def _canonicalize_index_graph(
        self,
        g: nx.DiGraph[N],
        labels: dict[N, L],
        variables: dict[N, Variables[V]],
    ) -> tuple[
        nx.DiGraph[CanonicalNode],
        dict[CanonicalNode, L],
        dict[CanonicalNode, CanonicalVariables],
        dict[CanonicalNode, N],
        dict[VariableClass, dict[CanonicalVariable, V]],
    ]:
        order = _topology_order(g, labels)
        canonical_variables_by_original = self._canonicalize_index_variables(order, variables)
        canonical_node_by_original = {
            original_node: canonical_node
            for canonical_node, original_node in enumerate(order)
        }
        canonical_to_witness_node = {
            canonical_node: original_node
            for original_node, canonical_node in canonical_node_by_original.items()
        }
        canonical_graph: nx.DiGraph[CanonicalNode] = nx.DiGraph()
        canonical_graph.graph.update(g.graph)
        canonical_graph.graph["_glabvartrie_canonical"] = True
        canonical_labels: dict[CanonicalNode, L] = {}
        canonical_variables: dict[CanonicalNode, CanonicalVariables] = {}
        canonical_to_original_variables: defaultdict[VariableClass, dict[CanonicalVariable, V]] = defaultdict(dict)

        for original_node in order:
            canonical_node = canonical_node_by_original[original_node]
            canonical_vars = canonical_variables_by_original[original_node]
            node_attrs = dict(g.nodes[original_node])
            if node_attrs.get("vars") == variables[original_node]:
                node_attrs["vars"] = canonical_vars
            if node_attrs.get("label") == labels[original_node]:
                node_attrs["label"] = labels[original_node]
            node_attrs["_glabvartrie_original_node"] = original_node
            node_attrs["_glabvartrie_label"] = labels[original_node]
            node_attrs["_glabvartrie_vars"] = canonical_vars
            canonical_graph.add_node(canonical_node, **node_attrs)
            canonical_labels[canonical_node] = labels[original_node]
            canonical_variables[canonical_node] = canonical_vars
            for variable_class, original_identifier in enumerate(variables[original_node]):
                canonical_identifier = canonical_vars[variable_class]
                canonical_to_original_variables[variable_class].setdefault(canonical_identifier, original_identifier)

        for source, target, edge_attrs in g.edges(data=True):
            canonical_graph.add_edge(
                canonical_node_by_original[source],
                canonical_node_by_original[target],
                **dict(edge_attrs),
            )

        return (
            canonical_graph,
            canonical_labels,
            canonical_variables,
            canonical_to_witness_node,
            dict(canonical_to_original_variables),
        )

    def _equivalence_graph(self, stored: _StoredGraph[N, L, V]) -> nx.DiGraph[N]:
        graph = nx.DiGraph()
        for node in stored.graph.nodes:
            graph.add_node(
                node,
                canon_label=repr((stored.labels[node], len(stored.variables[node]))),
            )
        graph.add_edges_from(stored.graph.edges)
        return graph

    def _canonical_bucket_key(self, stored: _StoredGraph[N, L, V]) -> tuple[Any, ...]:
        equivalence_graph = self._equivalence_graph(stored)
        return (
            stored.graph.number_of_nodes(),
            stored.graph.number_of_edges(),
            tuple(sorted(stored.constant_counts.items(), key=lambda item: repr(item[0]))),
            tuple(sorted(stored.variable_group_sizes.items(), key=lambda item: item[0])),
            weisfeiler_lehman_graph_hash(equivalence_graph, node_attr="canon_label"),
        )

    def _stored_graphs_equivalent(
        self,
        left: _StoredGraph[N, L, V],
        right: _StoredGraph[N, L, V],
    ) -> bool:
        if left.graph.number_of_nodes() != right.graph.number_of_nodes():
            return False
        if left.graph.number_of_edges() != right.graph.number_of_edges():
            return False
        if left.constant_counts != right.constant_counts:
            return False
        if left.variable_group_sizes != right.variable_group_sizes:
            return False

        left_graph = self._equivalence_graph(left)
        right_graph = self._equivalence_graph(right)
        matcher = DiGraphMatcher(
            left_graph,
            right_graph,
            node_match=lambda left_attrs, right_attrs: left_attrs["canon_label"] == right_attrs["canon_label"],
        )

        for mapping in matcher.isomorphisms_iter():
            slot_mapping: dict[int, dict[CanonicalVariable, CanonicalVariable]] = defaultdict(dict)
            reverse_slot_mapping: dict[int, dict[CanonicalVariable, CanonicalVariable]] = defaultdict(dict)
            for left_node, right_node in mapping.items():
                left_vars = left.variables[left_node]
                right_vars = right.variables[right_node]
                if len(left_vars) != len(right_vars):
                    break
                for slot, left_identifier in enumerate(left_vars):
                    right_identifier = right_vars[slot]
                    existing = slot_mapping[slot].get(left_identifier)
                    if existing is not None and existing != right_identifier:
                        break
                    reverse_existing = reverse_slot_mapping[slot].get(right_identifier)
                    if reverse_existing is not None and reverse_existing != left_identifier:
                        break
                    slot_mapping[slot][left_identifier] = right_identifier
                    reverse_slot_mapping[slot][right_identifier] = left_identifier
                else:
                    continue
                break
            else:
                return True

        return False

    def index(self, g: nx.DiGraph[N], ident: I) -> None:
        labels = {node: self._node_label(g.nodes[node]) for node in g.nodes}
        variables = {node: self._node_vars(g.nodes[node]) for node in g.nodes}
        canonical_graph, canonical_labels, canonical_variables, canonical_to_witness_nodes, canonical_to_witness_variables = self._canonicalize_index_graph(g, labels, variables)
        stored = self._build_stored_graph(
            canonical_graph,
            canonical_labels,
            canonical_variables,
            witness_graph=g.copy(),
            canonical_to_witness_nodes=canonical_to_witness_nodes,
            canonical_to_witness_variables=canonical_to_witness_variables,
        )
        identifier_witness = self._identifier_witness(
            canonical_to_witness_nodes,
            canonical_to_witness_variables,
        )
        bucket_key = self._canonical_bucket_key(stored)
        for graph_index in self._canonical_buckets[bucket_key]:
            if self._stored_graphs_equivalent(self._graphs[graph_index], stored):
                self._idents[graph_index][ident] = identifier_witness
                return
        graph_index = len(self._graphs)
        self._graphs.append(stored)
        self._idents.append({ident: identifier_witness})
        self._canonical_buckets[bucket_key].append(graph_index)

        node = self._root
        node.descendant_graph_indices.add(graph_index)
        for depth, topology_pattern in enumerate(stored.topology_patterns, start=1):
            child = node.children.get(topology_pattern)
            if child is None:
                child = _TrieNode(depth=depth, topology_pattern=topology_pattern)
                node.children[topology_pattern] = child
            child.descendant_graph_indices.add(graph_index)
            child.label_pattern_graph_indices.setdefault(stored.label_patterns[depth - 1], set()).add(graph_index)
            child.condition_options.add(_option_for_depth(stored.full_conditions, depth - 1))
            node = child

        node.terminal_graph_indices.append(graph_index)
        if node.full_conditions is None:
            node.full_conditions = stored.full_conditions

    def query(self, g: nx.DiGraph[N]) -> Iterator[MatchResult[N, V, I]]:
        query_labels = {node: self._node_label(g.nodes[node]) for node in g.nodes}
        query_variables = {node: self._node_vars(g.nodes[node]) for node in g.nodes}
        query_data = _build_query_data(g, query_labels, query_variables)

        eligible = {
            index
            for index, stored in enumerate(self._graphs)
            if self._can_match(stored, query_data)
        }
        if not eligible:
            return
            yield

        seen_results: set[tuple[int, tuple[N, ...], I]] = set()
        seen_graphs: set[int] = set()
        yield from self._query_direct_first_matches(
            query_data,
            eligible,
            seen_results,
            seen_graphs,
            self._native_ops,
        )
        unseen_graphs = eligible - seen_graphs
        if unseen_graphs:
            yield from self._query_direct_first_matches(
                query_data,
                unseen_graphs,
                seen_results,
                seen_graphs,
                self._native_ops * 4,
            )
        if len(eligible) <= 4:
            yield from self._query_direct(query_data, eligible, seen_results)
            return

        used_query_nodes: set[N] = set()
        matched_query_nodes: list[N] = []

        for child in sorted(self._root.children.values(), key=self._node_priority, reverse=True):
            if child.descendant_graph_indices.isdisjoint(eligible):
                continue
            yield from self._search(child, query_data, eligible, matched_query_nodes, used_query_nodes, seen_results)

    def _query_direct_first_matches(
        self,
        query_data: _QueryData[N, L, V],
        eligible: set[int],
        seen_results: set[tuple[int, tuple[N, ...], I]],
        seen_graphs: set[int],
        first_match_budget_ops: int,
    ) -> Iterator[MatchResult[N, V, I]]:
        graph_order = sorted(
            eligible,
            key=lambda graph_index: (
                self._direct_graph_priority(self._graphs[graph_index], query_data),
                graph_index,
            ),
        )
        for graph_index in graph_order:
            yield from self._collect_single_graph_matches(
                graph_index,
                query_data,
                {graph_index},
                seen_results,
                1,
                use_lookahead=True,
                allow_timeout_fallback=False,
                first_match_budget_ops=first_match_budget_ops,
                seen_graphs=seen_graphs,
            )

    def _query_direct(
        self,
        query_data: _QueryData[N, L, V],
        eligible: set[int],
        seen_results: set[tuple[int, tuple[N, ...], I]],
    ) -> Iterator[MatchResult[N, V, I]]:
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
            yield from self._collect_single_graph_matches(
                graph_index,
                query_data,
                remaining,
                seen_results,
                self._all_match_limit(self._graphs[graph_index], query_data),
                use_lookahead=True,
            )

    def _all_match_limit(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
    ) -> int:
        return math.perm(query_data.graph.number_of_nodes(), stored.graph.number_of_nodes())

    def _canonical_node_mapping_signature(
        self,
        target_to_query: CanonicalNodeMapping[N],
    ) -> frozenset[tuple[CanonicalNode, N]]:
        return frozenset(target_to_query.items())

    def _canonical_variable_mapping_signature(
        self,
        target_var_to_query_identifier: CanonicalVariableMapping[V],
    ) -> frozenset[tuple[VariableClass, frozenset[tuple[CanonicalVariable, V]]]]:
        return frozenset(
            (
                variable_class,
                frozenset(identifier_mapping.items()),
            )
            for variable_class, identifier_mapping in target_var_to_query_identifier.items()
        )

    def _used_identifier_signature(
        self,
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> tuple[()] | frozenset[tuple[VariableClass, frozenset[V]]]:
        if not used_query_identifiers:
            return ()
        return frozenset(
            (
                variable_class,
                frozenset(identifiers),
            )
            for variable_class, identifiers in used_query_identifiers.items()
        )

    def _mask_signature(
        self,
        masks: dict[CanonicalNode, int],
    ) -> tuple[int, ...]:
        # Mask dictionaries are constructed in canonical target-node order and only
        # copied or value-updated afterwards, so insertion order is stable enough
        # to serve as an internal cache key.
        return tuple(masks.values())

    def _state_signature(
        self,
        target_to_query: CanonicalNodeMapping[N],
        masks: dict[CanonicalNode, int],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
    ) -> tuple[Any, ...]:
        return (
            self._canonical_node_mapping_signature(target_to_query),
            self._mask_signature(masks),
            self._canonical_variable_mapping_signature(target_var_to_query_identifier),
        )

    def _direct_graph_priority(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
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
        query_data: _QueryData[N, L, V],
        eligible: set[int],
        seen_results: set[tuple[int, tuple[N, ...], I]],
        match_limit: int,
        use_lookahead: bool,
        allow_timeout_fallback: bool = True,
        first_match_budget_ops: int | None = None,
        seen_graphs: set[int] | None = None,
    ) -> Iterator[MatchResult[N, V, I]]:
        if graph_index not in eligible or match_limit <= 0:
            return
            yield

        stored = self._graphs[graph_index]
        emitted = False
        if match_limit > 1:
            try:
                for node_mapping, variable_mapping in self._iter_single_graph_matches(
                    stored,
                    query_data,
                    {},
                    set(),
                    {},
                    {},
                    match_limit,
                    budget=self._native_budget(stored),
                    search_caches=None,
                ):
                    emitted = True
                    yield from self._iter_match_results(
                        graph_index,
                        node_mapping,
                        variable_mapping,
                        seen_results,
                        seen_graphs,
                    )
            except _SearchTimeout:
                fallback_match = self._timeout_fallback_match(stored, query_data) if allow_timeout_fallback else None
                if fallback_match is not None:
                    emitted = True
                    yield from self._iter_match_results(
                        graph_index,
                        fallback_match[0],
                        fallback_match[1],
                        seen_results,
                        seen_graphs,
                    )
        else:
            search_caches = _SearchCaches[N, V]() if allow_timeout_fallback else None
            match: tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None
            try:
                budget = self._native_budget(stored)
                if first_match_budget_ops is not None and not allow_timeout_fallback:
                    budget = _OperationBudget(first_match_budget_ops)
                elif budget is None and not allow_timeout_fallback:
                    # The first-match prepass should not let small but ambiguous graphs
                    # monopolize the iterator before later eligible identifiers are tried.
                    budget = _OperationBudget(self._native_ops)
                match = self._find_single_graph_match(
                    stored,
                    query_data,
                    {},
                    set(),
                    {},
                    {},
                    budget=budget,
                    search_caches=search_caches,
                    use_search_caches=allow_timeout_fallback,
                )
            except _SearchTimeout:
                match = self._timeout_fallback_match(stored, query_data) if allow_timeout_fallback else None
            if match is not None:
                emitted = True
                yield from self._iter_match_results(
                    graph_index,
                    match[0],
                    match[1],
                    seen_results,
                    seen_graphs,
                )

        if emitted:
            eligible.discard(graph_index)

    def _copy_variable_state(
        self,
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> tuple[CanonicalVariableMapping[V], dict[VariableClass, set[V]]]:
        if not used_query_identifiers:
            return (
                {
                    variable_class: dict(identifier_mapping)
                    for variable_class, identifier_mapping in target_var_to_query_identifier.items()
                },
                {},
            )
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

    def _internal_match_signature(
        self,
        graph_index: int,
        stored: _StoredGraph[N, L, V],
        node_mapping: CanonicalNodeMapping[N],
    ) -> InternalMatchSignature:
        return (
            graph_index,
            tuple(node_mapping[target_node] for target_node in stored.order),
        )

    def _iter_match_results(
        self,
        graph_index: int,
        node_mapping: CanonicalNodeMapping[N],
        variable_mapping: CanonicalVariableMapping[V],
        seen_results: set[tuple[int, tuple[N, ...], I]],
        seen_graphs: set[int] | None = None,
    ) -> Iterator[MatchResult[N, V, I]]:
        stored = self._graphs[graph_index]
        ordered_query_nodes = tuple(node_mapping[target_node] for target_node in stored.order)

        for ident, witness in self._idents[graph_index].items():
            signature = (graph_index, ordered_query_nodes, ident)
            if signature in seen_results:
                continue
            seen_results.add(signature)
            if seen_graphs is not None:
                seen_graphs.add(graph_index)

            witness_node_mapping = {
                witness.canonical_to_witness_nodes[target_node]: query_node
                for target_node, query_node in node_mapping.items()
            }
            witness_variable_mapping: dict[VariableClass, dict[V, V]] = {}
            for variable_class, identifier_mapping in variable_mapping.items():
                witness_identifier_mapping: dict[V, V] = {}
                canonical_to_witness_identifiers = witness.canonical_to_witness_variables.get(variable_class, {})
                for canonical_identifier, query_identifier in identifier_mapping.items():
                    witness_identifier_mapping[canonical_to_witness_identifiers[canonical_identifier]] = query_identifier
                witness_variable_mapping[variable_class] = witness_identifier_mapping

            yield witness_node_mapping, witness_variable_mapping, ident

    def _timeout_fallback_match(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        if stored.graph.number_of_nodes() < 50:
            return None
        if self._heuristic_fallbacks_enabled and self._direct_graph_priority(stored, query_data)[0] <= 16:
            match = self._find_single_graph_match_scc_decomposed(stored, query_data)
            if match is not None:
                return match
            match = self._find_single_graph_match_anchored(stored, query_data)
            if match is not None:
                return match
        if self._ortools_enabled:
            match = self._find_single_graph_match_ortools(stored, query_data)
            if match is not None:
                return match
        if self._z3_enabled:
            return self._find_single_graph_match_z3(stored, query_data)
        return None

    def _stored_subgraph(
        self,
        stored: _StoredGraph[N, L, V],
        nodes: frozenset[CanonicalNode],
    ) -> _StoredGraph[N, L, V]:
        subgraph = nx.DiGraph(stored.graph.subgraph(nodes))
        sublabels = {node: stored.labels[node] for node in subgraph.nodes}
        subvars = {node: stored.variables[node] for node in subgraph.nodes}
        witness_nodes = [stored.canonical_to_witness_nodes[node] for node in subgraph.nodes]
        witness_subgraph = nx.DiGraph(stored.witness_graph.subgraph(witness_nodes))
        canonical_to_witness_nodes = {
            node: stored.canonical_to_witness_nodes[node]
            for node in subgraph.nodes
        }
        canonical_to_witness_variables = {
            variable_class: {
                canonical_identifier: witness_identifier
                for canonical_identifier, witness_identifier in identifier_mapping.items()
                if any(canonical_identifier in stored.variables[node] for node in subgraph.nodes)
            }
            for variable_class, identifier_mapping in stored.canonical_to_witness_variables.items()
        }
        return self._build_stored_graph(
            subgraph,
            sublabels,
            subvars,
            witness_subgraph,
            canonical_to_witness_nodes,
            canonical_to_witness_variables,
        )

    def _find_single_graph_match_scc_decomposed(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        sccs = [frozenset(component) for component in nx.strongly_connected_components(stored.graph)]
        if len(sccs) <= 1:
            return None

        component_index = {
            node: index
            for index, component in enumerate(sccs)
            for node in component
        }
        condensation = nx.DiGraph()
        condensation.add_nodes_from(range(len(sccs)))
        for source, target in stored.graph.edges:
            source_component = component_index[source]
            target_component = component_index[target]
            if source_component != target_component:
                condensation.add_edge(source_component, target_component)
        budget = _OperationBudget(self._scc_ops)
        subgraph_cache: dict[frozenset[CanonicalNode], _StoredGraph[N, L, V]] = {}

        def component_setup(
            component_id: int,
            target_to_query: CanonicalNodeMapping[N],
        ) -> tuple[
            _StoredGraph[N, L, V],
            tuple[CanonicalNode, ...],
            CanonicalNodeMapping[N],
            set[N],
            CanonicalVariableMapping[V],
            dict[VariableClass, set[V]],
            tuple[int, int, int, str],
        ] | None:
            component_nodes = sccs[component_id]
            boundary_nodes = {
                neighbour
                for node in component_nodes
                for neighbour in stored.neighbours[node]
                if neighbour not in component_nodes and neighbour in target_to_query
            }
            subgraph_nodes = frozenset(component_nodes | boundary_nodes)
            substored = subgraph_cache.get(subgraph_nodes)
            if substored is None:
                substored = self._stored_subgraph(stored, subgraph_nodes)
                subgraph_cache[subgraph_nodes] = substored

            partial_mapping = {
                node: query_node
                for node, query_node in target_to_query.items()
                if node in subgraph_nodes
            }
            used_query_nodes = set(target_to_query.values())
            target_var_to_query_identifier, used_query_identifiers = self._variable_state(
                stored,
                target_to_query,
                query_data,
            )
            masks = self._initial_single_graph_masks(
                substored,
                query_data,
                partial_mapping,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
                budget,
            )
            if masks is None:
                return None
            refined = self._refine_single_graph_masks(
                substored,
                query_data,
                partial_mapping,
                masks,
                budget,
            )
            if refined is None:
                return None
            priority = min(
                (
                    (
                        -len(component_nodes),
                        refined[target_node].bit_count(),
                        -(substored.in_degrees[target_node] + substored.out_degrees[target_node]),
                        repr(target_node),
                    )
                    for target_node in component_nodes
                ),
                default=(0, 0, 0, ""),
            )
            return (
                substored,
                tuple(target_node for target_node in stored.order if target_node in component_nodes),
                partial_mapping,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
                priority,
            )

        def recurse(
            target_to_query: CanonicalNodeMapping[N],
            assigned_components: frozenset[int],
        ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
            _spend(budget)
            if len(assigned_components) == len(sccs):
                ordered_query_nodes = [target_to_query[target_node] for target_node in stored.order]
                if not _conditions_hold(stored.full_conditions, ordered_query_nodes, query_data.node_ranks):
                    return None
                return target_to_query, self._variable_mapping(stored, query_data, ordered_query_nodes)

            setups: list[tuple[int, _StoredGraph[N, L, V], tuple[CanonicalNode, ...], CanonicalNodeMapping[N], set[N], CanonicalVariableMapping[V], dict[VariableClass, set[V]], tuple[int, int, int, str]]] = []
            for component_id in condensation.nodes:
                if component_id in assigned_components:
                    continue
                setup = component_setup(component_id, target_to_query)
                if setup is None:
                    continue
                setups.append((component_id, *setup))
            if not setups:
                return None

            component_id, substored, component_nodes, partial_mapping, used_query_nodes, target_var_to_query_identifier, used_query_identifiers, _ = min(
                setups,
                key=lambda item: (item[-1], item[0]),
            )
            match_limit = 32 if len(component_nodes) == 1 else 12
            component_matches = self._enumerate_single_graph_matches(
                substored,
                query_data,
                partial_mapping,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
                match_limit,
                budget,
            )
            for component_mapping, _ in component_matches:
                next_target_to_query = dict(target_to_query)
                for node in component_nodes:
                    query_node = component_mapping.get(node)
                    if query_node is None:
                        break
                    next_target_to_query[node] = query_node
                else:
                    match = recurse(next_target_to_query, assigned_components | {component_id})
                    if match is not None:
                        return match
            return None

        try:
            return recurse({}, frozenset())
        except _SearchTimeout:
            return None

    def _enumerate_single_graph_best_first(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        results: list[tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]]],
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

            for penalty_delta, query_node in enumerate(ordered_candidates):
                new_target_to_query = dict(state.target_to_query)
                new_target_to_query[best_target] = query_node
                new_used_query_nodes = set(state.used_query_nodes)
                new_used_query_nodes.add(query_node)
                new_target_var_to_query_identifier, new_used_query_identifiers = self._copy_variable_state(
                    state.target_var_to_query_identifier,
                    state.used_query_identifiers,
                )

                for variable_class, target_identifier in enumerate(stored.variables[best_target]):
                    query_identifier = query_data.variables[query_node][variable_class]
                    variable_mapping = new_target_var_to_query_identifier.setdefault(variable_class, {})
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
        query_data: _QueryData[N, L, V],
        eligible: set[int],
        matched_query_nodes: list[N],
        used_query_nodes: set[N],
        seen_results: set[tuple[int, tuple[N, ...], I]],
        candidate_plan: list[tuple[N, frozenset[int]]] | None = None,
    ) -> Iterator[MatchResult[N, V, I]]:
        active_graphs = node.descendant_graph_indices & eligible
        if not active_graphs or node.topology_pattern is None:
            return

        if len(active_graphs) == 1:
            graph_index = next(iter(active_graphs))
            yield from self._search_single_graph(
                graph_index,
                node.depth - 1,
                query_data,
                eligible,
                matched_query_nodes,
                used_query_nodes,
                seen_results,
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
                        yield from self._iter_match_results(
                            graph_index,
                            target_to_query,
                            self._variable_mapping(stored, query_data, matched_query_nodes),
                            seen_results,
                        )

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
                    yield from self._search(
                        child,
                        query_data,
                        remaining_active,
                        matched_query_nodes,
                        used_query_nodes,
                        seen_results,
                        child_candidates,
                    )

            matched_query_nodes.pop()
            used_query_nodes.remove(candidate)

    def _search_single_graph(
        self,
        graph_index: int,
        position: int,
        query_data: _QueryData[N, L, V],
        eligible: set[int],
        matched_query_nodes: list[N],
        used_query_nodes: set[N],
        seen_results: set[tuple[int, tuple[N, ...], I]],
        candidate_plan: list[tuple[N, frozenset[int]]] | None = None,
    ) -> Iterator[MatchResult[N, V, I]]:
        del position
        del candidate_plan

        if graph_index not in eligible:
            return
            yield

        stored = self._graphs[graph_index]
        target_to_query = {
            target_node: matched_query_nodes[current_position]
            for current_position, target_node in enumerate(stored.order[: len(matched_query_nodes)])
        }
        target_var_to_query_identifier, used_query_identifiers = self._variable_state(stored, target_to_query, query_data)
        try:
            for node_mapping, variable_mapping in self._iter_single_graph_matches(
                stored,
                query_data,
                target_to_query,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
                self._all_match_limit(stored, query_data),
                budget=self._native_budget(stored),
                search_caches=None,
            ):
                yield from self._iter_match_results(
                    graph_index,
                    node_mapping,
                    variable_mapping,
                    seen_results,
                )
        except _SearchTimeout:
            fallback_match = self._timeout_fallback_match(stored, query_data)
            if fallback_match is not None:
                yield from self._iter_match_results(
                    graph_index,
                    fallback_match[0],
                    fallback_match[1],
                    seen_results,
                )

    def _variable_state(
        self,
        stored: _StoredGraph[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
    ) -> tuple[CanonicalVariableMapping[V], dict[VariableClass, set[V]]]:
        target_var_to_query_identifier: defaultdict[VariableClass, dict[CanonicalVariable, V]] = defaultdict(dict)
        used_query_identifiers: defaultdict[VariableClass, set[V]] = defaultdict(set)

        for target_node, query_node in target_to_query.items():
            if stored.labels[target_node] != query_data.labels[query_node]:
                raise ValueError("Matched node labels disagree")
            target_variables = stored.variables[target_node]
            query_variables = query_data.variables[query_node]
            if len(target_variables) != len(query_variables):
                raise ValueError("Matched node variable counts disagree")
            for variable_class, target_identifier in enumerate(target_variables):
                query_identifier = query_variables[variable_class]
                existing = target_var_to_query_identifier[variable_class].get(target_identifier)
                if existing is not None and existing != query_identifier:
                    raise ValueError("Matched target variable maps to multiple query identifiers")
                target_var_to_query_identifier[variable_class][target_identifier] = query_identifier

        return dict(target_var_to_query_identifier), {}

    def _partial_conditions_hold(
        self,
        stored: _StoredGraph[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
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
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> dict[CanonicalNode, frozenset[N]] | None:
        domains: dict[CanonicalNode, frozenset[N]] = {}

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
        stored: _StoredGraph[N, L, V],
        target_node: CanonicalNode,
        query_node: N,
        domains: dict[CanonicalNode, frozenset[N]],
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
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
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        domains: dict[CanonicalNode, frozenset[N]],
    ) -> dict[CanonicalNode, frozenset[N]] | None:
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
        stored: _StoredGraph[N, L, V],
        target_node: CanonicalNode,
        query_node: N,
        domains: dict[CanonicalNode, frozenset[N]],
        query_data: _QueryData[N, L, V],
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
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        domains: dict[CanonicalNode, frozenset[N]],
        assigned_target: CanonicalNode,
        assigned_query: N,
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> tuple[dict[CanonicalNode, frozenset[N]], CanonicalVariableMapping[V], dict[VariableClass, set[V]]] | None:
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

        target_variables = stored.variables[assigned_target]
        query_variables = query_data.variables[assigned_query]
        for variable_class, target_identifier in enumerate(target_variables):
            query_identifier = query_variables[variable_class]
            variable_mapping = next_target_var_to_query_identifier.setdefault(variable_class, {})
            existing_query_identifier = variable_mapping.get(target_identifier)
            if existing_query_identifier is not None:
                if existing_query_identifier != query_identifier:
                    return None
            else:
                variable_mapping[target_identifier] = query_identifier

            same_identifier_domain = query_data.variable_identifier_nodes.get((variable_class, query_identifier), frozenset())
            for variable_target in stored.variable_key_nodes.get((variable_class, target_identifier), ()):
                if variable_target == assigned_target:
                    continue
                current_domain = next_domains[variable_target]
                reduced = current_domain & same_identifier_domain
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
        stored: _StoredGraph[N, L, V],
        target_node: CanonicalNode,
        query_data: _QueryData[N, L, V],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> int:
        del used_query_identifiers
        mask = query_data.constant_masks.get(stored.labels[target_node], 0)
        if not mask:
            return 0
        for variable_class, target_identifier in enumerate(stored.variables[target_node]):
            existing = target_var_to_query_identifier.get(variable_class, {}).get(target_identifier)
            if existing is None:
                continue
            mask &= query_data.variable_identifier_masks.get((variable_class, existing), 0)
            if not mask:
                return 0
        return mask

    def _initial_single_graph_masks(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        budget: _OperationBudget | None = None,
    ) -> dict[CanonicalNode, int] | None:
        used_mask = 0
        for query_node in used_query_nodes:
            used_mask |= 1 << query_data.node_indices[query_node]

        masks: dict[CanonicalNode, int] = {}
        for target_node in stored.order:
            _spend(budget)
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
                _spend(budget)
                query_node = query_data.index_to_node[query_index]
                if len(query_data.variables[query_node]) != len(stored.variables[target_node]):
                    continue
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
        stored: _StoredGraph[N, L, V],
        target_node: CanonicalNode,
        query_index: int,
        masks: dict[CanonicalNode, int],
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
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
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        masks: dict[CanonicalNode, int],
        budget: _OperationBudget | None = None,
        search_caches: _SearchCaches[N, V] | None = None,
    ) -> dict[CanonicalNode, int] | None:
        cache_key: tuple[Any, ...] | None = None
        if search_caches is not None:
            cache_key = (
                self._canonical_node_mapping_signature(target_to_query),
                self._mask_signature(masks),
            )
            cached = search_caches.refined_masks.get(cache_key)
            if cached is not None:
                return dict(cached)
            if cache_key in search_caches.refined_masks and search_caches.refined_masks[cache_key] is None:
                return None

        refined = dict(masks)

        changed = True
        while changed:
            _spend(budget)
            changed = False
            for target_node in stored.order:
                current_mask = refined[target_node]
                filtered_mask = 0
                for query_index in _iter_mask_indices(current_mask):
                    _spend(budget)
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
                    if search_caches is not None and cache_key is not None:
                        search_caches.refined_masks[cache_key] = None
                    return None
                if filtered_mask != current_mask:
                    refined[target_node] = filtered_mask
                    changed = True

        if search_caches is not None and cache_key is not None:
            search_caches.refined_masks[cache_key] = dict(refined)
        return refined

    def _mask_candidate_order_key(
        self,
        stored: _StoredGraph[N, L, V],
        target_node: CanonicalNode,
        query_index: int,
        masks: dict[CanonicalNode, int],
        query_data: _QueryData[N, L, V],
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
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        masks: dict[CanonicalNode, int],
        assigned_target: CanonicalNode,
        assigned_query_index: int,
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        budget: _OperationBudget | None = None,
        search_caches: _SearchCaches[N, V] | None = None,
    ) -> tuple[dict[CanonicalNode, int], CanonicalVariableMapping[V], dict[VariableClass, set[V]]] | None:
        cache_key: tuple[Any, ...] | None = None
        if search_caches is not None:
            cache_key = (
                self._canonical_node_mapping_signature(target_to_query),
                self._mask_signature(masks),
                assigned_target,
                assigned_query_index,
                self._canonical_variable_mapping_signature(target_var_to_query_identifier),
                self._used_identifier_signature(used_query_identifiers),
            )
            cached = search_caches.propagated_masks.get(cache_key)
            if cached is not None:
                cached_masks, cached_mapping, cached_used_ids = cached
                return dict(cached_masks), cached_mapping, cached_used_ids
            if cache_key in search_caches.propagated_masks and search_caches.propagated_masks[cache_key] is None:
                return None

        _spend(budget)
        assigned_bit = 1 << assigned_query_index
        next_masks = {target_node: (assigned_bit if target_node == assigned_target else mask & (query_data.all_nodes_mask ^ assigned_bit)) for target_node, mask in masks.items()}
        if any(mask == 0 for mask in next_masks.values()):
            if search_caches is not None and cache_key is not None:
                search_caches.propagated_masks[cache_key] = None
            return None

        next_target_var_to_query_identifier = target_var_to_query_identifier
        next_used_query_identifiers = used_query_identifiers
        variable_mapping_copied = False

        assigned_query = query_data.index_to_node[assigned_query_index]
        target_variables = stored.variables[assigned_target]
        query_variables = query_data.variables[assigned_query]
        for variable_class, target_identifier in enumerate(target_variables):
            query_identifier = query_variables[variable_class]
            variable_mapping = next_target_var_to_query_identifier.get(variable_class)
            if variable_mapping is None:
                if not variable_mapping_copied:
                    next_target_var_to_query_identifier = {
                        existing_class: dict(identifier_mapping)
                        for existing_class, identifier_mapping in target_var_to_query_identifier.items()
                    }
                    variable_mapping_copied = True
                variable_mapping = next_target_var_to_query_identifier.setdefault(variable_class, {})
            existing_query_identifier = variable_mapping.get(target_identifier)
            if existing_query_identifier is not None:
                if existing_query_identifier != query_identifier:
                    return None
            else:
                if not variable_mapping_copied:
                    next_target_var_to_query_identifier = {
                        existing_class: dict(identifier_mapping)
                        for existing_class, identifier_mapping in target_var_to_query_identifier.items()
                    }
                    variable_mapping = next_target_var_to_query_identifier.setdefault(
                        variable_class,
                        dict(variable_mapping),
                    )
                    variable_mapping_copied = True
                variable_mapping[target_identifier] = query_identifier

            same_identifier_mask = query_data.variable_identifier_masks.get((variable_class, query_identifier), 0)
            for variable_target in stored.variable_key_nodes.get((variable_class, target_identifier), ()):
                if variable_target == assigned_target:
                    continue
                current_mask = next_masks[variable_target]
                reduced_mask = current_mask & same_identifier_mask
                if not reduced_mask:
                    if search_caches is not None and cache_key is not None:
                        search_caches.propagated_masks[cache_key] = None
                    return None
                next_masks[variable_target] = reduced_mask

        successor_mask = query_data.successor_masks[assigned_query_index]
        predecessor_mask = query_data.predecessor_masks[assigned_query_index]
        for successor in stored.successors[assigned_target]:
            if successor == assigned_target:
                continue
            reduced_mask = next_masks[successor] & successor_mask
            if not reduced_mask:
                if search_caches is not None and cache_key is not None:
                    search_caches.propagated_masks[cache_key] = None
                return None
            next_masks[successor] = reduced_mask
        for predecessor in stored.predecessors[assigned_target]:
            if predecessor == assigned_target:
                continue
            reduced_mask = next_masks[predecessor] & predecessor_mask
            if not reduced_mask:
                if search_caches is not None and cache_key is not None:
                    search_caches.propagated_masks[cache_key] = None
                return None
            next_masks[predecessor] = reduced_mask

        refined = self._refine_single_graph_masks(
            stored,
            query_data,
            target_to_query,
            next_masks,
            budget,
            search_caches,
        )
        if refined is None:
            if search_caches is not None and cache_key is not None:
                search_caches.propagated_masks[cache_key] = None
            return None

        if search_caches is not None and cache_key is not None:
            search_caches.propagated_masks[cache_key] = (dict(refined), next_target_var_to_query_identifier, next_used_query_identifiers)
        return refined, next_target_var_to_query_identifier, next_used_query_identifiers

    def _find_single_graph_match(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        budget: _OperationBudget | None = None,
        search_caches: _SearchCaches[N, V] | None = None,
        use_search_caches: bool = True,
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        if not self._partial_conditions_hold(stored, target_to_query, query_data):
            return None

        active_search_caches = search_caches or (_SearchCaches[N, V]() if use_search_caches else None)
        _spend(budget)
        masks = self._initial_single_graph_masks(
            stored,
            query_data,
            target_to_query,
            used_query_nodes,
            target_var_to_query_identifier,
            used_query_identifiers,
            budget,
        )
        if masks is None:
            return None

        refined = self._refine_single_graph_masks(
            stored,
            query_data,
            target_to_query,
            masks,
            budget,
            active_search_caches,
        )
        if refined is None:
            return None

        return self._search_single_graph_masks(
            stored,
            query_data,
            dict(target_to_query),
            dict(refined),
            target_var_to_query_identifier,
            used_query_identifiers,
            budget,
            active_search_caches,
        )

    def _enumerate_single_graph_matches(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        match_limit: int,
        budget: _OperationBudget | None = None,
        search_caches: _SearchCaches[N, V] | None = None,
    ) -> list[tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]]]:
        return list(
            self._iter_single_graph_matches(
                stored,
                query_data,
                target_to_query,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
                match_limit,
                budget,
                search_caches,
            )
        )

    def _iter_single_graph_matches(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        match_limit: int,
        budget: _OperationBudget | None = None,
        search_caches: _SearchCaches[N, V] | None = None,
    ) -> Iterator[tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]]]:
        if not self._partial_conditions_hold(stored, target_to_query, query_data):
            return
            yield

        active_search_caches = search_caches
        masks = self._initial_single_graph_masks(
            stored,
            query_data,
            target_to_query,
            used_query_nodes,
            target_var_to_query_identifier,
            used_query_identifiers,
            budget,
        )
        if masks is None:
            return
            yield

        refined = self._refine_single_graph_masks(
            stored,
            query_data,
            target_to_query,
            masks,
            budget,
            active_search_caches,
        )
        if refined is None:
            return
            yield

        yield from self._iter_search_single_graph_masks_collect(
            stored,
            query_data,
            dict(target_to_query),
            dict(refined),
            target_var_to_query_identifier,
            used_query_identifiers,
            match_limit,
            budget,
            active_search_caches,
        )

    def _solver_candidates_from_masks(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        masks: dict[CanonicalNode, int],
    ) -> tuple[list[CanonicalNode], list[N], dict[CanonicalNode, int], list[list[int]]] | None:
        order = list(stored.order)
        query_nodes = list(query_data.graph.nodes)
        target_position = {node: position for position, node in enumerate(order)}
        candidates = [
            list(_iter_mask_indices(masks[target_node]))
            for target_node in order
        ]
        if any(not target_candidates for target_candidates in candidates):
            return None
        return order, query_nodes, target_position, candidates

    def _refined_solver_candidates(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        budget: _OperationBudget | None = None,
    ) -> tuple[list[CanonicalNode], list[N], dict[CanonicalNode, int], list[list[int]]] | None:
        masks = self._initial_single_graph_masks(
            stored,
            query_data,
            {},
            set(),
            {},
            {},
            budget,
        )
        if masks is None:
            return None

        refined = self._refine_single_graph_masks(
            stored,
            query_data,
            {},
            masks,
            budget,
        )
        if refined is None:
            return None

        return self._solver_candidates_from_masks(stored, query_data, refined)

    def _fallback_anchor_targets(
        self,
        stored: _StoredGraph[N, L, V],
        masks: dict[CanonicalNode, int],
    ) -> tuple[CanonicalNode, ...]:
        product = 1
        anchors: list[CanonicalNode] = []
        for target_node in sorted(
            stored.order,
            key=lambda current_target: (
                masks[current_target].bit_count(),
                -(stored.in_degrees[current_target] + stored.out_degrees[current_target]),
                repr(current_target),
            ),
        ):
            domain_size = masks[target_node].bit_count()
            if domain_size <= 1:
                continue
            next_product = product * domain_size
            if anchors and next_product > 512:
                break
            anchors.append(target_node)
            product = next_product
            if len(anchors) >= 4:
                break
        return tuple(anchors)

    def _find_single_graph_match_anchored(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        try:
            budget = _OperationBudget(self._anchored_ops)
            masks = self._initial_single_graph_masks(
                stored,
                query_data,
                {},
                set(),
                {},
                {},
                budget,
            )
            if masks is None:
                return None

            refined = self._refine_single_graph_masks(
                stored,
                query_data,
                {},
                masks,
                budget,
            )
            if refined is None:
                return None

            anchor_targets = self._fallback_anchor_targets(stored, refined)
            if not anchor_targets:
                return None

            return self._search_anchored_fallback(
                stored,
                query_data,
                {},
                refined,
                {},
                {},
                anchor_targets,
                0,
                budget,
            )
        except _SearchTimeout:
            return None

    def _search_anchored_fallback(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        masks: dict[CanonicalNode, int],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        anchor_targets: tuple[CanonicalNode, ...],
        anchor_index: int,
        budget: _OperationBudget,
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        _spend(budget)

        if anchor_index >= len(anchor_targets):
            variable_state = self._copy_variable_state(target_var_to_query_identifier, used_query_identifiers)
            try:
                return self._search_single_graph_masks(
                    stored,
                    query_data,
                    dict(target_to_query),
                    dict(masks),
                    variable_state[0],
                    variable_state[1],
                    budget,
                )
            except _SearchTimeout:
                return None

        target_node = anchor_targets[anchor_index]
        if target_node in target_to_query:
            return self._search_anchored_fallback(
                stored,
                query_data,
                target_to_query,
                masks,
                target_var_to_query_identifier,
                used_query_identifiers,
                anchor_targets,
                anchor_index + 1,
                budget,
            )

        candidate_indices = list(_iter_mask_indices(masks[target_node]))
        candidate_indices.sort(
            key=lambda query_index: self._mask_candidate_order_key(
                stored,
                target_node,
                query_index,
                masks,
                query_data,
            ),
        )

        for query_index in candidate_indices:
            _spend(budget)
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
                budget,
            )
            if propagated is None:
                continue

            next_masks, next_target_var_to_query_identifier, next_used_query_identifiers = propagated
            match = self._search_anchored_fallback(
                stored,
                query_data,
                next_target_to_query,
                next_masks,
                next_target_var_to_query_identifier,
                next_used_query_identifiers,
                anchor_targets,
                anchor_index + 1,
                budget,
            )
            if match is not None:
                return match

        return None

    def _find_single_graph_match_z3(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        if not self._z3_enabled or z3 is None:
            return None

        solver_candidates = self._refined_solver_candidates(
            stored,
            query_data,
            _OperationBudget(self._native_ops),
        )
        if solver_candidates is None:
            return None
        return self._find_single_graph_match_z3_from_candidates(stored, query_data, solver_candidates)

    def _find_single_graph_match_z3_from_masks(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        masks: dict[CanonicalNode, int],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        if not self._z3_enabled or z3 is None:
            return None
        solver_candidates = self._solver_candidates_from_masks(stored, query_data, masks)
        if solver_candidates is None:
            return None
        return self._find_single_graph_match_z3_from_candidates(stored, query_data, solver_candidates)

    def _find_single_graph_match_z3_from_candidates(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        solver_candidates: tuple[list[CanonicalNode], list[N], dict[CanonicalNode, int], list[list[int]]],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        z3m: Any = z3
        order, query_nodes, target_position, candidates = solver_candidates

        solver = z3m.Solver()
        solver.set(rlimit=self._z3_rlimit)

        assignment_variables: dict[tuple[int, int], Any] = {
            (target_index, query_index): z3m.Bool(f"x_{target_index}_{query_index}")
            for target_index, target_candidates in enumerate(candidates)
            for query_index in target_candidates
        }

        for target_index, target_candidates in enumerate(candidates):
            solver.add(z3m.PbEq([(assignment_variables[(target_index, query_index)], 1) for query_index in target_candidates], 1))

        for query_index in range(len(query_nodes)):
            uses = [
                assignment_variables[(target_index, query_index)]
                for target_index, target_candidates in enumerate(candidates)
                if query_index in target_candidates
            ]
            if uses:
                solver.add(z3m.PbLe([(query_use, 1) for query_use in uses], 1))

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
                        z3m.Or(
                            z3m.Not(assignment_variables[(source_index, source_query_index)]),
                            z3m.Not(assignment_variables[(target_index, target_query_index)]),
                        )
                    )

        variables_by_class: dict[VariableClass, dict[CanonicalVariable, list[int]]] = {}
        for target_index, target_node in enumerate(order):
            for variable_class, identifier in enumerate(stored.variables[target_node]):
                variables_by_class.setdefault(variable_class, {}).setdefault(identifier, []).append(target_index)

        for variable_class, target_groups in variables_by_class.items():
            query_identifiers = sorted(
                {
                    query_data.variables[query_nodes[query_index]][variable_class]
                    for target_indices in target_groups.values()
                    for target_index in target_indices
                    for query_index in candidates[target_index]
                },
                key=repr,
            )
            identifier_variables = {
                (identifier, query_identifier): z3m.Bool(
                    f"y_{repr(variable_class)}_{repr(identifier)}_{repr(query_identifier)}"
                )
                for identifier in target_groups
                for query_identifier in query_identifiers
            }
            for identifier in target_groups:
                solver.add(
                    z3m.PbEq(
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
                            query_identifier = query_data.variables[query_nodes[query_index]][variable_class]
                            solver.add(
                                z3m.Implies(
                                    assignment_variables[(target_index, query_index)],
                                identifier_variables[(identifier, query_identifier)],
                            )
                        )

        if solver.check() != z3m.sat:
            return None
        model = solver.model()

        node_mapping = {
            target_node: query_nodes[query_index]
            for target_index, target_node in enumerate(order)
            for query_index in candidates[target_index]
            if z3m.is_true(model.evaluate(assignment_variables[(target_index, query_index)], model_completion=True))
        }
        if len(node_mapping) != len(order):
            return None

        ordered_query_nodes = [node_mapping[target_node] for target_node in order]
        return node_mapping, self._variable_mapping(stored, query_data, ordered_query_nodes)

    def _find_single_graph_match_ortools(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        if not self._ortools_enabled or cp_model is None:
            return None

        solver_candidates = self._refined_solver_candidates(
            stored,
            query_data,
            _OperationBudget(self._native_ops),
        )
        if solver_candidates is None:
            return None
        return self._find_single_graph_match_ortools_from_candidates(stored, query_data, solver_candidates)

    def _find_single_graph_match_ortools_from_masks(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        masks: dict[CanonicalNode, int],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        if not self._ortools_enabled or cp_model is None:
            return None
        solver_candidates = self._solver_candidates_from_masks(stored, query_data, masks)
        if solver_candidates is None:
            return None
        return self._find_single_graph_match_ortools_from_candidates(stored, query_data, solver_candidates)

    def _find_single_graph_match_ortools_from_candidates(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        solver_candidates: tuple[list[CanonicalNode], list[N], dict[CanonicalNode, int], list[list[int]]],
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        cp: Any = cp_model
        order, query_nodes, target_position, candidates = solver_candidates

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

        variables_by_class: dict[VariableClass, dict[CanonicalVariable, list[int]]] = {}
        for target_index, target_node in enumerate(order):
            for variable_class, identifier in enumerate(stored.variables[target_node]):
                variables_by_class.setdefault(variable_class, {}).setdefault(identifier, []).append(target_index)

        for variable_class, target_groups in variables_by_class.items():
            query_identifiers = sorted(
                {
                    query_data.variables[query_nodes[query_index]][variable_class]
                    for target_indices in target_groups.values()
                    for target_index in target_indices
                    for query_index in candidates[target_index]
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
                            query_identifier = query_data.variables[query_nodes[query_index]][variable_class]
                            model.AddImplication(
                                assignment_variables[(target_index, query_index)],
                                identifier_variables[(identifier, query_identifier)],
                        )

        solver: Any = cp.CpSolver()
        solver.parameters.max_deterministic_time = self._ortools_deterministic_time
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
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        masks: dict[CanonicalNode, int],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        budget: _OperationBudget | None = None,
        search_caches: _SearchCaches[N, V] | None = None,
    ) -> tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]] | None:
        state_signature: tuple[Any, ...] | None = None
        if search_caches is not None:
            state_signature = self._state_signature(
                target_to_query,
                masks,
                target_var_to_query_identifier,
            )
            if state_signature in search_caches.dead_single_states:
                return None

        _spend(budget)
        if len(target_to_query) == len(stored.order):
            ordered_query_nodes = [target_to_query[target_node] for target_node in stored.order]
            if not _conditions_hold(stored.full_conditions, ordered_query_nodes, query_data.node_ranks):
                if search_caches is not None and state_signature is not None:
                    search_caches.dead_single_states.add(state_signature)
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

        candidate_indices: Iterator[int]
        if budget is None:
            candidate_indices = iter(
                sorted(
                    _iter_mask_indices(masks[target_node]),
                    key=lambda query_index: self._mask_candidate_order_key(stored, target_node, query_index, masks, query_data),
                )
            )
        else:
            candidate_indices = _iter_mask_indices(masks[target_node])

        for query_index in candidate_indices:
            _spend(budget)
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
                budget,
                search_caches,
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
                budget,
                search_caches,
            )
            if match is not None:
                return match

        if search_caches is not None and state_signature is not None:
            search_caches.dead_single_states.add(state_signature)
        return None

    def _iter_search_single_graph_masks_collect(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        masks: dict[CanonicalNode, int],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        match_limit: int,
        budget: _OperationBudget | None = None,
        search_caches: _SearchCaches[N, V] | None = None,
    ) -> Iterator[tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]]]:
        _spend(budget)
        if len(target_to_query) == len(stored.order):
            ordered_query_nodes = [target_to_query[target_node] for target_node in stored.order]
            if _conditions_hold(stored.full_conditions, ordered_query_nodes, query_data.node_ranks):
                yield target_to_query.copy(), self._variable_mapping(stored, query_data, ordered_query_nodes)
            return

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

        candidate_indices = sorted(
            _iter_mask_indices(masks[target_node]),
            key=lambda query_index: self._mask_candidate_order_key(stored, target_node, query_index, masks, query_data),
        )
        emitted_count = 0
        for query_index in candidate_indices:
            if emitted_count >= match_limit:
                return
            _spend(budget)
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
                budget,
                search_caches,
            )
            if propagated is None:
                continue

            next_masks, next_target_var_to_query_identifier, next_used_query_identifiers = propagated
            for match in self._iter_search_single_graph_masks_collect(
                stored,
                query_data,
                next_target_to_query,
                next_masks,
                next_target_var_to_query_identifier,
                next_used_query_identifiers,
                match_limit,
                budget,
                search_caches,
            ):
                yield match
                emitted_count += 1
                if emitted_count >= match_limit:
                    return

    def _dynamic_label_candidates(
        self,
        stored: _StoredGraph[N, L, V],
        target_node: CanonicalNode,
        query_data: _QueryData[N, L, V],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> frozenset[N]:
        del used_query_identifiers
        candidates = query_data.constant_nodes.get(stored.labels[target_node], frozenset())
        for variable_class, target_identifier in enumerate(stored.variables[target_node]):
            existing = target_var_to_query_identifier.get(variable_class, {}).get(target_identifier)
            if existing is None:
                continue
            candidates = candidates & query_data.variable_identifier_nodes.get((variable_class, existing), frozenset())
            if not candidates:
                return frozenset()
        return candidates

    def _dynamic_label_compatible(
        self,
        stored: _StoredGraph[N, L, V],
        target_node: CanonicalNode,
        query_node: N,
        query_data: _QueryData[N, L, V],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
    ) -> bool:
        del used_query_identifiers
        if stored.labels[target_node] != query_data.labels[query_node]:
            return False
        target_variables = stored.variables[target_node]
        query_variables = query_data.variables[query_node]
        if len(target_variables) != len(query_variables):
            return False
        for variable_class, target_identifier in enumerate(target_variables):
            existing = target_var_to_query_identifier.get(variable_class, {}).get(target_identifier)
            if existing is not None and query_variables[variable_class] != existing:
                return False
        return True

    def _dynamic_candidates_for_target(
        self,
        stored: _StoredGraph[N, L, V],
        target_node: CanonicalNode,
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
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
        stored: _StoredGraph[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        use_lookahead: bool,
    ) -> tuple[CanonicalNode | None, list[N]]:
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

        best_target: CanonicalNode | None = None
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
            for query_node in best_candidates:
                target_to_query[best_target] = query_node
                used_query_nodes.add(query_node)

                added_mappings: list[tuple[VariableClass, CanonicalVariable]] = []
                for variable_class, target_identifier in enumerate(stored.variables[best_target]):
                    query_identifier = query_data.variables[query_node][variable_class]
                    variable_mapping = target_var_to_query_identifier.setdefault(variable_class, {})
                    if target_identifier not in variable_mapping:
                        variable_mapping[target_identifier] = query_identifier
                        added_mappings.append((variable_class, target_identifier))

                next_key = self._dynamic_next_choice_key(
                    stored,
                    target_to_query,
                    query_data,
                    used_query_nodes,
                    target_var_to_query_identifier,
                    used_query_identifiers,
                )
                candidate_scores.append((next_key, query_node))

                for variable_class, target_identifier in reversed(added_mappings):
                    del target_var_to_query_identifier[variable_class][target_identifier]
                    if not target_var_to_query_identifier[variable_class]:
                        del target_var_to_query_identifier[variable_class]

                used_query_nodes.remove(query_node)
                del target_to_query[best_target]

            candidate_scores.sort(key=lambda item: (item[0], repr(item[1])))
            ordered_candidates = [query_node for _, query_node in candidate_scores]

        return best_target, ordered_candidates

    def _complete_single_graph_dynamic(
        self,
        stored: _StoredGraph[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
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

        for query_node in ordered_candidates:
            target_to_query[best_target] = query_node
            used_query_nodes.add(query_node)

            added_mappings: list[tuple[VariableClass, CanonicalVariable]] = []
            for variable_class, target_identifier in enumerate(stored.variables[best_target]):
                query_identifier = query_data.variables[query_node][variable_class]
                variable_mapping = target_var_to_query_identifier.setdefault(variable_class, {})
                if target_identifier not in variable_mapping:
                    variable_mapping[target_identifier] = query_identifier
                    added_mappings.append((variable_class, target_identifier))

            if self._complete_single_graph_dynamic(
                stored,
                target_to_query,
                query_data,
                used_query_nodes,
                target_var_to_query_identifier,
                used_query_identifiers,
            ):
                return True

            for variable_class, target_identifier in reversed(added_mappings):
                del target_var_to_query_identifier[variable_class][target_identifier]
                if not target_var_to_query_identifier[variable_class]:
                    del target_var_to_query_identifier[variable_class]

            used_query_nodes.remove(query_node)
            del target_to_query[best_target]

        return False

    def _enumerate_single_graph_dynamic(
        self,
        stored: _StoredGraph[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
        used_query_identifiers: dict[VariableClass, set[V]],
        results: list[tuple[CanonicalNodeMapping[N], CanonicalVariableMapping[V]]],
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

        for query_node in ordered_candidates:
            target_to_query[best_target] = query_node
            used_query_nodes.add(query_node)

            added_mappings: list[tuple[VariableClass, CanonicalVariable]] = []
            for variable_class, target_identifier in enumerate(stored.variables[best_target]):
                query_identifier = query_data.variables[query_node][variable_class]
                variable_mapping = target_var_to_query_identifier.setdefault(variable_class, {})
                if target_identifier not in variable_mapping:
                    variable_mapping[target_identifier] = query_identifier
                    added_mappings.append((variable_class, target_identifier))

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

            for variable_class, target_identifier in reversed(added_mappings):
                del target_var_to_query_identifier[variable_class][target_identifier]
                if not target_var_to_query_identifier[variable_class]:
                    del target_var_to_query_identifier[variable_class]

            used_query_nodes.remove(query_node)
            del target_to_query[best_target]

            if stop:
                return True

        return False

    def _dynamic_next_choice_key(
        self,
        stored: _StoredGraph[N, L, V],
        target_to_query: CanonicalNodeMapping[N],
        query_data: _QueryData[N, L, V],
        used_query_nodes: set[N],
        target_var_to_query_identifier: CanonicalVariableMapping[V],
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
        stored: _StoredGraph[N, L, V],
        position: int,
        query_data: _QueryData[N, L, V],
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
        query_data: _QueryData[N, L, V],
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
        query_data: _QueryData[N, L, V],
    ) -> AbstractSet[N]:
        anchored: AbstractSet[N] | None = None

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
        query_data: _QueryData[N, L, V],
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
        label_pattern: _LabelPattern[L],
        matched_query_nodes: list[N],
        query_data: _QueryData[N, L, V],
    ) -> frozenset[N]:
        candidates = query_data.constant_nodes.get(label_pattern.node_label, frozenset())
        for variable_class, repeated_from in enumerate(label_pattern.repeated_from):
            if repeated_from is None or repeated_from >= len(matched_query_nodes):
                continue
            repeated_query_node = matched_query_nodes[repeated_from]
            repeated_identifier = query_data.variables[repeated_query_node][variable_class]
            candidates = candidates & query_data.variable_identifier_nodes.get(
                (variable_class, repeated_identifier),
                frozenset(),
            )
            if not candidates:
                return frozenset()
        return candidates

    def _label_pattern_compatible(
        self,
        label_pattern: _LabelPattern[L],
        matched_query_nodes: list[N],
        query_data: _QueryData[N, L, V],
        candidate: N,
    ) -> bool:
        if query_data.labels[candidate] != label_pattern.node_label:
            return False
        candidate_variables = query_data.variables[candidate]
        if len(candidate_variables) != len(label_pattern.repeated_from):
            return False
        for variable_class, repeated_from in enumerate(label_pattern.repeated_from):
            if repeated_from is None:
                continue
            if repeated_from >= len(matched_query_nodes):
                continue
            repeated_identifier = query_data.variables[matched_query_nodes[repeated_from]][variable_class]
            if candidate_variables[variable_class] != repeated_identifier:
                return False
        return True

    def _future_requirement_key(
        self,
        label_pattern: _LabelPattern[L],
        prefix_query_nodes: list[N],
        query_data: _QueryData[N, L, V],
    ) -> tuple[Any, ...]:
        required_identifiers = tuple(
            (
                variable_class,
                query_data.variables[prefix_query_nodes[repeated_from]][variable_class],
            )
            for variable_class, repeated_from in enumerate(label_pattern.repeated_from)
            if repeated_from is not None and repeated_from < len(prefix_query_nodes)
        )
        return ("label", label_pattern.node_label, len(label_pattern.repeated_from), required_identifiers)

    def _neighbor_matches_requirement(
        self,
        requirement: tuple[Any, ...],
        query_node: N,
        query_data: _QueryData[N, L, V],
    ) -> bool:
        kind, required_label, variable_count, required_identifiers = requirement
        if kind != "label":
            return False
        if query_data.labels[query_node] != required_label:
            return False
        query_variables = query_data.variables[query_node]
        if len(query_variables) != variable_count:
            return False
        return all(query_variables[variable_class] == identifier for variable_class, identifier in required_identifiers)

    def _future_neighbor_feasible(
        self,
        stored: _StoredGraph[N, L, V],
        position: int,
        candidate: N,
        query_data: _QueryData[N, L, V],
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

    def _can_match(self, stored: _StoredGraph[N, L, V], query_data: _QueryData[N, L, V]) -> bool:
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
            candidates = query_data.constant_nodes.get(target_label, frozenset())

            if not any(
                query_data.in_degrees[query_node] >= target_in_degree
                and query_data.out_degrees[query_node] >= target_out_degree
                and len(stored.variables[target_node]) == len(query_data.variables[query_node])
                for query_node in candidates
            ):
                return False

        return True

    def _variable_mapping(
        self,
        stored: _StoredGraph[N, L, V],
        query_data: _QueryData[N, L, V],
        matched_query_nodes: list[N],
    ) -> CanonicalVariableMapping[V]:
        variable_mapping: defaultdict[VariableClass, dict[CanonicalVariable, V]] = defaultdict(dict)

        for target_node, query_node in zip(stored.order, matched_query_nodes, strict=True):
            if stored.labels[target_node] != query_data.labels[query_node]:
                raise ValueError("Matched node labels disagree")
            target_variables = stored.variables[target_node]
            query_variables = query_data.variables[query_node]
            if len(target_variables) != len(query_variables):
                raise ValueError("Matched node variable counts disagree")
            for variable_class, target_identifier in enumerate(target_variables):
                query_identifier = query_variables[variable_class]
                existing = variable_mapping[variable_class].get(target_identifier)
                if existing is not None and existing != query_identifier:
                    raise ValueError("Matched target variable maps to multiple query identifiers")
                variable_mapping[variable_class][target_identifier] = query_identifier

        return dict(variable_mapping)
