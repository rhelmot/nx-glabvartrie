from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Hashable

import networkx as nx

from .common import (
    I,
    L,
    N,
    V,
    TopologyPattern,
    _default_label_matches,
    _default_node_order_key,
    _stable_value_key,
    graphs_are_isomorphic,
    topologies_are_isomorphic,
    topology_patterns_for_order,
)


@dataclass(slots=True, frozen=True)
class _OccurrenceRef(Generic[N, I]):
    source: I
    nodeset: frozenset[N]


@dataclass(slots=True)
class _MotifClass(Generic[N, I]):
    representative: _OccurrenceRef[N, I]
    occurrences: list[tuple[I, tuple[N, ...]]] = field(default_factory=list)
    occurrence_keys: set[tuple[I, frozenset[N]]] = field(default_factory=set)


@dataclass(slots=True)
class _MotifTrieNode(Generic[N, I]):
    depth: int
    topology_pattern: TopologyPattern | None
    children: dict[TopologyPattern, _MotifTrieNode[N, I]] = field(default_factory=dict)
    motif_classes: list[_MotifClass[N, I]] = field(default_factory=list)


@dataclass(slots=True)
class _TopologyClass(Generic[N, I]):
    representative: _OccurrenceRef[N, I]
    terminal: _MotifTrieNode[N, I] | None = None
    motif_classes: list[_MotifClass[N, I]] = field(default_factory=list)
    motif_classes_by_hash: dict[Hashable, list[_MotifClass[N, I]]] = field(default_factory=lambda: defaultdict(list))


@dataclass(slots=True)
class _ParentSamples(Generic[N]):
    graph: nx.DiGraph[N]
    nodesets: frozenset[frozenset[N]]
    subset_parent_cache: dict[frozenset[N], frozenset[N] | None] = field(default_factory=dict)
    degree_cache: dict[frozenset[N], dict[N, tuple[int, int, bool]]] = field(default_factory=dict)
    subgraph_cache: dict[frozenset[N], nx.DiGraph[N]] = field(default_factory=dict)


class MotifFinder(Generic[N, L, V, I]):
    def __init__(
        self,
        samples: Mapping[I, tuple[nx.DiGraph[N], frozenset[frozenset[N]]]],
        node_label: Callable[[dict[str, Any]], L],
        node_vars: Callable[[dict[str, Any]], tuple[V, ...]],
        *,
        node_order_key: Callable[[N], Any] | None = None,
        label_matches: Callable[[L, L], bool] | None = None,
        max_motif_size: int | None = None,
    ) -> None:
        self._node_label = node_label
        self._node_vars = node_vars
        self._node_order_key = node_order_key or _default_node_order_key  # type: ignore[assignment]
        self._label_matches = label_matches or _default_label_matches
        self._default_label_matches = label_matches is None or label_matches is _default_label_matches
        self._max_motif_size = max_motif_size
        self._root: _MotifTrieNode[N, I] = _MotifTrieNode(depth=0, topology_pattern=None)
        self._motif_classes_by_size: dict[int, list[_MotifClass[N, I]]] = defaultdict(list)
        self._parents: dict[I, _ParentSamples[N]] = {}

        ordered_occurrences: list[_OccurrenceRef[N, I]] = []
        for source, (graph, nodesets) in sorted(
            samples.items(),
            key=lambda item: _stable_value_key(item[0]),
        ):
            filtered_nodesets = frozenset(
                nodeset
                for nodeset in nodesets
                if nodeset
                and (self._max_motif_size is None or len(nodeset) <= self._max_motif_size)
            )
            self._parents[source] = _ParentSamples(graph=graph, nodesets=filtered_nodesets)
            ordered_occurrences.extend(
                _OccurrenceRef(source=source, nodeset=nodeset)
                for nodeset in filtered_nodesets
            )

        ordered_occurrences.sort(
            key=lambda occurrence: (
                len(occurrence.nodeset),
                _stable_value_key(occurrence.source),
                tuple(sorted(occurrence.nodeset, key=self._node_order_key)),
            )
        )

        topology_classes_by_size: dict[int, dict[Hashable, list[_TopologyClass[N, I]]]] = defaultdict(lambda: defaultdict(list))

        for occurrence in ordered_occurrences:
            sample_size = len(occurrence.nodeset)
            sample_graph = self._subgraph(occurrence)

            topology_hash = self._topology_hash(occurrence)
            topology_classes = topology_classes_by_size[sample_size][topology_hash]
            topology_class: _TopologyClass[N, I] | None = None
            for candidate in topology_classes:
                if topologies_are_isomorphic(self._subgraph(candidate.representative), sample_graph):
                    topology_class = candidate
                    break

            if topology_class is None:
                topology_class = _TopologyClass(representative=occurrence)
                topology_classes.append(topology_class)

            labeled_hash = self._labeled_hash(occurrence)
            motif_class_bucket = topology_class.motif_classes_by_hash[labeled_hash]
            occurrence_nodes = tuple(sorted(occurrence.nodeset, key=self._node_order_key))
            occurrence_key = (occurrence.source, occurrence.nodeset)
            for motif_class in motif_class_bucket:
                if graphs_are_isomorphic(
                    self._subgraph(motif_class.representative),
                    sample_graph,
                    self._node_label,
                    self._node_vars,
                    self._label_matches,
                ):
                    if occurrence_key not in motif_class.occurrence_keys:
                        motif_class.occurrences.append((occurrence.source, occurrence_nodes))
                        motif_class.occurrence_keys.add(occurrence_key)
                    break
            else:
                motif_class = _MotifClass(
                    representative=occurrence,
                    occurrences=[(occurrence.source, occurrence_nodes)],
                    occurrence_keys={occurrence_key},
                )
                motif_class_bucket.append(motif_class)
                topology_class.motif_classes.append(motif_class)
                self._motif_classes_by_size[sample_size].append(motif_class)

        self._topology_classes_by_size = topology_classes_by_size
        self._trie_built = False

        self._motifs_by_size: dict[int, tuple[tuple[tuple[I, tuple[N, ...]], ...], ...]] = {}
        for size, motif_classes in self._motif_classes_by_size.items():
            self._motifs_by_size[size] = tuple(
                tuple(sorted(motif_class.occurrences, key=lambda occurrence: (_stable_value_key(occurrence[0]), occurrence[1])))
                for motif_class in motif_classes
            )

    def _ensure_trie(self) -> None:
        if self._trie_built:
            return

        self._root = _MotifTrieNode(depth=0, topology_pattern=None)
        for topology_classes_by_hash in self._topology_classes_by_size.values():
            for topology_classes in topology_classes_by_hash.values():
                for topology_class in topology_classes:
                    topology_signatures = self._topology_node_signatures(topology_class.representative)
                    terminal = self._insert_topology(topology_class.representative, topology_signatures)
                    terminal.motif_classes.extend(topology_class.motif_classes)
                    topology_class.terminal = terminal
        self._trie_built = True

    def _subgraph(self, occurrence: _OccurrenceRef[N, I]) -> nx.DiGraph[N]:
        parent = self._parents[occurrence.source]
        cached = parent.subgraph_cache.get(occurrence.nodeset)
        if cached is not None:
            return cached
        subgraph = parent.graph.subgraph(occurrence.nodeset)
        assert isinstance(subgraph, nx.DiGraph)
        parent.subgraph_cache[occurrence.nodeset] = subgraph
        return subgraph

    def _subset_parent(self, occurrence: _OccurrenceRef[N, I]) -> frozenset[N] | None:
        parent = self._parents[occurrence.source]
        cached = parent.subset_parent_cache.get(occurrence.nodeset)
        if cached is not None or occurrence.nodeset in parent.subset_parent_cache:
            return cached
        if len(occurrence.nodeset) <= 1:
            parent.subset_parent_cache[occurrence.nodeset] = None
            return None

        ordered_nodes = tuple(sorted(occurrence.nodeset, key=self._node_order_key))
        for node in ordered_nodes:
            candidate = occurrence.nodeset.difference((node,))
            if candidate in parent.nodesets:
                parent.subset_parent_cache[occurrence.nodeset] = candidate
                return candidate
        parent.subset_parent_cache[occurrence.nodeset] = None
        return None

    def _degree_signatures(self, occurrence: _OccurrenceRef[N, I]) -> dict[N, tuple[int, int, bool]]:
        parent = self._parents[occurrence.source]
        cached = parent.degree_cache.get(occurrence.nodeset)
        if cached is not None:
            return cached

        graph = parent.graph
        parent_nodeset = self._subset_parent(occurrence)
        if parent_nodeset is None:
            degrees = {
                node: (
                    sum(1 for predecessor in graph.pred[node] if predecessor in occurrence.nodeset),
                    sum(1 for successor in graph.succ[node] if successor in occurrence.nodeset),
                    graph.has_edge(node, node),
                )
                for node in occurrence.nodeset
            }
            parent.degree_cache[occurrence.nodeset] = degrees
            return degrees

        subset_occurrence = _OccurrenceRef(source=occurrence.source, nodeset=parent_nodeset)
        degrees = dict(self._degree_signatures(subset_occurrence))
        added_node = next(iter(occurrence.nodeset - parent_nodeset))
        in_degree = 0
        out_degree = 0
        self_loop = graph.has_edge(added_node, added_node)
        if self_loop:
            in_degree += 1
            out_degree += 1

        for neighbour in parent_nodeset:
            neighbour_in, neighbour_out, neighbour_loop = degrees[neighbour]
            if graph.has_edge(neighbour, added_node):
                neighbour_out += 1
                in_degree += 1
            if graph.has_edge(added_node, neighbour):
                neighbour_in += 1
                out_degree += 1
            degrees[neighbour] = (neighbour_in, neighbour_out, neighbour_loop)

        degrees[added_node] = (in_degree, out_degree, self_loop)
        parent.degree_cache[occurrence.nodeset] = degrees
        return degrees

    def _insert_topology(
        self,
        occurrence: _OccurrenceRef[N, I],
        topology_signatures: dict[N, tuple[Any, ...]],
    ) -> _MotifTrieNode[N, I]:
        graph = self._subgraph(occurrence)
        topology_order = self._topology_insertion_order(occurrence, topology_signatures)
        topology_patterns = topology_patterns_for_order(graph, topology_order)
        node = self._root
        for depth, topology_pattern in enumerate(topology_patterns, start=1):
            child = node.children.get(topology_pattern)
            if child is None:
                child = _MotifTrieNode(depth=depth, topology_pattern=topology_pattern)
                node.children[topology_pattern] = child
            node = child
        return node

    def _topology_insertion_order(
        self,
        occurrence: _OccurrenceRef[N, I],
        topology_signatures: dict[N, tuple[Any, ...]],
    ) -> tuple[N, ...]:
        nodes = tuple(occurrence.nodeset)
        if len(nodes) <= 1:
            return nodes

        ordered_nodes = tuple(sorted(nodes, key=self._node_order_key))
        rank_by_node = {node: position for position, node in enumerate(ordered_nodes)}
        return tuple(
            sorted(
                nodes,
                key=lambda node: (
                    topology_signatures[node],
                    rank_by_node[node],
                ),
            )
        )

    def _topology_node_signatures(self, occurrence: _OccurrenceRef[N, I]) -> dict[N, tuple[Any, ...]]:
        graph = self._parents[occurrence.source].graph
        base = self._degree_signatures(occurrence)
        return {
            node: (
                base[node],
                tuple(sorted(base[neighbor] for neighbor in graph.pred[node] if neighbor in occurrence.nodeset)),
                tuple(sorted(base[neighbor] for neighbor in graph.succ[node] if neighbor in occurrence.nodeset)),
            )
            for node in occurrence.nodeset
        }

    def _topology_hash(self, occurrence: _OccurrenceRef[N, I]) -> Hashable:
        degrees = self._degree_signatures(occurrence)
        return (
            len(occurrence.nodeset),
            sum(out_degree for _, out_degree, _ in degrees.values()),
            tuple(sorted(degrees.values())),
        )

    def _labeled_hash(self, occurrence: _OccurrenceRef[N, I]) -> Hashable:
        graph = self._parents[occurrence.source].graph
        degrees = self._degree_signatures(occurrence)
        node_signatures: list[tuple[Any, ...]] = []
        for node in occurrence.nodeset:
            variables = self._node_vars(graph.nodes[node])
            local_variable_pattern: dict[V, int] = {}
            in_degree, out_degree, self_loop = degrees[node]
            node_signatures.append(
                (
                    self._node_label(graph.nodes[node]) if self._default_label_matches else None,
                    tuple(local_variable_pattern.setdefault(identifier, len(local_variable_pattern)) for identifier in variables),
                    len(variables),
                    in_degree,
                    out_degree,
                    self_loop,
                )
            )
        return tuple(sorted(node_signatures))

    def motifs(self, size: int | None = None, *, descending: bool = False) -> Iterator[list[tuple[I, tuple[N, ...]]]]:
        if size is not None:
            for motif_class in self._motifs_by_size.get(size, ()):
                yield list(motif_class)
            return

        for current_size in sorted(self._motifs_by_size, reverse=descending):
            for motif_class in self._motifs_by_size[current_size]:
                yield list(motif_class)
