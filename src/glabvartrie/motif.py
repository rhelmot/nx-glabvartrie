from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
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
    graphs_are_isomorphic,
    topologies_are_isomorphic,
    topology_patterns_for_order,
)


@dataclass(slots=True)
class _Representative:
    graph: nx.DiGraph[Any]


@dataclass(slots=True)
class _MotifClass(Generic[N, I]):
    representative: _Representative
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
    representative: _Representative
    terminal: _MotifTrieNode[N, I] | None = None
    motif_classes: list[_MotifClass[N, I]] = field(default_factory=list)
    motif_classes_by_hash: dict[Hashable, list[_MotifClass[N, I]]] = field(default_factory=lambda: defaultdict(list))


class MotifFinder(Generic[N, L, V, I]):
    def __init__(
        self,
        samples: Iterable[tuple[nx.DiGraph[N], I]],
        key: Callable[[tuple[nx.DiGraph[N], I]], Any],
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

        ordered_samples = sorted(
            samples,
            key=lambda sample: (
                key(sample),
                sample[0].number_of_nodes(),
                sample[0].number_of_edges(),
                sample[1],
                tuple(sorted(sample[0].nodes, key=self._node_order_key)),
            ),
        )

        seen_occurrences: set[tuple[I, frozenset[N]]] = set()
        topology_classes_by_size: dict[int, dict[Hashable, list[_TopologyClass[N, I]]]] = defaultdict(lambda: defaultdict(list))

        for sample_graph, source in ordered_samples:
            sample_size = sample_graph.number_of_nodes()
            if sample_size == 0:
                continue
            if self._max_motif_size is not None and sample_size > self._max_motif_size:
                continue

            occurrence_nodes = tuple(sorted(sample_graph.nodes, key=self._node_order_key))
            occurrence_key = (source, frozenset(occurrence_nodes))
            if occurrence_key in seen_occurrences:
                continue
            seen_occurrences.add(occurrence_key)

            topology_hash = self._topology_hash(sample_graph)
            topology_classes = topology_classes_by_size[sample_size][topology_hash]
            topology_class: _TopologyClass[N, I] | None = None
            for candidate in topology_classes:
                if topologies_are_isomorphic(candidate.representative.graph, sample_graph):
                    topology_class = candidate
                    break

            if topology_class is None:
                topology_signatures = self._topology_node_signatures(sample_graph)
                topology_class = _TopologyClass(
                    representative=_Representative(sample_graph),
                )
                topology_classes.append(topology_class)

            labeled_hash = self._labeled_hash(sample_graph)
            motif_class_bucket = topology_class.motif_classes_by_hash[labeled_hash]
            for motif_class in motif_class_bucket:
                if graphs_are_isomorphic(
                    motif_class.representative.graph,
                    sample_graph,
                    self._node_label,
                    self._node_vars,
                    self._label_matches,
                ):
                    if occurrence_key not in motif_class.occurrence_keys:
                        motif_class.occurrences.append((source, occurrence_nodes))
                        motif_class.occurrence_keys.add(occurrence_key)
                    break
            else:
                motif_class = _MotifClass(
                    representative=_Representative(sample_graph),
                    occurrences=[(source, occurrence_nodes)],
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
                tuple(
                    sorted(
                        motif_class.occurrences,
                        key=lambda occurrence: (occurrence[0], occurrence[1]),
                    )
                )
                for motif_class in motif_classes
            )

    def _ensure_trie(self) -> None:
        if self._trie_built:
            return

        self._root = _MotifTrieNode(depth=0, topology_pattern=None)
        for topology_classes_by_hash in self._topology_classes_by_size.values():
            for topology_classes in topology_classes_by_hash.values():
                for topology_class in topology_classes:
                    topology_signatures = self._topology_node_signatures(topology_class.representative.graph)
                    terminal = self._insert_topology(topology_class.representative.graph, topology_signatures)
                    terminal.motif_classes.extend(topology_class.motif_classes)
                    topology_class.terminal = terminal
        self._trie_built = True

    def _insert_topology(
        self,
        graph: nx.DiGraph[N],
        topology_signatures: dict[N, tuple[Any, ...]],
    ) -> _MotifTrieNode[N, I]:
        topology_order = self._topology_insertion_order(graph, topology_signatures)
        topology_patterns = topology_patterns_for_order(
            graph,
            topology_order,
        )
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
        graph: nx.DiGraph[N],
        topology_signatures: dict[N, tuple[Any, ...]],
    ) -> tuple[N, ...]:
        nodes = tuple(graph.nodes)
        if len(nodes) <= 1:
            return nodes

        ordered_nodes = tuple(sorted(nodes, key=self._node_order_key))
        rank_by_node = {
            node: position
            for position, node in enumerate(ordered_nodes)
        }
        return tuple(
            sorted(
                nodes,
                key=lambda node: (
                    topology_signatures[node],
                    rank_by_node[node],
                ),
            )
        )

    def _topology_node_signatures(self, graph: nx.DiGraph[N]) -> dict[N, tuple[Any, ...]]:
        base = {
            node: (
                graph.in_degree(node),
                graph.out_degree(node),
                graph.has_edge(node, node),
            )
            for node in graph.nodes
        }
        return {
            node: (
                base[node],
                tuple(sorted(base[neighbor] for neighbor in graph.pred[node])),
                tuple(sorted(base[neighbor] for neighbor in graph.succ[node])),
            )
            for node in graph.nodes
        }

    def _topology_hash(self, graph: nx.DiGraph[N]) -> Hashable:
        return (
            graph.number_of_nodes(),
            graph.number_of_edges(),
            tuple(
                sorted(
                    (
                        graph.in_degree(node),
                        graph.out_degree(node),
                        graph.has_edge(node, node),
                    )
                    for node in graph.nodes
                )
            ),
        )

    def _labeled_hash(self, graph: nx.DiGraph[N]) -> Hashable:
        node_signatures: list[tuple[Any, ...]] = []
        for node in graph.nodes:
            variables = self._node_vars(graph.nodes[node])
            local_variable_pattern: dict[V, int] = {}
            node_signatures.append(
                (
                    self._node_label(graph.nodes[node]) if self._default_label_matches else None,
                    tuple(local_variable_pattern.setdefault(identifier, len(local_variable_pattern)) for identifier in variables),
                    len(variables),
                    graph.in_degree(node),
                    graph.out_degree(node),
                    graph.has_edge(node, node),
                )
            )
        return tuple(sorted(node_signatures))

    def motifs(self, size: int) -> Iterator[list[tuple[I, tuple[N, ...]]]]:
        for motif_class in self._motifs_by_size.get(size, ()):
            yield list(motif_class)
