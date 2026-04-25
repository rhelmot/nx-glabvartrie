from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
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
    graph_isomorphism_mapping,
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
    representative_order: tuple[N, ...]
    occurrences: list[tuple[I, tuple[N, ...]]] = field(default_factory=list)
    occurrence_keys: set[tuple[I, frozenset[N]]] = field(default_factory=set)
    occurrence_mappings: dict[tuple[I, frozenset[N]], tuple[N, ...]] = field(default_factory=dict)


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
    occurrences: list[_OccurrenceRef[N, I]] = field(default_factory=list)
    motif_classes: list[_MotifClass[N, I]] = field(default_factory=list)
    motif_classes_by_hash: dict[Hashable, list[_MotifClass[N, I]]] = field(default_factory=lambda: defaultdict(list))


@dataclass(slots=True)
class _ParentSamples(Generic[N]):
    graph: nx.DiGraph[N]
    nodesets: frozenset[frozenset[N]]
    nodesets_by_size: dict[int, tuple[frozenset[N], ...]]
    subset_parent_cache: dict[frozenset[N], frozenset[N] | None] = field(default_factory=dict)
    degree_cache: dict[frozenset[N], dict[N, tuple[int, int, bool]]] = field(default_factory=dict)
    subgraph_cache: dict[frozenset[N], nx.DiGraph[N]] = field(default_factory=dict)


class MotifSession(Generic[N, L, V, I]):
    def __init__(
        self,
        finder: MotifFinder[N, L, V, I],
        motif_class: _MotifClass[N, I],
    ) -> None:
        self._finder = finder
        self._motif_class = motif_class
        self._occurrences = tuple(motif_class.occurrences)

    def __iter__(self) -> Iterator[tuple[I, tuple[N, ...]]]:
        return iter(self._occurrences)

    def __len__(self) -> int:
        return len(self._occurrences)

    def __getitem__(self, index: int) -> tuple[I, tuple[N, ...]]:
        return self._occurrences[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MotifSession):
            return self._occurrences == other._occurrences
        if isinstance(other, list) or isinstance(other, tuple):
            return list(self._occurrences) == list(other)
        return NotImplemented  # type: ignore[return-value]

    def state_for(
        self,
        witness: tuple[I, Iterable[N]],
    ) -> ExpansionState[N, L, V, I]:
        source, witness_nodes = witness
        witness_nodeset = frozenset(witness_nodes)
        witness_key = (source, witness_nodeset)
        if witness_key not in self._motif_class.occurrence_mappings:
            raise ValueError("witness is not an occurrence of this motif")
        return self._finder.expansion().state_for_motif_class(self._motif_class)

    def expand_from(
        self,
        witness: tuple[I, Iterable[N]],
        expanded_nodes: Iterable[N],
    ) -> MotifSession[N, L, V, I] | None:
        source, witness_nodes = witness
        next_state = self._finder.expansion().validate_expansion(
            self.state_for((source, witness_nodes)),
            source,
            witness_nodes,
            expanded_nodes,
        )
        if next_state is None:
            return None
        return self._finder.expansion().expand_from(next_state)


class ExpansionState(Generic[N, L, V, I]):
    def __init__(
        self,
        finder: MotifFinder[N, L, V, I],
        motif_class: _MotifClass[N, I],
        *,
        predecessor_state: ExpansionState[N, L, V, I] | None = None,
        base_size: int | None = None,
        scan_complete: bool = True,
    ) -> None:
        self._finder = finder
        self._motif_class = motif_class
        self._predecessor_state = predecessor_state
        self._base_size = base_size
        self._scan_complete = scan_complete
        self._expansion_cache: dict[tuple[I, frozenset[N], frozenset[N]], ExpansionState[N, L, V, I] | None] = {}
        self._candidate_supersets_cache: dict[tuple[I, frozenset[N], int], tuple[frozenset[N], ...]] = {}
        self._slot_domain_cache: dict[tuple[I, frozenset[N], Hashable], tuple[N, ...]] = {}


class MotifExpansion(Generic[N, L, V, I]):
    def __init__(
        self,
        finder_or_parents: MotifFinder[N, L, V, I] | Mapping[I, nx.DiGraph[N]],
        occurrences: Iterable[tuple[I, Iterable[N]]] | None = None,
        node_label: Callable[[dict[str, Any]], L] | None = None,
        node_vars: Callable[[dict[str, Any]], tuple[V, ...]] | None = None,
        *,
        node_order_key: Callable[[N], Any] | None = None,
        label_matches: Callable[[L, L], bool] | None = None,
        sampled_nodesets: Mapping[I, frozenset[frozenset[N]]] | None = None,
    ) -> None:
        provided_occurrence_keys: set[tuple[I, frozenset[N]]] = set()
        if isinstance(finder_or_parents, MotifFinder):
            self._finder = finder_or_parents
        else:
            if occurrences is None or node_label is None or node_vars is None:
                raise TypeError(
                    "standalone MotifExpansion requires occurrences, node_label, and node_vars"
                )

            grouped_occurrences: defaultdict[I, set[frozenset[N]]] = defaultdict(set)
            for source, witness_nodes in occurrences:
                nodeset = frozenset(witness_nodes)
                grouped_occurrences[source].add(nodeset)
                provided_occurrence_keys.add((source, nodeset))

            samples: dict[I, tuple[nx.DiGraph[N], frozenset[frozenset[N]]]] = {}
            for source, graph in finder_or_parents.items():
                nodesets = set(sampled_nodesets.get(source, frozenset()) if sampled_nodesets is not None else ())
                nodesets.update(grouped_occurrences.get(source, ()))
                samples[source] = (graph, frozenset(nodesets))

            self._finder = MotifFinder(
                samples,
                node_label,
                node_vars,
                node_order_key=node_order_key,
                label_matches=label_matches,
            )

        self._states_by_motif_class: dict[int, ExpansionState[N, L, V, I]] = {}
        self._occurrence_index: dict[tuple[I, frozenset[N]], ExpansionState[N, L, V, I]] = {}

        for motif_classes in self._finder._motifs_by_size.values():
            for motif_class in motif_classes:
                state = self.state_for_motif_class(motif_class)
                self._register_known_occurrences(state)

        if not isinstance(finder_or_parents, MotifFinder):
            matching_state_ids = {
                id(self.state_for_occurrence(source, nodeset)._motif_class)
                for source, nodeset in provided_occurrence_keys
            }
            if len(matching_state_ids) != 1:
                raise ValueError("provided occurrences do not form a single repeated motif class")

    def state_for_motif_class(
        self,
        motif_class: _MotifClass[N, I],
    ) -> ExpansionState[N, L, V, I]:
        key = id(motif_class)
        state = self._states_by_motif_class.get(key)
        if state is None:
            state = ExpansionState(self._finder, motif_class)
            self._states_by_motif_class[key] = state
        return state

    def state_for_occurrence(
        self,
        source: I,
        witness_nodes: Iterable[N],
    ) -> ExpansionState[N, L, V, I]:
        witness_nodeset = frozenset(witness_nodes)
        occurrence_key = (source, witness_nodeset)
        state = self._occurrence_index.get(occurrence_key)
        if state is None:
            raise ValueError("witness is not a known occurrence")
        return state

    def _register_known_occurrences(
        self,
        state: ExpansionState[N, L, V, I],
    ) -> None:
        for occurrence_source, occurrence_nodes in state._motif_class.occurrences:
            self._occurrence_index.setdefault((occurrence_source, frozenset(occurrence_nodes)), state)

    def _bootstrap_successor_state(
        self,
        predecessor_state: ExpansionState[N, L, V, I],
        source: I,
        witness_nodeset: frozenset[N],
        expanded_nodeset: frozenset[N],
    ) -> ExpansionState[N, L, V, I]:
        motif_class = predecessor_state._motif_class
        representative_base_order = motif_class.occurrence_mappings[(source, witness_nodeset)]
        representative_expanded_order = self._finder._expanded_order(source, representative_base_order, expanded_nodeset)
        expanded_motif_class = _MotifClass(
            representative=_OccurrenceRef(source=source, nodeset=expanded_nodeset),
            representative_order=representative_expanded_order,
            occurrences=[(source, tuple(sorted(expanded_nodeset, key=self._finder._node_order_key)))],
            occurrence_keys={(source, expanded_nodeset)},
            occurrence_mappings={(source, expanded_nodeset): representative_expanded_order},
        )
        state = self.state_for_motif_class(expanded_motif_class)
        state._predecessor_state = predecessor_state
        state._base_size = len(motif_class.representative_order)
        state._scan_complete = False
        self._register_known_occurrences(state)
        return state

    def validate_expansion(
        self,
        predecessor_state: ExpansionState[N, L, V, I],
        source: I,
        witness_nodes: Iterable[N],
        expanded_nodes: Iterable[N],
    ) -> ExpansionState[N, L, V, I] | None:
        witness_nodeset = frozenset(witness_nodes)
        expanded_nodeset = frozenset(expanded_nodes)
        cache_key = (source, witness_nodeset, expanded_nodeset)
        if cache_key in predecessor_state._expansion_cache:
            return predecessor_state._expansion_cache[cache_key]

        motif_class = predecessor_state._motif_class
        witness_key = (source, witness_nodeset)
        if witness_key not in motif_class.occurrence_mappings:
            raise ValueError("witness is not an occurrence of this motif")
        if not witness_nodeset.issubset(expanded_nodeset):
            raise ValueError("expanded nodes must be a superset of the witness")

        known_state = self._occurrence_index.get((source, expanded_nodeset))
        if known_state is not None:
            predecessor_state._expansion_cache[cache_key] = known_state
            return known_state

        result = self._bootstrap_successor_state(
            predecessor_state,
            source,
            witness_nodeset,
            expanded_nodeset,
        )
        support = 0
        for _, _, _ in self._finder._iter_state_occurrences(result, stop_after=2):
            support += 1
            if support >= 2:
                break
        if support < 2:
            predecessor_state._expansion_cache[cache_key] = None
            return None

        self._register_known_occurrences(result)
        predecessor_state._expansion_cache[cache_key] = result
        return result

    def materialize(
        self,
        state: ExpansionState[N, L, V, I],
    ) -> MotifSession[N, L, V, I]:
        motif_class = self._finder._materialize_state(state)
        self._register_known_occurrences(state)
        return MotifSession(self._finder, motif_class)

    def expand_from(
        self,
        state: ExpansionState[N, L, V, I],
    ) -> MotifSession[N, L, V, I]:
        return self.materialize(state)


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
        self._expander: MotifExpansion[N, L, V, I] | None = None

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
            nodesets_by_size_lists: defaultdict[int, list[frozenset[N]]] = defaultdict(list)
            for nodeset in filtered_nodesets:
                nodesets_by_size_lists[len(nodeset)].append(nodeset)
            self._parents[source] = _ParentSamples(
                graph=graph,
                nodesets=filtered_nodesets,
                nodesets_by_size={
                    size: tuple(nodesets_for_size)
                    for size, nodesets_for_size in nodesets_by_size_lists.items()
                },
            )
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

        topology_hashes: dict[_OccurrenceRef[N, I], Hashable] = {}
        topology_hash_counts: defaultdict[tuple[int, Hashable], int] = defaultdict(int)
        for occurrence in ordered_occurrences:
            topology_hash = self._topology_hash(occurrence)
            topology_hashes[occurrence] = topology_hash
            topology_hash_counts[(len(occurrence.nodeset), topology_hash)] += 1

        topology_classes_by_size: dict[int, dict[Hashable, list[_TopologyClass[N, I]]]] = defaultdict(lambda: defaultdict(list))

        for occurrence in ordered_occurrences:
            sample_size = len(occurrence.nodeset)
            topology_hash = topology_hashes[occurrence]
            if topology_hash_counts[(sample_size, topology_hash)] < 2:
                continue

            sample_graph = self._subgraph(occurrence)
            topology_classes = topology_classes_by_size[sample_size][topology_hash]
            topology_class: _TopologyClass[N, I] | None = None
            for candidate in topology_classes:
                if topologies_are_isomorphic(self._subgraph(candidate.representative), sample_graph):
                    topology_class = candidate
                    break

            if topology_class is None:
                topology_class = _TopologyClass(representative=occurrence)
                topology_classes.append(topology_class)
            topology_class.occurrences.append(occurrence)

        for sample_size, topology_classes_by_hash in topology_classes_by_size.items():
            for topology_hash, topology_classes in list(topology_classes_by_hash.items()):
                surviving_topology_classes: list[_TopologyClass[N, I]] = []
                for topology_class in topology_classes:
                    if len(topology_class.occurrences) < 2:
                        continue

                    labeled_hashes: dict[_OccurrenceRef[N, I], Hashable] = {}
                    labeled_hash_counts: defaultdict[Hashable, int] = defaultdict(int)
                    for occurrence in topology_class.occurrences:
                        labeled_hash = self._labeled_hash(occurrence)
                        labeled_hashes[occurrence] = labeled_hash
                        labeled_hash_counts[labeled_hash] += 1

                    for occurrence in topology_class.occurrences:
                        labeled_hash = labeled_hashes[occurrence]
                        if labeled_hash_counts[labeled_hash] < 2:
                            continue

                        sample_graph = self._subgraph(occurrence)
                        motif_class_bucket = topology_class.motif_classes_by_hash[labeled_hash]
                        occurrence_nodes = tuple(sorted(occurrence.nodeset, key=self._node_order_key))
                        occurrence_key = (occurrence.source, occurrence.nodeset)
                        for motif_class in motif_class_bucket:
                            mapping = graph_isomorphism_mapping(
                                self._subgraph(motif_class.representative),
                                sample_graph,
                                self._node_label,
                                self._node_vars,
                                self._label_matches,
                            )
                            if mapping is not None:
                                if occurrence_key not in motif_class.occurrence_keys:
                                    motif_class.occurrences.append((occurrence.source, occurrence_nodes))
                                    motif_class.occurrence_keys.add(occurrence_key)
                                    motif_class.occurrence_mappings[occurrence_key] = tuple(
                                        mapping[node] for node in motif_class.representative_order
                                    )
                                break
                        else:
                            motif_class = _MotifClass(
                                representative=occurrence,
                                representative_order=occurrence_nodes,
                                occurrences=[(occurrence.source, occurrence_nodes)],
                                occurrence_keys={occurrence_key},
                                occurrence_mappings={occurrence_key: occurrence_nodes},
                            )
                            motif_class_bucket.append(motif_class)

                    surfaced_motif_classes = [
                        motif_class
                        for motif_classes in topology_class.motif_classes_by_hash.values()
                        for motif_class in motif_classes
                        if len(motif_class.occurrences) >= 2
                    ]
                    if not surfaced_motif_classes:
                        topology_class.motif_classes_by_hash.clear()
                        continue

                    topology_class.motif_classes = surfaced_motif_classes
                    for motif_class in surfaced_motif_classes:
                        self._motif_classes_by_size[sample_size].append(motif_class)
                    surviving_topology_classes.append(topology_class)

                if surviving_topology_classes:
                    topology_classes_by_hash[topology_hash] = surviving_topology_classes
                else:
                    del topology_classes_by_hash[topology_hash]

        self._topology_classes_by_size = topology_classes_by_size
        self._trie_built = False

        self._motifs_by_size: dict[int, tuple[_MotifClass[N, I], ...]] = {}
        for size, motif_classes in self._motif_classes_by_size.items():
            for motif_class in motif_classes:
                motif_class.occurrences.sort(key=lambda occurrence: (_stable_value_key(occurrence[0]), occurrence[1]))
            surfaced_motif_classes = tuple(
                motif_class
                for motif_class in motif_classes
                if len(motif_class.occurrences) >= 2
            )
            if surfaced_motif_classes:
                self._motifs_by_size[size] = surfaced_motif_classes

    def expansion(self) -> MotifExpansion[N, L, V, I]:
        if self._expander is None:
            self._expander = MotifExpansion(self)
        return self._expander

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

    def _expanded_order(
        self,
        source: I,
        base_order: tuple[N, ...],
        expanded_nodeset: frozenset[N],
    ) -> tuple[N, ...]:
        graph = self._parents[source].graph
        base_nodes = frozenset(base_order)
        extra_nodes = expanded_nodeset - base_nodes
        extra_signatures: list[tuple[tuple[Any, ...], N]] = []
        for node in extra_nodes:
            variables = self._node_vars(graph.nodes[node])
            local_variable_pattern: dict[V, int] = {}
            extra_signatures.append(
                (
                    (
                        self._node_label(graph.nodes[node]) if self._default_label_matches else None,
                        tuple(local_variable_pattern.setdefault(identifier, len(local_variable_pattern)) for identifier in variables),
                        len(variables),
                        tuple(int(graph.has_edge(base_node, node)) for base_node in base_order),
                        tuple(int(graph.has_edge(node, base_node)) for base_node in base_order),
                        graph.has_edge(node, node),
                        sum(1 for predecessor in graph.pred[node] if predecessor in expanded_nodeset),
                        sum(1 for successor in graph.succ[node] if successor in expanded_nodeset),
                    ),
                    node,
                )
            )
        ordered_extra = tuple(
            node
            for _, node in sorted(
                extra_signatures,
                key=lambda item: (item[0], self._node_order_key(item[1])),
            )
        )
        return base_order + ordered_extra

    def _expansion_signature(
        self,
        graph: nx.DiGraph[N],
        base_order: tuple[N, ...],
        expanded_nodeset: frozenset[N],
    ) -> tuple[Any, ...]:
        base_nodes = frozenset(base_order)
        extra_nodes = expanded_nodeset - base_nodes
        extra_signatures: list[tuple[Any, ...]] = []
        for node in extra_nodes:
            variables = self._node_vars(graph.nodes[node])
            local_variable_pattern: dict[V, int] = {}
            to_base = tuple(int(graph.has_edge(base_node, node)) for base_node in base_order)
            from_base = tuple(int(graph.has_edge(node, base_node)) for base_node in base_order)
            in_degree = sum(1 for predecessor in graph.pred[node] if predecessor in expanded_nodeset)
            out_degree = sum(1 for successor in graph.succ[node] if successor in expanded_nodeset)
            extra_signatures.append(
                (
                    self._node_label(graph.nodes[node]) if self._default_label_matches else None,
                    tuple(local_variable_pattern.setdefault(identifier, len(local_variable_pattern)) for identifier in variables),
                    len(variables),
                    to_base,
                    from_base,
                    graph.has_edge(node, node),
                    in_degree,
                    out_degree,
                )
        )
        return tuple(sorted(extra_signatures))

    def _anchored_slot_signature(
        self,
        graph: nx.DiGraph[N],
        base_order: tuple[N, ...],
        node: N,
    ) -> tuple[Any, ...]:
        variables = self._node_vars(graph.nodes[node])
        return (
            self._node_label(graph.nodes[node]) if self._default_label_matches else None,
            len(variables),
            tuple(int(graph.has_edge(base_node, node)) for base_node in base_order),
            tuple(int(graph.has_edge(node, base_node)) for base_node in base_order),
            graph.has_edge(node, node),
        )

    def _candidate_domain_for_slot(
        self,
        state: ExpansionState[N, L, V, I],
        occurrence_source: I,
        occurrence_nodeset: frozenset[N],
        base_order: tuple[N, ...],
        slot_signature: Hashable,
    ) -> tuple[N, ...]:
        cache_key = (occurrence_source, occurrence_nodeset, slot_signature)
        cached = state._slot_domain_cache.get(cache_key)
        if cached is not None:
            return cached

        graph = self._parents[occurrence_source].graph
        candidates = tuple(
            candidate
            for candidate in graph.nodes
            if candidate not in occurrence_nodeset
            and self._anchored_slot_signature(graph, base_order, candidate) == slot_signature
        )
        state._slot_domain_cache[cache_key] = candidates
        return candidates

    def _find_anchored_expansion(
        self,
        representative_graph: nx.DiGraph[N],
        representative_order: tuple[N, ...],
        base_size: int,
        target_graph: nx.DiGraph[N],
        target_base_order: tuple[N, ...],
        candidate_domains: dict[N, tuple[N, ...]] | None = None,
    ) -> tuple[N, ...] | None:
        used_targets = set(target_base_order)
        mapping = {
            representative_order[position]: target_base_order[position]
            for position in range(base_size)
        }
        source_to_target: defaultdict[int, dict[V, V]] = defaultdict(dict)
        target_to_source: defaultdict[int, dict[V, V]] = defaultdict(dict)

        def bind_variables(source_node: N, target_node: N) -> bool:
            source_vars = self._node_vars(representative_graph.nodes[source_node])
            target_vars = self._node_vars(target_graph.nodes[target_node])
            if len(source_vars) != len(target_vars):
                return False
            for variable_class, source_identifier in enumerate(source_vars):
                target_identifier = target_vars[variable_class]
                existing_target = source_to_target[variable_class].get(source_identifier)
                existing_source = target_to_source[variable_class].get(target_identifier)
                if existing_target is not None and existing_target != target_identifier:
                    return False
                if existing_source is not None and existing_source != source_identifier:
                    return False
            for variable_class, source_identifier in enumerate(source_vars):
                target_identifier = target_vars[variable_class]
                source_to_target[variable_class][source_identifier] = target_identifier
                target_to_source[variable_class][target_identifier] = source_identifier
            return True

        for position in range(base_size):
            if not bind_variables(representative_order[position], target_base_order[position]):
                return None

        target_nodes = tuple(target_graph.nodes)
        search_positions = list(range(base_size, len(representative_order)))
        if candidate_domains is None:
            candidate_domains = {}
            for position in search_positions:
                source_node = representative_order[position]
                source_label = self._node_label(representative_graph.nodes[source_node])
                source_vars = self._node_vars(representative_graph.nodes[source_node])
                candidate_domains[source_node] = tuple(
                    candidate
                    for candidate in target_nodes
                    if candidate not in used_targets
                    and self._label_matches(source_label, self._node_label(target_graph.nodes[candidate]))
                    and len(source_vars) == len(self._node_vars(target_graph.nodes[candidate]))
                    and target_graph.has_edge(candidate, candidate) == representative_graph.has_edge(source_node, source_node)
                    and all(
                        representative_graph.has_edge(representative_order[previous_position], source_node)
                        == target_graph.has_edge(target_base_order[previous_position], candidate)
                        and representative_graph.has_edge(source_node, representative_order[previous_position])
                        == target_graph.has_edge(candidate, target_base_order[previous_position])
                        for previous_position in range(base_size)
                    )
                )
        search_positions.sort(key=lambda position: len(candidate_domains[representative_order[position]]))
        if search_positions and len(candidate_domains[representative_order[search_positions[0]]]) == 0:
            return None

        def search(depth: int) -> bool:
            if depth == len(search_positions):
                return True

            position = search_positions[depth]
            source_node = representative_order[position]
            source_label = self._node_label(representative_graph.nodes[source_node])
            source_vars = self._node_vars(representative_graph.nodes[source_node])
            for candidate in candidate_domains[source_node]:
                if candidate in used_targets:
                    continue
                for previous_source, previous_target in mapping.items():
                    if representative_graph.has_edge(previous_source, source_node) != target_graph.has_edge(previous_target, candidate):
                        break
                    if representative_graph.has_edge(source_node, previous_source) != target_graph.has_edge(candidate, previous_target):
                        break
                else:
                    snapshot_source = {variable_class: dict(value) for variable_class, value in source_to_target.items()}
                    snapshot_target = {variable_class: dict(value) for variable_class, value in target_to_source.items()}
                    if not bind_variables(source_node, candidate):
                        source_to_target.clear()
                        target_to_source.clear()
                        source_to_target.update({variable_class: dict(value) for variable_class, value in snapshot_source.items()})
                        target_to_source.update({variable_class: dict(value) for variable_class, value in snapshot_target.items()})
                        continue
                    mapping[source_node] = candidate
                    used_targets.add(candidate)
                    if search(depth + 1):
                        return True
                    used_targets.remove(candidate)
                    del mapping[source_node]
                    source_to_target.clear()
                    target_to_source.clear()
                    source_to_target.update({variable_class: dict(value) for variable_class, value in snapshot_source.items()})
                    target_to_source.update({variable_class: dict(value) for variable_class, value in snapshot_target.items()})

            return False

        if not search(0):
            return None
        return tuple(mapping[node] for node in representative_order)

    def _expand_motif_class(
        self,
        state: ExpansionState[N, L, V, I],
        source: I,
        witness_nodeset: frozenset[N],
        expanded_nodeset: frozenset[N],
    ) -> ExpansionState[N, L, V, I] | None:
        motif_class = state._motif_class
        witness_key = (source, witness_nodeset)
        if witness_key not in motif_class.occurrence_mappings:
            raise ValueError("witness is not an occurrence of this motif")
        if not witness_nodeset.issubset(expanded_nodeset):
            raise ValueError("expanded nodes must be a superset of the witness")

        representative_base_order = motif_class.representative_order
        origin_base_order = motif_class.occurrence_mappings[witness_key]
        representative_expanded_order = self._expanded_order(source, origin_base_order, expanded_nodeset)
        representative_graph = self._parents[source].graph.subgraph(expanded_nodeset)
        assert isinstance(representative_graph, nx.DiGraph)

        expanded_motif_class = _MotifClass(
            representative=_OccurrenceRef(source=source, nodeset=expanded_nodeset),
            representative_order=representative_expanded_order,
            occurrences=[(source, tuple(sorted(expanded_nodeset, key=self._node_order_key)))],
            occurrence_keys={(source, expanded_nodeset)},
            occurrence_mappings={(source, expanded_nodeset): representative_expanded_order},
        )

        for occurrence_source, occurrence_nodes in motif_class.occurrences:
            occurrence_nodeset = frozenset(occurrence_nodes)
            occurrence_key = (occurrence_source, occurrence_nodeset)
            base_order = motif_class.occurrence_mappings[occurrence_key]

            sampled_match: tuple[N, ...] | None = None
            for candidate_nodeset in self._candidate_supersets_for(
                state,
                occurrence_source,
                occurrence_nodeset,
                len(expanded_nodeset),
            ):
                candidate_graph = self._parents[occurrence_source].graph.subgraph(candidate_nodeset)
                assert isinstance(candidate_graph, nx.DiGraph)
                mapping = self._find_anchored_expansion(
                    representative_graph,
                    representative_expanded_order,
                    len(representative_base_order),
                    candidate_graph,
                    base_order,
                )
                if mapping is not None:
                    sampled_match = mapping
                    break

            if sampled_match is None:
                sampled_match = self._find_anchored_expansion(
                    representative_graph,
                    representative_expanded_order,
                    len(representative_base_order),
                    self._parents[occurrence_source].graph,
                    base_order,
                )

            if sampled_match is None:
                continue

            matched_nodeset = frozenset(sampled_match)
            occurrence_value = (occurrence_source, tuple(sorted(matched_nodeset, key=self._node_order_key)))
            occurrence_cache_key = (occurrence_source, matched_nodeset)
            if occurrence_cache_key in expanded_motif_class.occurrence_keys:
                continue
            expanded_motif_class.occurrences.append(occurrence_value)
            expanded_motif_class.occurrence_keys.add(occurrence_cache_key)
            expanded_motif_class.occurrence_mappings[occurrence_cache_key] = sampled_match

        if len(expanded_motif_class.occurrences) < 2:
            return None
        return self.expansion().state_for_motif_class(expanded_motif_class)

    def _candidate_supersets_for(
        self,
        state: ExpansionState[N, L, V, I],
        source: I,
        witness_nodeset: frozenset[N],
        target_size: int,
    ) -> tuple[frozenset[N], ...]:
        cache_key = (source, witness_nodeset, target_size)
        cached = state._candidate_supersets_cache.get(cache_key)
        if cached is not None:
            return cached

        parent = self._parents[source]
        candidates = tuple(
            candidate_nodeset
            for candidate_nodeset in parent.nodesets_by_size.get(target_size, ())
            if witness_nodeset.issubset(candidate_nodeset)
        )
        state._candidate_supersets_cache[cache_key] = candidates
        return candidates

    def _match_occurrence_expansion(
        self,
        state: ExpansionState[N, L, V, I],
        occurrence_source: I,
        occurrence_nodeset: frozenset[N],
        base_order: tuple[N, ...],
        representative_signature: tuple[Any, ...],
    ) -> tuple[N, ...] | None:
        if state._base_size is None:
            raise ValueError("expansion state missing base size")

        representative_graph = self._parents[state._motif_class.representative.source].graph.subgraph(
            state._motif_class.representative.nodeset
        )
        assert isinstance(representative_graph, nx.DiGraph)
        cache_owner = state._predecessor_state if state._predecessor_state is not None else state
        candidate_domains: dict[N, tuple[N, ...]] = {}
        for source_node in state._motif_class.representative_order[state._base_size:]:
            slot_signature = self._anchored_slot_signature(
                representative_graph,
                state._motif_class.representative_order[:state._base_size],
                source_node,
            )
            candidate_domains[source_node] = self._candidate_domain_for_slot(
                cache_owner,
                occurrence_source,
                occurrence_nodeset,
                base_order,
                slot_signature,
            )
            if not candidate_domains[source_node]:
                return None

        sampled_match: tuple[N, ...] | None = None
        for candidate_nodeset in self._candidate_supersets_for(
            state._predecessor_state if state._predecessor_state is not None else state,
            occurrence_source,
            occurrence_nodeset,
            len(state._motif_class.representative.nodeset),
        ):
            if (
                self._expansion_signature(
                    self._parents[occurrence_source].graph,
                    base_order,
                    candidate_nodeset,
                )
                != representative_signature
            ):
                continue
            candidate_graph = self._parents[occurrence_source].graph.subgraph(candidate_nodeset)
            assert isinstance(candidate_graph, nx.DiGraph)
            mapping = self._find_anchored_expansion(
                representative_graph,
                state._motif_class.representative_order,
                state._base_size,
                candidate_graph,
                base_order,
                candidate_domains,
            )
            if mapping is not None:
                sampled_match = mapping
                break

        if sampled_match is None:
            sampled_match = self._find_anchored_expansion(
                representative_graph,
                state._motif_class.representative_order,
                state._base_size,
                self._parents[occurrence_source].graph,
                base_order,
                candidate_domains,
            )

        return sampled_match

    def _iter_state_occurrences(
        self,
        state: ExpansionState[N, L, V, I],
        *,
        stop_after: int | None = None,
    ) -> Iterator[tuple[I, frozenset[N], tuple[N, ...]]]:
        yielded: set[tuple[I, frozenset[N]]] = set()
        for occurrence_source, occurrence_nodes in state._motif_class.occurrences:
            occurrence_nodeset = frozenset(occurrence_nodes)
            occurrence_key = (occurrence_source, occurrence_nodeset)
            yielded.add(occurrence_key)
            yield (
                occurrence_source,
                occurrence_nodeset,
                state._motif_class.occurrence_mappings[occurrence_key],
            )
            if stop_after is not None and len(yielded) >= stop_after:
                return

        if state._scan_complete:
            return
        if state._predecessor_state is None or state._base_size is None:
            state._scan_complete = True
            return

        representative_graph = self._parents[state._motif_class.representative.source].graph
        representative_signature = self._expansion_signature(
            representative_graph,
            state._motif_class.representative_order[:state._base_size],
            state._motif_class.representative.nodeset,
        )

        for occurrence_source, occurrence_nodeset, base_order in self._iter_state_occurrences(state._predecessor_state):
            occurrence_key = (occurrence_source, occurrence_nodeset)
            if occurrence_key in yielded:
                continue
            mapping = self._match_occurrence_expansion(
                state,
                occurrence_source,
                occurrence_nodeset,
                base_order,
                representative_signature,
            )
            if mapping is None:
                continue
            matched_nodeset = frozenset(mapping)
            matched_key = (occurrence_source, matched_nodeset)
            if matched_key in state._motif_class.occurrence_keys:
                continue
            occurrence_value = (occurrence_source, tuple(sorted(matched_nodeset, key=self._node_order_key)))
            state._motif_class.occurrences.append(occurrence_value)
            state._motif_class.occurrence_keys.add(matched_key)
            state._motif_class.occurrence_mappings[matched_key] = mapping
            yielded.add(matched_key)
            yield (occurrence_source, matched_nodeset, mapping)
            if stop_after is not None and len(yielded) >= stop_after:
                return

        state._scan_complete = True

    def _materialize_state(
        self,
        state: ExpansionState[N, L, V, I],
    ) -> _MotifClass[N, I]:
        if not state._scan_complete:
            for _ in self._iter_state_occurrences(state):
                pass
            state._motif_class.occurrences.sort(
                key=lambda occurrence: (_stable_value_key(occurrence[0]), occurrence[1])
            )
        return state._motif_class

    def validate_expansion(
        self,
        predecessor_state: ExpansionState[N, L, V, I],
        source: I,
        witness_nodes: Iterable[N],
        expanded_nodes: Iterable[N],
    ) -> ExpansionState[N, L, V, I] | None:
        return self.expansion().validate_expansion(predecessor_state, source, witness_nodes, expanded_nodes)

    def motifs(self, size: int | None = None, *, descending: bool = False) -> Iterator[MotifSession[N, L, V, I]]:
        if size is not None:
            for motif_class in self._motifs_by_size.get(size, ()):
                yield MotifSession(self, motif_class)
            return

        for current_size in sorted(self._motifs_by_size, reverse=descending):
            for motif_class in self._motifs_by_size[current_size]:
                yield MotifSession(self, motif_class)
