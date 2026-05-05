from collections import defaultdict
from typing import Generic, Mapping

import networkx as nx

from .common import (
    I,
    L,
    N,
    V,
    LabVarDiGraphExpansionMatcher,
    LabVarDiGraphMatcher,
)

class AcyclicMotifFinder(Generic[N, L, V, I]):
    def __init__(
        self,
        node_label_attr: str,
        edge_label_attr: str,
    ) -> None:
        self._parents: dict[I, nx.DiGraph[N]] = {}
        self._motifs: defaultdict[tuple[L, ...], list[list[tuple[I, tuple[N, ...]]]]] = defaultdict(list)
        self._node_label_attr = node_label_attr
        self._edge_label_attr = edge_label_attr

    def add_parent(self, ident: I, graph: nx.DiGraph[N]):
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Must be a DAG")
        for node, attrs in graph.nodes.items():
            if "_self" not in attrs:
                attrs["_self"] = node
            if self._node_label_attr:
                raise ValueError("Missing node label attribute")
        for attrs in graph.edges.items():
            if self._edge_label_attr not in attrs:
                raise ValueError("Missing edge label attribute")
        self._parents[ident] = graph

    def add_subgraph(self, ident: I, nodes: frozenset[N]):
        subg = self._parents[ident].subgraph(nodes)
        lbl = self.graph_label(subg)
        motifcls = self._motifs[lbl]
        for motif in motifcls:
            ident0, nodes0 = motif[0]
            subg0 = self._parents[ident0].subgraph(nodes0)
            matcher = self.matcher(subg0, subg)
            if not matcher.is_isomorphic():
                continue
            motif.append((ident, tuple(matcher.mapping[n] for n in nodes0)))
        else:
            motifcls.append([(ident, tuple(nodes))])

    def graph_label(self, graph: nx.DiGraph[N]):
        return tuple(graph.nodes[n][self._node_label_attr] for n in nx.lexicographical_topological_sort(graph, key=lambda n: graph.nodes[n][self._node_label_attr]))

    def matcher(self, g0: nx.DiGraph[N], g1: nx.DiGraph[N]) -> LabVarDiGraphMatcher[N]:
        return LabVarDiGraphMatcher(g0, g1, self._node_label_attr, self._edge_label_attr)

    def motifs(self):
        for motifcls in self._motifs.values():
            for motif in motifcls:
                if len(motif) > 1:
                    yield motif

    def expand(self, base_mapping: Mapping[N, N], g0: nx.DiGraph[N], g1: nx.DiGraph[N]) -> Mapping[N, N] | None:
        matcher = LabVarDiGraphExpansionMatcher(g0, g1, self._node_label_attr, self._edge_label_attr, base_mapping)
        if matcher.is_isomorphic():
            return matcher.mapping
        return None
