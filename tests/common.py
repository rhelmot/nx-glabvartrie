from __future__ import annotations

from random import Random
from collections import defaultdict
from typing import Any, Callable, Iterable

import networkx as nx

from glabvartrie.common import N, L, V, graphs_are_isomorphic

def random_connected_graph(r: Random, nodes: range, labels: range, variables: range, variable_density: float, edge_density: float) -> nx.DiGraph[int]:
    g = nx.DiGraph()
    for n in nodes:
        base_label = r.choice(labels)
        if r.random() < variable_density:
            arity = r.randrange(1, 4)
            node_vars = tuple(r.choice(variables) for _ in range(arity))
        else:
            node_vars = ()
        g.add_node(n, label=base_label, vars=node_vars)
    wccs = [[n] for n in nodes]
    while len(wccs) > 1:
        r.shuffle(wccs)
        wcc1 = wccs.pop()
        wcc2 = wccs.pop()
        n1 = r.choice(wcc1)
        n2 = r.choice(wcc2)
        g.add_edge(n1, n2)
        wccs.append(wcc1 + wcc2)

    all_edges = len(nodes) ** 2
    expected_edges = all_edges * edge_density
    current_edges = len(nodes) - 1
    desired_expected_edges = expected_edges - current_edges
    remaining_tries = all_edges - current_edges
    new_density = desired_expected_edges / remaining_tries

    for n1 in nodes:
        for n2 in nodes:
            if g.has_edge(n1, n2):
                continue
            if r.random() < new_density:
                g.add_edge(n1, n2)

    return g

class CantEmbed(Exception):
    pass

def embed_graph(query: nx.DiGraph[int], t: nx.DiGraph[int], available_nodes: list[int]) -> dict[int, int]:
    target_nodes = tuple(t.nodes)
    if len(available_nodes) < len(target_nodes):
        raise CantEmbed()

    node_mapping = {
        target_node: available_nodes.pop()
        for target_node in target_nodes
    }
    used_query_nodes = tuple(node_mapping.values())

    for source in used_query_nodes:
        for target in used_query_nodes:
            if query.has_edge(source, target):
                query.remove_edge(source, target)

    for target_node in target_nodes:
        query_node = node_mapping[target_node]
        query.nodes[query_node]['label'] = t.nodes[target_node]['label']
        query.nodes[query_node]['vars'] = t.nodes[target_node]['vars']

    for source, target in t.edges:
        query.add_edge(node_mapping[source], node_mapping[target])

    return node_mapping

def is_isomorphic_match(g1: nx.DiGraph[N], g2: nx.DiGraph[N], node_label: Callable[[dict[str, Any]], L], node_vars: Callable[[dict[str, Any]], tuple[V, ...]], label_matches: Callable[[L, L], bool] | None = None) -> bool:
    return graphs_are_isomorphic(g1, g2, node_label, node_vars, label_matches)

def generate_random_subgraph_nodesets(r: Random, g: nx.DiGraph[N], n: int) -> Iterable[frozenset[N]]:
    assert nx.is_weakly_connected(g)
    nodes = tuple(g)
    for _ in range(n):
        size = r.randrange(1, len(g) + 1)
        startnode = nodes[r.randrange(len(nodes))]
        seen = {startnode}
        frontier_set = set(g.pred[startnode]) | set(g.succ[startnode])
        frontier_set.difference_update(seen)
        frontier = list(frontier_set)
        while len(seen) < size:
            if not frontier:
                break
            node_index = r.randrange(len(frontier))
            node = frontier[node_index]
            frontier[node_index] = frontier[-1]
            frontier.pop()
            frontier_set.remove(node)
            seen.add(node)
            for neighbor in g.pred[node]:
                if neighbor in seen or neighbor in frontier_set:
                    continue
                frontier_set.add(neighbor)
                frontier.append(neighbor)
            for neighbor in g.succ[node]:
                if neighbor in seen or neighbor in frontier_set:
                    continue
                frontier_set.add(neighbor)
                frontier.append(neighbor)
        yield frozenset(seen)


def generate_random_subgraphs(r: Random, g: nx.DiGraph[N], n: int) -> Iterable[nx.DiGraph[N]]:
    for nodeset in generate_random_subgraph_nodesets(r, g, n):
        subgraph = g.subgraph(nodeset)
        assert isinstance(subgraph, nx.DiGraph)
        yield subgraph
