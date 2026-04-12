from __future__ import annotations

import os
import unittest
from collections.abc import Mapping
from collections import defaultdict
from random import Random

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

from glabvartrie import Database

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

def is_valid_match(
    target_graph: nx.DiGraph[int],
    target_ident: int,
    query_graph: nx.DiGraph[int],
    node_mapping: dict[int, int],
    variable_mapping: Mapping[int, Mapping[int, int]],
    found_ident: int,
) -> bool:
    if set(node_mapping) != set(target_graph.nodes):
        return False
    if len(set(node_mapping.values())) != len(node_mapping):
        return False
    if found_ident != target_ident:
        return False

    computed_variable_mapping: defaultdict[int, dict[int, int]] = defaultdict(dict)
    for target_node, query_node in node_mapping.items():
        if query_node not in query_graph:
            return False

        if target_graph.nodes[target_node]["label"] != query_graph.nodes[query_node]["label"]:
            return False

        target_vars = target_graph.nodes[target_node]["vars"]
        query_vars = query_graph.nodes[query_node]["vars"]
        if len(target_vars) != len(query_vars):
            return False
        for slot, target_identifier in enumerate(target_vars):
            query_identifier = query_vars[slot]
            existing = computed_variable_mapping[slot].get(target_identifier)
            if existing is not None and existing != query_identifier:
                return False
            computed_variable_mapping[slot][target_identifier] = query_identifier

    for source, target in target_graph.edges:
        if not query_graph.has_edge(node_mapping[source], node_mapping[target]):
            return False

    return {
        variable_class: dict(identifier_mapping)
        for variable_class, identifier_mapping in computed_variable_mapping.items()
    } == dict(variable_mapping)

def embed_graph(query: nx.DiGraph[int], t: nx.DiGraph[int], available_nodes: list[int], variables: list[int]):
    var_mapping: defaultdict[int, dict[int, int]] = defaultdict(dict)
    node_mapping: dict[int, int] = {}
    for n in t:
        if n in node_mapping:
            matching = node_mapping[n]
        else:
            matching = available_nodes.pop()
            node_mapping[n] = matching

        for succ in t.succ[n]:
            if succ in node_mapping:
                succ_matching = node_mapping[succ]
            else:
                succ_matching = available_nodes.pop()
                node_mapping[succ] = succ_matching
            query.add_edge(matching, succ_matching)
        query.nodes[matching]['label'] = t.nodes[n]['label']
        query.nodes[matching]['vars'] = tuple(
            var_mapping[slot].setdefault(identifier, variables.pop())
            for slot, identifier in enumerate(t.nodes[n]['vars'])
        )


def normalize_match_result(
    node_mapping: Mapping[int, int],
    variable_mapping: Mapping[int, Mapping[int, int]],
    ident: int,
) -> tuple[
    int,
    tuple[tuple[int, int], ...],
    tuple[tuple[int, tuple[tuple[int, int], ...]], ...],
]:
    return (
        ident,
        tuple(sorted(node_mapping.items())),
        tuple(
            (variable_class, tuple(sorted(identifier_mapping.items())))
            for variable_class, identifier_mapping in sorted(variable_mapping.items())
        ),
    )


def normalize_subgraph_result(
    node_mapping: Mapping[int, int],
    ident: int,
) -> tuple[int, frozenset[int]]:
    return ident, frozenset(node_mapping.values())


def brute_force_matches(
    target_graph: nx.DiGraph[int],
    target_ident: int,
    query_graph: nx.DiGraph[int],
) -> set[
    tuple[
        int,
        tuple[tuple[int, int], ...],
        tuple[tuple[int, tuple[tuple[int, int], ...]], ...],
    ]
]:
    target_nodes = list(target_graph.nodes)
    candidate_nodes = {
        target_node: [
            query_node
            for query_node in query_graph.nodes
            if target_graph.nodes[target_node]["label"] == query_graph.nodes[query_node]["label"]
            and len(target_graph.nodes[target_node]["vars"]) == len(query_graph.nodes[query_node]["vars"])
        ]
        for target_node in target_nodes
    }
    order = sorted(
        target_nodes,
        key=lambda target_node: (
            len(candidate_nodes[target_node]),
            -target_graph.in_degree(target_node) - target_graph.out_degree(target_node),
            target_node,
        ),
    )
    results: set[
        tuple[
            int,
            tuple[tuple[int, int], ...],
            tuple[tuple[int, tuple[tuple[int, int], ...]], ...],
        ]
    ] = set()

    def visit(
        depth: int,
        node_mapping: dict[int, int],
        used_query_nodes: set[int],
        variable_mapping: dict[int, dict[int, int]],
    ) -> None:
        if depth == len(order):
            for source, target in target_graph.edges:
                if not query_graph.has_edge(node_mapping[source], node_mapping[target]):
                    return
            results.add(normalize_match_result(node_mapping, variable_mapping, target_ident))
            return

        target_node = order[depth]
        for query_node in candidate_nodes[target_node]:
            if query_node in used_query_nodes:
                continue

            valid = True
            for predecessor in target_graph.pred[target_node]:
                if predecessor in node_mapping and not query_graph.has_edge(node_mapping[predecessor], query_node):
                    valid = False
                    break
            if not valid:
                continue
            for successor in target_graph.succ[target_node]:
                if successor in node_mapping and not query_graph.has_edge(query_node, node_mapping[successor]):
                    valid = False
                    break
            if not valid:
                continue

            next_variable_mapping = {
                variable_class: dict(identifier_mapping)
                for variable_class, identifier_mapping in variable_mapping.items()
            }
            target_vars = target_graph.nodes[target_node]["vars"]
            query_vars = query_graph.nodes[query_node]["vars"]
            for variable_class, target_identifier in enumerate(target_vars):
                query_identifier = query_vars[variable_class]
                existing = next_variable_mapping.setdefault(variable_class, {}).get(target_identifier)
                if existing is not None and existing != query_identifier:
                    valid = False
                    break
                next_variable_mapping[variable_class][target_identifier] = query_identifier
            if not valid:
                continue

            node_mapping[target_node] = query_node
            used_query_nodes.add(query_node)
            visit(depth + 1, node_mapping, used_query_nodes, next_variable_mapping)
            used_query_nodes.remove(query_node)
            del node_mapping[target_node]

    visit(0, {}, set(), {})
    return results


class TestRegressions(unittest.TestCase):
    def test_label_aware_symmetry_conditions(self):
        d = Database(node_label=lambda attrs: attrs['label'], node_vars=lambda attrs: attrs['vars'])

        target = nx.DiGraph()
        target.add_node(1, label=3, vars=())
        target.add_node(3, label=0, vars=())
        target.add_node(2, label=32, vars=(0,))
        target.add_node(0, label=5, vars=(3,))
        target.add_edges_from([(1, 0), (1, 2), (1, 3)])
        d.index(target, 999)

        query = nx.DiGraph()
        query.add_node(100, label=3, vars=())
        query.add_node(30, label=0, vars=())
        query.add_node(20, label=32, vars=(152,))
        query.add_node(10, label=5, vars=(251,))
        query.add_edges_from([(100, 10), (100, 20), (100, 30)])

        result = list(d.query_best_effort(query))

        assert len(result) == 1
        found_node_mapping, found_var_mapping, found_ident = result[0]
        assert is_valid_match(target, 999, query, found_node_mapping, found_var_mapping, found_ident)

class TestUnits(unittest.TestCase):
    def test_scc_fallback(self):
        d = Database(node_label=lambda attrs: attrs['label'], node_vars=lambda attrs: attrs['vars'])

        target = nx.DiGraph()
        next_node = 0
        previous_exit = None
        for component in range(20):
            cycle_nodes = [next_node, next_node + 1, next_node + 2]
            next_node += 3
            for offset, node in enumerate(cycle_nodes):
                target.add_node(node, label=component * 10 + offset, vars=())
            target.add_edge(cycle_nodes[0], cycle_nodes[1])
            target.add_edge(cycle_nodes[1], cycle_nodes[2])
            target.add_edge(cycle_nodes[2], cycle_nodes[0])
            if previous_exit is not None:
                target.add_edge(previous_exit, cycle_nodes[0])
            previous_exit = cycle_nodes[2]

        query = nx.DiGraph()
        for node, attrs in target.nodes(data=True):
            query.add_node(node + 1000, **attrs)
        for source, target_node in target.edges:
            query.add_edge(source + 1000, target_node + 1000)
        for distractor in range(200):
            query.add_node(distractor, label=10_000 + distractor, vars=())
        for distractor in range(199):
            query.add_edge(distractor, distractor + 1)

        d.index(target, 999)
        d._native_ops = 1
        d._ortools_enabled = False
        d._z3_enabled = False

        scc_calls = 0
        original_scc = d._find_single_graph_match_scc_decomposed

        def wrapped_scc(*args, **kwargs):
            nonlocal scc_calls
            scc_calls += 1
            return original_scc(*args, **kwargs)

        d._find_single_graph_match_scc_decomposed = wrapped_scc

        result = list(d.query_best_effort(query))

        assert scc_calls == 1
        assert len(result) == 1
        found_node_mapping, found_var_mapping, found_ident = result[0]
        assert is_valid_match(target, 999, query, found_node_mapping, found_var_mapping, found_ident)

    def test_multiple_identifiers(self):
        d = Database(node_label=lambda attrs: attrs['label'], node_vars=lambda attrs: attrs['vars'])

        first_target = nx.DiGraph()
        first_target.add_node(1, label=3, vars=())
        first_target.add_node(3, label=0, vars=())
        first_target.add_node(2, label=32, vars=(0,))
        first_target.add_node(0, label=5, vars=(3,))
        first_target.add_edges_from([(1, 0), (1, 2), (1, 3)])
        d.index(first_target, 999)

        second_target = nx.DiGraph()
        second_target.add_node(0, label=3, vars=())
        second_target.add_node(1, label=0, vars=())
        second_target.add_node(2, label=32, vars=(1,))
        second_target.add_node(3, label=5, vars=(4,))
        second_target.add_edges_from([(0, 1), (0, 2), (0, 3)])
        d.index(second_target, 998)

        query = nx.DiGraph()
        query.add_node(100, label=3, vars=())
        query.add_node(30, label=0, vars=())
        query.add_node(20, label=32, vars=(152,))
        query.add_node(10, label=5, vars=(251,))
        query.add_edges_from([(100, 10), (100, 20), (100, 30)])

        result = list(d.query(query))

        assert len(result) == 2
        by_ident = {found_ident: (found_node_mapping, found_var_mapping) for found_node_mapping, found_var_mapping, found_ident in result}
        assert set(by_ident) == {998, 999}
        first_mapping, first_vars = by_ident[999]
        assert is_valid_match(first_target, 999, query, first_mapping, first_vars, 999)
        second_mapping, second_vars = by_ident[998]
        assert is_valid_match(second_target, 998, query, second_mapping, second_vars, 998)
        assert set(first_vars) == {0}
        assert set(first_vars[0]) == {0, 3}
        assert set(first_vars[0].values()) == {152, 251}

    def test_multiple_matches(self):
        d = Database(node_label=lambda attrs: attrs['label'], node_vars=lambda attrs: attrs['vars'])

        target = nx.DiGraph()
        target.add_node(1, label=None, vars=())
        target.add_node(2, label=None, vars=())
        target.add_node(3, label=None, vars=())
        target.add_edge(1, 2)
        target.add_edge(2, 3)

        d.index(target, 0)

        query = nx.DiGraph()
        query.add_node(11, label=None, vars=())
        query.add_node(12, label=None, vars=())
        query.add_node(13, label=None, vars=())
        query.add_node(14, label=None, vars=())
        query.add_edge(11, 12)
        query.add_edge(12, 13)
        query.add_edge(13, 14)

        result = list(d.query(query))

        assert len(result) == 2
        found_nodes = frozenset(frozenset(mapping.values()) for mapping, _, _ in result)
        assert found_nodes == frozenset({frozenset({11,12,13}),frozenset({12,13,14})})


FUZZ_COUNT = int(os.environ.get("FUZZ_COUNT", 100))
FUZZ_OFFSET = int(os.environ.get("FUZZ_OFFSET", 0))
EXACT_FUZZ_COUNT = int(os.environ.get("EXACT_FUZZ_COUNT", 20))
EXACT_FUZZ_OFFSET = int(os.environ.get("EXACT_FUZZ_OFFSET", 0))

class TestFuzz(unittest.TestCase):
    def test_fuzz(self):
        def rcg(r: Random, n: int | None = None):
            if n is None:
                n = r.randrange(3, 100)
            return random_connected_graph(r, range(n), range(r.randrange(1, 10)), range(r.randrange(1, 100)), r.random(), r.random() * 0.1)

        for i in range(FUZZ_OFFSET, FUZZ_OFFSET + FUZZ_COUNT):
            r = Random(i)
            d: Database[int, int, int, int] = Database(node_label=lambda attrs: attrs['label'], node_vars=lambda attrs: attrs['vars'])

            for ident in range(r.randrange(100)):
                d.index(rcg(r), ident)
            targets = [rcg(r) for _ in range(r.randrange(1, 5))]
            for ident, t in enumerate(targets):
                d.index(t, ident + 100)

            needed_nodes = sum(len(t) for t in targets)
            query = rcg(r, r.randrange(needed_nodes, needed_nodes * 10))
            available_nodes = list(query)
            variables = list(range(1000))
            r.shuffle(variables)
            r.shuffle(available_nodes)
            for t in targets:
                embed_graph(query, t, available_nodes, variables)

            remaining_target_graphs = {
                target_ident + 100: target_graph
                for target_ident, target_graph in enumerate(targets)
            }
            for found_node_mapping, found_var_mapping, found_ident in d.query_best_effort(query):
                target_graph = remaining_target_graphs.get(found_ident)
                if target_graph is None:
                    continue
                assert is_valid_match(
                    target_graph,
                    found_ident,
                    query,
                    found_node_mapping,
                    found_var_mapping,
                    found_ident,
                )
                del remaining_target_graphs[found_ident]
                if not remaining_target_graphs:
                    break

            assert not remaining_target_graphs, "There is no corresponding finding for this target"

            print("OK", i)

    def test_fuzz_independent(self):
        label_start = 0
        def rcg(r: Random, n: int | None = None):
            nonlocal label_start
            if n is None:
                n = r.randrange(3, 100)
            label_count = r.randrange(1, 10)
            label_end = label_start + label_count
            result = random_connected_graph(r, range(n), range(label_start, label_end), range(r.randrange(1, 100)), r.random(), r.random() * 0.1)
            label_start = label_end
            return result

        for i in range(FUZZ_OFFSET, FUZZ_OFFSET + FUZZ_COUNT):
            r = Random(i)
            d = Database(node_label=lambda attrs: attrs['label'], node_vars=lambda attrs: attrs['vars'])
            label_start = 0

            for ident in range(r.randrange(100)):
                d.index(rcg(r), ident)
            targets = [rcg(r) for _ in range(r.randrange(1, 5))]
            for ident, t in enumerate(targets):
                d.index(t, ident + 100)

            needed_nodes = sum(len(t) for t in targets)
            query = rcg(r, r.randrange(needed_nodes, needed_nodes * 10))
            available_nodes = list(query)
            variables = list(range(1000))
            r.shuffle(variables)
            r.shuffle(available_nodes)
            for t in targets:
                embed_graph(query, t, available_nodes, variables)

            remaining_target_graphs = {
                target_ident + 100: target_graph
                for target_ident, target_graph in enumerate(targets)
            }
            for found_node_mapping, found_var_mapping, found_ident in d.query_best_effort(query):
                target_graph = remaining_target_graphs.get(found_ident)
                if target_graph is None:
                    continue
                assert is_valid_match(
                    target_graph,
                    found_ident,
                    query,
                    found_node_mapping,
                    found_var_mapping,
                    found_ident,
                )
                del remaining_target_graphs[found_ident]
                if not remaining_target_graphs:
                    break

            assert not remaining_target_graphs, "There is no corresponding finding for this target"

            print("OK", i)

    def test_fuzz_many_small_sparse_query(self):
        def rcg(r: Random, n: int | None = None):
            if n is None:
                n = r.randrange(3, 11)
            return random_connected_graph(
                r,
                range(n),
                range(r.randrange(1, 10)),
                range(r.randrange(1, 100)),
                r.random(),
                0.02,
            )

        for i in range(FUZZ_OFFSET, FUZZ_OFFSET + FUZZ_COUNT):
            r = Random(i)
            d = Database(node_label=lambda attrs: attrs['label'], node_vars=lambda attrs: attrs['vars'])

            for ident in range(r.randrange(500, 2000)):
                d.index(rcg(r), ident)
            targets = [rcg(r) for _ in range(r.randrange(1, 5))]
            for ident, t in enumerate(targets):
                d.index(t, ident + 2000)

            needed_nodes = sum(len(t) for t in targets)
            query = random_connected_graph(
                r,
                range(r.randrange(max(50, needed_nodes), 101)),
                range(r.randrange(1, 10)),
                range(r.randrange(1, 100)),
                r.random(),
                0.02,
            )
            available_nodes = list(query)
            variables = list(range(1000))
            r.shuffle(variables)
            r.shuffle(available_nodes)
            for t in targets:
                embed_graph(query, t, available_nodes, variables)

            remaining_target_graphs = {
                target_ident + 2000: target_graph
                for target_ident, target_graph in enumerate(targets)
            }
            for found_node_mapping, found_var_mapping, found_ident in d.query_best_effort(query):
                target_graph = remaining_target_graphs.get(found_ident)
                if target_graph is None:
                    continue
                assert is_valid_match(
                    target_graph,
                    found_ident,
                    query,
                    found_node_mapping,
                    found_var_mapping,
                    found_ident,
                )
                del remaining_target_graphs[found_ident]
                if not remaining_target_graphs:
                    break

            assert not remaining_target_graphs, "There is no corresponding finding for this target"

            print("OK", i)

    def test_fuzz_exact_small(self):
        def rcg(r: Random, n: int | None = None):
            if n is None:
                n = r.randrange(2, 6)
            return random_connected_graph(
                r,
                range(n),
                range(r.randrange(1, 5)),
                range(r.randrange(1, 8)),
                r.random() * 0.4,
                r.random() * 0.35,
            )

        for i in range(EXACT_FUZZ_OFFSET, EXACT_FUZZ_OFFSET + EXACT_FUZZ_COUNT):
            r = Random(i)
            d: Database[int, int, int, int] = Database(node_label=lambda attrs: attrs["label"], node_vars=lambda attrs: attrs["vars"])
            indexed_graphs: dict[int, nx.DiGraph[int]] = {}

            for ident in range(r.randrange(4, 9)):
                graph = rcg(r)
                indexed_graphs[ident] = graph
                d.index(graph, ident)

            targets = [rcg(r, r.randrange(2, 5)) for _ in range(r.randrange(1, 3))]
            next_ident = len(indexed_graphs)
            for target in targets:
                indexed_graphs[next_ident] = target
                d.index(target, next_ident)
                next_ident += 1

            needed_nodes = max(1, sum(len(t) for t in targets))
            query = rcg(r, r.randrange(max(needed_nodes, 3), max(needed_nodes + 1, 8)))
            available_nodes = list(query)
            variables = list(range(1000))
            r.shuffle(variables)
            r.shuffle(available_nodes)
            for target in targets:
                if len(available_nodes) < len(target):
                    break
                embed_graph(query, target, available_nodes, variables)

            actual_results = list(d.query(query))
            actual_subgraphs = set()
            for found_node_mapping, found_var_mapping, found_ident in actual_results:
                assert is_valid_match(
                    indexed_graphs[found_ident],
                    found_ident,
                    query,
                    found_node_mapping,
                    found_var_mapping,
                    found_ident,
                )
                actual_subgraphs.add(normalize_subgraph_result(found_node_mapping, found_ident))

            expected_subgraphs = {
                normalize_subgraph_result(dict(node_mapping), ident)
                for ident, target_graph in indexed_graphs.items()
                for _, node_mapping, _ in brute_force_matches(target_graph, ident, query)
            }

            assert len(actual_results) == len(actual_subgraphs)
            assert actual_subgraphs == expected_subgraphs

            print("EXACT OK", i)


if __name__ == '__main__':
    unittest.main()
