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
    target_idents: set[int],
    query_graph: nx.DiGraph[int],
    node_mapping: dict[int, int],
    variable_mapping: Mapping[int, Mapping[int, int]],
    found_idents: set[int],
) -> bool:
    if set(node_mapping) != set(target_graph.nodes):
        return False
    if len(set(node_mapping.values())) != len(node_mapping):
        return False
    if not target_idents.issubset(found_idents):
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


def corresponds_to_witness_graph(target_graph: nx.DiGraph[int], witness_graph: nx.DiGraph[int]) -> bool:
    if target_graph.number_of_nodes() != witness_graph.number_of_nodes():
        return False
    if target_graph.number_of_edges() != witness_graph.number_of_edges():
        return False

    matcher = DiGraphMatcher(
        target_graph,
        witness_graph,
        node_match=lambda left_attrs, right_attrs: (
            left_attrs["label"] == right_attrs["label"]
            and len(left_attrs["vars"]) == len(right_attrs["vars"])
        ),
    )

    for mapping in matcher.isomorphisms_iter():
        target_to_witness_vars: defaultdict[int, dict[int, int]] = defaultdict(dict)
        witness_to_target_vars: defaultdict[int, dict[int, int]] = defaultdict(dict)
        for target_node, witness_node in mapping.items():
            target_vars = target_graph.nodes[target_node]["vars"]
            witness_vars = witness_graph.nodes[witness_node]["vars"]
            if len(target_vars) != len(witness_vars):
                break
            for slot, target_identifier in enumerate(target_vars):
                witness_identifier = witness_vars[slot]
                existing = target_to_witness_vars[slot].get(target_identifier)
                if existing is not None and existing != witness_identifier:
                    break
                reverse_existing = witness_to_target_vars[slot].get(witness_identifier)
                if reverse_existing is not None and reverse_existing != target_identifier:
                    break
                target_to_witness_vars[slot][target_identifier] = witness_identifier
                witness_to_target_vars[slot][witness_identifier] = target_identifier
            else:
                continue
            break
        else:
            return True

    return False

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

        result = list(d.query(query))

        assert len(result) == 1
        witness_graph, found_node_mapping, found_var_mapping, found_idents = result[0]
        assert corresponds_to_witness_graph(target, witness_graph)
        assert is_valid_match(witness_graph, {999}, query, found_node_mapping, found_var_mapping, found_idents)

class TestUnits(unittest.TestCase):
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

        assert len(result) == 1
        witness_graph, found_node_mapping, found_var_mapping, found_idents = result[0]
        assert corresponds_to_witness_graph(first_target, witness_graph)
        assert corresponds_to_witness_graph(second_target, witness_graph)
        assert is_valid_match(witness_graph, {999, 998}, query, found_node_mapping, found_var_mapping, found_idents)
        assert set(found_var_mapping) == {0}
        assert set(found_var_mapping[0]) == {0, 3}
        assert set(found_var_mapping[0].values()) == {152, 251}

FUZZ_COUNT = int(os.environ.get("FUZZ_COUNT", 100))
FUZZ_OFFSET = int(os.environ.get("FUZZ_OFFSET", 0))

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

            result = list(d.query(query))
            for target_ident, target_graph in enumerate(targets):
                for witness_graph, found_node_mapping, found_var_mapping, found_idents in result:
                    if (
                        corresponds_to_witness_graph(target_graph, witness_graph)
                        and is_valid_match(witness_graph, {target_ident + 100}, query, found_node_mapping, found_var_mapping, found_idents)
                    ):
                        break
                else:
                    assert False, "There is no corresponding finding for this target"

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

            result = list(d.query(query))
            for target_ident, target_graph in enumerate(targets):
                for witness_graph, found_node_mapping, found_var_mapping, found_idents in result:
                    if (
                        corresponds_to_witness_graph(target_graph, witness_graph)
                        and is_valid_match(witness_graph, {target_ident + 100}, query, found_node_mapping, found_var_mapping, found_idents)
                    ):
                        break
                else:
                    assert False, "There is no corresponding finding for this target"

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

            result = list(d.query(query))
            for target_ident, target_graph in enumerate(targets):
                for witness_graph, found_node_mapping, found_var_mapping, found_idents in result:
                    if (
                        corresponds_to_witness_graph(target_graph, witness_graph)
                        and is_valid_match(witness_graph, {target_ident + 2000}, query, found_node_mapping, found_var_mapping, found_idents)
                    ):
                        break
                else:
                    assert False, "There is no corresponding finding for this target"

            print("OK", i)


if __name__ == '__main__':
    unittest.main()
