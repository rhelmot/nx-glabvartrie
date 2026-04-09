from __future__ import annotations

import os
import unittest
from collections.abc import Hashable, Mapping
from collections import defaultdict
from random import Random

import networkx as nx

from glabvartrie import Database

def random_connected_graph(r: Random, nodes: range, labels: range, variable_kinds: range, variables: range, variable_density: float, edge_density: float) -> nx.DiGraph[int]:
    g = nx.DiGraph()
    for n in nodes:
        if r.random() < variable_density:
            label = (r.choice(variable_kinds), r.choice(variables))
        else:
            label = (None, r.choice(labels))
        g.add_node(n, label=label)
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
    query_graph: nx.DiGraph[int],
    node_mapping: dict[int, int],
    variable_mapping: Mapping[Hashable, Mapping[int, int]],
) -> bool:
    if set(node_mapping) != set(target_graph.nodes):
        return False
    if len(set(node_mapping.values())) != len(node_mapping):
        return False

    computed_variable_mapping: defaultdict[Hashable, dict[int, int]] = defaultdict(dict)
    for target_node, query_node in node_mapping.items():
        if query_node not in query_graph:
            return False

        target_class, target_identifier = target_graph.nodes[target_node]["label"]
        query_class, query_identifier = query_graph.nodes[query_node]["label"]
        if target_class is None:
            if query_class is not None or query_identifier != target_identifier:
                return False
        else:
            if query_class != target_class:
                return False
            existing = computed_variable_mapping[target_class].get(target_identifier)
            if existing is not None and existing != query_identifier:
                return False
            computed_variable_mapping[target_class][target_identifier] = query_identifier

    for source, target in target_graph.edges:
        if not query_graph.has_edge(node_mapping[source], node_mapping[target]):
            return False

    return {
        variable_class: dict(identifier_mapping)
        for variable_class, identifier_mapping in computed_variable_mapping.items()
    } == dict(variable_mapping)

def embed_graph(query: nx.DiGraph[int], t: nx.DiGraph[int], available_nodes: list[int], variables: list[int]):
    var_mapping: defaultdict[str, dict[int, int]] = defaultdict(dict)
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
        varkind, idx = t.nodes[n]['label']
        if varkind is None:
            query.nodes[matching]['label'] = (None, idx)
        elif idx in var_mapping[varkind]:
            query.nodes[matching]['label'] = (varkind, var_mapping[varkind][idx])
        else:
            new_var = variables.pop()
            var_mapping[varkind][idx] = new_var
            query.nodes[matching]['label'] = (varkind, new_var)

class TestRegressions(unittest.TestCase):
    def test_label_aware_symmetry_conditions(self):
        d: Database[int, int] = Database(node_label=lambda attrs: attrs['label'])

        target = nx.DiGraph()
        target.add_node(1, label=(None, 3))
        target.add_node(3, label=(None, 0))
        target.add_node(2, label=(32, 0))
        target.add_node(0, label=(5, 3))
        target.add_edges_from([(1, 0), (1, 2), (1, 3)])
        d.index(target)

        query = nx.DiGraph()
        query.add_node(100, label=(None, 3))
        query.add_node(30, label=(None, 0))
        query.add_node(20, label=(32, 152))
        query.add_node(10, label=(5, 251))
        query.add_edges_from([(100, 10), (100, 20), (100, 30)])

        result = list(d.query(query))

        self.assertEqual(len(result), 1)
        found_graph, found_node_mapping, found_var_mapping = result[0]
        self.assertIs(found_graph, target)
        self.assertTrue(is_valid_match(target, query, found_node_mapping, found_var_mapping))


class TestFuzz(unittest.TestCase):
    def test_fuzz(self):
        fuzz_count = int(os.environ.get("FUZZ_COUNT", 100))
        fuzz_offset = int(os.environ.get("FUZZ_OFFSET", 0))

        def rcg(r: Random, n: int | None = None):
            if n is None:
                n = r.randrange(3, 100)
            return random_connected_graph(r, range(n), range(r.randrange(1, 10)), range(r.randrange(1, 100)), range(r.randrange(1, 10)), r.random(), r.random() * 0.1)

        for i in range(fuzz_offset, fuzz_offset + fuzz_count):
            r = Random(i)
            d: Database[int, int] = Database(node_label=lambda attrs: attrs['label'])

            for _ in range(r.randrange(100)):
                d.index(rcg(r))
            targets = [rcg(r) for _ in range(r.randrange(1, 5))]
            for t in targets:
                d.index(t)

            needed_nodes = sum(len(t) for t in targets)
            query = rcg(r, r.randrange(needed_nodes, needed_nodes * 10))
            available_nodes = list(query)
            variables = list(range(1000))
            r.shuffle(variables)
            r.shuffle(available_nodes)
            for t in targets:
                embed_graph(query, t, available_nodes, variables)

            result = list(d.query(query))
            for target_graph in targets:
                for found_graph, found_node_mapping, found_var_mapping in result:
                    if target_graph is found_graph and is_valid_match(target_graph, query, found_node_mapping, found_var_mapping):
                        break
                else:
                    assert False, "There is no corresponding finding for this target"

            print("OK", i)

    def test_fuzz_independent(self):
        fuzz_count = int(os.environ.get("FUZZ_COUNT", 100))
        fuzz_offset = int(os.environ.get("FUZZ_OFFSET", 0))

        label_start = 0
        def rcg(r: Random, n: int | None = None):
            nonlocal label_start
            if n is None:
                n = r.randrange(3, 100)
            label_count = r.randrange(1, 10)
            label_end = label_start + label_count
            result = random_connected_graph(r, range(n), range(label_start, label_end), range(r.randrange(1, 100)), range(r.randrange(1, 10)), r.random(), r.random() * 0.1)
            label_start = label_end
            return result

        for i in range(fuzz_offset, fuzz_offset + fuzz_count):
            r = Random(i)
            d: Database[int, int] = Database(node_label=lambda attrs: attrs['label'])
            label_start = 0

            for _ in range(r.randrange(100)):
                d.index(rcg(r))
            targets = [rcg(r) for _ in range(r.randrange(1, 5))]
            for t in targets:
                d.index(t)

            needed_nodes = sum(len(t) for t in targets)
            query = rcg(r, r.randrange(needed_nodes, needed_nodes * 10))
            available_nodes = list(query)
            variables = list(range(1000))
            r.shuffle(variables)
            r.shuffle(available_nodes)
            for t in targets:
                embed_graph(query, t, available_nodes, variables)

            result = list(d.query(query))
            for target_graph in targets:
                for found_graph, found_node_mapping, found_var_mapping in result:
                    if target_graph is found_graph and is_valid_match(target_graph, query, found_node_mapping, found_var_mapping):
                        break
                else:
                    assert False, "There is no corresponding finding for this target"

            print("OK", i)


if __name__ == '__main__':
    unittest.main()
