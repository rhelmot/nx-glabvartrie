import unittest
from random import Random

import networkx as nx

from glabvartrie import Database

def random_connected_graph(r: Random, nodes: int, labels: int, variable_kinds: int, variables: int, variable_density: float, edge_density: float) -> nx.DiGraph[int]:
    g = nx.DiGraph()
    for n in range(nodes):
        if r.random() < variable_density:
            label = (r.randrange(variable_kinds), r.randrange(variables))
        else:
            label = (None, r.randrange(labels))
        g.add_node(n, label=label)
    wccs = [[n] for n in range(nodes)]
    while len(wccs) > 1:
        r.shuffle(wccs)
        wcc1 = wccs.pop()
        wcc2 = wccs.pop()
        n1 = r.choice(wcc1)
        n2 = r.choice(wcc2)
        g.add_edge(n1, n2)
        wccs.append(wcc1 + wcc2)

    all_edges = nodes * nodes
    expected_edges = all_edges * edge_density
    current_edges = nodes - 1
    desired_expected_edges = expected_edges - current_edges
    remaining_tries = all_edges - current_edges
    new_density = desired_expected_edges / remaining_tries

    for n1 in range(nodes):
        for n2 in range(nodes):
            if g.has_edge(n1, n2):
                continue
            if r.random() < new_density:
                g.add_edge(n1, n2)

    return g


class TestFuzz(unittest.TestCase):
    def test_fuzz(self):
        for i in range(100):
            r = Random(i)
            d = Database(node_label=lambda attrs: attrs['label'])
            def rcg(n=None):
                if n is None:
                    n = r.randrange(3, 100)
                return random_connected_graph(r, n, r.randrange(1, 10), r.randrange(1, 100), r.randrange(1, 10), r.random(), r.random())

            for _ in range(r.randrange(100)):
                d.update(rcg())
            targets = [rcg() for _ in range(r.randrange(1, 5))]
            for t in targets:
                d.update(t)

            query = rcg(r.randrange(100, 1000))
            nodes = list(query)
            variables = list(range(1000))
            r.shuffle(variables)
            r.shuffle(nodes)
            target_info = []
            for t in targets:
                m = t.copy()
                var_mapping = {}
                node_mapping = {}
                for n in m:
                    matching = nodes.pop()
                    node_mapping[n] = m
                    varkind, idx = m.nodes[n]['label']
                    if varkind is None:
                        query.nodes[matching]['label'] = (None, idx)
                    elif (varkind, idx) in var_mapping:
                        query.nodes[matching]['label'] = (varkind, var_mapping[varkind, idx])
                    else:
                        new_var = variables.pop()
                        var_mapping[varkind, idx] = new_var
                        query.nodes[matching]['label'] = (varkind, new_var)

                target_info.append((node_mapping, var_mapping))

            result = d.query(query)
            for found_graph, found_node_mapping, found_var_mapping in result:
                for idx, (target_graph, (node_mapping, var_mapping)) in enumerate(zip(targets, target_info)):
                    if target_graph is found_graph and node_mapping == found_node_mapping and var_mapping == found_var_mapping:
                        break
                else:
                    assert False, "Could not find a target subgraph in the query"


if __name__ == '__main__':
    unittest.main()

