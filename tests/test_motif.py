from __future__ import annotations

import os
import unittest
from typing import Callable
from random import Random

import networkx as nx

from glabvartrie import MotifFinder
from common import CantEmbed, random_connected_graph, embed_graph, is_isomorphic_match, generate_random_subgraphs

FUZZ_COUNT = int(os.environ.get("FUZZ_COUNT", 100))
FUZZ_OFFSET = int(os.environ.get("FUZZ_OFFSET", 0))

def node_label(attrs):
    return attrs['label']
def node_vars(attrs):
    return attrs['vars']

def generate_and_test_corpus(r: Random, graph_generator: Callable[[Random], nx.DiGraph[int]], motif_generator: Callable[[Random], nx.DiGraph[int]], num_graphs: int, num_motifs: int, max_num_embeddings: int, max_num_variables: int = 1000, label_matches = None):
    graphs = [(graph_generator(r), []) for i in range(num_graphs)]
    motifs: list[tuple[nx.DiGraph[int], set[int]]] = [(motif_generator(r), set()) for i in range(num_motifs)]
    motif_embeddings: list[list[tuple[int, tuple[int, ...]]]] = [[] for _ in range(num_motifs)]
    max_motif_size = max((len(motif) for motif, _ in motifs), default=0)

    for g, l in graphs:
        l.extend(g)
        r.shuffle(l)

    for motif_index, (motif, found_in) in enumerate(motifs):
        num_embeddings = r.randrange(2, max_num_embeddings)
        for _ in range(num_embeddings):
            graphshuf = list(enumerate(graphs))
            r.shuffle(graphshuf)
            for graph_idx, (graph, remaining_nodes) in graphshuf:
                graph_idx = r.randrange(0, len(graphs))
                graph, remaining_nodes = graphs[graph_idx]
                try:
                    embedding = embed_graph(graph, motif, remaining_nodes)
                except CantEmbed:
                    continue

                found_in.add(graph_idx)
                motif_embeddings[motif_index].append((graph_idx, tuple(sorted(embedding.values()))))
                break

    to_index: list[tuple[nx.DiGraph[int], int]] = []
    for i, (motif, _) in enumerate(motifs):
        for gidx, embedded_nodes in motif_embeddings[i]:
            g = graphs[gidx][0]
            motif_in_g = g.subgraph(embedded_nodes).copy()
            assert isinstance(motif_in_g, nx.DiGraph)
            to_index.append((motif_in_g, gidx))
            for g2 in generate_random_subgraphs(r, motif_in_g, 10):
                to_index.append((g2, gidx))
    for gidx, (g, _) in enumerate(graphs):
        for g2 in generate_random_subgraphs(r, g, 1000):
            to_index.append((g2, gidx))

    database: MotifFinder[int, int, int, int] = MotifFinder(
        ((g, s) for g, s in to_index),
        lambda v: len(v[0]),
        node_label,
        node_vars,
        label_matches=label_matches,
        max_motif_size=max_motif_size,
    )

    motif_sizes = list(set(len(g) for g, _ in motifs))
    r.shuffle(motif_sizes)

    for motif_size in motif_sizes:
        wanted_motifs = {(m, frozenset(gs)) for m, gs in motifs if len(m) == motif_size and gs}
        for found in database.motifs(motif_size):
            source_set = {inst_source for inst_source, _ in found}
            for motif, ideal_sources in wanted_motifs:
                if not ideal_sources.issubset(source_set):
                    continue
                for inst_source, inst_nodes in found:
                    inst_g, _ = graphs[inst_source]
                    inst_subg = inst_g.subgraph(inst_nodes)
                    assert isinstance(inst_subg, nx.DiGraph)
                    if not is_isomorphic_match(motif, inst_subg, node_label, node_vars, label_matches):
                        # this isn't the right motif
                        break
                else:
                    # every found instance of this found motif was isomorphic to one of our target motifs
                    # don't try to match against this target motif again
                    wanted_motifs.remove((motif, ideal_sources))
                    # move onto the next found motif
                    break
            else:
                # no wanted motif matches this motif. this is potentially fine, just move on
                pass
        assert not wanted_motifs, f"We missed {len(wanted_motifs)} embedded motif(s) of size {motif_size}"


class TestFuzz(unittest.TestCase):
    def test_fuzz_small(self):
        for seed in range(FUZZ_OFFSET, FUZZ_OFFSET + FUZZ_COUNT):
            rr = Random(seed)
            generate_and_test_corpus(
                rr,
                graph_generator=lambda r: random_connected_graph(
                    r,
                    nodes=range(r.randrange(20, 26)),
                    labels=range(100),
                    variables=range(50),
                    variable_density=1.0,
                    edge_density=0.03,
                ),
                motif_generator=lambda r: random_connected_graph(
                    r,
                    nodes=range(r.randrange(3, 5)),
                    labels=range(100),
                    variables=range(50),
                    variable_density=1.0,
                    edge_density=0.15,
                ),
                num_graphs=2,
                num_motifs=2,
                max_num_embeddings=3,
            )
            print("OK", seed)

    def test_fuzz_big(self):
        for seed in range(FUZZ_OFFSET, FUZZ_OFFSET + FUZZ_COUNT):
            rr = Random(seed)
            num_graphs = rr.randrange(50, 200)
            max_num_embeddings = 10
            max_motif_size = 50
            min_motif_size = 3
            minimum_graph_size = 100
            minimum_available_nodes = num_graphs * minimum_graph_size
            max_motifs = minimum_available_nodes // max_motif_size
            num_motifs = rr.randrange(1, max_motifs)

            generate_and_test_corpus(
                rr,
                graph_generator=lambda r: random_connected_graph(
                    r,
                    nodes=range(r.randrange(minimum_graph_size, minimum_graph_size * 3)),
                    labels=range(10000),
                    variables=range(50),
                    variable_density=1.,
                    edge_density=0.1,
                ),
                motif_generator=lambda r: random_connected_graph(
                    r,
                    nodes=range(r.randrange(min_motif_size, max_motif_size+1)),
                    labels=range(10000),
                    variables=range(50),
                    variable_density=1,
                    edge_density=0.2,
                ),
                num_graphs=num_graphs,
                num_motifs=num_motifs,
                max_num_embeddings=max_num_embeddings,
            )
            print("OK", seed)

if __name__ == '__main__':
    unittest.main()
