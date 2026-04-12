gLabVarTrie
===========

An extension of `gLabTrie: A Data Structure for Motif Discovery with Constraints` to also handle variables. Variables in this context refers to the idea that in addition to a label, nodes also have an ordered list of variables which need to be consistently renamed across the graph to match a subgraph.

### Interface

construct a `Database` object. It takes two parameters, each a callable that takes as a single parameter a node attributes `dict[str, Any]`. One should return the node's label, and one should return a tuple of the node's variables. Then, you can index graphs by calling `db.index(graph, ident)`. Finally, you can query for indexed graphs which match any subgraph of a query graph by calling `db.query(graph)`. This method returns a list of matches, including the node and variable mappings from the discovered match to the query graph and the identifier from the indexed graphs that we matched. It also takes an optional boolean parameter indicating whether the search should be best-effort (with deterministic timeouts, not the default) or exhaustive. `Database` takes four type parameters - the networkx node type, the node label type, the node variable type, and the identifier type. Note that if you supply an un-sortable node type you will need to supply an extra function parameter to `Database` implementing a sort key.

### Warning

This repo is slop. I needed this implementation as a dependency for something and do not have the time to do this right so I had gpt5.4 do it for me over the course of two days of aggressive handholding and a lot of testing. I wrote the public interfaces and the fuzz tests exercising them by hand, and like, they pass, so it's not complete garbage, but I really wouldn't trust this code as a correct implementation of anything. Trust but verify, you know?

### License

Accordingly, I put this code into the public domain. As the old prophets once said, go nuts, show nuts, whatever.
