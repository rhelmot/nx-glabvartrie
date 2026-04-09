gLabVarTrie
===========

An extension of `gLabTrie: A Data Structure for Motif Discovery with Constraints` to also handle variables. Variables in this context refers to the idea that in addition to a label, nodes also have an ordered list of variables which need to be consistently renamed across the graph to match a subgraph.

### Interface

construct a `Database` object. It takes two parameters, each a callable that takes as a single parameter a node attributes `dict[str, Any]`. One should return the node's label, and one should return a tuple of the node's variables. Then, you can index graphs by calling `db.index(graph, ident)`. Finally, you can query for indexed graphs which match any subgraph of a query graph by calling `db.query(graph)`. This method returns a list of matches, including a sample subgraph, the node and variable mappings from the sample subgraph to the query graph, and the identifiers that correspond to the indexed graphs that we matched. `Database` takes four type parameters - the networkx node type, the node label type, the node variable type, and the identifier type.

### Warning

This repo is slop. I needed this implementation as a dependency for something and do not have the time to do this right so I had gpt5.4 do it for me over the course of two days of aggressive handholding and a lot of testing. I wrote the public interfaces and the test cases exercising them by hand, and like, they pass, so it's not complete garbage, but I really wouldn't trust this code as a correct implementation of anything. The main fuzz test takes long enough on some of its matching that I have never observed it terminate, but like it is kind of a brutal test.

### License

Accordingly, I put this code into the public domain. As the old prophets once said, go nuts, show nuts, whatever.
