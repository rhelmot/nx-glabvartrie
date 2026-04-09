gLabVarTrie
===========

An extension of `gLabTrie: A Data Structure for Motif Discovery with Constraints` to also handle variables. Variables in this context refers to the idea that some labels can be renamed as long as they are renamed along with the rest of the instances of that variable.

### Warning

This repo is slop. I needed this implementation as a dependency for something and do not have the time to do this right so I had gpt5.4 do it for me over the course of two days of aggressive handholding and a lot of testing. I wrote the public interfaces and the test cases exercising them by hand, and like, they pass, so it's not complete garbage, but I really wouldn't trust this code as a correct implementation of anything. The main fuzz test takes long enough on some of its iterations that I have never observed it terminate, but like it is kind of a brutal test.

### License

Accordingly, I put this code into the public domain. As the old prophets once said, go nuts, show nuts, whatever.
