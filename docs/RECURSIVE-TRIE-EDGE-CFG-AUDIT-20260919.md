# Rejection audit: recursive trie-edge CFG lane

`5cbfe4a4` is invalid constructive evidence. The trie recursion only listed
individual words; the subsequent loop paired words and checked an outermost
character. It carried no complete-sentence grammar state or mirrored
character obligation through the full tape, and its rendered rows were
fragments rather than clauses. Do not integrate this lane. The next attempt
must be a genuine grammar-state product over trie edges with complete parses.
