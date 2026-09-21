# What was actually executed

Source review found that algorithm names in the experiment registry overstate
the implementations. These results cannot establish that online grammatical
palindrome intersection has been exhausted.

* `shared_tape_finite_automata_role_ledger_20260921.py` builds complete clauses,
  enumerates their Cartesian product, and calls `audit(text)`. No transition
  propagates a character obligation. Its method and novelty metadata have been
  corrected, preserving the historical results.
* `lexicalized_macro_grammar_20260921.py` compares the last character of each
  clause. That is not the outer endpoint equation for their concatenation.
* `semantic_slot_lattice_smt.py` contains nested loops over complete strings;
  it does not invoke SMT or propagate partial character equations.
* `cfg_earley_character_intersection_20260916.py` scans a fixed authored text.
  Its initial chart item never predicts a lexical production; the scan cannot
  substantiate a working Earley generator.

These are source-level findings about these particular scripts, not a claim
that every previous implementation is invalid.

## Executable replacement

`experiments/online_regular_language_palindrome_20260921.py` compiles lexical
alternatives into an acyclic NFA. Its state consists of a forward state and a
backward state of the same accepting path. Each transition chooses an equal
character from both frontiers. Only a meeting state or a single central edge
permits rendering. Every resulting path is checked for automaton connectivity,
then the rendered string is checked with an independent two-pointer loop and
forward/reverse SHA-256.

This is an implementation of a known intersection construction, not a claim of
a new theoretical algorithm. Merging equivalent frontier states retains one
witness per state/depth: it preserves existence, not every surface variant.
Three tiny finite languages are compared with exhaustive oracles. The known
38-letter seed is a declared regression input only, excluded from the generation
grammar. The fresh grammar represents 2,304 complete two-clause realizations.
No readability score or rendered-string repair participates in search.

The first grammar has zero exact closures. Its deepest failure occurs after
five matched character pairs: plural verb endings cannot supply the required
letter. The immediately executed follow-up makes the second subject and its
agreeing singular verb a single alternative. The run records its actual
frontier and closure results separately. Both tests are bounded diagnostics;
no new reader-worthy palindrome is claimed.

The next useful investment is a lexicon with grammatical agreement attached
to transitions and a shared semantic frame. The engine can expose missing
character transitions before rendering. It does not establish that a large
grammar will contain coherent long palindromes, and human readership remains
the final evidence for that requirement.
