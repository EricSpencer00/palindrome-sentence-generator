# Bidirectional boundary beam (2026-09-17)

This experiment searches a new degree of freedom: keep an authored English
left clause, reverse its letters, and use a trie beam to choose *different*
word boundaries on the right.  A Zipf lexical cost prefers ordinary words;
the right side is also passed through a small clause-shape gate.  No word
sequence is mirrored and no self-palindromic unit is inserted.  The run is
bounded to 176 seeds (148 authored lines plus 28 labelled fixture pairs),
beam 80, and words of at most 12 letters.

The best mechanically valid output was:

```
no one was at home | em oh ta sa we no on   (28 letters)
```

It is not readable: the lexical beam exploits short words and abbreviations.
Other top outputs (`a nation was on it | ti no saw no it ana`, 28 letters)
show the same failure.  This is a useful negative result, not an accepted
palindrome.

Every reported candidate was independently checked with a two-pointer scan,
and the normalized full tape's SHA-256 is emitted (for the first candidate:
`f55f48f1da984c1298cf1c1c8658cc01a7288da22e93086074c33fee33db1b90`).  The
reverse-tape check separately proves that the right segmentation consumes
exactly the reversed left tape.

Novelty: unlike word-order-only symmetry, this searches segmentation boundaries
while scoring lexical cost; unlike catalogue mining, provenance is authored
seeds plus explicitly labelled smoke fixtures.  Next repair is to replace the
permissive shape fallback with a tagged POS trie and impose a minimum Zipf
frequency/closed-class budget, then add a boundary-aware phrase corpus score.
