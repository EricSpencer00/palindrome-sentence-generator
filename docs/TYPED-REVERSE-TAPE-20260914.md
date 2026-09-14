# Typed reverse-tape search (14 September 2026)

This experiment changes the construction state rather than adding a reward
signal. A finite, authored lexical grammar enumerates complete left clauses.
For each clause, a trie dynamic program reverses its normalized letters and
enumerates legal word breaks for the right side. The engine records the exact
surface and provenance; no corpus sentence is copied and no language model is
queried per candidate. Word-frequency and frozen Brown bigram counts are only
traversal orderings.

## Replay

```bash
python3 experiments/typed_reverse_tape_search_20260914.py \
  --out runs/typed_reverse_300k_v4.json --max-clauses 300000 --seed 5
```

The run generated 300,000 finite-grammar clauses and 1,033 exact closures.
The independent validator was then run over every returned surface. The
longest rows were 50 letters, for example:

> We noticed that good gardener; ren ed rag doo gta ht dec ito new.

The normalized tape is exactly symmetric and its provenance identifies the
finite lexical product and trie inventory. These rows are **not** reader-worthy:
the right-hand word breaks are visibly fragments. They are retained as a
negative construction result, not presented as generated English prose.

The stricter short-word guard (at most two words of two letters or fewer) still
returned closures, but requiring every right-hand adjacent pair to be attested
in the frozen Brown transition table yielded zero admissible closures in the
same 300,000-clause budget. This is a concrete repair operator: short-word and
unseen-transition escape routes are removed from the search state rather than
defended after the fact.

No human readability claim is made. A candidate can enter the reader package
only after intact prose is independently authored, exactly audited, and then
blind-rated against intact and shuffled controls.
