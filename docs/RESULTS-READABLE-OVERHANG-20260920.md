# Readable live-overhang search — 2026-09-20

This run tested a fresh construction lane after the readable reference was
separated from the 286k lexical benchmark. It mined ordinary forward n-grams
from the Brown corpus, then grew the two rendered sides with a center-out
character overhang. The right side was emitted in ordinary reading order; no
finished sentence was reversed or repaired afterward.

The 140,000-unit inventory was searched from 120 seeded frontiers. It found
one exact closure at 40 normalized letters:

> as a was it is it it is away ya was it it is it is a was a

The independent two-pointer audit and forward/reverse SHA-256 agree, but the
candidate is not readable and contains a proper hidden palindromic span. It is
therefore retained only as a negative diagnostic; `longest_clean_exact` is 0.

A separate forward-grammar sweep paired 22,686 complete Brown sentences with
an independently generated reverse-side CFG. It rendered no exact closures.

The run does not change the admitted readable benchmark: **38 normalized
letters**, `An aide rips nine memos; some men inspire Diana.` The long Norvig
construction remains a separate lexical-tile result, not readable prose.

The executable lane is
`experiments/brown_phrase_live_overhang_20260920.py`; the wordwise grammar
diagnostic is `experiments/wordwise_grammar_search_20260920.py`.
