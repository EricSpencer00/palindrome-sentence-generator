# Live CFG character intersection (21 September 2026)

`experiments/live_cfg_character_intersection_20260921.py` is a distinct
construction lane from complete-tape resegmentation and fixed slot products.
It maintains two unfinished CFG derivation stacks. The left stack expands in
ordinary reading order; the right stack expands from the final symbol using
reversed production right-hand sides. Each terminal is admitted only if it
consumes the current character debt, so the solver never renders a sentence
and reverses it afterward.

The bounded run visited 38 unique chart states, pruned 116 incompatible
terminal transitions, and reached two complete derivations. One is the known
calibration seed, rendered directly from live terminal expansions:

> An aide rips nine memos; some men inspire Diana.

It has 38 normalized letters. Independent pointer comparison and SHA-256 of
the normalized tape and its reverse both pass (`ce71723a...184c6`). The words
are distinct and no word is self-palindromic. This is exact construction
evidence, not human-readability evidence; the reader gate remains closed.

The run produced no novel output longer than 38 letters. The next concrete
operator is a typed adjunct production added to the highest-scoring live chart
frontier, with an independent grammatical-control test and the same audit.
Evidence: `runs/live-cfg-character-intersection-20260921.json`.
