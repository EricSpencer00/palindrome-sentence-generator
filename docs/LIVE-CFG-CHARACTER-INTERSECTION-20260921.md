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

## Typed adjunct continuation

The continuation adds temporal `LAdj`/`RAdj` productions (`at dawn`, `by noon`,
`near dusk`, `after rain`) to both CFG sides. They are expanded as ordinary
terminals while obligations remain live. It visited 58 states, pruned 182
transitions, and reached six terminal derivations. The longest rendered
frontier was:

> An aide rips nine memos near dusk; some men inspire Diana.

This is 46 letters but not exact; the remaining live debt is `neardusk`.
Other 44-letter adjunct controls also fail closure. The seed remains the only
shortcut-clean exact result; these controls are retained as concrete frontiers,
not promoted candidates. The next operator is a typed adjunct whose lexical
terminals are selected by the opposing residual rather than appended after a
complete derivation. A follow-up grammar expansion makes adjunct placement
explicitly variable: the left CFG permits a temporal frame before the subject
or after the object, while the right CFG permits a frame after the predicate
argument. The bounded run visited 83 states and pruned 262 transitions. It
still recovers the 38-letter seed and records the same 44--47-letter temporal
frontiers, but no new exact candidate; no output is promoted as readable
evidence.
