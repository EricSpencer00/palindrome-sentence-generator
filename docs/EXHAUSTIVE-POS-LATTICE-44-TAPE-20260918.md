# Exhaustive POS-lattice resegmentation of the 44-letter tape (2026-09-18)

This is a bounded, exhaustive resegmentation audit of the fixed diagnostic
tape `wasnoelaneraagasanitemmetinasagaarenaleonsaw`. It is not a new readable
palindrome result.

The generator intersects every tape substring with `data/lexicon.txt`, then
enumerates every complete dictionary segmentation (no language-model beam and
no post-hoc reversal). In this lexicon intersection there are 30 usable edges
but only **one** complete segmentation:

> Was Noel an era, a gas, an item met in a saga, arena, Leon saw.

An authored, deliberately permissive POS recognizer flags that path as a
candidate-shaped question/verb sequence, but it does not certify syntax,
meaning, or human readability. There are 0 alternate complete POS-lattice
renderings and 0 strict admissions.

The independent audit confirms 44 letters and forward/reverse tape equality.
The word sequence is not itself a reverse-token list, but the normalized tape
contains proper palindromic islands, including `anitemmetina`, `itemmeti`,
`temmet`, `emme`, `aga`, and `asa` (among larger nested spans). Therefore the
shortcut-free gate is closed before any reader test. This does not claim that a
different lexicon or grammar could not produce another segmentation; it records
exactly what this declared finite lattice found.

Evidence is replayable from
`experiments/exhaustive_pos_lattice_44_tape_20260918.py` and
`runs/exhaustive-pos-lattice-44-tape-20260918.json`, including generator and
lexicon SHA-256 provenance. The next constructive operation is to author a
complete finite clause at a changed outer seam and rerun the hard span audit;
continuing to resegment this immutable tape cannot remove its internal islands.
