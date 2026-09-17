# Paired typed-CFG chart (2026-09-17)

This run is a new chart/intersection construction, not a beam enlargement. Two
independent derivations use the typed frame `DET SUBJ VERB OBJ PREP PLACE`.
The verb production carries a transitive valency feature; lexical choices on
the left and right are made independently. A chart state is
`(production-index, residual-side, residual-characters, valency)`. Terminal
characters are consumed from opposite edges as productions are expanded, so a
state is rejected on its first mirrored-character conflict before a complete
sentence is rendered.

## Result

The run produced chart sizes `[1, 0]`: all 30 paired first-production
expansions conflicted at the first terminal, and there were no complete
derivations or exact candidates. This is an exclusion of the current lexical
boundary domains, not evidence that typed CFG construction is impossible.
The independent audit is implemented in the run itself and records zero
rendered candidates; no readability claim is made.

## Concrete repair

The next implementation should author a larger, independently checked lexical
boundary inventory for the outer `DET`/`PLACE` pair (including inflected and
proper-name place expressions) and propagate character-level terminal edges
within each word. It must retain the same semantic frame and chart-state
definition, then run a blinded reader screen only if an exact closure survives.
