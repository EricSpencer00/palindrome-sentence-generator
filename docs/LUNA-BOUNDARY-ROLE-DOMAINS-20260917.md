# Boundary-conditioned role domains (2026-09-17)

This repair extends the typed clause automaton with role-specific lexical
domains bucketed by each word's first and last letter. The chart compares the
exposed terminal character of the ordinary-order clause pair before expanding
the next grammatical role, and retains grammatical role, valency, and lexical
provenance in every state. It does not reverse a completed tape, copy words,
or use catalogue material.

## Result

The bounded run used a 300,000-state limit and a 45-letter admission floor.
The chart counts were `[1, 4, 0]`: four determiner/place boundary states
survived, but all subject/preposition expansions conflicted at the next live
character seam. It produced 0 rendered candidates and 0 exact closures. The
independent audit is performed by `mechanical_admission_checks` plus a second
normalization-and-reversal equality check in the run artifact.

The failure is informative but not a paper result. The next concrete repair is
to author boundary-compatible synonym sets jointly for the `SUBJ/PREP` pair,
then split multiword place expressions into terminal-bearing lexical units so
that a grammatical phrase can carry the residual debt across a boundary.
