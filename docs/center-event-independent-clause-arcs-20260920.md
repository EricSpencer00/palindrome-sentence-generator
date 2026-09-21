# Complete center event with independent clause arcs (2026-09-20)

This run selects a complete, non-palindromic center utterance first, then
independently authors a forward clause on each side. Arc characters are checked
against the opposing live edge before admission; no finished tape is reversed
and no candidate is repaired after rendering.

The bounded bank contains 3 center events and 12 arc pairs. All 12 are retained
as prose controls with their first live mismatch, normalized character count,
and forward/reverse SHA-256 values in
`runs/center-event-independent-clause-arcs-20260920.json`. There are no exact
survivors above 38 letters. The novelty preflight explicitly rejects overlap
with the semantic-atom-transducer, center-relation, semantic-spine,
phrase-boundary, and chunk-composer lanes.

Next construction: hold the center utterance fixed and add held-out arc pairs
indexed by the live two-character residual, preserving all five preflight
checks.
