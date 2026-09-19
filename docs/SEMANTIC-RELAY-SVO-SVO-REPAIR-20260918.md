# Semantic-relay SVO/SVO repair (2026-09-18)

## Question

Can a small, fresh inventory of two complete, semantically connected clauses
make progress without using the polar-question scaffold or literal
reversible-word pairs?

## Construction

The run contains eight authored event relays.  Each has:

- a grammatical left SVO clause plus a typed setting;
- a grammatical right SVO consequence clause;
- an anaphoric subject on the right linked to the left actor; and
- a shared event and setting label recorded before character matching.

The solver emits left tokens normally and right tokens from their outer edge,
carrying unmatched character debt across token boundaries.  It rejects a row
before admission if any two literal output tokens are reverses or if a proper
multiword palindromic span occurs.  It never reverses a completed text to
construct an output.

## Result

All eight authored probes are complete sentence pairs, for example:

> The archivist marks a ledger at dusk; she files one chart.

No probe closed as an exact palindrome.  The run therefore has zero exact
closures, zero mechanically admitted rows, and zero reader-eligible rows.
The stored artifact includes every rendered candidate, its live first mismatch,
and independent two-pointer plus forward/reverse SHA-256 checks.

This is a useful bounded negative result: it tests a reader-facing grammar
whose semantic relation is present before matching, not a re-punctuation of
the 44-letter diagnostic or a direct semordnilap insertion.  It is not human
readability evidence.

## Next repair

Keep the same two-clause event relay and add one predeclared alternative for
the right-hand consequence object, chosen to address the recorded first
character mismatch of the best live frontier.  Preserve the direct-token and
hidden-span rejections; do not enlarge the inventory into a generic sweep.
