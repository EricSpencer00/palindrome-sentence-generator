# Semantic event-argument CSP (2026-09-18)

This lane is a bounded repair of the semantic event-frame search. It builds
2,316 complete typed event frames, carrying actor class, subject number,
verb valency, object class, and an independently selected setting. It then
joins independently selected frames through five seam types (`;`, `.`,
`because`, `while`, and `as`) and audits each rendered string with a
two-pointer normalized-tape check plus independent forward/reverse SHA-256.

The first implementation accidentally enumerated the entire Cartesian product
even when a diagnostic cap was requested. That was corrected before the run;
the recorded run is explicitly bounded at 10,000 pair worlds.

## Result

The run checked 10,000 pair worlds and found two exact rows, both the same
38-letter benchmark rendered with different punctuation:

> An aide rips nine memos; some men inspire Diana.

The exact tape is
`anaideripsninememossomemeninspirediana`, with forward and reverse SHA-256
`ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6`.
Both rows pass the mechanical gates, but both are baseline duplicates and are
not new progress. No new reader-eligible candidate was produced. The result
therefore preserves an independently audited control while rejecting the lane
as a construction method for longer prose.

The 44-letter sentence remains the best longer human-looking diagnostic, not a
promoted result, because it contains a proper self-palindromic multiword span:

> Was Noel an era, a gas, an item? Met in a, saga, arena, Leon saw.

## Concrete next repair

Do not enlarge this frame product. Carry the seam residual through typed
argument tokens before complete-frame joining, and add held-out event roles so
the solver can change lexical boundaries while it is satisfying the character
equations. Any resulting candidate must be printed with provenance and pass
independent exact validation before reader testing.

Reader evidence is not claimed for this lane; programmatic checks diagnose and
filter but do not certify English readability.
