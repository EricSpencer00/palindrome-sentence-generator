# Semantic scene stack machine (2026-09-16)

This experiment uses a different state dimension from clause-pair and fixed-tape
families: a recursive discourse scene grows complete semantic frames, pushing
the letters emitted by each left-side frame onto a live stack. On recursive
return, a normal-order right-side continuation is admitted only when its next
word pops the exact top obligations. No immutable tape, reverse resegmentation,
catalogue lookup, repeated unit, or clause replay is used.

The four authored frames produce a 62-letter intact surface:

> live on time; drawer; stressed; deliver; reviled; desserts; reward; emit no evil

Exact, independent two-pointer, SHA-256(reverse), and mechanical checks all
pass. A deliberate `reviled` → `revil**e**d` mismatch is captured at the first
bad obligation, then repaired by restoring the held-out frame choice. The JSON
run contains the push/pop provenance and generator digest.

This is diagnostic construction evidence, not a reader-facing claim. The
novelty preflight was reviewed against the registry: the nearest entry is
`recursive-obligation-clause-growth`, which uses relative adjunction; this
machine instead uses recursive return continuations and stack-pop word
admission, so it remains a disjoint construction state.
