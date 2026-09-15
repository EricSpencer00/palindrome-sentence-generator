# Scalable exact-length constructor (2026-09-15)

The current goal is no longer a one-off attempt to extend the 38-letter seed.
The construction core now treats the requested letter count as an explicit
input.  It grows two independently segmented lexical sides around a declared
centre, carries the live character residual, and memoizes lexical states.  A
state is closed only when the residual is empty **and** its rendered tape has
the requested length.  The same code therefore handles even and odd targets
without changing the palindrome definition.

The strict mode reports `no_construction` when the supplied lexical inventory
cannot tile a requested length.  An explicit one-letter fallback demonstrates
that the length engine itself is total, but those rows are marked
`exact_fallback`, are not readable prose, and cannot enter a reader packet.
This separation prevents an arbitrary-length exact core from being mistaken
for a readability result.

## Replay

```text
python experiments/scalable_exact_length_sweep_20260915.py \
  --out runs/scalable-exact-length-sweep-20260915.json \
  --targets 31 47 63 95
```

Every row records the target, rendered text (when a strict or fallback closure
exists), independent exact validation, vocabulary hash, and the full shared
mechanical admission result.  The only acceptable reader-facing next step is
to replace the fallback with a larger typed, boundary-crossing lexical
inventory and then run the intact-prose/shuffled-control blinded study.

Callers can set `require_admitted=True` to continue past exact closures that
fail the independent mechanical admission checks.  This is the mode required
for candidate collection; a closure that merely has a symmetric tape is not
silently promoted.

## Why this is scalable

The search does not enumerate complete left/right sentence products.  It
indexes legal lexical units by forward and reversed letter prefixes and carries
only `(left units, right units, residual, owner, length)` states.  Length
pruning removes overshoot before expansion; memoization prevents duplicate
residual states.  Increasing a target therefore changes the budgeted frontier,
not the algorithm or the palindrome proof.  Readability remains a separate,
human-evidenced property.
