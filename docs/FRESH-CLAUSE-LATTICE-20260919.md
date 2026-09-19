# Fresh clause lattice (2026-09-19)

This lane independently authors a finite ordinary-English scene grammar:
subject, agreement-valid finite verb, object, and adjunct. The two sides are
chosen from that grammar and a center is tried from `a`, `i`, `eve`, and `one`.
Before rendering, the normalized outer characters are consumed as a live
obligation; the first impossible orbit is retained as a residual. No catalogue
phrases, copied text, repeated modules, or word-order symmetry are used.

The bounded `hst-bench` run generated 1,296 clauses across six subjects, six
objects, six adjuncts, two agreement classes, and four centers. It found **0
exact closures above 38 letters**. The first residual was:

```text
the sailor maps quiet rivers at dawn; a; the sailor maps quiet rivers at dawn
first mismatch: normalized positions 0/60 (`t`/`n`)
```

The residual is decisive: ordinary clause generation alone cannot satisfy the
outer character obligation, even before clause interiors become relevant. The
next constructive discriminator is a typed subject/object lattice indexed by
the required boundary character, while retaining the same grammar and live
orbit pruning. This is a new construction direction, not a repair to a
finished candidate.

Artifact: `runs/fresh-clause-lattice-20260919.json`.
