# Multiword scene-boundary search (2026-10-01)

This lane tests a different construction from terminal semordnilap pairing.
Eight independently authored ordinary scene clauses were composed into two
clause sequences on each side. The bilateral search compared characters while
allowing the obligation to cross word and sentence boundaries; no side was
constructed by reversing a finished tape.

The 64 × 64 search visited 4,096 independently selected sequence pairs and
found zero exact closures. Its strongest outer frontier matched one letter;
the best rendered pair was:

> Near the river, Lena found a blue button. At dawn, Mira opened the garden gate. At dawn, Mira opened the garden gate. Near the river, Lena found a blue button.

This is an ordinary-prose control, not a palindrome: 122 letters, independent
two-pointer `false`, validator `false`, and unequal forward/reverse SHA-256
digests. It is retained because the source clauses are fresh, intact prose and
the failure occurred under a live character obligation rather than a
post-hoc reversal. The reader gate is closed.

Run artifact: `runs/multispan-scene-boundary-search-20261001.json`.

The concrete next construction is a three-clause bilateral chart: author a new
ordinary clause against the deepest residual suffix, then permit its word and
sentence boundaries to resegment before adding the next clause. This preserves
the multiword-span hypothesis while changing the search state rather than
repeating the same two-clause sweep.
