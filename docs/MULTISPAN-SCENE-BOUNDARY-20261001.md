# Multiword scene-boundary search (2026-10-01)

This lane tests a different construction from terminal semordnilap pairing.
Eight independently authored ordinary scene clauses were composed into two
clause sequences on each side. The bilateral search compared characters while
allowing the obligation to cross word and sentence boundaries; no side was
constructed by reversing a finished tape.

The 64 × 64 search rejected 2,416 sequence pairs that reused a clause unit
across the two sides, then visited 1,680 four-distinct-clause pairs. It found
zero exact closures. Its strongest outer frontier still matched one letter;
the best permitted rendered pair was:

> Near the river, Lena found a blue button. At dawn, Mira opened the garden gate. A patient fox watched the quiet road. By noon, the keeper had mended the lantern.

This is an ordinary-prose control, not a palindrome: 125 letters, independent
two-pointer `false`, validator `false`, and unequal forward/reverse SHA-256
digests. It is retained because the source clauses are fresh, intact prose and
the failure occurred under a live character obligation rather than a
post-hoc reversal. Repeated units are explicitly rejected; the reader gate is
closed.

Run artifact: `runs/multispan-scene-boundary-search-20261001.json`.

The concrete next construction is a three-clause bilateral chart: author a new
ordinary clause against the deepest residual suffix, then permit its word and
sentence boundaries to resegment before adding the next clause. This preserves
the multiword-span hypothesis while changing the search state rather than
repeating the same two-clause sweep.
