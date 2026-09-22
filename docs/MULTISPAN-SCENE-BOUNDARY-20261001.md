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

## Three-clause bilateral chart (follow-up)

The follow-up keeps the first two-clause scene choices intact, records the live
residual at their deepest seam, and only then selects a separately authored
third-clause pair. The fresh pair is placed at the actual outer edges: the left
third clause opens the tape and the right third clause closes it. Consequently,
the live edge test compares the left clause's first character with the right
clause's last character. The third clauses are ordinary prose and are not
reversals or catalogue entries. The final comparison still runs over normalized
characters, so a seam may cross a word or sentence boundary.

The chart visited 1,680 distinct two-clause residual frontiers and 360
three-clause bilateral states after conditioning the actual outer edge
obligation. It rejected 2,416 base choices with
a repeated clause unit. No state closed exactly. The strongest rendered state
was:

> Near the hill, Mira waited for rain. A patient fox watched the quiet road. The baker carried warm bread to the school. Near the river, Lena found a blue button. By noon, the keeper had mended the lantern. The keeper checked the latch before rain.

This is a 193-letter ordinary-prose control, not a palindrome: its independent
two-pointer audit fails at `(1, e, i)`, the project validator is false, and its
forward/reverse SHA-256 digests differ. The base residual was recorded before
the third-clause choice (`required_next_left_char = n`,
`required_next_right_char = s`), so the failure is attributable to the live
obligation rather than a post-hoc reversal. No human readability claim is made;
the reader gate remains closed.

The prior experiment's first-character gate was invalid because appending to the
right makes its closing character face the left opening. This revision fixes
that orientation. The next concrete operator is to use the recorded residual
after consuming the outer pair and author the next clause at the seam, while
retaining this explicit opening/closing edge assertion.
