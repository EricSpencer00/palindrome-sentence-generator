# Asymmetric 5-word/4-word partial-seam probe (2026-09-23)

## Scope

This bounded experiment tries a source-authored, readable local grammar at a
pair of mirrored *partial-word* locations in the 568-letter working tape. It
does not promote the inherited 568 text or the later rejected 588 audit.

The recorded source cuts are normalized offsets `[99,99]` and `[469,469]`:
`del|ivers` and `revi|led`. If `T` is the 568 tape, a prospective child would
have the equation:

```
T[:99] + L + T[99:469] + R + T[469:]
```

It is exact only when `L = reverse(R)`. This is an equation for verification,
not a recipe that reverses a finished string to make an output.

## Grammar and guards

The fixed inventory has eight pairs of ordinary finite clauses. Every left
side has five words and every right side has four. The pair states an everyday
shared setting, such as baking and dining or navigation and flight planning.
Before an exact closure could be considered, the program rejects literal
cross-clause reversed tokens, self-palindromic tokens, repeated multiword
units, and proper multiword palindromic spans. A baseline repository scan
also found no exact source phrase before this experiment's own paths were
included.

Each pair is read from left start and right end with live token owners,
per-letter cursors, and a residual. An independent normalizer, opposing
pointer walk, and SHA-256 comparison audit the outcome separately. The
novelty preflight checks both rendered clauses for each pair against tracked
repository text.

## Result

There is no exact closure and therefore no rendered child, hash, or admission
claim. The best clean probe was:

> A baker cools one tart. / Some diners share data.

Its tapes are `abakercoolsonetart` (18 letters) and `somedinerssharedata`
(19 letters). The outer cursors match only `a`; the next live obligation is
`b` on the left against `t` from the right. This makes the obstruction
explicit: punctuation or the partial-word seam cannot repair a mismatch that
precedes either seam.

The selected seam itself is not viable for standalone clauses: it splits
`Aidan del|ivers maps.` and `Spam's revi|led, Nadia.`. Delimiting an inserted
clause leaves `del`, `ivers`, `revi`, and `led` as fragments; fusing across
those inherited owners would re-enter the already-tried boundary-shift and
morphology-transducer families. This seam and its generic splice-lattice
escape are therefore rejected, not renamed as a new method.

## Follow-up and next action

The clean seam `[148,163]` / `[405,420]` was tested. Its exact 578-letter
trial is preserved separately in
[`INCUMBENT-568-PHRASEWISE-EQUATION-AUDIT-20260923.md`](INCUMBENT-568-PHRASEWISE-EQUATION-AUDIT-20260923.md),
but rejected because a shared reflected word boundary splits the inserted
equation into two reversed phrase chunks. The next construction changes the
event topology: an auxiliary-plus-participle clause against a finite-verb plus
object-complement clause, with boundary masks fixed before lexicalization.
