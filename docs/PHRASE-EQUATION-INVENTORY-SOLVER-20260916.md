# Phrase-equation inventory solver (2026-09-16)

This experiment treats readability as a first-class construction state. It
starts from independently authored semantic role phrases (agent, action,
object, place, and purpose), composes each side in ordinary word order, and
then compares complete scenes under one global equation:

```
left_letter_tape = reverse(right_letter_tape)
```

The search is meet-in-the-middle over character-count vectors, so a candidate
is not accepted because a local prefix happens to match. Every complete scene
is rendered, independently checked with a two-pointer scan and forward/reverse
SHA-256, and passed through the shared mechanical anti-shortcut gate. Programmatic
scores are diagnostic only; no row is reader-eligible without an exact closure
and a blinded intact-versus-shuffled reader study.

Novelty preflight inspected 197 registry entries, found no exact signature
collision, and recorded related phrase-lattice families as intentionally
distinct. The run evaluated 243 complete authored scenes. The best rendered
probe is:

> the careful porter carries the sealed parcel beside the quiet gate for the waiting child.

It has 74 letters. Its equation has equal 74-letter sides, but the first
reversed mismatch is `(0, 't', 'd')`; the independent two-pointer and hash
audits both reject it. It is therefore a readable diagnostic, not a claimed
palindrome. The run found zero closures and zero reader-eligible rows.

The concrete next repair is held-out whole-phrase authoring: replace one place
or purpose phrase using the recorded global character debt, then recompute the
complete equation and both exact audits. A one-sided character edit is not an
allowed repair because it would destroy the semantic phrase inventory.

The held-out repair phase then replaced one complete place/purpose phrase in
12 base scenes with four unseen authored alternatives. It produced 48 repair
trials (36 unique renderings), including this 81-letter intact prose probe:

> the careful porter carries the sealed parcel beside the lantern-lit school for the waiting child.

Its global equation has left/right lengths 81/73 and debt
`h:1, c:2, r:1, f:1, l:4, o:2` against surplus `g:1, n:2`; the two-pointer
and SHA-256 audits both reject it. No repair closed exactly, so it is not
reader-eligible. The next repair must author a second complete phrase against
that recorded debt, not edit individual letters.

Evidence and provenance are preserved in
`runs/phrase-equation-inventory-solver-20260916.json`.
