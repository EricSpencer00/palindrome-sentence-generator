# Why the typed valency zero does not test valency

The original 54-clause domain cannot match even its first character pair.
Subjects start with {t, s, a}; tails end with {n, m, r}. All nine exterior
subject/tail combinations are impossible regardless of every interior choice.
Changing verbs, object valency, discourse topology, or search budget cannot
affect this certificate.

A separate domain replaces the three tails with independently authored
"at sunset", "beneath the stars", and "in the plaza". This admits 18 of 54
derivations at the first pair; all fail at the second. Neither domain has an
exact palindrome. This is a bounded falsification of endpoint-only widening,
not a new construction success, lexical exhaustion, or a readability result.

Actual diagnostic prose: "the careful pilot marked the weathered chart at
sunset." It has 46 normalized letters and fails at zero-based pair 1 (h/e).
Forward SHA-256: d08eb7ed620ecb67eeaf15044b237a20d023edf86c253bfcccc53aa720c6b5f3.
Reverse SHA-256: ad99dcbf54a5c983a92130374d2fbdfcd837e1a5f7a9704bfc9e04ad2a8f12c5.
It is deliberately excluded from `reader_candidates`.

The original `typed_boundary_valency_20260921.py` additionally calls its
admission function with [tail, complement], although the right-hand surface
is [complement, tail], and only compares endpoint characters. That predicate
is neither full residual matching nor an accurate exterior test of its
rendered sentence. The certificate independently uses actual surface order;
it does not change the old record or its provenance.

## Algorithmic direction

The ABBA family scripts inspect a handful of independently completed prose
surfaces. Their negative audits say nothing about completeness of a paragraph
grammar. Likewise, the existing bidirectional CFG script enumerates complete
left/right derivations before matching, and additionally requires the left
sentence alone to be palindromic. Requiring both a reversed pair and a
self-palindromic left half excludes general two-clause palindromes.

The strongest next direction suggested by these inspected failures is
support-driven grammar expansion, measured before another large search:

1. Compile grammatical continuations to a shared character graph. Preserve
   lexical positions, valency obligations, agreement, discourse referents,
   and exact emitted length in the state. The existing world-preimage product
   is a better foundation than independently completed phrase banks.
2. Compute productive opposite-frontier states: both sides must still reach
   a center meeting. Carry unmatched terminal characters across word
   boundaries, rather than requiring phrase or clause boundaries to align.
3. Extract the earliest cut where all reachable grammar pairs lose support.
   Expand the relevant syntactic/lexical domains jointly, keeping scene
   arguments and semantic roles fixed. A model can propose a reusable domain
   expansion for this cut; it should not score each candidate or repair an
   already completed tape.
4. Admit a new domain only when it increases supported paired-character
   depth and surviving non-seed derivations on held-out scene plans. Stop
   widening when it only changes the location of an unavoidable mismatch.
5. Reconstruct one shared start-to-accept path on a frontier meeting and
   audit that whole surface. Do not require either half to be palindromic.

This is guidance, not a claim that conflict-directed methods are new to the
repository: prior counterexample-guided and character-product experiments
exist. The missing evidence in the inspected lanes is that domain expansion
actually restores deep character support while preserving semantic freedom.
Use support depth and reachable state counts as diagnostics, not readability
scores; exact outputs still need independent reading.

## Verification

Four lightweight standard-library tests pass, covering shifted word
boundaries, actual right-side order, agreement between exterior certificates
and full scans, and independent pointer/hash audit agreement. The complete
result and source hash are in `runs/endpoint-feasibility-certificate-20260921.json`.
The run contains only 108 derivations and required no model or heavy compute.
