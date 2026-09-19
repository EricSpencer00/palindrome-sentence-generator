# Semantic template lattice with joint equations (2026-09-19)

This orthogonal lane tested whether a small human-authored lattice of complete,
readable English clauses could choose scene/valency roles jointly with their
mirrored character equations. Each row composes a subject, inflected verb,
object, and adjunct on both sides; it does not generate prose and then reverse
or filter a fixed tape.

## Result

The run evaluated 9 scene pairs (scribe, gardener, pilot), rendering 9 intact
two-clause prose rows. Every row failed the live outer-inward equation and the
independent exact audit: 0 equation passes, 0 exact palindromes. The longest
rendered row contained 62 normalized letters. Each row stores the rendered
prose, first mirrored mismatch, normalized/reversed SHA-256 values, and a
separate two-pointer replay.

## Provenance and anti-shortcut record

All lexical material was authored in the experiment file for this run. No
catalogue, known palindrome, seed sentence, tape reversal, or generated-prose
filter was used. The search space is compositional (`scene × scene`) and can be
extended by adding typed role alternatives rather than copying candidate text.

## Failure frontier and next repair

The frontier is immediate: the first mirrored edge mismatches before clause
interiors can contribute. Complete-clause pairing therefore cannot repair its
own boundary debt. The next concrete repair is a two-word subject/object
lattice with inflectional variants that carry residual character domains across
the clause boundary while preserving role typing.

Run artifact: `runs/semantic-template-lattice-joint-equations-20260919.json`.
