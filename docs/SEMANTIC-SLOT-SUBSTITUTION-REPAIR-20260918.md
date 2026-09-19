# Semantic slot substitution repair (2026-09-18)

This lane starts with intact micro-clauses and changes typed subject, verb,
object, and adjunct slots only. Agreement and valency are enforced before a
pair is considered. For each left/right pair, the normalized character tape
and its reverse are compared before punctuation is rendered; this exposes the
actual residual rather than using a prose or language-model score.

## Run

`runs/semantic-slot-substitution-repair-20260918-v2.json`

- 710 typed clauses; 10,000 bounded pair worlds checked
- 2 exact rows, both punctuation renderings of the existing 38-letter seed
- 2 mechanically admitted; 0 new admissions; 0 reader-eligible rows
- exact audit uses an independent two-pointer comparison and forward/reverse
  SHA-256 equality

The exact rendered rows were:

> an aide rips nine memos;some men inspire Diana.

> an aide rips nine memos. some men inspire Diana.

Both normalize to 38 letters and SHA-256
`ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6`.
They are retained as controls, not claimed as new progress. No candidate from
this run has human-readability evidence; programmatic checks do not certify
English prose.

## Interpretation and next repair

The residual-before-render check correctly rediscovered the seed but found no
new closure in the first 10,000 worlds. The next construction should retain
residual-compatible subject/verb choices while introducing adjective–noun
slots and held-out authored clauses. That is a new operator, not a larger
duplicate sweep.
