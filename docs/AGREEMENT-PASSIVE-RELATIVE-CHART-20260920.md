# Agreement/passive-relative shared-tape chart

This is a new construction topology after the object-relative bound-gap lane.
The grammar carries singular/plural state through both the matrix clause and
the relative clause. A relative clause can be active (`that the writer reads`)
or passive/participle (`that the writer has read`), with the auxiliary selected
by the relative subject's number. The solver is still a single whole-sentence
packed CFG: complete-parse support is propagated through mirrored character
domains before branching on the most constrained character orbit.

## Falsifiable plan

- **Decision:** does explicit agreement plus an auxiliary/participle relative
  frame open a new exact-palindrome frontier above 38 letters?
- **Gate:** an exact closure at 39, 40, 44, 48, 52, 60, 72, 90, or 100
  letters, independently checked by the outside-in pointer and matching
  forward/reverse SHA-256 digests. Any closure would still require the
  shortcut screen and blinded reader study.
- **Control:** three complete prose parses, including active, passive, and
  plural relative subjects.
- **Falsifier:** all held-out target lengths root-UNSAT with zero exact rows.
- **Promotion rule:** only a shortcut-clean exact row enters the reader package;
  a failure must select a new grammar state, not a larger duplicate sweep.

## Remote result

The bounded run was executed on `hst-bench` with 4,000 search nodes per target.
The 38-letter calibration recovered the known seed in three nodes and seven
support-propagation rounds, but was not promoted. All held-out targets were
UNSAT with zero exact candidates:

| target letters | nodes | propagation rounds | exact |
|---:|---:|---:|---:|
| 39 | 12 | 20 | 0 |
| 40 | 12 | 20 | 0 |
| 44 | 12 | 19 | 0 |
| 48 | 12 | 19 | 0 |
| 52 | 11 | 22 | 0 |
| 60 | 1 | 5 | 0 |
| 72 | 1 | 3 | 0 |
| 90 | 1 | 2 | 0 |
| 100 | 1 | 1 | 0 |

The complete controls were independently parseable:

- “the poet that the writer reads studies a memo” (37 letters)
- “the poet that the writer has read studies a poem” (39 letters)
- “some poets who writers read carry two maps” (35 letters)

None is an exact palindrome, so none is a reader candidate. Evidence is in
`runs/agreement-passive-relative-chart-20260920-remote.json`; the implementation
and lightweight control tests are in
`experiments/agreement_passive_relative_chart_20260920.py` and
`tests/test_agreement_passive_relative_chart_20260920.py`.

## Next construction

The next branch is a subject-relative frame with an explicit relative-pronoun
role and a center-capable finite clause. It must retain the same agreement
state and root-support propagation; adding another text-noun bank would be a
duplicate sweep.
