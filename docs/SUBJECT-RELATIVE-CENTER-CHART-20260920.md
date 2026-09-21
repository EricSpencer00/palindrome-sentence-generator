# Subject-relative center chart

This lane changes the relative-clause topology again: the head noun supplies
the relative subject (`the poet who reads a memo`) instead of introducing an
overt relative subject (`the poet that the writer reads`). Active and
auxiliary/participle forms are represented in the same agreement-bearing
packed CFG, and the matrix clause remains center-capable through the shared
whole-tape chart.

## Falsifiable plan

- **Decision:** does subject-relative head binding open an exact frontier above
  the 38-letter benchmark?
- **Gate:** exact closure at 39, 40, 44, 48, 52, 60, 72, 90, or 100 letters,
  independently audited by an outside-in pointer and matching forward/reverse
  SHA-256 digests.
- **Controls:** active, auxiliary/participle, and plural subject-relative
  sentences must all parse as complete prose.
- **Falsifier:** every held-out target is root-UNSAT with zero exact rows.
- **Promotion:** only a shortcut-clean exact row can reach the randomized
  blinded intact/shuffled reader package.

## Remote result

The bounded run executed on `hst-bench` with 4,000 nodes per target. The
separate 38-letter calibration recovered the known seed in three nodes and
seven propagation rounds. Every held-out target was root-UNSAT with zero exact
closures: each target visited 11 nodes and 21 propagation rounds. Controls were
complete parses at 31, 33, and 34 letters:

- “the poet who reads a memo studies a map”
- “the poet who has read a poem studies a map”
- “some poets who carry two maps study a memo”

None is palindromic, so the reader gate remains closed. Raw evidence is in
`runs/subject-relative-center-chart-20260920-remote.json`, with implementation
and control tests in `subject_relative_center_chart_20260920.py` and
`tests/test_subject_relative_center_chart_20260920.py`.

## Next construction

The next branch adds a subject-relative center bridge with an explicit
complementizer and a held-out semantic-role frame. This is a new center
construction; widening the current lexical inventory would be a duplicate
sweep.
