# v3 against the north star

Run 24 August 2026, re-run 25 August after the presentation fix below.
`experiments/north_star_v3.py`, 24 seeds per length, `novel=true` throughout.

`docs/NORTH-STAR.md` reserved "v3" for the goal before any code carried the
name. The endpoint carries it now, so the six mechanical criteria have to be
run against it and the answer published whichever way it comes out. It came out badly
at length, for one reason, which has since been fixed.

## The six mechanical criteria, after the fix

| letters | 1 ≥100 words | 2 mirrors | 3 ≤1 self-pal | 4 no repeat | 5 disjoint halves | 9 novel |
|--------:|---:|---:|---:|---:|---:|---:|
| 400     | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| 1,200   | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| 4,000   | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| 14,500  | 24/24 | 24/24 | **13/24** | 24/24 | 24/24 | 24/24 |

As first measured, before the fix:

| letters | 3 ≤1 self-pal | 4 no repeat | 5 disjoint halves |
|--------:|---:|---:|---:|
| 400     | 24/24 | **21/24** | **23/24** |
| 1,200   | 24/24 | **1/24**  | **22/24** |
| 4,000   | 24/24 | **0/24**  | **11/24** |
| 14,500  | **14/24** | **0/24** | **0/24** |

Criteria 4 and 5 were one bug, diagnosed below and fixed in
`llm_palindrome/present.py`: a cut that would close a sentence already used is
refused, and the run is absorbed into the current sentence instead. Criterion
3 at full length is a different cause and is unfixed.

Control, same checks, same script: `/api/v2/paragraph` at 9 sentences is
24/24 on all six. v2 is the endpoint `tests/test_north_star.py` was written
against and it still passes; nothing here weakens that.

Criteria 6, 7 and 8 — grammatical, has a subject a reader can name, judged as
prose rather than as sentences that happen to parse — are not computed. They
need a blinded batch with salad and real-prose controls, and four automated
proxies have disagreed with blind ranking in this project without one ever
agreeing. Nothing in this file is evidence about them.

## Why criterion 4 fails, stated precisely

**No chunk ever repeats. That is a different property from no sentence
repeating, and only the first one is guaranteed.**

| letters | chunks repeated | sentence types repeated | worst |
|--------:|---:|---:|---|
| 400   | 0 | 1  | `bar a met` x2 |
| 1,200 | 0 | 2  | `bar a met` x2 |
| 4,000 | 0 | 10 | `bar a met` x4 |

The endpoint deduplicates on the half text and refuses degenerate pairs, and
that holds — `repeats` is 0 in every response above. Punctuation is then
applied to the assembled word run as a whole, not per chunk, so two unrelated
chunks that happen to contain the same short word run are cut into the same
sentence. `bar a met` is not one chunk appearing four times; it is four
different chunks each yielding it.

The fix was not in the assembly. `present.py` now tracks the sentences it has
emitted and refuses a cut that would close one already used, downgrading the
full stop to a comma so the run joins the current sentence. Letters are
untouched, so the mirror cannot be affected; only where the marks fall
changes. Criterion 5 was the same collision seen from the other side and it
came back to 24/24 with no separate change.

One case survives by construction: a duplicate landing on the final run, where
there is no later cut to defer to. It did not occur in 96 draws.

## The 14,500-letter row

Criterion 3 still fails at full capacity, 13/24, and it is not the sentence-cut
bug: with the bank exhausted the composition is drawing from a small residue of
short pairs, and short self-palindromic sentences like `level` survive the cut
at that density. More than one sentence is then a palindrome on its own, which
criterion 3 forbids. That is a material problem, not a presentation one, so the
route to it is a larger bank rather than another change to `present.py`.

The honest reading is that v3 now clears five of the six mechanical criteria at
every length measured, and the sixth only at full capacity. Criteria 6, 7 and 8
remain unmeasured and nothing here is evidence about them; see
`experiments/RESULTS-seams.md`, where blind judging says assembly loses to a
single chunk at the very first seam.
