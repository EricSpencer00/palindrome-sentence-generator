# v3 against the north star

Run 24 August 2026. `experiments/north_star_v3.py`, 24 seeds per length,
`novel=true` throughout.

`docs/NORTH-STAR.md` reserved "v3" for the goal before any code carried the
name. The endpoint carries it now, so the six mechanical criteria have to be
run against it and the answer published whichever way it comes out. It comes
out badly at length.

## The six mechanical criteria

| letters | 1 ≥100 words | 2 mirrors | 3 ≤1 self-pal | 4 no repeat | 5 disjoint halves | 9 novel |
|--------:|---:|---:|---:|---:|---:|---:|
| 400     | 24/24 | 24/24 | 24/24 | **21/24** | **23/24** | 24/24 |
| 1,200   | 24/24 | 24/24 | 24/24 | **1/24**  | **22/24** | 24/24 |
| 4,000   | 24/24 | 24/24 | 24/24 | **0/24**  | **11/24** | 24/24 |
| 14,500  | 24/24 | 24/24 | **14/24** | **0/24** | **0/24** | 24/24 |

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

So the fix is not in the assembly. It is either in `present.py` — refuse a cut
that produces a sentence already used — or in chunk selection, which would
have to look at word runs rather than at whole halves. The first is cheap and
local; the second is the one that would also help criterion 5, since the
halves collide for the same reason.

## Criterion 5 and the 14,500-letter row

Disjoint halves degrades with length on the same mechanism, and at full
capacity it fails every time: with the bank exhausted the two halves are
drawing from the same small residue of short pairs. Criterion 3 also starts
failing there (14/24) because short self-palindromic sentences like `level`
survive the cut at that density.

The honest reading is that v3 clears the structural criteria at short length
and does not hold them at the length it advertises. `/dev` reports the
repeated-sentence count next to the repeated-chunk count so the page cannot
imply otherwise.
