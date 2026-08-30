# Does coherence cost per seam or per letter?

Run 25 August 2026. `experiments/seam_count.py` builds the ladder,
`runs/punct/make_seam_blind.py` builds the blind batch,
`runs/punct/score_seam.py` scores it against `runs/punct/seam_key.json`.

## The question

Nesting mirror-pairs makes length free. Four pairs from the bank —

```
non academia + reno sir parasites + set i sara prisoner + aimed a canon
```

— is 54 letters and a valid palindrome. That is longer than
`doc note i dissent a fast never prevents a fatness i diet on cod`, the
51-letter human best, and it reads worse than either palindrome it was built
from. Two explanations imply opposite strategies:

- **per-seam** — each join between two unrelated pairs costs something, so the
  route to a long readable palindrome is few, long chunks, and search should
  hunt bigger single finds;
- **per-letter** — length itself is what costs, chunk count is irrelevant, and
  no assembly scheme can help.

## Design

Every item is one palindrome against a nest built from **that same palindrome
plus k−1 others**, so the comparison is "this text, or this text with more
joined onto it". Material is held constant; only the seam count moves.

k ∈ {2, 4, 8}, 7 items each, plus 6 calibration items — 24 words of WikiText
prose against its own shuffle, presented through the same `present.py` path.
27 items. Side assignment is balanced by coin flip, item order is shuffled,
and judge 2 sees every pair flipped, so a judge answering by position scores
κ=0 against judge 1 while a judge reading the text scores κ=+1. Both judges
are blind subagents with no repository access; the key is written to disk
before either runs.

Length across the ladder, same construction, one seed a rung:

| pairs | letters |
|---:|---:|
| 1 | 28 |
| 2 | 60 |
| 3 | 90 |
| 5 | 148 |
| 8 | 236 |

## Result

| arm | n | single preferred | agreement | left picks |
|---|---:|---|---:|---|
| calibration | 6 | 6/6 & 6/6 = **12/12**, p=0.000 | 6/6 | j1 1/6, j2 5/6 |
| k=2 | 7 | 7/7 & 7/7 = **14/14**, p=0.000 | 7/7 | j1 4/7, j2 3/7 |
| k=4 | 7 | 7/7 & 7/7 = **14/14**, p=0.000 | 7/7 | j1 2/7, j2 5/7 |
| k=8 | 7 | 7/7 & 7/7 = **14/14**, p=0.000 | 7/7 | j1 4/7, j2 3/7 |

Calibration is 12/12, so the protocol has full power — these judges can tell
prose from its own shuffle every time. Left-pick counts are near half in every
palindrome arm and reverse between judges in calibration, which is what
position-blindness looks like.

**The collapse is complete at k=2.** One seam already loses 14/14. Going to
four and to eight seams does not lose more, because by then there is nothing
left to lose. This is not per-seam degradation with a sweet spot at two or
three chunks, and it is not smooth per-letter decay: it is a step function at
the first join.

## What it licenses

The 54-letter construction above is a real palindrome and does beat the human
record on letters. It does not beat anything on reading, and this experiment
says no amount of care in choosing or ordering chunks will change that, because
the loss is already total at a single join.

So assembly is closed as a route to a *coherent* long palindrome, and the
remaining route is bigger single finds. That route is priced by the yield
exponent in `RESULTS-polaris-yield.md`: yield falls about 1.7x per letter, so
36 to 51 letters is roughly 1.7^15 ≈ 2,900x more search — arithmetic rather
than a wall, but far outside a debug-queue job.

The uncomfortable comparison is that the 51-letter record was found by hand.
Whoever found it was not drawing a billion candidates; they were steering
semantically in a way the walk cannot. That is the same conclusion several
other routes in this repository have reached from different directions, and it
points at a guided proposal distribution rather than at more compute.

## What it does not license

Nothing here is evidence about criteria 6-8 in `docs/NORTH-STAR.md` in the
absolute. The judges ranked pairs; they were never asked whether either side
was good. A k=1 palindrome winning 14/14 against a k=8 nest is consistent with
both sides being unreadable, and the seed texts in the ladder above
(`Lee felt till. Let tell little. Feel.`) suggest they largely are.

The nesting endpoint is unchanged by this. It is honest about being assembly,
and `/dev` already reports repeat counts, so the page does not claim what this
experiment just refuted.
