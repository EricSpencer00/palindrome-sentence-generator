# Vocabulary against length, on 32 cores

Polaris debug queue, 1 node / 32 ranks, 23 August 2026, about one node-hour.
Driver `tools/polaris/shard_yield.py`, submitted by
`tools/polaris/yield.pbs`. Raw output in `runs/polaris/yield_20260823/`.

Five cells, each given all 32 ranks for 420 seconds. Ranks shard on the opening
unit, which partitions the space exactly: measured pairwise overlap between
shards is zero, so no rank repeats another's work.

## What came back

| vocab | band | draws | hits | rate/hit | distinct cores | rate/core |
|---|---|---|---|---|---|---|
| 1,200 | 27-31 | 2,092,650 | 48 | 2.3e-05 | 17 | 8.1e-06 |
| 6,000 | 27-31 | 402,577 | 212 | 5.3e-04 | 12 | 3.0e-05 |
| 28,402 | 27-31 | 79,263 | 12 | 1.5e-04 | 3 | **3.8e-05** |
| 28,402 | 32-36 | 423,534 | 1 | 2.4e-06 | 1 | 2.4e-06 |
| 28,402 | 37-41 | 79,555 | 0 | 0 | 0 | 0 |

All 273 emitted texts pass `is_palindrome`. Zero invalid.

## The claim that does not survive

The local run said a 1,200-word vocabulary produced **zero** sentence-shaped
results in 99,446 draws at 27-31 letters, and this repository then said that
vocabulary "cannot express a single readable palindrome at that length",
reasoning from the fact that it reaches none of the fourteen catalogued ones.

It produces 48, at a rate of 2.3e-5. The local zero was ordinary bad luck:
99,446 draws at that rate expects 2.3 hits, and seeing none has probability
0.10.

The canon-reachability table was measuring something narrower than the sentence
it was used to support. A vocabulary that cannot reproduce any *cataloged*
palindrome at a length can still find its own, and it does:

    one note legal law wall age let one no
    sure pay named art trade many a per us
    deep son named art trade man no speed

So the correction stands in weakened form. Vocabulary matters a great deal;
it is not the difference between possible and impossible.

## Counting hits and counting findings are different measurements

By raw hits the 6,000-word cell looks 23x better than 1,200 and 3.5x better
than 28,402, which reads as an optimum in the middle. It is not one.

Clustering each text by its middle fourteen letters --- what the two halves are
built around --- the 212 hits at 6,000 words are **12 cores**, and one of them
accounts for 76:

| count | core | example |
|---|---|---|
| 76 | `timatessetamit` | `no take estimates set am it seek a ton` |
| 61 | `oworderredrowo` | `mad pan snow order red row on snap dam` |
| 29 | `hsubmittimbush` | `no cap path submit tim bush tap pa con` |
| 14 | `seatsettestaes` | `go let at seat set test a estate log` |
| 14 | `lpmetroortempl` | `pa let a help metro or temple hate lap` |

Per independent core the ordering is monotone in vocabulary size --- 8.1e-6,
3.0e-5, 3.8e-5 --- and the apparent optimum disappears. What a narrow
vocabulary loses is findings; what it gains is many rewordings of each one,
because fewer words fit each overhang and the same centre gets reused.

Report the core rate. The hit rate is a measure of how many ways a vocabulary
can decorate a find, and it moves in the opposite direction from the thing
anyone cares about.

## Length

32-36 letters is not empty:

    reno sir parasites set i sara prisoner        (32 letters)

One core in 423,534 draws, 2.4e-6. Against 3.8e-5 at 27-31, that is a factor of
16 over five letters, or about 1.7x rarer per letter --- close to the 1.93x the
local 19-to-24 slope gave, and nowhere near a wall.

37-41 letters returned nothing in 79,555 draws, which bounds the rate at
**< 3.8e-5** rather than establishing zero. At the observed decay the expected
count there is well under one, so the cell is uninformative and needs roughly
an order of magnitude more draws to say anything.

## What the job settles

- The wide vocabulary that could not be walked on a laptop **can** be walked,
  and it has the best per-core yield of the three at 27-31 letters.
- The narrow vocabulary is worse but not incapable, so the earlier "cannot
  express" phrasing is withdrawn.
- Readable material exists at 32 letters and the decline with length is
  geometric at roughly 1.7x per letter, consistent with the bits argument and
  not with a cliff.
- Nothing here is evidence about 37+ letters either way.

## What it does not settle

`sentence_like` is a generous filter. `reno sir parasites set i sara prisoner`
is close to reading; `no level did level level did level on` is not, and both
pass. The rates above are rates of *tag-shaped* results, and the ratio between
tag-shaped and readable is unmeasured. A blind judging pass over the 273 texts
is the obvious next step and costs no allocation.

The core-clustering threshold of fourteen middle letters is a choice, not a
derivation. It groups the obvious families correctly at this scale and would
need checking before being carried to longer texts.
