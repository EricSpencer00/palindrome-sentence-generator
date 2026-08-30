# Where the enumerable regime actually ends

Run 22 August 2026. Script: `experiments/regime_boundary.py`.

The paper said "below about 30 letters the space is enumerable". The figure was
never measured. It appears as 24 in the `exhaustive.py` docstring, "at most 28"
in `docs/training.md`, "~30" in the paper's Table 1, and the next band anyone
searched was 40-60, leaving 31-39 unexamined. Two things turn out to be wrong.

## 1. A unit cap, not a letter count, was closing the long bands

`enumerate_palindromes` defaults to `max_units=12`. A 60-word vocabulary of
mostly short words cannot reach forty letters in twelve units, so the walk
returned almost nothing up there and the emptiness read as a property of the
space. Lifting the cap to 30, same vocabulary, same 1M node budget:

| band | max_units=12 | max_units=30 |
|---|---|---|
| 22-26 | 41,205 | 87,842 |
| 27-31 | 1,759 | 44,785 |
| 32-36 | **17** | **87,693** |
| 37-41 | 0 | 44,818 |
| 42-46 | 0 | 88,934 |

Any earlier reading of "the space is empty above N letters" from a walk at the
default should be checked against this before it is believed.

## 2. Enumeration is not exhaustive anywhere worth being

"Exhaustive" is checkable without trusting anything: run the same band at two
budgets and see whether the count moves. A completed walk returns the same
number however much more budget it gets, so the ratio is 1.00. A walk sampling
a vast space keeps finding new results in proportion to its budget, so the
ratio is 2.00.

Node budget 1.5M against 750k, `max_units=40`:

| band | 60-word vocab | 200-word vocab |
|---|---|---|
| 12-16 | **1.00 complete** | 1.32 |
| 17-21 | 1.99 | 2.00 |
| 22-26 | 2.02 | 1.92 |
| 27-31 | 2.00 | 1.99 |
| 32-36 | 2.02 | 2.04 |
| 37-41 | 2.00 | 2.09 |
| 42-46 | 1.99 | 2.03 |

At sixty words only the 12-16 band completes. At two hundred words nothing
completes at all. And the ratio is flat at 2.00 from 17 letters to 46: the
walk is no closer to finishing at 20 letters than at 45, so **there is no
letter at which enumeration stops being exhaustive, because it was not
exhaustive to begin with** above a toy vocabulary and a very short band. The
2.55M figure in `docs/training.md` is a yield from a 30-minute walk, not a
census of the space.

## 3. What does change with length is the density of readable material

This is the real regime structure, and it is far sharper than 30. Enumerating
each band at a 1,200-word vocabulary and 400k nodes, and asking
`syntax.sentence_like` of every 3-9 unit result:

| band | distinct found | sentence-shaped | rate | first example |
|---|---|---|---|---|
| 12-16 | 9,410 | 416 | **4.4%** | `an i a my as say main a` |
| 17-21 | 3,962 | 69 | **1.7%** | `sir of may as say am for is` |
| 22-26 | 9,340 | 6 | **0.064%** | `to led i stay as say at side lot` |
| 27-31 | 4,234 | 0 | 0 | — |
| 32-36 | 9,304 | 0 | 0 | — |
| 37-41 | 3,887 | 0 | 0 | — |
| 42-46 | 9,304 | 0 | 0 | — |

A 70-fold collapse from 12-16 to 22-26, and zero from 27 letters up.

We read that as a cliff. It is not one, and §4 is why: this whole table is a
1,200-word vocabulary, and the zeros above 27 letters are a property of that
vocabulary rather than of the space. The curve is kept here because the
retraction is more useful than the deletion.

## 4. The zeros were the vocabulary, not the space

Everything above was run at a 1,200-word frequency-ranked vocabulary. That
vocabulary cannot express a single readable palindrome the record contains at
those lengths.

Checking every catalogued palindrome against the frequency ranking of the words
it needs:

| top N words | canon reachable, <=26 letters | canon reachable, 27+ letters |
|---|---|---|
| 1,200 | 1/57 | **0/14** |
| 3,000 | 5/57 | **0/14** |
| 10,000 | 17/57 | 1/14 |
| 30,000 | 35/57 | 7/14 |

Zero of fourteen. The walk that returned no sentence-shaped result above 27
letters was drawing from a vocabulary in which no known readable palindrome of
that length can be written. The measurement was of the vocabulary.

The reason is visible in the rarest word each one needs: `erasmus` at rank
23,454, and `oscillate`, `metallic`, `sonatas`, `myriad`, `dairymen`,
`garageman` outside the top 30,000 entirely. Long readable palindromes are made
of unusual words, because an unusual letter sequence is what lets a long mirror
close and still say something. A frequency cut selects for the words that make
ordinary prose read, which is close to the opposite pressure.

Direct test, same band and same node budget, only the vocabulary changed:

| vocabulary | distinct found | sentence-shaped | rate |
|---|---|---|---|
| 1,200 | 20,796 | 0 | 0 |
| 6,000 | 10,806 | **1** | 0.000093 |
| 30,000 | — | — | branching too wide to finish at this budget |

`level note wanna ann a wet on level`, 27 letters, verified by
`is_palindrome`. Half the sample, and the zero becomes non-zero. The rate at
27-31 letters is vocabulary-limited well before it is length-limited.

The 30,000-word row not finishing is the other half of the trade: the
vocabulary that can express the canon branches too widely to walk at this
budget. That is the real tension, and it is a compute-and-proposal problem
rather than a fact about English.

## 5. Superseded by the Polaris run

`experiments/RESULTS-polaris-yield.md` re-ran the decisive cells on 32 cores
and corrects §4 in one place. The 1,200-word vocabulary is not incapable at
27-31 letters: it yields 48 sentence-shaped results in 2.09M draws. The local
zero in 99,446 draws was bad luck at a rate of 2.3e-5, where 2.3 hits were
expected and P(0) = 0.10.

The direction of §4 holds and the strength does not. Per independent find the
rate rises monotonically with vocabulary — 8.1e-6 at 1,200 words, 3.0e-5 at
6,000, 3.8e-5 at 28,402 — so vocabulary is the dominant lever. It is not the
difference between possible and impossible, and the sentence claiming a
1,200-word vocabulary "cannot express" a readable palindrome at that length is
withdrawn.

## What this does to the claims

- "Below 30 letters the space is enumerable" is **withdrawn**. Enumeration
  completes only for a toy vocabulary at 12-16 letters.
- "Zero sentence-shaped above 27 letters" is **withdrawn**. It held for a
  1,200-word vocabulary that cannot express any of the fourteen readable canon
  palindromes at that length, and a 6,000-word vocabulary produces a hit at
  half the sample size.
- The yield curve of 4.4% / 1.7% / 0.064% stands **only** as a statement about
  that vocabulary. It is not a curve in length; it is a curve in length
  confounded with what a frequency-ranked slice can express.
- The three-regime table as previously drafted claimed a boundary in letters.
  There is no such boundary in evidence. What there is: yield falls with
  length, falls faster when the vocabulary is small, and the vocabulary that
  fixes it is the one that is too wide to walk.
- `non academia aimed a canon` remains a real find at 22 letters, and the
  reason a walk of millions was needed is unchanged.
- The practical advice becomes different and more useful: before concluding a
  length is out of reach, check whether the vocabulary can express any known
  solution at that length. Ours could not, and we spent a while measuring that
  fact and calling it a property of English.

## Limits

One vocabulary ordering (frequency-ranked), one seed, one node budget per cell.
The saturation ratio assumes results accrue at a roughly constant rate within a
band, which is what a shuffled frontier over a large space gives, and would
mislead on a space small enough to be nearly finished. The 12-16 cell at 200
words, ratio 1.32, is that intermediate case and is reported as `sampling`
rather than split more finely.
