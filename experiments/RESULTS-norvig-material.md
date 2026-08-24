# This project against Norvig's, on the material

Run 24 August 2026. `experiments/norvig_material.py`, plus one live v1 run
taken from the deployed `/api/generate`.

## Why not length

Norvig's version 3 is 21,012 words and 90,439 letters. Nothing here is within
an order of magnitude of it, and the gap is the difference between two
questions rather than a result about either. His program chooses, at each step,
the branch that lets the search keep going — at random in versions 1 and 2, and
in version 3 by the letter that "leads to the most completions of words on both
left and right." This one chooses the branch that reads. Both use the same
two-sided overhang search, which is his, building on Hoey's 1984 program.

## The material

Both programs spend a dictionary, and the difference between the dictionaries
is most of the difference between the outputs. Measured with this project's own
filters — `lexicon.is_real_word` against `data/lexicon.txt` (52,927 headwords,
a dictionary intersected with a frequency list) and `shortwords.is_real_short`.

| text | words | 1–2 letters | in lexicon | passes both | mean word |
|---|---:|---:|---:|---:|---:|
| Norvig v3, the 21,012-word palindrome | 21,196 | 22.9% | **53.0%** | 52.4% | 4.27 |
| Norvig `npdict.txt`, as tokens | 210,257 | 33.9% | 60.7% | 60.1% | 5.58 |
| English, frequency-weighted sample | 20,000 | 21.7% | 94.8% | 94.0% | 4.39 |
| here, v1, one live run, 1,010 letters | 327 | 32.7% | 98.5% | 98.5% | 3.09 |
| here, v3 composition, 400 letters | 107 | 18.7% | **100%** | 100% | 3.51 |
| here, v3 composition, 1,200 letters | 341 | 24.9% | **100%** | 100% | 3.49 |
| here, v3 composition, 4,000 letters | 1,151 | 27.5% | **100%** | 100% | 3.46 |

**47.6% of the words in the 21,012-word palindrome are not words in that
sense.** That is not a defect in it — it is where the length comes from. `PETN`,
`ILGWU`, `Roydd`, `Aeniah` and the rest of the 126,342-entry Moby extract are
exactly the reversible material a length search needs, and Norvig says as much:
"there are 126,000 words in the dictionary, but only about 10% of them are
easily reversible." Every proper noun and abbreviation ruled out here is length
given up deliberately.

**The last column does not flatter this method.** Mean word length is 3.09
letters for the search and about 3.5 for the composition, against 4.39 for
English and 4.27 for his palindrome. His text uses words of roughly English
length that are mostly not English words; this one uses real words that are
mostly short. Both are ways of being unlike English prose and the second is not
obviously better. The one- and two-letter share says the same: 22.9% for his
against 21.7% for English — his is normal — and v1's 32.7% is not.

## A repository claim that does not reproduce

`server/v3.py` and `README.md` say v1's output is "52.6% one- and two-letter
words against real English's 18.5%." Neither number reproduces here. The live
v1 run measures **32.7%**, and English measures **21.7%** on a
frequency-weighted draw from wordfreq's top 30,000. The gap is real in the same
direction and about half the claimed size. Either the original was measured
before the vocabulary hardening in `safe_vocab.py`, or against a different
English baseline; it is not sourced in the repository, so this file does not
try to say which. **The claim should not be requoted until it is re-measured.**

## Method notes

- The English baseline is a frequency-WEIGHTED draw, not the raw vocabulary
  list. The raw top-30k is mostly long rare words and would put the bar in the
  wrong place: what a reader meets is the weighted distribution.
- Norvig's palindrome is tokenised out of `pal21txt.html` with tags stripped,
  which gives 21,196 tokens against his stated 21,012 words. The difference is
  the page's own framing text and the hyphen and apostrophe handling; it is
  under 1% and moves no percentage in the table.
- `v1` is one run, not a distribution. It is there to place the method, not to
  be a measurement of it.
