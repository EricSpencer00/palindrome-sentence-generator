# Beating the published Norvig v3 length

5 September 2026. This is a length-search experiment using the existing Norvig
phrase dictionary. It does not claim coherent prose or a world record.

## Reference and exact comparison

Source algorithm: https://norvig.com/pal-alg.html

Unmodified code: https://raw.githubusercontent.com/norvig/pytudes/main/py/pal3.py

Cached at `runs/norvig/pal3.py`; SHA-256:
`c21a79f77e3021c098b4223e6f63976479733a699b3dc0bf4b33499a711758e2`.
Both of the reference program's own tests pass. Its dictionary has 125,512
normalized phrase keys from 126,342 lines. Fourteen keys contain non-ASCII-letter
characters; the adapted searches exclude those keys from emitted material.

Dictionary SHA-256:
`3f28b8a95d92be6c2f73a63bfee0c80e6ad2718c1ee191b7032cdd5f09dbb51f`.

The published version-3 text (`runs/norvig/pal21txt.html`) has **90,439 letters**,
independently verified as palindromic after stripping its heading and author
footer. The page reports 21,012 words; our ASCII word-token rule counts 21,194.
Use the same tokenizer for both candidates and the reference. Never claim a
word-count win just because a different tokenizer increased the number.

## Implemented arms

`experiments/norvig_letters.py` imports the unmodified reference and overrides
its phrase admission, bookkeeping, closure recording, and optional inventory
counts. The underlying paired-letter action and undo mechanics are Norvig's.
This is an attributed adaptation, not a newly invented palindrome algorithm.

- **Static:** reference product-of-prefix-count-and-suffix-count letter ordering.
- **Unused:** decrement both counters when a phrase is completed; restore them
  when backtracking. Used phrases no longer inflate the branching estimates.
- **Feasible:** additionally remove phrases whose content-word counts would
  exceed the cap, updating only phrases indexed under affected words.

All strict arms enforce unique normalized phrases, no adjacent repeated words,
and at most three uses per non-function word. Function exceptions are identical
to the earlier phrase DFS. Closure recording maximizes letters rather than the
reference implementation's phrase count. Therefore "static" describes the
reference's ordering under our guards, not a byte-for-byte replay of its
original search policy.

An explicitly separate comparable arm removes the extra three-use cap while
retaining phrase uniqueness and no adjacent repeated words. It must not be
presented as satisfying the stricter cap.

## Initial results

| arm | budget | letters | ASCII word tokens |
|---|---:|---:|---:|
| static, cap 3 | 45 s | 68,286 | 15,822 |
| unused, cap 3 | 45 s | 76,979 | 17,926 |
| unused, cap 3 | 120 s | 88,101 | 20,553 |
| feasible, cap 3 | 120 s | 88,095 | 20,505 |
| unused, no word cap | 120 s | 88,455 | 20,690 |

The 45-second pair is a promising local observation: unused inventory found
12.7% more letters, with slightly fewer search-loop steps (4.00M versus 4.10M).
These are single deterministic search orders run concurrently on the same
machine; timing is not a replicated performance result. At 120 seconds,
feasible-inventory bookkeeping essentially tied unused-only bookkeeping.

Restarting at the saved best midpoint exhausted immediately with no improvement:
the completed palindrome need not offer any unused extension there. Four unused
reversible phrase-pairs survived the strict cap, only 28 potential letters, so
that shortcut cannot bridge the gap. Further growth needs continuing the
backtracking search, not appending repeated material or borrowing a new source.

## Reproduction

```
curl -fsSL https://raw.githubusercontent.com/norvig/pytudes/main/py/pal3.py -o runs/norvig/pal3.py
.venv-v3/bin/python -m experiments.norvig_letters --seconds 300 --dynamic --feasible --out runs/norvig-letter-feasible-300
.venv-v3/bin/python -m experiments.audit_norvig_result runs/norvig-letter-feasible-300
.venv-v3/bin/python -m pytest -q tests/test_norvig_letters.py tests/test_norvig_long.py tests/test_growth_convention.py
```

The original dictionary and published palindrome are already cached by
`experiments/norvig_material.py`. Source hashes above pin the inputs. Wall-clock
budgets vary with machine load; saved phrase sequences preserve exact outputs.

Validation includes reference self-tests and independent rollback checks for
prefix/suffix counters, active inventory, phrase ownership, and word counts.
The output auditor uses only standard-library parsing, checks dictionary
membership, phrase identity/uniqueness, exact letter reversal, word repetition,
and preserved opening/ending. It does not trust the search's validator.

## Final result

The five-minute feasible-inventory run produced **90,937 letters**:
**498 more than Norvig's 90,439**, an increase of
0.551%. It contains **16,168 distinct phrases**.

Under identical ASCII word tokenization, the result has 21,222
words against the reference's 21,194, a gain of
28. Under Norvig's own phrase-count-plus-internal-spaces
convention it has 21,088 words against his published 21,012.

All phrases belong to the original dictionary. No normalized phrase repeats,
no adjacent words repeat, and the maximum non-function-word count remains 3.
The exact reversal, dictionary membership, opening/ending, counts, and file
hash were independently audited. The actual beginning and ending were inspected;
it remains a list of names and noun phrases, with no readability claim.

Deliverables: `artifacts/norvig-v3/palindrome.txt`, `phrases.json`, `result.json`,
and `audit.json`. The existing public `/panama` page was not changed by this
research run.

Validation: **14 tests passed**, including counter restoration, word-cap
exclusion/restoration, seed resume, independent corruption rejection, the
previous phrase search, growth conventions, and graph closure checks. A resume
regression initially exposed choosing the wrong side of a phrase boundary;
choosing the nearest boundary fixed it. This did not affect the from-scratch
winning run. `git diff --check` passed.

This beats the published Norvig v3 result on length within the same dictionary,
with stricter repetition constraints. It does not establish a global maximum,
a world record, or an improvement in meaning. The useful search change is
maintaining counts of available phrases as the walk uses and restores them.
