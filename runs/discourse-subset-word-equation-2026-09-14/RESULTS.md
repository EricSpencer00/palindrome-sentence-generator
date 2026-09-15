# Whole-discourse word-equation experiment

The new construction treats the desired paragraph as an ordered subset of
complete source sentences. Its reversal can cross any number of sentence
boundaries; individual sentences need not have mirror partners. This tests a
different language from two-clause reverse intersection, without scoring
readability or constructing a paragraph out of palindromic units.

The frozen bank contains 60 newly authored sentences, split into three coherent
scenes: an art studio, a repair workshop, and a community garden. Each selected
sentence stays intact and may occur only once. The solver explores up to eight
sentences and 39–200 letters using exact residual cancellation. The run also
preserves 15 complete 100–160-letter diagnostic controls, each with independent
ASCII/two-pointer exactness evidence and all shared admission rejections.

Result: zero exact closures. All 1,140 ordered outer-sentence pairings fail
before a second sentence can enter a live residual state. There are 63 explored
states, and the state budget is not exhausted. Thus this bank never reaches
the multi-sentence mechanism being tested; the result diagnoses its sentence
inventory, not the usefulness of staggered sentence boundaries. No output is
promoted, and there is no independent reader evidence.

The next concrete operator must construct a **live boundary transition before
freezing a complete sentence**: choose a grammatical opening clause and a
grammatical final clause jointly, retaining their first mismatch; unlock the
two constituents containing that mismatch, including adjacent word boundaries;
enumerate grammatical phrase realizations until one complete clause cancels
and leaves a nonempty residual. Preserve the transition's two full sentence
renderings and their syntax witnesses, then feed that witnessed transition
into this multi-sentence solver. Abort that branch if the transition set is
empty. Adding more sentence permutations to the present inventory cannot help.

Four focused tests pass: an exact three-sentence crossing, agreement with an
independent exhaustive tiny-space oracle, residual-owner accounting, and an
independent audit that rejects mismatches and unsupported alphabetic input.

```sh
.venv-v3/bin/python -m pytest -q tests/test_discourse_subset_word_equation_20260914.py
.venv-v3/bin/python experiments/discourse_subset_word_equation_20260914.py --out /tmp/discourse-subset-replay.json
```

`result-01.json` contains the exact configuration, full source inventory,
program and provenance hashes, every rendered retained proposal, and rejection
details. Diagnostic control ordering uses exact character mismatches only;
it does not affect the exact search and is not a readability metric.
