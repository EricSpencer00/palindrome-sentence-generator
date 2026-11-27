# Low fragment-penalty development probe — October 8, 2026

## Protocol

Jev’s suggested low-strength range was converted into a bounded paired development probe. The final frozen revision tested penalties **0, 1, 2, and 4** on two seeds (9100, 9101), two bands (120–159 and 800–999 letters), and both existing search methods, for **32 search jobs**. Every seed–band–method block contains all four penalties in deterministic randomized order. Penalty 0 passed through the same `FragmentScorer` wrapper and was verified to return the unmodified Zipf score.

The run used the existing six frozen target trigrams, beam 256, candidate limit 800, 32 children per parent, 400 steps, diversity 0.4, overhang 24, a 256-word cap, and a 60-second cooperative deadline per job. Search ran on the NUC with two workers pinned to CPUs 2–3. The final manifest was frozen before search and contains 66 source hashes.

## Execution integrity

- Final manifest: `manifest-v4.json` and the executed copy `manifest-v4-executed.json` have the same SHA-256: `621200d3799fc2a50aaca07cfe483293658cc0d5d421e5f32e394ec9c3703e0c`.
- Results: `results-v3.json` SHA-256 `62fde404b60d18d1c41c42928abd9eecf9bf76ce2148a7969ede28240c25bb9e`; summary: `summary-v3.json`. All 32 planned IDs appear once, no search-job execution errors occurred, and all selected outputs independently pass exact-letter palindrome validation.
- Three earlier setup/preflight attempts are archived separately and excluded from the 32-job denominator: two runner setup failures before search expansion, and one preflight command typo. The frozen revision 4 records those histories.
- This run is development evidence. Brown scores, a grammar model, language-model selection, held-out human ratings, and readability evaluations were not run. Mechanical strict checks are structural diagnostics, not grammar ratings.

## Feasibility-first results

| Penalty | Basic-valid exact outputs / 8 | Strict-admitted / selected | Deadline jobs | Execution errors | Mean letters among outputs |
|---:|---:|---:|---:|---:|---:|
| 0 | 4/8 | 0/4 | 1 | 0 | 475.0 |
| 1 | 4/8 | 0/4 | 1 | 0 | 489.0 |
| 2 | 4/8 | 0/4 | 1 | 0 | 466.5 |
| 4 | 4/8 | 0/4 | 1 | 0 | 483.0 |

All eight paired blocks had identical feasibility across penalties: seed 9100 produced no admissible closure for any arm; seed 9101 produced one output in each of the four seed–band–method blocks for every arm. Thus each penalty yielded 4/8 outputs, and every paired output-yield difference versus penalty 0 was **0**. This small development set cannot establish equivalence or identify a population-optimal penalty.

One outside-in 800–999-letter job per penalty hit its 60-second deadline and returned its best valid output so far. There were no deadline jobs without an output.

### Paired output lengths

Rows below are matched seed 9101 blocks; seed 9100 had no output under any arm.

| Band | Method | Penalty 0 | Penalty 1 | Penalty 2 | Penalty 4 |
|---:|---|---:|---:|---:|---:|
| 120–159 | outside_in | 121 | 159 | 121 | 159 |
| 120–159 | center_out | 158 | 158 | 126 | 156 |
| 800–999 | outside_in | 807 | 815 | 807 | 805 |
| 800–999 | center_out | 814 | 824 | 812 | 812 |

Penalty 1 had the longest selected output (824 letters, 185 words), and mean selected length was 489 letters versus 475 at penalty 0. Penalty 4 averaged 483 and penalty 2 averaged 466.5. These are four outputs from one successful seed block, with one deadline return per arm; they are descriptive only. There were 16 distinct selected surfaces overall.

## Structural residuals

No selected output passed the full strict gate (0/16). Among the 16 selected outputs, strict failures occurred as follows: `no_self_palindromic_word` in 14, `no_self_palindromic_proper_multiword_span` in 8, and `distinct_words` in 6. These overlapping structural checks say nothing direct about grammaticality or reader judgments.

Across all searched closures, basic-gate rejections were only `ordinary_short_words`: penalty 0 had 3, penalty 1 had 3, penalty 2 had 5, and penalty 4 had 4. The 16 no-output jobs (all seed 9100) found no closure at or above the requested minimum; they are search failures, not execution errors.

### Longest observed candidate

**s9101-n800-center_out-p1 — 824 letters, 185 words.** It independently passes exact-letter palindrome and basic admission, but fails three mechanical strict checks listed above. The text is preserved verbatim here as a candidate artifact, not as evidence of readability:

> Push par got oh psi me trade tai cos sara declare billionaire bot code tam it seat admit civil ear side sub believe gala tip so help missy as separate gwen knee beta ret illinois ice drag delivered rosa war against understand a ordered is no clear sign in everett also like beg del won know almanac ire mania model bastille were hotel poe peninsula collar eve set armor first ah those it rep or page more talk cabinet for ever off one definitely let are bile duo yet at seem and raw design i villa no it a not to otto national living is edward name estate you deliberately let in if eden of forever often i back later omega properties oh that sri from rate several local us nine people to here well its able domain american am law on knowledge be kilos latter evening israel considered road nats red nuts niagara was order evil edgar decision illiterate been knew get a rap essays simple hospital age veil ebb used israeli victim data estimated october ian oil liberal cedar associated artemis photographs up.

## Interpretation

The probe offers no feasibility basis to prefer 1, 2, or 4 over 0: exact-output yield tied in every paired block. Penalty 1 produced the longest candidate and the highest mean selected length in this tiny development set, but it also left a targeted-fragment occurrence in each of its four outputs; each other arm had zero such occurrences. None of these measurements establish human readability. The right next evidence for that claim is a separate held-out, blinded human evaluation; this probe did not use or consume one.

## Files

- `manifest-v4.json`: final frozen protocol and setup-attempt history.
- `manifest-v4-executed.json`: copy retrieved from the NUC and hash-matched to the frozen manifest.
- `results-v3.json`: complete per-job outputs, counters, exactness checks, strict failures, timings, and scorer calls.
- `summary-v3.json`: per-penalty outcome summary.
- `preflight-v4.json`: source-hash, import, signature, vocabulary, dependency, and zero-strength checks.
- `attempt-1/`, `attempt-2/`, `setup-attempt-3.json`: retained setup failure evidence, separate from search results.

## Independent offline reproduction

Run `python3 experiments/verify_parameter_probe_20261008.py` to verify all32
job joins, frozen source hashes, every selected full reversal, and the complete
mechanical check maps. Root independently reproduced these checks after
retrieval. Both task-created NUC workspaces are removed.
