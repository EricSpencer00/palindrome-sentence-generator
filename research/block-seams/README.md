# Typed block and seam palindrome research

This research extends exact residual cancellation to finite typed constituents. A state can add a word or phrase to either end when the letter streams remain compatible and a source-independent grammar admits the frontier. Provenance records identify possible source occurrences; they do not define grammatical admission.

## Usage

Run the focused checks without model calls:

```sh
python3 -m pytest tests/test_bidirectional_block_search.py tests/test_block_seams.py tests/test_compositional_grammar.py tests/test_paragraph_review_regressions.py tests/test_block_seam_comparison_20261009.py
```

`llm_palindrome.block_search.block_beam_search` accepts an inventory, a word-additive scorer, grammar and closure callbacks, and explicit beam, action, word, letter and time bounds. `compatible_actions` exposes the actual two-sided menu. The version is `typed-two-sided-block-beam-v1`.

The development runner is `python3 -m experiments.block_seam_comparison_20261009`. It reserves run 003, refuses an existing output configuration, and uses one worker with an aggregate 60-second budget. No run-003 experimental outcomes are included in this release. Its empty-start design has 12 cells: two arms, seeds 921–923, and letter bands 60–119 and 120–239. Eligible closures require a single global exact palindrome and two to four distinct rendered clauses. The reference beam receives single words; the block arm uses both sides, phrase actions, grammar frontiers and diversity pruning. Scheduling and pruning therefore change along with action granularity. The arms' action counters have different denominators.

## Frozen observations

| Pilot | Method | Observed outcome |
| --- | --- | --- |
| 001 | Fixed endpoints, legacy source-bound grammar | Zero closures in 12 cells; five starts had no initial letter match. |
| 002 | Empty starts, repaired grammar, inherited opposite-side expansion | Baseline: 24 exact closure presentations, 21 distinct raw texts of 60–80 letters, zero grammar-eligible paragraphs. Block arm: zero closures. |

Pilot 002 used 38 words, 54 blocks and nine grammar productions. Each baseline cell recorded 2,000 action logs; its counter includes one additional cap-trigger attempt. Each block cell recorded 74 grammar-callback proposals: 51 rejected and 23 retained. Index construction and search took 3.424 seconds; the first result write brought elapsed time to 4.092 seconds, with zero recorded deadline overrun.

The decisive witness was left `eva carries`, right `a cave`, with residual `rries`. Grammar lookahead licensed six left additions, while the inherited expansion menu offered none. The new two-sided engine exposes all six. Tests establish this repair and compare terminal sets against exhaustive enumeration on 11 small fixtures. These are implementation checks, not results from a new paragraph pilot.

The legacy inventory audit parsed 144 of 148 authored rows and tested 99,387 interfaces derived from 18,392 whole-VP templates. Its source-bound grammar makes those counts a description of that finite method rather than evidence of lexical scarcity.

## Evidence and status

`evidence/` contains frozen configurations, results, action logs, timing and analyses. The largest JSON is gzip-compressed; Python's `gzip.open(path, 'rt')` reads it directly. `export-manifest.json` records original receipt hashes and public-export hashes. Public exports remove local coordination and service metadata while retaining scientific records; original receipts remain preserved locally. Historical receipt hashes refer to the original files, while export hashes identify the supplied public files.

The finite grammar does not certify human meaning or cross-clause coherence. No pilot produced an independently accepted paragraph. Beam or action truncation limits coverage; an action-cap interruption returns the last complete frontier and discards a partly constructed next frontier. `remaining_beam` consequently does not enumerate every attempted continuation. Human calibration judgments are categorical and separate from model evidence; no numeric scores have been inferred. The three named calibration examples are controls, with novelty unverified or reported as already existing.

## Attribution and inputs

Project code uses the repository MIT license. The runner reads existing repository wordlists in place and distributes no additional wordlist or corpus copies. The Norvig/Hoey residual-search adaptation and wordfreq attribution remain in the repository documentation and licenses. The small illustrative catalogue retains its [provenance record](../../data/readable_palindrome_centres.PROVENANCE.md): widely circulated examples are controls, with individual source and rights determinations still unresolved. They are not a licensed training corpus or newly authored discoveries. The grammar and scene drafts are finite project constructions.
