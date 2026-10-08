# Search frontier audit — 2026-10-08

## Finding

There is no single artifact that is both the longest exact output and a
reader-validated result. The Oct. 2 seam lineage reaches **650 normalized
letters**, while a separate Sep. 22 comparison artifact reaches **666**.
Neither has human readability ratings. The repo's short status pages still
designate the **568-letter causal-scene construction** as the working
incumbent; this is a conservative lineage designation, not a claim that it is
the longest exact string or readable prose. Some Oct. 2 run metadata calls the
650 row a working-length incumbent, so those labels should not be conflated.

## Scope correction after the broader audit

The650/666 figures below concern the specific inspected lineages, not the
repository-wide longest construction. Older residual-edit artifacts reach752
letters. The October8 cap probe adds804-letter exact outputs that fail strict
screens and are word salad. A new774-letter center insertion is retained as a
rejected shortcut diagnostic; a punctuation-only752-letter repair fixes one
local fragment without changing the tape. None has blinded reader validation.
See `runs/center-vocative-repunctuation-20261008.json`,
`runs/central-identity-clause-extension-20261008.json`, and
`experiments/LENGTH-CAP-PROBE-RESULTS-20261008.md`.

## Independently checked candidates

| Candidate | Normalized letters | Independent reversal check | Provenance and language evidence |
|---|---:|---|---|
| Causal-scene working incumbent | 568 | Parent tape SHA-256 `6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380`; later exact children retain independent checks | `runs/incumbent-560-outer-causal-scene-20261002.json`; retained as the clearest 568 lineage. No human ratings located. |
| Remaining-shell child | 594 | My `A-Z`-only lowercase two-pointer scan: exact; SHA-256 `2b85c0ef4654cf9a7f1ef15b7a599512f20b0c93daba7354c15594a37ab89a07` | `runs/incumbent-568-remaining-shell-global-gate-20261002.json`; three residual-matched event substitutions, with inherited names and rough prose. No reader validation. |
| Repaired-shell lineage maximum | **650** | My independent normalized scan: exact; SHA-256 `2bd92686cbd01945be3869fbeee7ae9415cfcdae5cd616b26d89ec4f0acc54a9` | `runs/incumbent-608-repeated-shell-repair-20261002.json`; direct parent is exact 608 child (`eacd84ecc82aacb84fe19b557faf91494370e86c986248be5e37de9ad7298248`). Two deterministic mirrored event-shell replacements add 42 letters. The replacements reduce “delivers maps” and “spam's reviled” from four instances each to two, but leave repeated names, `I` (10 occurrences), and formulaic scaffolding. Run marks rough prose and `human_certified: false`. |
| Comparison alternative | **666** | My independent normalized scan: exact; SHA-256 `cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297` | `runs/incumbent-666-comparison-alternative-20260922.json`; repeated `sees` (16), `Mara` (9), `Aram` (8), `Nora` (8), `saw`/`was` (8 each); explicitly no human reader validation. Its 666 naturalness-seam child is also exact, but is unpromoted and likewise has no human validation (`runs/incumbent-666-naturalness-seam-repair-20260922.json`). |

The exactness check independently applies the project normalizer (remove
non-ASCII letters, lowercase), counts the normalized tape, compares opposing
characters, and hashes the normalized bytes. These checks establish exact
construction, not grammaticality. The paired Oct. 2 AWS fragment-penalty study
is separate: none of its 148 distinct 30–119-letter generated texts passed
the model's joint grammar/meaning threshold, but those are not the 650/666
texts and are not human judgments. No primary human ratings or administered
reader protocol for the long candidates were located in the cited current
status records.

## Recent work and open seam

The repository history has no new search implementation after Oct. 2 through
Oct. 8; intervening commits are about paper framing, reproducibility, and the
paired fragment-penalty analysis. The Oct. 2 seam work includes live partial
word ownership at `w|on ... no|w` (568→608), then two exact mirrored event
substitutions (608→650). Its recorded next operator is the still-repeated
central `delivers/reviled` shell. Separate 568-family probes generated 594–600
letter alternatives, but the global-gate 594 was explicitly non-reader-ready;
the 650 and 666 artifacts must stay distinct because they have different
parents, construction histories, and status labels.

## Next bounded search

The concrete bounded driver and frozen protocol are in
`experiments/central_shell_dual_parse_20261008.py`, with its frozen inputs,
frame domains, acceptance gates, caps, and run contract in
`experiments/central_shell_dual_parse_20261008.PROTOCOL.md`. It uses the
existing `llm_palindrome.dual_parse.word_residual_search` over three typed
event-clause slots per side, but checks endpoints before constructing the
chart. The frozen domains have **0 prefix-compatible pairs out of 120**
between the first left actor and the reversed final right patient. Mismatch
counts are 115 at character 1, four at character 2, and one at character 3.
The closest pair, `Tara` / `boat`, yields `tara` versus `taob`, diverging at
character 3. Therefore this frozen domain is rejected as **infeasible before
search** and the 500,000-state chart is not launched. This is not a substantive
negative search result. The existing shell also has direct whole-word
reversals and reflected internal word boundaries.

The rest of the 650 tape is immutable. It already fails whole-text
distinct-content, repeated-unit, and proper-palindrome-span gates, so even a
successful seam child remains a construction frontier and cannot become a
reader-study item without broader repair and blinded human evaluation.

The alternative topology proposal is to expose the rightmost patient phrase
as separate determiner and noun slots. A concrete edge witness is left outer
word `Mara` against right final phrase `a ram`: `mara == reverse("a ram")`,
with a word boundary only on the right. `Tara sees a ram.` is one possible
right final clause. This proves endpoint compatibility only; it is not a
complete exact child. A revised chart should carry the determiner's residual
into the next left token before expanding the inner events. This changes the
grammar topology, rather than enlarging the infeasible actor/patient lists.

The driver was run locally only through this preflight. It wrote
`runs/central-shell-dual-parse-20261008.json` with status
`infeasible_before_search`, 0/120 compatible endpoint pairs, and
`search_launched: false`. Its source SHA-256 is
`c5d1c09e029fb465c684e853e6c8abeefc2dbadffdec332b3d134933f3339b90`.

Proposed NUC invocation for reproducing that preflight certificate, after the
source and frozen parent artifact are copied to that checkout:

```sh
cd /home/eric/palindrome-sentence-generator && python3 experiments/central_shell_dual_parse_20261008.py --parent runs/incumbent-608-repeated-shell-repair-20261002.json --out runs/central-shell-dual-parse-20261008.json --max-states 500000 --max-results 5000
```

The driver and parent input now exist in the local repo; this NUC path assumes
they have been copied to the stated checkout. It emits the same endpoint
infeasibility certificate and returns before the residual chart. No SSH or
compute search was started, and no chart result is claimed.
