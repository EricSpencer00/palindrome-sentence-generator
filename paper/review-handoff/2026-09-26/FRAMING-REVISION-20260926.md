# Existing-results framing revision — September 26, 2026

The user requested a manuscript revision using existing results, with no new
experiments. The revised title is *Evaluating Exact Palindrome Generation
Beyond Character Validity*. The central result is the coexistence of positive
shuffle-relative likelihood and floor-level absolute model ratings on exact
generated outputs. Construction yield, output variety, and the length
analysis explain the setting in which those outcomes occur.

## What changed

- Rewrote the title, abstract, introduction, and discussion around an empirical
  evaluation case study.
- Moved the saved length-by-construction summaries into the main paper.
- Included an exact output from the frozen selection order so readers can
  inspect the language directly.
- Kept the 38/54-letter Diana pair in a separate retrospective appendix.
- Defined relative likelihood, absolute rubric outcomes, control screens,
  and denominators without treating any of them as human readability.
- Updated the professor brief and handoff builder to copy the current PDF
  and source together.

## Claim-to-source map

Paths below are relative to the repository root. The anonymous supplement
contains the corresponding frozen inputs, saved outputs, and analyzers.

| Manuscript claim | Saved source |
| --- | --- |
| 240 tasks; 0/120 versus 66/120; paired seed-cluster interval | runs/directional-search-resource-sensitivity-20260925/confirmatory-v4-resource-sensitivity.json; manifest.json and summary-v4.json in the same directory |
| Closure rules, vocabulary, lexical gate, resource settings, search heuristic | experiments/paired_directional_search_resource_sensitivity_20260925.py and the frozen manifest |
| 52 distinct strings; 54/66 stricter mechanical passes; recurring phrase; quoted selected output | runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json; confirmatory-v4-resource-sensitivity.json and summary-v4.json in the same directory |
| Brown 40/66; GPT-2 37/66; DeepSeek 66 ties | runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json and underlying saved scorer records |
| Length table, score units, subgroup counts and descriptive comparisons | experiments/READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md and its included standard-library analyzer |
| Qwen3 zero candidate scores, 11/12 control passes, 0/12 control-shuffle passes, 156 candidate shuffles, repeat agreement | experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md; archived run responses and frozen rubric |
| Qwen2.5 failed calibration and post-hoc single-sentence check | experiments/LOCAL-LLM-READABILITY-AUDIT-RESULTS-20260926.md; archived protocols and raw responses |
| Diana surfaces, construction provenance, token/sentence counts and Brown gains | runs/readability-length-stratified-sentence-aware-20260925.json; paper/READABILITY-SCORE-AUDIT.md; length/construction report |
| Relation to existing work | paper/refs.bib; cited Norvig, Papadopoulos, Bonlarron/Régin, Howcroft et al., and Khashabi et al. sources |

## Interpretation retained

The original resource bundle produces no texts, so language quality cannot
be compared across resource bundles. Four settings changed together after
earlier zero-yield runs. Length summaries condition on selected successes and
do not estimate a causal length effect. Repeated outputs and shared phrases
limit diversity.

The Qwen3 screen is small and partly multi-sentence. Model pretraining
exposure to Brown is unknown; hosted responses lack a provider seed and
weight digest. Saved responses can be aggregated exactly, but new API calls
need not reproduce them. No human ratings or prior-generator superiority
results have been added.

## Review and delivery checks

An independent subagent reviewed the rewritten manuscript against saved
records. It confirmed the counts, score directions, length cells, and Diana
figures. Its two substantive wording fixes were applied: the recurring phrase
is described as appearing across outputs, and the discussion no longer
implies a cross-bundle language-quality comparison.

The current PDF is output/pdf/palindrome-evaluation/naacl2027.pdf. The build,
text extraction, page inspection, supplement manifest checks, and professor
packet source/PDF equality checks are part of this editorial turnaround.
They introduce no new search or scoring observations. Historical PDF outputs
are preserved separately.

## Final review corrections: complete

The final mock ARR review recommended 3/5 overall (Findings). Its four
editorial requests are resolved:

| Review concern | Completed correction | Evidence |
| --- | --- | --- |
| Incremental contribution unclear | Introduction states the recorded findings and why coverage, variety, likelihood, and absolute ratings require separate interpretation; discussion identifies the question each criterion leaves unresolved | Existing v4 outputs and scorer records; cited prior work |
| Rubric and control interpretation | Methods include all 0–3 anchors and the exact-text instruction; the screen is explicitly limited to ordinary versus shuffled prose, without validating quality differences among palindromes | Frozen local_llm_readability_audit_20260926.py prompt, reused by Qwen3 |
| Seed mechanics and experimental unit | Methods specify child-ranking perturbations in [0, 0.4), their exclusion from the final objective, 32-word and 24-letter-overhang limits, and whole-seed resampling of all 12 conditions | Frozen search.py, centerout.py, resource manifest, and analyzer |
| Descriptive length results | Subsection heading and interpretation describe scorer-specific patterns and closure-length concentration without asserting effects of length or strategy | Existing length-by-construction table |

The supplementary report and its rendering function now name the 0.280
GPT-2 contrast as the outside-in-minus-center-out difference in raw candidate
mean token log probability. Its definition is preserved in the saved
static-verifier summary and summarizer source. It is distinct from the
candidate-minus-shuffle endpoint used by the manuscript's length table.
No numbers, search/scoring code, frozen protocols, or raw outcomes changed;
the report renderer changed only explanatory labels.

The final source, PDF, professor brief, and packages are synchronized. The
editorial work is complete, with no further internal review or experiment
scheduled. The earlier review entries above remain a chronological record.
