# NAACL paper readiness plan

**Current manuscript, revised October 2, 2026:** *Evaluating Exact Palindromes Beyond Word-Order Shuffles*. Read the [anonymous PDF](../output/pdf/naacl2027-v1.pdf) or [source](naacl2027-v1.tex). The public version label remains v1. `naacl2027.tex` is a synchronized source alias; historical run identifiers are unchanged.

## Scope and venue

The paper targets a focused short-paper contribution in evaluation: a retrospective analysis of automatic diagnostics on exact palindromes. It does not present a new readable generator or a matched comparison with published systems.

The [NAACL 2027 call](https://2027.naacl.org/calls/main_conference_papers/), checked October 2, sets the ARR deadline at **October 12, 2026, 11:59 p.m. AoE**, followed by a separate NAACL commitment deadline of December 23. Evaluation and reproducibility are in scope. The paper fits the four-page short-paper content limit.

The [ARR requirements](https://aclrollingreview.org/cfp) require an anonymous ACL-format manuscript, a dedicated Limitations section, and anonymized supplementary material. Limitations and optional ethical considerations follow the conclusion; references and subsequent appendices are outside the content limit. Software and data have separate archive fields. The combined local evidence archive and professor handoff are review materials, not a completed OpenReview submission. Authorship, profile/service requirements, submission history, and the responsible NLP checklist must be verified when preparing the submission form.

## What changed

The abstract and results now lead with continuous diagnostics and lexical recurrence. Two full-width figures contain five panels: Brown versus GPT-2 contrasts, all 32 candidate-minus-shuffle GPT-2 differences per record, trigram prevalence, and phrase-group score distributions for each scorer. The judge floor remains a brief supporting result, with detailed judge settings in the appendix.

| Descriptive result | Saved-data value |
| --- | --- |
| Brown/GPT-2 contrast signs disagree | 28/52 distinct texts |
| Candidate GPT-2 score falls inside its 32-shuffle range | 63/66 records |
| Candidate exceeds every shuffle / is below every shuffle | 2/66 / 1/66 records |
| Distinct texts containing “no it a” | 46/52 (88.5%) |
| Distinct texts containing “one poem association” | 24/52 |
| Mean Brown contrast, focal phrase present / absent | +0.201 / -0.003 nats per transition |
| Mean GPT-2 contrast, focal phrase present / absent | -0.013 / +0.110 per predicted token |

Each text-level point averages contrasts across that exact text's records. Each record-level range uses its own 32 archived controls. The figures show empirical permutation spreads, not confidence intervals. Phrase groups and overlapping trigrams are retrospective descriptions; length, vocabulary, and construction are confounded with membership.

Run `python3 paper/build_diagnostic_figures.py` with `paper/figure-requirements.txt` installed. The builder validates the existing hash joins and writes both vector figures, previews, and `paper/fig/diagnostic-data.json`. Numerical data can be re-aggregated from a clean extraction of the anonymized archive. No inference is needed. The earlier cross-tab remains available through `paper/audit_diagnostic_overlap.py`.

## Evidence boundary

The collection covers one vocabulary, three project-defined length bands, and 20 search seeds. Only successful outputs have language diagnostics. The resource follow-up changes four settings together, and the implementations have different closure rules. Of 66 records, 36 share the phrase “one poem association.” Duplicate collapse does not remove that dependence.

Brown and GPT-2 use different shuffles and scoring units. The Qwen3 screen separates 11/12 prose controls from 0/12 prose shuffles, but has no human calibration and provides no quality ranking within the all-zero candidate set. Qwen2.5 fails its screen. DeepSeek ties every pair. No human ratings or matched external-generator comparison have been added. The Diana appendix remains a separate construction example with unresolved authorship and no human ratings.

The [September baseline audit](BASELINE-PROVENANCE-AUDIT-V1.md) remains applicable. The [earlier simulated review](mock-reviews/naacl2027-v1-arr-review.md) is a historical snapshot with its original 2.5/5 assessment, not a score for this revision or an ARR decision.

## Verification

The main content and both figures fit within four pages. The complete anonymous PDF has six pages, including limitations, references, and appendices, in the existing ACL review style. The full project compiles with Tectonic. The built-in single-file compiler cannot resolve the separate `acl.sty` dependency; the saved source remains available in the editor.

The figure regression checks verify saved numerical data, permutation accounting, fragment counts, and equal weighting of distinct texts. The offline audit verifies candidate exactness, score-input hashes, complete score inventories, duplicate-sign consistency, and candidate/response text joins. Regression checks reject stale likelihood inputs and mismatched Qwen text hashes. The supplementary builder checks archive hashes, anonymity, linked provenance, and deterministic builds. Raw search, model responses, and frozen reader items remain unchanged. No generation or inference was run for this revision.
