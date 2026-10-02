# NAACL paper readiness plan

**Current manuscript, revised October 2, 2026:** *Evaluating Exact Palindromes Beyond Word-Order Shuffles*. Read the [anonymous PDF](../output/pdf/naacl2027-v1.pdf) or [source](naacl2027-v1.tex). The public version label remains v1. `naacl2027.tex` is a synchronized source alias; historical run identifiers are unchanged.

## Scope and venue

The paper targets a focused short-paper contribution in evaluation: a retrospective analysis of automatic diagnostics on exact palindromes. It does not present a new readable generator or a matched comparison with published systems.

The [NAACL 2027 call](https://2027.naacl.org/calls/main_conference_papers/), checked October 2, sets the ARR deadline at **October 12, 2026, 11:59 p.m. AoE**, followed by a separate NAACL commitment deadline of December 23. Evaluation and reproducibility are in scope. The paper fits the four-page short-paper content limit.

The [ARR requirements](https://aclrollingreview.org/cfp) require an anonymous ACL-format manuscript, a dedicated Limitations section, and anonymized supplementary material. Limitations and optional ethical considerations follow the conclusion; references and subsequent appendices are outside the content limit. Software and data have separate archive fields. The combined local evidence archive and professor handoff are review materials, not a completed OpenReview submission. Authorship, profile/service requirements, submission history, and the responsible NLP checklist must be verified when preparing the submission form.

## What changed

The title and abstract now describe the evidence actually available. The introduction states one research question and relates the analysis to criterion-specific evaluation checks. The methods distinguish sample collection from evaluation and separate the two native search closure rules from the bundled resource settings. The discussion explains the relative-score interpretation and gives a bounded reporting recommendation. The conclusion states the finite-sample result.

A new retrospective cross-tab joins the saved Brown and GPT-2 scores and the Qwen3 responses by exact text hashes:

| Brown contrast | GPT-2 contrast | Output records | Distinct texts |
| --- | --- | ---: | ---: |
| Positive | Positive | 21 | 15 |
| Positive | Nonpositive | 19 | 12 |
| Nonpositive | Positive | 16 | 16 |
| Nonpositive | Nonpositive | 10 | 9 |
| Total | | 66 | 52 |

All 52 distinct texts receive zero on both Qwen3 dimensions. The 15-text overlap shows that the observation survives duplicate collapse. Repeated copies retain item-specific shuffle controls; every copy agrees on both signs before collapsing. These are descriptive counts, not independent trials or a significance test. Run `python3 paper/audit_diagnostic_overlap.py` to reproduce the table and Qwen3 joins without inference, corpus downloads, or network access.

## Evidence boundary

The collection covers one vocabulary, three project-defined length bands, and 20 search seeds. Only successful outputs have language diagnostics. The resource follow-up changes four settings together, and the implementations have different closure rules. Of 66 records, 36 share the phrase “one poem association.” Duplicate collapse does not remove that dependence.

Brown and GPT-2 use different shuffles and scoring units. The Qwen3 screen separates 11/12 prose controls from 0/12 prose shuffles, but has no human calibration and provides no quality ranking within the all-zero candidate set. Qwen2.5 fails its screen. DeepSeek ties every pair. No human ratings or matched external-generator comparison have been added. The Diana appendix remains a separate construction example with unresolved authorship and no human ratings.

The [September baseline audit](BASELINE-PROVENANCE-AUDIT-V1.md) remains applicable. The [earlier simulated review](mock-reviews/naacl2027-v1-arr-review.md) is a historical snapshot with its original 2.5/5 assessment, not a score for this revision or an ARR decision.

## Verification

The main text ends on page four. The complete anonymous PDF has five pages, including limitations, references, and appendices, in the existing ACL review style. The full project compiles with Tectonic. The built-in single-file compiler cannot resolve the separate `acl.sty` dependency; the saved source remains available in the editor.

The offline audit verifies candidate exactness, score-input hashes, complete score inventories, duplicate-sign consistency, and candidate/response text joins. Regression checks reject stale likelihood inputs and mismatched Qwen text hashes. The supplementary builder checks archive hashes, anonymity, linked provenance, and deterministic builds. Raw search, model responses, and frozen reader items remain unchanged. No generation or inference was run for this revision.
