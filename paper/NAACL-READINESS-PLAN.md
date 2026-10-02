# NAACL paper readiness plan

**Current manuscript, revised October 2, 2026:** *Fragment Penalties Shift Repetition in Exact Palindrome Search*. Read the [anonymous PDF](../output/pdf/naacl2027-v1.pdf) or [source](naacl2027-v1.tex). The version label remains v1; `naacl2027.tex` is a synchronized source alias.

## Scope and venue

The paper targets a focused short-paper contribution in evaluation: a paired experiment on fragment penalties in exact palindrome search. It does not present a new readable generator or a matched comparison with published systems.

The [NAACL 2027 call](https://2027.naacl.org/calls/main_conference_papers/), checked October 2, sets the ARR deadline at **October 12, 2026, 11:59 p.m. AoE**, followed by a separate NAACL commitment deadline of December 23. Evaluation and reproducibility are in scope. The paper fits the four-page short-paper content limit.

The [ARR requirements](https://aclrollingreview.org/cfp) require an anonymous ACL-format manuscript, a dedicated Limitations section, and anonymized supplementary material. Limitations and optional ethical considerations follow the conclusion; references and subsequent appendices are outside the content limit. Software and data have separate archive fields. The combined local evidence archive and professor handoff are review materials, not a completed OpenReview submission. Authorship, profile/service requirements, submission history, and the responsible NLP checklist must be verified when preparing the submission form.

## New experimental evidence

The paper now tests a scoring intervention, rather than centering the earlier scorer disagreement or all-zero rubric. Six trigrams were selected from the historical 52-text collection. A frozen protocol crosses 20 fresh seeds, three length bands, two native search implementations, and penalties 0/4/16. All 360 AWS jobs completed without errors or deadlines.

Basic exact yield remains 90/120 per arm. Distinct texts increase from 69 to 78 and mean trigram-set distance rises from 0.889 to 0.939 at penalty 4. Yet new fragments occur in 46/78 distinct penalized outputs, and strict structural acceptance falls from 59/120 to 28/120. The main figures show these tradeoffs. Paired seed-bootstrap intervals retain method, band, and arm dependencies.

See [the full report](../experiments/FRAGMENT-PENALTY-RESULTS-20261002.md), [frozen protocol](../runs/fragment-penalty-20261002/manifest.json), and [saved analysis](../runs/fragment-penalty-20261002/analysis.json). Reproduction uses `python3 experiments/analyze_fragment_penalty_20261002.py` without new inference.

## Evidence boundary

This is a within-implementation intervention on one vocabulary, not an external generator benchmark. Penalty targets were chosen after inspecting historical outputs, then fixed before the new seed block. The two implementations retain different native closure rules. Diversity conditions on success; strict structural checks do not measure readability. The model rubric is blinded to experimental arms and screened on prose/shuffle controls, but has no human calibration. No human ratings have been collected.

The prior 52-item first-person packet, historical provenance audits, and older simulated review are retained as historical material. They do not assess the new 148 distinct outputs. The earlier 2.5/5 mock-review score is not a review of this experiment or an ARR decision.

## Verification

The new scorer's potential differences were checked against exhaustive toy-sequence recounts for both growth directions, including broken-join refunds and the zero-penalty baseline. Analysis verifies all 360 jobs, exactness and text hashes, and recomputes pairwise distances independently. The anonymous archive preserves outcomes and ratings while rebinding input hashes to projected metadata. AWS result transfers and removal of temporary resources are recorded locally.

The paper uses the existing ACL review style. The built-in single-file compiler cannot resolve the separate `acl.sty`; the full project is compiled with Tectonic and visually inspected before delivery. The submission itself and author-side forms have not been completed.
