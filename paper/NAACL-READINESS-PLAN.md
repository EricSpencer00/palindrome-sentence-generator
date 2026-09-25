# NAACL 2027 paper readiness plan

**Target.** Prepare a short-paper draft and anonymous evidence package for the
NAACL 2027 ARR October cycle. The official call lists a four-page short-paper
limit and an ARR submission deadline of 12 October 2026. NAACL uses ARR reviews
before conference commitment; preparing this draft does not submit it.

## Claim boundary

The paper's question is what a reproducible word-bigram diagnostic reports for
selected exact palindrome texts against held-out corpus and same-word controls.
It will not use length increase as evidence of language quality. It will not
call the automatic value a readability score, and it will not imply human
readability, coherence, novelty, or superiority over another generator.

## Work plan and current status

| Step | Completion check | Status |
| --- | --- | --- |
| 1. Freeze claims to evidence | Exactness, samples, sources, and the local-order interpretation are defined in the manuscript and audit. | Complete |
| 2. Audit readability scoring | Recompute with the document-held-out Brown split and compare every saved score/control with `--expected-report`. Include all eleven candidates and retain the short inherited example. | Complete; exact rerun passed |
| 3. Rewrite the paper | Center the local-order question; remove the character-growth table and full-length endpoint from the main narrative; cite prior palindrome generation, Brown/NLTK, and human-evaluation methodology. | Complete |
| 4. Make reproduction portable | Pin scorer dependencies, generate the anonymous source/evidence archives, and run the included exactness verifier and Brown-score rerun. | Complete; exported archives replayed |
| 5. Build and inspect the PDF | Compile with the ACL review template, check the page limit, references, line breaks, table, and every rendered page. | Complete; three pages rendered and inspected |
| 6. Decide submission readiness | Confirm the main-paper contribution and whether claims require human ratings or a comparative language-quality baseline. | Open research gate; evidence is insufficient to call this NAACL-ready |

## Evidence map

| Paper claim | Source and repeatable check |
| --- | --- |
| The eleven saved renderings are exact letter palindromes. | `paper/week_results.json`; `paper/audit_week_results.py`; `paper/verify_anonymous_evidence.py` |
| Ten of eleven matched Brown spans outscore their candidate; the 38-letter inherited text is the exception. | `runs/readability-length-stratified-sentence-aware-20260925.json`; `experiments/score_length_stratified_readability.py --expected-report ...` |
| All nine longer constructions have a positive prose-minus-candidate gap; the mean gap is 1.631 nats per transition. | The same frozen report; `paper/build_readability_table.py` |
| All 108 held-out length controls outscore their own shuffles. | The length-control rows and summary in the frozen report; the expected-report rerun compares every row. |
| The criterion is local word order, not human readability. | Scorer definition in `experiments/audit_programmatic_readability.py`; limits stated in paper and audit. |

The previous 2026-09-25 score report trained on the full Brown corpus and drew
controls from that corpus. It reproduces numerically, but it is in-sample; it
is excluded from the manuscript. The selected texts are retrospective and
related. Neither report supports a causal or population-level conclusion.

## Reproduction commands

From the repository root:

```sh
python3 paper/audit_week_results.py
python3 paper/check_seam_invariant.py
python3 paper/replay_clause_search.py
python3 -m pip install -r paper/readability-requirements.txt
python3 -m nltk.downloader brown
python3 experiments/score_length_stratified_readability.py \
  --expected-report runs/readability-length-stratified-sentence-aware-20260925.json \
  --output /tmp/readability-length-stratified-rerun.json
python3 paper/build_readability_table.py
python3 paper/export_submission.py
python3 output/naacl-submission/anonymous-evidence/verify_anonymous_evidence.py
```

Then compile and visually inspect the manuscript PDF. The paper's own summary
must stay within the official four-page short-paper limit, and all novelty,
method, result, limitation, ethics, and assistance statements must remain in
the main text rather than relying on reviewers to read the supplement.

## Open research gate

The present revision supports a narrow, reproducible claim about local word
order. It cannot claim that readers find the outputs grammatical, coherent, or
readable. It also contains no same-input language-quality comparison with a
prior palindrome system and no evidence that the observed score pattern
generalizes beyond this selected set and Brown controls. These gaps prevent an
honest claim of NAACL main-paper readiness. The score audit is supporting
diagnostic evidence; it is not a substitute for language-quality evidence.

To make a paper about readable palindrome construction reviewable, the next
research gate is an authorized, blinded reader study on frozen outputs, with
intact-prose and shuffled controls, separate grammaticality and coherence
rubrics, and completed responses. The recruitment route and any required
ethics review must be settled before people are contacted. A same-input
comparison with a relevant prior palindrome method and a broader evaluation
design would still be needed to support a general method claim. Until those
results exist, keep this draft explicitly scoped to the saved audit.

## External requirements checked

- [NAACL 2027 main-conference call](https://2027.naacl.org/calls/main_conference_papers/): short-paper length and ARR-to-conference process.
- [ARR author guidelines](https://aclrollingreview.org/authors): self-contained main text and the role of the limitations section.
- [Howcroft et al. (2020)](https://aclanthology.org/2020.inlg-1.23/): reporting human-evaluation criteria and protocol.
- [Khashabi et al. (2022)](https://aclanthology.org/2022.emnlp-main.787/): reproducible, standardized human evaluation for generation.
