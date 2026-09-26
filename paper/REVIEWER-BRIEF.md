# Reviewer brief

**Public release v1:** *Methods of Finding Longer, Readable Palindromes*

## Central result

The manuscript examines how construction success, output length, relative
likelihood, and absolute model ratings relate on the same exact outputs.
Brown word-order gain is positive for 40/66 selected outputs and GPT-2 for
37/66. All 52 distinct candidates nevertheless receive zero grammaticality
and coherent-meaning scores under a Qwen3 32B rubric. That judge passes a
small control screen: 11/12 prose spans and 0/12 prose shuffles pass jointly.
None of the 156 candidate shuffles passes either.

This is an empirical evaluation case study. Positive gain over deliberately
disordered text and an absolute rubric score measure different properties.
Their coexistence shows why the former cannot establish sentence quality in
this output set. No human ratings validate either interpretation.

The contribution is a reproducible record connecting construction coverage,
output variety, relative likelihood, and absolute ratings. It shows which
questions remain unanswered by each favorable result: construction yield
does not measure diversity, shuffle gains do not establish sentence quality,
and prose-control separation does not validate distinctions among palindromes.

## Construction and length

The frozen study contains 240 runs: 20 seed clusters, three target length
bands, two search implementations, and two resource bundles. The original
bundle returns 0/120 basic-valid outputs and the larger bundle 66/120.
Four settings change together in an adaptive follow-up. The comparison
establishes output coverage under those conditions; no original-bundle texts
exist for comparing language quality. These are project implementations,
without a same-input reproduction of a published generator.

Seeds control uniform perturbations in [0, 0.4) added to child scores for
beam ranking. These perturbations do not enter the objective used to select
completed outputs. Both searches cap partial texts at 32 words and residual
overhang at 24 letters. Bootstrap resampling retains all 12 runs belonging
to each seed, measuring search-seed variation under a fixed vocabulary.

Table 1 presents the existing length and likelihood summaries by search
implementation. The trends are descriptive and conditional on successful
search. All 52 distinct strings are letter-level palindromes; none reverses
the word sequence. Thirty-six of the 66 selected outputs contain the exact
phrase “one poem association.”

The Diana appendix is a separate, retrospective 38/54-letter construction
pair. It demonstrates a recorded extension with exactness preserved and
contrasts own-word shuffle scores with prose matched on scorer-token count.
It is outside the main study and Qwen audits. Its reference has no established authorship in the
project record; neither text has project human ratings.

## Interpretation limits

Qwen2.5 fails its control gate, so its candidate scores are inconclusive.
Qwen3 passes the small screen, but seven prose controls span multiple
sentences, pretraining exposure to Brown is unknown, and 11/12 hidden repeats
agree. A passed screen does not make its rubric a validated readability
scale. DeepSeek ties all 66 candidate–shuffle pairs.

The paper claims neither a readable-generator advance nor superiority over
prior systems. The September 26 revision changes framing and presentation;
it includes no new experiments or model scoring calls.

## Final revision status

All four requests from the final internal review are incorporated: a more
explicit contribution, the complete rubric anchors and control-screen
boundary, search randomization and fixed limits, and descriptive length
wording. The supplementary report also labels the 0.280 GPT-2 contrast as a
raw candidate-likelihood difference, distinct from candidate-minus-shuffle
gains. No search or model scoring was rerun.

The final mock ARR assessment before these corrections recommended 3/5
overall (Findings), with soundness 3.5/5, excitement 2.5/5, and reproducibility
4/5. An independent Astra review also recommended 3/5. These are internal
assessments, not peer-review decisions or acceptance guarantees. The
manuscript and evidence package are finalized for handoff; no additional
internal review or experiment is scheduled.
