# Reviewer brief

**Draft:** *Evaluating Exact Palindrome Generation Beyond Character Validity*

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

## Construction and length

The frozen study contains 240 runs: 20 seed clusters, three target length
bands, two search implementations, and two resource bundles. The original
bundle returns 0/120 basic-valid outputs and the larger bundle 66/120.
Four settings change together in an adaptive follow-up. The comparison
establishes output coverage under those conditions; no original-bundle texts
exist for comparing language quality. These are project implementations,
without a same-input reproduction of a published generator.

Table 1 presents the existing length and likelihood summaries by search
implementation. The trends are descriptive and conditional on successful
search. All 52 distinct strings are letter-level palindromes; none reverses
the word sequence. Thirty-six of the 66 selected outputs contain the exact
phrase “one poem association.”

The Diana appendix is a separate, retrospective 38/54-letter construction
pair. It demonstrates a recorded extension with exactness preserved and
contrasts own-word shuffle scores with prose matched on scorer-token count. It is outside the v4
study and Qwen audits. Its reference has no established authorship in the
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

## Questions for the professor

1. Does the abstract make the evaluation contribution clear and proportional
   to the evidence?
2. Are the denominators, calibration limits, and descriptive length findings
   understandable on a first reading?
3. Which passages need a clearer explanation or a narrower claim?

The prior resource-centered draft received an informal internal estimate of
2.5/5 overall. That estimate predates this rewrite. The current independent
AI review checked factual support and framing; it did not assign a new score
and is not peer review.
