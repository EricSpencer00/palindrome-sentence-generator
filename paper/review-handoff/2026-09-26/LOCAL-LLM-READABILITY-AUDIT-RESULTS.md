# Local model readability audit: results

**Model:** Qwen2.5 3B Instruct, Ollama Q4_K_M
**Digest:** `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`
**Run:** `local-llm-readability-audit-20260926`
**Interpretation:** diagnostic only; calibration failed, so candidate scores are inconclusive.

## First frozen audit

The scorer rated 52 unique v4 candidate strings, three deterministic same-word
shuffles for each string, 12 Brown corpus controls and their shuffles, and 12
hidden duplicate presentations: 244 presentations total. It scored each text
on grammaticality and coherent meaning from 0 to 3. A joint pass required at
least 2 on both dimensions. Presentation order, prompt, structured schema,
model digest, decoding settings, IDs, and response hashes are archived in the
run JSON. No hosted inference or AWS call was used.

| Group | Joint passes |
| --- | ---: |
| Palindrome candidates | 0/52 |
| Candidate word shuffles | 0/156 |
| Brown control spans | 1/12 |
| Brown control shuffles | 0/12 |
| Hidden repeats with exact same scores | 12/12 |

Every candidate received 0/3 for both grammaticality and coherent meaning.
The prespecified control gate required at least 8/12 Brown controls to pass
and no more than 3/12 shuffles to pass. The gate failed because only 1/12
controls passed. The 12/12 duplicate agreement shows repeatability under this
setup; it does not establish that the scores are valid readability ratings.

## Post-hoc individual-sentence sensitivity

After the first audit, source-span inspection found that 7 of its 12 Brown
control spans contained 2–4 adjacent sentences; 5 were single sentences. This
is a protocol-description defect, not a demonstrated cause of the first
failure. A separate, pre-inference protocol then scored 12 previously unused
sentence indices from the held-out Brown split and one deterministic same-word
shuffle per sentence. Sentence tokens were joined with spaces and case-
normalized, and each sentence matched its candidate target exactly in word
count (0-word absolute difference for all 12). Model, prompt, rubric, decoding,
and joint-pass threshold were unchanged. No palindrome candidates were
rescored.

| Group | Joint passes | Mean grammaticality | Mean meaning |
| --- | ---: | ---: | ---: |
| Individual Brown sentences | 4/12 | 1.50 | 1.08 |
| Matched shuffles | 0/12 | 0.25 | 0.00 |

The same 8/12 and 3/12 calibration criterion was not met. The model assigned
zero joint passes to the shuffles and passed four source sentences, but the
follow-up changes both source content and presentation units. It is a
post-hoc sensitivity analysis only; it cannot revise the original gate or
validate any candidate score. Brown sentence boundaries also do not guarantee
standalone coherence or human-validated positive examples.

## What these scores support

The verifier confirms that all 52 unique candidates satisfy the paper's
letter-level palindrome definition; none reverses the word order. The local
model's all-zero candidate scores are available for inspection, but the failed
control criterion makes them inconclusive. They are not evidence that 52
readability judgments were correct, and they are not human ratings. The honest
paper claim remains that the current automatic diagnostics do not establish
whether people find the constructions grammatical or understandable.

## Reproduction and source limits

The frozen first-run prompt, code, presentation manifest, raw structured model
responses, and analysis are in
`runs/local-llm-readability-audit-20260926.json`. Re-run with:

```sh
python3 experiments/local_llm_readability_audit_20260926.py --prepare-only
python3 experiments/local_llm_readability_audit_20260926.py --resume
```

The post-hoc follow-up protocol and runner are
`experiments/LOCAL-LLM-SINGLE-SENTENCE-SENSITIVITY-20260926.md` and
`experiments/local_llm_single_sentence_brown_sensitivity_20260926.py`; its
result is `runs/local-llm-single-sentence-brown-sensitivity-20260926.json`.
The separate user-punctuation-variant request and raw response are recorded
in `paper/local-llm-user-example-score-20260926.json` (SHA-256
`849dd1424f57b095638cea3801990addc1328d4baf0d9bc6f9c54abde2044706`); that
single request is excluded from the frozen 244-presentation totals.
Brown source sentences are not copied into result artifacts. Results retain
document IDs, sentence indices, hashes, model metadata, and all scores so a
licensed local Brown corpus can reconstruct the selected controls.
