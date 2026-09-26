# Post-hoc local-judge calibration sensitivity

**Status:** protocol frozen before this follow-up inference. This is a
post-hoc sensitivity check, not a replacement for the original audit.

## Why this check exists

The first Qwen 2.5 3B audit scored only 1 of 12 Brown controls as both
grammatical and meaningful at 2/3 or above, so it failed its prespecified
calibration gate. Post-run source inspection also found that 7 of those 12
control spans contained more than one Brown sentence. Five were single
sentences. The span-length issue is a separate protocol-description defect;
it does not explain or erase the failed first gate.

This follow-up asks only whether the same model, rubric, prompt, and decoding
can distinguish 12 previously unused, individual held-out Brown sentences
from deterministic permutations of their own words. It does not rescore the
palindrome candidates. A successful result would establish separation on this
small, newly selected control set only. It would not revise the original gate,
validate the candidate ratings, establish human agreement, or validate this
judge for readability claims.

## Frozen evaluation

- **Model:** the exact local model identified by the first run's saved model
  metadata; the expected model is `qwen2.5:3b`.
- **Prompt, schema, decoding, and rubric:** imported unchanged from
  `local_llm_readability_audit_20260926.py`. The two rated dimensions are
  grammaticality and coherent meaning, each 0–3. A joint pass is at least 2
  on both.
- **Controls:** 12 Brown corpus sentences from previously unused held-out
  documents/spans, paired with one deterministic same-word permutation each.
  Sentences are selected without Qwen scores, never reused, drawn from distinct
  Brown files, and matched by word count to hash-selected palindrome-item
  targets. All presentation text is normalized with the same casing rule.
- **Order and blinding:** the 24 presentations are deterministically shuffled;
  the model sees no item labels, conditions, pairs, or control status.
- **Gate, retained from the original frozen protocol:** at least 8/12
  individual Brown sentences must jointly pass, and at most 3/12 matched
  shuffles may jointly pass. Report all 24 outputs and dimension-level score
  distributions even if the gate fails.
- **Reporting:** descriptive counts and score histograms only. No inferential
  test, human-readability estimate, candidate rerating, or prompt/model
  adjustment based on outcomes.
- **Data handling:** preserve response JSON, model digest, prompt/script/data
  hashes, deterministic IDs, and run fingerprint. Do not write Brown sentence
  text into the result artifact; retain source file IDs, sentence indices, and
  hashes for reproducibility. The Brown corpus source is distributed for
  non-commercial use; keep source text out of public artifacts.
- **Compute:** use the already available local model on the remote Mini through
  its private inference endpoint. No AWS, hosted model, or model download.

## Reproduction

From the repository root:

```sh
python3 -m py_compile experiments/local_llm_single_sentence_brown_sensitivity_20260926.py
python3 experiments/local_llm_single_sentence_brown_sensitivity_20260926.py --prepare-only
python3 experiments/local_llm_single_sentence_brown_sensitivity_20260926.py
```

The final command expects the pre-established private Mini port forward at
`127.0.0.1:11435`. If interrupted, rerun it with `--resume`; the runner checks
that the frozen model, prompt, script, controls, and presentation manifest
match the saved run fingerprint before continuing.

## Interpretation boundary

Call these “individual Brown-sentence controls,” not validated human-positive
examples. Corpus sentence boundaries do not guarantee that a sentence is
grammatical, self-contained, or meaningful in isolation. Do not attribute the
original failure to its multi-sentence spans: five original controls were
already single sentences, and this follow-up changes both source content and
presentation units. Whatever the result, preserve the first audit as a failed
calibration run and describe the follow-up as post-hoc.
