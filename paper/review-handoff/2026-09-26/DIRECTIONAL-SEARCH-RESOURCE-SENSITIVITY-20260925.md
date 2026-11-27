# Resource-sensitivity study v4: results

## Why this follow-up exists

The preregistered v3 comparison produced no eligible closures: 0/90 tasks in each arm. Toy-vocabulary controls then verified that both search APIs can reach and admit a known target. v4 was frozen on fresh seeds to test whether a larger resource bundle changes coverage. It is an adaptive follow-up to v3, not an independent confirmation of a pre-v3 hypothesis.

The v4 manifest hash is `f97b76d4094bd5f767a11c3f304ae831edc6ac66df30de5690ddd6b9f8d026cc` and the confirmatory result hash is `b7f5c9082030903b3e3c29f2ad6e5de3d5fe44da57186e5fecc5ffdc68e76070`. The frozen manifest retains hashes for v1–v3 results. This confirmatory phase contains 240 method/configuration runs across 60 seed-by-length blocks.

## Primary result

| Resource configuration | Tasks | Basic-valid outputs | Coverage | Eligible closure tasks | Deadline reached | Median seconds / task |
|---|---:|---:|---:|---:|---:|---:|
| `original_v3` | 120 | 0 | 0.000 | 0 | 0 | 0.014 |
| `escalated_v4` | 120 | 66 | 0.550 | 66 | 0 | 8.954 |

The paired coverage difference (escalated minus original) was 0.550, with a descriptive paired seed-cluster bootstrap 95% interval of `[0.350, 0.750]`.

## Method-level coverage

| Resource configuration | Algorithm | Tasks | Basic-valid | Eligible closure tasks | Strict-pass / selected | Unique outputs | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| `original_v3` | `outside_in_norvig_hoey_adaptation` | 60 | 0 | 0 | 0 / 0 | 0 | 0.014 |
| `original_v3` | `center_out_project` | 60 | 0 | 0 | 0 / 0 | 0 | 0.014 |
| `escalated_v4` | `outside_in_norvig_hoey_adaptation` | 60 | 33 | 33 | 28 / 33 | 26 | 8.928 |
| `escalated_v4` | `center_out_project` | 60 | 33 | 33 | 26 / 33 | 26 | 9.110 |

The resource settings change beam width, candidate limit, children per parent, and cooperative time budget together. The estimate therefore compares two resource bundles, not the isolated effect of runtime. The two algorithms also retain different native closure rules, so their contrast is descriptive rather than a causal effect of growth direction.
Among the 66 selected escalated outputs, 54 passed every additional mechanical screen; this is a selected-output pass rate, not strict search coverage. The outside-in and center-out methods produced 26 and 26 unique texts, respectively. The exact phrase `one poem association` occurs in 36 of 66 outputs, illustrating that exactness and lexical screens do not ensure coherent prose.

## Language diagnostics

We completed the audit for 66 escalated_v4 outputs; no original_v3 outputs existed. Brown order gain, pinned GPT-2 token likelihood, and one counterbalanced DeepSeek V3.2 comparison against a deterministic same-word shuffle are only computed for selected basic-valid outputs. Brown regenerates its 32 item-keyed permutations with its scorer RNG; GPT-2 scores the 32 archived permutations generated from task ID and replicate. They preserve the same words and sentence lengths but are different control sets. No model diagnostic is a human rating.

| Diagnostic, computed only among selected valid outputs | Outside-in (n=33) | Center-out (n=33) |
|---|---:|---:|
| Brown order gain vs 32 same-word shuffles, mean (positive count) | 0.303 (25/33) | 0.124 (15/33) |
| GPT-2 candidate minus 32-shuffle mean token log probability (positive count) | 0.068 (23/33) | -0.003 (14/33) |
| DeepSeek V3.2 choice: candidate / shuffle / tie | 0 / 0 / 33 | 0 / 0 / 33 |

For the 33 seed-by-band tasks where both methods returned an output, the descriptive paired outside-in-minus-center-out differences were 0.178 Brown order-gain nats/transition (seed-cluster 95% interval `[0.041, 0.317]`) and 0.280 GPT-2 raw candidate mean log-probability nats/token (interval `[0.205, 0.353]`). The GPT-2 paired contrast uses raw candidate likelihood, whereas the preceding table and length-stratified report use candidate-minus-shuffle gains. These endpoints differ. These are post-run, conditional model diagnostics, not human judgments or population-level method effects. DeepSeek tied all 66 separate candidate-versus-shuffle pairs; it did not directly compare the two algorithms.
The DeepSeek run used 11,538 input and 1,410 output tokens. Its charge was estimated as USD 0.00976 from recorded token counts and the published [Amazon Bedrock DeepSeek V3.2 rates](https://aws.amazon.com/bedrock/pricing/); the preflight upper bound was USD 0.06293.
The static verifier summary SHA-256 is `b9efa930ec8ff42c7b05175cb0fdda454028f47be281efdb560a1ae7713a5ed4`; its frozen input SHA-256 is `9229cdfdf761dace9824686bd2f9ed285b30a12c798045995d6fa834e3130bf0`.

## Separate local absolute-rating diagnostic

The reproducible Qwen2.5-3B local audit and its calibration failures are
reported in [`LOCAL-LLM-READABILITY-AUDIT-RESULTS-20260926.md`](LOCAL-LLM-READABILITY-AUDIT-RESULTS-20260926.md), with raw responses and run
provenance in `runs/local-llm-readability-audit-20260926.json`. The 52 unique
candidate strings all received 0/3 for grammar and meaning, but only 1/12
original Brown spans and 4/12 in a separate post-hoc individual-sentence check
met the joint threshold. Both checks failed the prespecified 8/12 control
criterion. The score is therefore inconclusive as a readability measure; it
does not turn the all-zero candidate outputs into a valid human-quality result.

## Reproduction

The manifest freezes the vocabulary, 240 jobs, resource settings, arm order, source hashes, model methods, and v1–v3 history. The raw run retains all requested rows and failures. The saved verifier input contains every selected text and deterministic same-word shuffle; model scores can be rerun against it on an authorized CPU host. Reproduce and validate with:

```sh
python3 experiments/paired_directional_search_resource_sensitivity_20260925.py run --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json --phase confirmatory --output /tmp/replayed-v4.json --workers 6
NLTK_DATA=/path/to/nltk_data python3 -m experiments.score_directional_search_brown_20260925 --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json --output /tmp/brown-v4.json
HF_HOME=/path/to/hf-cache OMP_NUM_THREADS=4 python3 -m experiments.score_gpt2_static_verifier_aws_20260925 --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json --output /tmp/gpt2-v4.json --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --batch-size 8 --execution-host 'authorized CPU host'
python3 experiments/infer_deepseek_static_verifier_20260925.py --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json --output /tmp/deepseek-v4.json --budget-usd 0.20 --interval-seconds 0.75
python3 experiments/summarize_directional_static_verifier_20260925.py --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json --brown /tmp/brown-v4.json --gpt2 /tmp/gpt2-v4.json --deepseek /tmp/deepseek-v4.json --output /tmp/static-verifier-summary-v4.json
python3 experiments/analyze_directional_search_resource_sensitivity_20260925.py --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json --results runs/directional-search-resource-sensitivity-20260925/confirmatory-v4-resource-sensitivity.json --static-verifier-summary /tmp/static-verifier-summary-v4.json --json-output /tmp/v4-summary.json --markdown-output /tmp/v4-summary.md
```
