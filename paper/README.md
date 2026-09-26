# Methods of Finding Longer, Readable Palindromes

The first **public release is v1**: read the [anonymous PDF](../output/pdf/methods-of-finding-longer-readable-palindromes-v1.pdf) or its [TeX source](naacl2027-v1.tex). Paper, supplement, source, and distribution packages use v1 labels. Raw experiment identifiers in saved records remain historical identifiers; they are not public release versions.

## What this study reports

The paper describes a collected sample of 66 selected output records from 240 search runs, containing 52 distinct strings. It reports construction records and automatic diagnostics; it does not claim a generation improvement, a length effect, or superiority over a published generator. No human ratings were collected, and no cited external generator was run under matched conditions.

Brown and GPT-2 compare each output with 32 permutations of its own words. Brown reports candidate-minus-shuffle mean add-0.1 word-bigram log probability in nats per word transition. GPT-2 reports candidate-minus-shuffle mean causal log probability per predicted token at a pinned model revision. The shuffle implementations differ, as do their scoring units. Positive contrasts describe model-specific sensitivity to word order; they are not absolute readability scores or generator comparisons.

Qwen3 rates grammaticality and coherent meaning on a 0–3 rubric. It gives zero joint passes among 52 candidates; 11/12 held-out Brown prose controls and 0/12 of their shuffles pass jointly. The frozen 8/12 prose and 3/12 shuffle thresholds are chosen diagnostic-screen cutoffs, not externally validated calibration criteria. The separate Qwen2.5 audit fails its control screen. Neither audit is human evaluation.

See the [baseline provenance audit](BASELINE-PROVENANCE-AUDIT-V1.md) for formal distinctions among shuffle controls, calibration checks, generation baselines, and external benchmarks. The [simulated ARR-style review](mock-reviews/naacl2027-v1-arr-review.md) is an internal mock review, not a real NAACL or ARR decision.

## Evidence and reproduction

The [readiness plan](NAACL-READINESS-PLAN.md) records the claim boundary and evidence status. [READABILITY-SCORE-AUDIT.md](READABILITY-SCORE-AUDIT.md) documents the Brown scorer and rerun. The experiment report, inputs, output records, protocols, and checksums are in [DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md](../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md) and [v1-reproducibility.zip](v1-reproducibility.zip), built with [build_v1_supplement.py](build_v1_supplement.py). The [Qwen3 audit report](../experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md) explains the saved rubric responses and offline reanalysis. The separate Diana construction analysis is documented in [READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md](../experiments/READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md); it is retrospective and outside the 52-candidate Qwen audit.

The v1 professor handoff is [NAACL2027-PROFESSOR-HANDOFF-v1.zip](review-handoff/NAACL2027-PROFESSOR-HANDOFF-v1.zip). The full evidence archive is separate because it includes the sealed mapping from review IDs to method and task provenance.

From the repository root, replay the frozen search and summarize its 240 outcomes:

```sh
python3 experiments/paired_directional_search_resource_sensitivity_20260925.py run \
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \
  --phase confirmatory --output /tmp/replayed-v1.json --workers 6
python3 experiments/analyze_directional_search_resource_sensitivity_20260925.py \
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \
  --results /tmp/replayed-v1.json \
  --static-verifier-summary runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json \
  --json-output /tmp/v1-summary.json --markdown-output /tmp/v1-summary.md
```

The archived Brown and GPT-2 scores can be recomputed with the commands and pinned dependencies in the experiment report. Brown regenerates its own deterministic item-keyed permutations; GPT-2 scores the archived permutations, so their controls do not share the same word orders. Saved Qwen3 responses can be re-aggregated offline; fresh Bedrock calls may differ because the service exposes no provider seed. The report's recorded Bedrock usage estimate is USD 0.00976206, derived from recorded tokens and [published rates](https://aws.amazon.com/bedrock/pricing/).

The release contains frozen manifests, saved outcomes, selected texts and controls, scorer sources, model revisions, prompts, and responses. Brown corpus text is omitted under its terms. No automatic metric is treated as a human readability score.
