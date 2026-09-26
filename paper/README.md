# Exact palindrome search: paper and evidence

The active anonymous manuscript is [naacl2027.tex](naacl2027.tex). It reports
the adaptive v4 comparison of two exact-palindrome search implementations
under original and escalated resource bundles. The earlier eleven-text Brown
audit is retrospective supporting evidence; it does not establish readability.

[NAACL-READINESS-PLAN.md](NAACL-READINESS-PLAN.md) records the claim boundary,
QA status, and readiness assessment. [READABILITY-SCORE-AUDIT.md](READABILITY-SCORE-AUDIT.md)
documents the separate Brown audit and its exact rerun. The complete v4
experiment summary and commands are in
[`DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md`](../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md).
The frozen inputs, raw outputs, source files and checksums are bundled in
[v4-reproducibility.zip](v4-reproducibility.zip), built with
[`build_v4_supplement.py`](build_v4_supplement.py).
The dated professor packet, including a rebuilt anonymous PDF, reviewer brief,
local-judge results, readiness plan, and a copy of the evidence archive, is
[`review-handoff/NAACL2027-PROFESSOR-HANDOFF-2026-09-26.zip`](review-handoff/NAACL2027-PROFESSOR-HANDOFF-2026-09-26.zip).

## Reproduce v4

From the repository root, run the frozen search and verify its 240 outputs:

```sh
python3 experiments/paired_directional_search_resource_sensitivity_20260925.py run \
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \
  --phase confirmatory --output /tmp/replayed-v4.json --workers 6
python3 experiments/analyze_directional_search_resource_sensitivity_20260925.py \
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \
  --results /tmp/replayed-v4.json \
  --static-verifier-summary runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json \
  --json-output /tmp/v4-summary.json --markdown-output /tmp/v4-summary.md
```

The frozen input contains all 66 selected texts and 32 deterministic same-word
shuffles for each. Brown regenerates its own 32 item-keyed permutations;
GPT-2 scores the archived shuffles, so the two diagnostics do not share the
same word orders. Both sets and their seed derivations are reproducible. The
local Qwen judge audit, including its failed control calibration and raw
structured responses, is documented in
[`LOCAL-LLM-READABILITY-AUDIT-RESULTS-20260926.md`](../experiments/LOCAL-LLM-READABILITY-AUDIT-RESULTS-20260926.md).
Brown and GPT-2 scores can be recomputed with the commands in the experiment
report and their pinned dependencies. DeepSeek's request,
visible response, usage and hash are archived. Bedrock did not provide a model
seed; reissuing an API request may return a different response. The reported
Bedrock charge is estimated at USD 0.00976206 from recorded tokens and
[published Bedrock rates](https://aws.amazon.com/bedrock/pricing/).

The v4 bundle records the full task denominator, source hashes, model
revisions, exactness checks, output screens, language diagnostics and prior
protocol history. It does not contain human ratings, and no automatic metric
is treated as a human readability score. The Qwen candidate scores are
inconclusive because the local judge did not meet the frozen Brown-control
calibration criterion.

## Build the anonymous review PDF

Build to a fresh temporary directory so existing PDFs and logs are preserved:

```sh
NAACL_BUILD_DIR="$(mktemp -d)"
tectonic --keep-logs -o "$NAACL_BUILD_DIR" paper/naacl2027.tex
```

The [NAACL 2027 call](https://2027.naacl.org/calls/main_conference_papers/)
sets the short-paper limit at four pages. Inspect the PDF and all rendered
pages before submission. The current worktree may contain unrelated PDF build
artifacts; do not use them as the manuscript output.
