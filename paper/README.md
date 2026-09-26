# Methods of Finding Longer, Readable Palindromes

The current **manuscript version is v1**: read the
[anonymous PDF](../output/pdf/methods-of-finding-longer-readable-palindromes-v1.pdf)
or its [TeX source](naacl2027-v1.tex). This manuscript reports the existing
v4 study and evidence archive; v1 names the paper edition and does not rename
the underlying runs or data. The paper is an empirical evaluation case
study: exact outputs can score above their own word shuffles under Brown and
GPT-2 while receiving floor-level grammar and meaning ratings from Qwen3.
Construction yield, output variety, and the existing length analysis explain
the conditions under which these signals arise.

The [simulated ARR-style review](mock-reviews/naacl2027-v1-arr-review.md)
assesses this v1 manuscript. It is an independent mock review, not a real
NAACL or ARR decision.

The September 26 framing revision uses only saved evidence. No new search
experiments or model scoring calls were run. Main content ends on page 4;
the PDF has five pages including limitations, references, and the separate
Diana construction appendix. [FRAMING-REVISION-20260926.md](FRAMING-REVISION-20260926.md)
records the changes and claim-to-source map.

The final editorial revision resolves the last review's four requests:
contribution positioning, rubric anchors and control limits, seeded search
mechanics, and descriptive length wording. The supplementary report clearly
distinguishes the raw GPT-2 likelihood contrast from shuffle-relative gains.
The manuscript and professor packet were finalized before the v1 relabeling;
this versioning pass runs no new experiment or model-scoring call.

[NAACL-READINESS-PLAN.md](NAACL-READINESS-PLAN.md) records the claim boundary,
QA status, and readiness assessment. [READABILITY-SCORE-AUDIT.md](READABILITY-SCORE-AUDIT.md)
documents the separate Brown audit and its exact rerun. The complete v4
experiment summary and commands are in
[`DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md`](../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md).
The frozen inputs, raw outputs, source files and checksums are bundled in
[v4-reproducibility.zip](v4-reproducibility.zip), built with
[`build_v4_supplement.py`](build_v4_supplement.py).
The dated professor packet, including the current anonymous PDF, reviewer brief,
diagnostic results, readiness plan, and a separate first-pass rating packet, is
[`review-handoff/NAACL2027-PROFESSOR-HANDOFF-2026-09-26.zip`](review-handoff/NAACL2027-PROFESSOR-HANDOFF-2026-09-26.zip).
The full evidence archive is kept separate because it contains the sealed
mapping from review IDs to method and task provenance.
The AWS Qwen3 run, frozen scorer, price snapshot, offline reanalysis tool, and
result report are included in `v4-reproducibility.zip`; see
[`AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md`](../experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md).
The length-by-construction cross-tab and Diana-frame case analysis are in
[`READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md`](../experiments/READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md);
its standard-library analyzer is included in the reproducibility archive.

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
The hosted Qwen3 run's saved responses can be re-aggregated offline; fresh
Bedrock calls need credentials and may differ because the API has no seed.
Brown and GPT-2 scores can be recomputed with the commands in the experiment
report and their pinned dependencies. DeepSeek's request,
visible response, usage and hash are archived. Bedrock did not provide a model
seed; reissuing an API request may return a different response. The reported
Bedrock charge is estimated at USD 0.00976206 from recorded tokens and
[published Bedrock rates](https://aws.amazon.com/bedrock/pricing/).

The v4 bundle records the full task denominator, source hashes, model
revisions, exactness checks, output screens, language diagnostics and prior
protocol history. It does not contain human ratings, and no automatic metric
is treated as a human readability score. Local Qwen2.5 failed its Brown
control gate; hosted Qwen3 passed a small control screen, but its candidates
and shuffles both had zero joint passes and its controls include seven
multi-sentence spans.

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
