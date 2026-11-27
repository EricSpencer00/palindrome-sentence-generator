# Directional search benchmark v3: results

## Scope

This was a preregistered operational comparison of the empty-center project search and an outside-in residual-search adaptation. The algorithms used their native closure rules. It was not a causal test of search direction. The v1 and v2 empty-residual pilots informed the protocol and are excluded from the confirmatory denominator.

The frozen manifest SHA-256 is `9529a7db32ac47df281167d2a5b804b46902fd328d10075895ddb3005f756257`; the confirmatory result SHA-256 is `e33ea6b6f3473a96bdeb63b2c86b7f33497fd4bf92a4a12d1f1186ec11b989c6`. The run contains 90 paired tasks and 180 requested method runs.

## Confirmatory results

| Configured algorithm | Tasks | Eligible closures | Basic-valid outputs | Strict-pass outputs / selected | Median seconds |
|---|---:|---:|---:|---:|---:|
| `outside_in_norvig_hoey_adaptation` | 90 | 0 tasks / 0 candidates | 0 | 0 / 0 | 0.0101 |
| `center_out_project` | 90 | 0 tasks / 0 candidates | 0 | 0 / 0 | 0.0098 |

| Length band (normalized letters) | Outside-in: valid / tasks | Center-out: valid / tasks |
|---|---:|---:|
| 30–49 | 0 / 30 | 0 / 30 |
| 50–79 | 0 / 30 | 0 / 30 |
| 80–119 | 0 / 30 | 0 / 30 |

Both arms reached zero eligible closures and produced 0/90 basic-valid outputs. There were no timeouts, no outputs to deduplicate, and no selected outputs to evaluate under the strict gate. The paired output-coverage difference (outside-in minus center-out) was 0.000. The seed-cluster bootstrap interval is `[0.0, 0.0]`; because every observed outcome was zero, this interval is degenerate and is not evidence of equivalence or zero underlying success probability.

Brown, GPT-2, and DeepSeek V3.2 scoring was not run on this benchmark: there were no selected outputs, and therefore no jointly successful pairs to send to Bedrock. This experiment used no AWS inference calls. These results say nothing about the language quality of outputs from either algorithm.

## Reproduction and artifacts

The exact vocabulary, jobs, settings, source hashes, citations, and excluded-pilot hashes are in `manifest.json`. The NUC run artifact preserves one row for each requested method task, including failures and search time. Run validation and summarization with:

```sh
python3 experiments/paired_directional_search_benchmark_20260925.py run --manifest runs/directional-search-benchmark-20260925/manifest.json --phase confirmatory --output /tmp/replayed-confirmatory.json --workers 6
python3 experiments/analyze_paired_directional_search_benchmark_20260925.py --manifest runs/directional-search-benchmark-20260925/manifest.json --results runs/directional-search-benchmark-20260925/confirmatory-v3-native-closure.json --json-output /tmp/summary.json --markdown-output /tmp/summary.md
```

The search command requires the pinned `wordfreq==3.1.1` dependency. Use the source, Python version, and host recorded in the result artifact; the experiment source hashes are checked before jobs start.
