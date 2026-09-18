# Controlled POS-pruning experiment

## Design

- 50 paired fixed opening subtrees, selected without POS tags or outcome data; ordering seeds 0--49.
- At most 50,000 popped states per arm and opening; a fully pruned or exhausted subtree stops earlier.
- Full frozen vocabulary intersection (21,073 admitted of 30,000 requested entries), Brown tag table, and the manuscript's 20--44-letter, 18-word, 16-overhang limits.
- Stable SHA-256 word ranks make expansion order a pure function of seed and word. Shared states receive identical sibling order in both arms.
- Arms use the same completed-candidate checks. Only the incremental arm applies POS feasibility before pushing a state.
- CPU time is process time. Peak RSS is measured in a fresh process per arm. Trials ran concurrently, so summed wall time is not an elapsed experiment duration.

## Results

Values below are means per paired trial unless marked as rates.

| Metric | Terminal POS | Incremental POS |
|---|---:|---:|
| States generated | 59,607.7 | 328,399.0 |
| States pushed | 59,607.7 | 54,897.5 |
| States popped | 50,000.0 | 48,000.6 |
| Eligible terminal closures | 688.1 | 368.2 |
| Distinct accepted outputs | 8.14 | 42.74 |
| CPU seconds | 12.650 | 15.220 |
| Accepted / million generated | 136.56 | 130.15 |
| Accepted / million popped | 162.80 | 890.40 |
| Accepted / CPU second | 0.643 | 2.808 |
| Peak RSS, MiB | 121.26 | 120.43 |
| Peak frontier states | 16,581.7 | 12,588.4 |

The incremental gate rejected 83.28% of otherwise generated states before insertion. Across all trials, the incremental/terminal rate ratios were:

| Rate ratio | Estimate | Paired bootstrap 95% interval |
|---|---:|---:|
| Accepted / generated state | 0.953x | [0.464, 3.199] |
| Accepted / popped state | 5.469x | [2.892, 17.653] |
| Accepted / CPU second | 4.364x | [2.298, 13.925] |

Incremental filtering returned more accepted outputs in 26 paired trials, tied in 24, and returned fewer in 0. The terminal and incremental arms produced at least one accepted output in 12 and 28 of 50 trials, respectively.

## Interpretation limits

The paired budget removes the earlier wall-time, output-cap, arm-order, and traversal-dependent-randomness confounds. It does not make every bounded walk exhaustive. The intervals measure variation across the fixed opening subtrees and their deterministic orderings over one frozen vocabulary; they are not corpus-level or hardware-population intervals. POS-shape admission is structural and is not evidence of grammaticality, meaning, or readability.
