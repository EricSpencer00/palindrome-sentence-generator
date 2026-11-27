# Baseline and historical evidence audit — public v1

## Standard and scope

The September 26 audit asks whether the saved runs establish a meaningful
comparison before calling any change an improvement. A comparison must name
its comparator, inputs, output and selection rules, resource budget, and
measured outcome. Its provenance must support the claimed replayability.
Reproducibility alone does not make a task grid a benchmark.

Three Luna subagents audited search comparisons, construction history, and
language controls. They inspected primary manifests, saved outputs, source,
and relevant prior work. Offline checks recomputed selected counts, exactness,
and hashes. No generation, model scoring, paid calls, or remote jobs ran.

## Findings and disposition

| Evidence | What the records establish | Public v1 decision |
| --- | --- | --- |
| Directional resource study | 240 runs from a project-selected grid. One configuration produced 66 records containing 52 distinct strings; the other produced none. Four resource settings changed together after earlier null runs. | Remove the coverage-gain headline, paired improvement interval, and length/strategy ranking table. Keep the grid and settings as sample provenance. No external or established “30/60” benchmark was found. |
| POS pruning, 50 paired openings | Raw totals reproduce: 407 terminal-only versus 2,137 incremental accepted word pairs; 12/50 versus 28/50 openings yielded any acceptance. The shared output gate and 50,000-popped-state cap define a bounded comparison. | Retain as historical internal ablation evidence, outside the manuscript's result set. Accepted per popped state gives 5.469×, but accepted per generated state gives 0.953×. Provenance says the historical source was not frozen; do not claim full source replayability or general efficiency. |
| Later traversal checks | DFS ascending: 295/877 terminal/incremental acceptances; descending: 91/470. BFS: 0/0 and 0/1. DFS per-generated ratios are 0.49× and 0.99×. | These records rule out a traversal-independent pruning-gain claim. A useful historical contrast survives, with an explicit outcome and budget; it does not establish readability. |
| Diana reference and extensions | Independent normalization confirms exactness and saved hashes for 38-, 54-, 568-, and 666-letter surfaces. The 38-letter text is inherited; its authorship is not established here. The 54-letter reader packet is unrun. | Correct the unsupported claim that 38 letters is an established human-readability benchmark. No primary human ratings, participant count, or administered protocol were located for it. Longer selected descendants remain exact construction examples, with no quality-improvement claim. |
| Brown and GPT-2 word permutations | Candidate mean log probability minus the mean over 32 own-word permutations is positive for 40/66 and 37/66 records, respectively. Brown uses nats per word transition; GPT-2 uses log probability per predicted token. Permutations and units differ. | Retain a formally defined word-order diagnostic. These controls supply neither a competing generator nor a human readability threshold. Counts include repeated strings. |
| Qwen3 rubric | All 52 distinct candidates score zero on grammar and meaning. Joint passes are 11/12 ordinary prose controls and 0/12 prose shuffles under chosen screen thresholds. | Retain this finite-set model observation. The control screen has no human calibration and does not rank palindrome quality. Qwen2.5 fails its screen. |
| Older mirror-cost records | 36 saved model/segmentation cells report 2.14–3.34 bits per normalized letter for a defined forward/reversed-prose contrast. The audit verifies saved text and arithmetic, not fresh logits. | Historical model-specific diagnostic only. No universal reversal law, generator baseline, or readability conclusion. |
| Published generators and constrained-sentence methods | Primary sources describe relevant methods, but no matched reproduction is present in the evaluated records. | Related work only. No superiority, state-of-the-art, or cross-paper length comparison. |

## Changes and acceptance criteria

- The manuscript defines each retained control and its denominator, units,
  selection conditions, and interpretation. Arbitrary search bands and limits
  are described as collection choices.
- The root README no longer turns historical per-letter score gains into
  fluency gains, pruning or proxy results into universal conclusions, or the
  mirror-cost diagnostic into a universal construction law.
- Current repository summaries no longer assert a human-validated 38-letter
  baseline. Historical logs are retained with a current correction at the top.
- The paper, source, supplement, and reviewer handoff remain public **v1**.
  Raw experiment identifiers and records are preserved.
- Historical reports and the earlier mock review are identified as historical
  snapshots. Their recorded outcomes and earlier judgments are not rewritten.
- The regenerated PDF and archives must match the current source, preserve
  all review items, and pass their build-time hash and privacy checks.

This audit strengthens the evidence boundary. It does not establish a 4/5
NAACL contribution or add human validation or a matched generator baseline.

## Detailed reports and ownership

| Luna lane | Report | Status |
| --- | --- | --- |
| Search baselines | [Historical search comparisons](audits/historical-search-baselines-v1.md) | Complete; raw counts and provenance checked |
| Construction history | [Construction evidence](audits/historical-construction-evidence-v1.md) | Complete; four surfaces independently checked |
| Metric controls | [Metrics and control definitions](audits/historical-metric-controls-v1.md) | Complete; units, totals, source definitions, and primary references checked |

The reports point to repository history beyond the main study's supplement;
including a report does not imply every historical run it cites is bundled.
The coordinator integrates the edits and builds the release artifacts.
No new worktrees, servers, remote jobs, or inference processes were created.
Temporary PDF renders and handoff staging are kept outside the repository
after inspection. Unrelated existing build artifacts are preserved.
