# Paper revision and controlled seam-order study

7 September 2026. Primary source files are in `runs/revision-2026-09-07/`.

The new 15-page research manuscript incorporates the September length and
sentence-search results and withdraws unsupported conclusions in the old draft.
No remote Overleaf update has occurred: both available browser sessions show
Restricted and require an authenticated account with project access.

## Seam construction

The 501 distinct midpoint-splittable source pairs allow two-chunk totals of
38–64 letters and eight-chunk totals of 206–250. There is no common support for
the reviewer's equal-length comparison. We did not pad or repeat chunks to
misrepresent such a comparison.

Instead, twelve sets of eight pairs are held fixed. All 40,320 permutations per
set are scored on both forced nesting boundaries using corpus bigram ordering
gain. The chosen order increases that objective by 0.20–26.19 summed log-score
units. This is an optimized training objective, not a readability result.
All twenty-four outputs are valid palindromes with exactly matched word
multisets, repetition counts, and lengths within a set. The central join can
change and is not part of the optimized boundary objective.

## Frozen evaluation

`protocol.json` and its hash were saved before judgments. The same five local
models face twelve held-out prose/shuffle controls. The gate requires 12/12
bare correct A/B answers; explanations and missing/truncated final answers fail.
The prompt and 512-output-token cap are retained in code/runtime metadata.

| model | exact gate | format/missing failures | experimental attempts |
|---|---:|---:|---:|
| gpt-oss:20b | 12/12 | 0 | 48 |
| mistral:latest | 0/12 | 12 | not evaluated |
| deepseek-r1:8b | 0/12 | 12 | not evaluated |
| gemma3:4b | 12/12 | 0 | 48 |
| llama3.1:8b | 8/12 | 4 | not evaluated |

Every parsed calibration preference was correct. Failure here is therefore
interface compliance/truncation, not demonstrated inability to prefer prose.
The fixed rule was not relaxed after looking at the answers. Failed models were
not given experimental items, as specified in the frozen protocol.

Whole-text scores use the frozen 0–3 rubric:

| model | catalogue | generated single | random nest | optimized nest |
|---|---:|---:|---:|---:|
| gpt-oss:20b | 2.00 (11) | 0.17 (12) | 0.00 (5) | 0.00 (10) |
| gemma3:4b | 1.58 (12) | 0.67 (12) | 0.00 (12) | 0.00 (12) |

Parentheses are valid ratings. Gpt-oss has ten missing final answers, mainly
long-item truncation. Zero is never substituted for missing. It has only five
complete order pairs; Gemma has all twelve. Both observed paired means are zero.
Worst-case bounds allowing missing values anywhere in 0–3 give the full gpt-oss
mean effect [-1.75, +0.50], not a confidence interval.

Ordinal Krippendorff alpha is 0.71 across 38 pairable items and two passing
models. That includes the catalogue/single gradient. Nest-only alpha is undefined
because all observed nest scores are zero. No human alpha is available.

**Finding:** the controls separate above the floor, but both nest arms remain
at the floor. We established a better corpus boundary objective and did not
establish better readability. This is not a per-seam causal estimate and does
not prove order irrelevant in a better source bank.

## Placement sensitivity

Eight requested vocabulary sizes (1,000 to 47,000; actual 926 to 44,232) crossed
with breadth-first and depth-first enumeration produce 792,356 edge observations
in total. Each arm stops at 60,000 or exhaustion. The candidate limit is 400 and
overhang limit 24. Mean signed overhang reduction ranges from -2.3606 to 1.0614;
mean settled positions range from 2.1976 to 3.7511. The old constant is rejected.

These are enumerated edges under the current candidate menu, not draws from
successful trajectories. Along an actual trajectory the signed reductions
sum exactly to initial minus final debt. A positive constant is not implied.

## Verification and remaining gaps

The independent Norvig auditor reproduces 90,937 letters, a 498-letter gain,
16,168 unique dictionary phrases, no adjacent word repeats, and content cap 3.
Thirteen existing length/intersection tests pass. The new packet checker verifies
its hash, all twenty-four palindrome invariants, twelve material matches, twelve
shuffle token inventories, and sixteen measurement identities. The alpha helper
passes perfect, degenerate, published binary-example, and missing-value checks.

No human panel, independent human punctuation, high-budget 42–43-letter sweep,
or readability best-of-N/beam sweep was run. Unsupported claims were deleted.
Blank human-rating and punctuation CSVs are prepared; the rater-facing IDs are
opaque and their key is separate. Historical subagent verdicts are not humans.
