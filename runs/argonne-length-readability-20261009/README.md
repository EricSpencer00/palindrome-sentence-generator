# Argonne length and language-quality screen — October 9, 2026

**The existing search makes exact text, but this run did not make readable
long prose.** Polaris job `7731461` finished successfully under `EVITA`, using
one node for 99 seconds. Queue wait was 9 minutes 21 seconds. The 72-search
grid produced 43 selected exact outputs, representing 33 distinct letter tapes.
The longest observed eligible closure was 447 letters. All returned tapes and
frozen source hashes were independently checked after retrieval.

## Length map

Each band has twelve attempts: two directions, two inventories and three seeds.
The native search selects by its existing objective; the longest observed
closure is saved separately. Counts include repeated outputs across jobs.

| Target letters | Selected exact outputs | Longest selected | Longest observed | Selected texts with machine grammar and meaning ≥2 |
|---|---:|---:|---:|---:|
| 30–79 | 12/12 | 39 | 79 | 0/12 |
| 80–159 | 12/12 | 145 | 159 | 0/12 |
| 160–319 | 12/12 | 227 | 319 | 0/12 |
| 320–639 | 7/12 | 362 | 447 | 0/7 |
| 640–999 | 0/12 | — | — | No outputs |
| 1,000–1,499 | 0/12 | — | — | No outputs |

The high-band failures mean **no eligible closure within this search budget**,
not that longer English palindromes are impossible. The repository's older
568–752-letter constructions remain separate, longer exact examples.

## Phrase comparison

| Direction and inventory | 30–79 | 80–159 | 160–319 | 320–639 | 640–999 | 1,000–1,499 |
|---|---:|---:|---:|---:|---:|---:|
| Outside-in, words | 3/3 | 3/3 | 3/3 | 1/3 | 0/3 | 0/3 |
| Outside-in, words + phrases | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 |
| Center-out, words | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| Center-out, words + phrases | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 |

At 320–639 letters, the phrase arms closed 6/6 jobs and the word-only arms 1/6.
Both remained at the low language-quality level. This is a small exploratory
comparison, with three seeds per cell and different closure semantics between
directions; it does not establish a general method ranking.

## Whole-text machine screen

A separate model-agent evaluator saw opaque IDs and complete texts, with
source, method and palindrome labels hidden. It used the same 0–3 grammar and
meaning rubric throughout, with a main-event paraphrase required at meaning 2
or 3. Its configured model was `gpt-6.1-sol` at `xhigh`. This evaluation was
performed separately from the Polaris CPU job; it is **not human validation**.

The frozen development calibration contained nine intact public prose spans
and nine same-word shuffles, covering short, middle and long texts. Eight of
nine intact passages and zero of nine shuffled passages scored at least 2 on
both dimensions, passing the specified narrow screen. This does not validate
the evaluator generally or establish agreement with readers.

All **67 distinct selected/longest surfaces** were then rated once:

- 63 scored grammar 1, meaning 1.
- Four scored grammar 0, meaning 1.
- None scored at least 2 on both dimensions.

In the separate archived-example screen, the 38-letter inherited reference
scored 3/3, and the 54-letter candidate scored 2/2. Every selected 498–752-letter
example scored meaning 1; grammar was 1 or 2. Those scores are preliminary
machine feedback, not new reader ratings, original authorship evidence or a
join to Eric's 183-item export.

A small presentation probe attempted punctuation, case and spacing changes
on the shortest and longest observed text in each successful band. All eight
letter tapes were preserved. Seven originals were retained; one gained a local
question, but the complete passage remained at grammar 1 and meaning 1. Only
that changed rendering was freshly scored. This limited probe cannot rule out
better presentation or segmentation by another method.

## Diagnostic scope

The Polaris job also scored 298 cohort records: all 72 search outcomes,
eleven archived exemplars, 189 credited Reddit palindromes and 26 intact public
prose controls. It used a fixed Norvig bigram table and 32 deterministic
same-word shuffle comparisons per text, preserving sentence token counts.
The bigram table also informs the search objective, so its local-order values
are descriptive features, not an independent grammar or meaning judge.

The 26 prose controls reproduce their frozen September 25 source-text hashes.
They comprise three length-selected controls per band plus all eleven matched
incumbent controls, deduplicated by ID. Their Brown document holdout does not
establish disjointness from Norvig's source corpus.

All 43 selected outputs are absent from the exact-tape lookups against the
189 Reddit references and local catalogue. That limited lookup does not
establish global novelty, construction-family diversity or readable meaning.
Historical strict admission was separately checked for 39 selected outputs;
none passed. Four outputs exceeded the 128-word diagnostic limit and explicitly
retain unrun strict status. These checks do not redefine the current request's
readability goal or exclude ordinary phrases and entity reuse automatically.

## Decision

Keep phrases, flexible boundaries and overhangs in the construction program.
The next improvement must establish connected meaning while solving both
readings jointly. More vocabulary, longer budgets or larger beams have not
earned promotion from this result.

The first long milestone is **1,000+ letters with reader scores of 3/3 for
grammar and whole-text meaning**, followed by further growth while preserving
those qualities. The full paragraph must be palindromic; individual sentences
need not be. This goal remains unachieved. The
[implementation and evaluation plan](../../docs/research/2026-10-09-argonne-evaluation-plan.md)
describes the next mechanisms and reader requirements.

## Reproduction and evidence

- [Execution and verification record](execution.json)
- [Scheduler record](scheduler-record.txt) and [job log](pbs-output.log)
- [Frozen search manifest](results/manifest.json), [resources](results/resources.json), and [all 72 outcomes](results/results.json)
- [Per-job progress and artifacts](results/jobs)
- [All diagnostic rows and input hashes](results/diagnostics.json)
- [Public prose controls and provenance](controls.json)
- [Calibration summary](calibration-summary.json), [blinded packet](blind-calibration.json), [labels](calibration-key.json), and [agent ratings](calibration-agent-ratings.json)
- [Joined baseline machine screen](baseline-machine-screen.json) and [joined new-output screen](new-machine-screen.json)
- [Blinded new-output packet](blind-new.json), [source key](new-key.json), and [raw machine ratings](new-agent-ratings.json)
- [Presentation probe](presentation-summary.json)

The runner uses 30,000 frozen words, 2,000 attested phrase units, beam 48,
candidate menu 200, parent quota 8, overhang cap 24, word cap 1,024 and step cap
1,500. Each search has a 20-second cooperative deadline and a 25-second hard
process deadline. Word and letter caps correctly flatten phrase constituents.
The phrase-aware repetition correction is applied in every arm. Native step
exhaustion and overhang pruning are not separately instrumented.

Submit [the PBS job](../../tools/polaris/length_readability_quick.pbs) from a
fresh staged checkout on Polaris. It runs the
[bounded search driver](../../tools/polaris/length_readability_quick.py), then
[the diagnostic evaluator](../../experiments/quick_length_readability_diagnostics_20261009.py).
Neither performs model training or claims a new generator. The original run
used `/usr/bin/python3.11` on its assigned compute node.

Results are preserved here. All task-created remote scratch and archives were
removed after retrieval and verification. No job from this task remains
running; the unrelated pre-existing held job was preserved.
