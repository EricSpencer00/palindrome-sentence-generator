# Hierarchical v3: words → paired sentences → paragraph

Date: 2026-09-04

## Decision

Promote structural sentence boundaries and hard anti-cycle constraints to the
v3 composition endpoint. Do not promote embedding similarity as a paragraph
ranker, the broadened LLM sense filter, or the short-beam sentence-pair
harvester. The remaining bottleneck is source material: the current bank has
enough sentence-shaped pairs for a long non-cycling palindrome, but not enough
meaningfully varied pairs for a coherent thematic chain.

## What changed

The old endpoint assembled mirror-pair fragments into one flat word stream and
then guessed punctuation. The default path now admits a pair only when both
halves pass the whole-sentence gate, preserves one structural chunk per
sentence, and nests those sentence-pairs around one centre.

Selection uses hard exclusions rather than a reward:

- no adjacent duplicate words inside a sentence;
- no bigram more than twice in the paragraph;
- no word-length sentence template more than twice;
- no non-connective content word more than three times.

Compressed sentences such as “Items draw award” remain legal. Mechanical
cycles such as “do do do do” cannot enter.

## Capacity and invariants

The generated v3 bank contains 498 non-degenerate mirror-pairs; 306 have both
halves passing the Brown whole-sentence gate. Before anti-cycle selection they
provide 8,888–8,916 letters of sentence-paired capacity, depending on centre.

Across 24 seeds at the largest request, the guarded endpoint produced:

| measure | minimum | mean | maximum |
|---|---:|---:|---:|
| letters | 832 | 881 | 980 |
| words | 236 | 255 | 282 |
| mirror-pairs | 28 | 29.5 | 33 |

The maximum content-word count was 3. The v3 mechanical north-star suite passed
all six checked criteria on 24/24 seeds at 400, 1,200, 4,000, and 14,500-letter
requests. Larger requests stop at the guarded material ceiling rather than
relaxing the repetition constraints.

## Rejected routes

### Embedding-connected paragraphs

A paired semantic graph required both transitions created by nesting to exceed
the same cosine threshold. Similarity was a hard edge gate, not a final score.
Without repetition guards, threshold 0.65 easily produced 16-pair paths by
repeating phrases such as “deep speed” and “trade man”: another coherence
proxy hack.

With sentence, bigram, template, and content-word guards, threshold 0.50
produced 4–11 pairs (mean 8.2; mean 76.6 words). Threshold 0.55 produced 2–7
pairs (mean 4.6; mean 46.8 words). The current bank therefore cannot sustain a
100-word thematic path without repetition.

### LLM binary sense filter

The filter was revised to treat “Items draw award” as an explicit positive and
to reject cycles rather than unusual style. `gpt-oss:20b` then accepted at
least one obvious salad control in every smoke batch. Both batches were
invalidated and none of their verdicts was used.

### Short-pair beam harvest

Polaris debug job `7592554.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` ran on
one node / 32 ranks for 29 seconds:

| attempts | closed | midpoint split | both halves sentence-shaped |
|---:|---:|---:|---:|
| 1,024 | 123 | 66 | **0** |

This corridor should not be scaled. It confirms that ordinary beam scoring
does not generate sentence-pair material; the existing bank required
million-candidate offline enumeration.

## Next research route

Generate mirror-pairs at the sentence level, not by ranking finished word
palindromes. A viable experiment must condition simultaneously on an explicit
sentence plan for each half, assert the letter mirror during decoding, and use
repetition only as a hard constraint. Promotion still requires blinded reading
against real prose and shuffled/salad controls; no scalar proxy may decide the
winner.

## Sentence-plan exhaustive expansion

Polaris debug job `7592586.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` ran a
matched A/B search on one node (32 disjoint opening shards, 600 seconds per arm
per rank). Both arms used the same Brown-known subset of the frozen vocabulary
and accepted only pairs whose halves each realized a complete subject-and-verb
tag shape.

| arm | nodes | closed states | state prunes | distinct pairs |
|---|---:|---:|---:|---:|
| terminal filter | 84,815,872 | 1,117,411 | 0 | 20,989 |
| incremental sentence plan | 115,236,212 | 536,805 | 138,210,789 | 86,511 |

Moving the same structural test into the walk produced **4.12x** as many
distinct pairs in the same aggregate wall budget. The left side is checked as
a feasible sentence suffix because it grows by prepending; the right side is
checked as a feasible prefix because it grows by appending.

This expands supply, not quality. Raw planned hits include proper-name and
acronym salad such as `a blade fire pears not tub | buttons rae per i fed
alba`. That is evidence against treating POS shape as a meaning score. Do not
promote this raw bank into v3. The next inventory pass must combine the plan
with attested internal joins, then use the calibrated semantic evaluator only
to rank surviving diverse pairs.

## Whole-sentence quality funnel

Polaris debug job `7592931.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`
combined the incremental plans above with Norvig bigram attestation:

| arm | nodes | closures | distinct pairs |
|---|---:|---:|---:|
| every internal join attested | 52,608,016 | 103,635 | 105 |
| one unattested join allowed | 63,922,176 | 124,538 | 227 |

The strict arm improved local phrasing (`as i went on`, `in a test on`) but did
not fix clauses as wholes (`in a testing is`, `sign it set an i`). A calibrated
absolute 0--3 pass with `gpt-oss:120b-cloud` then scored both halves of all 105
strict pairs independently. The controls passed: four ordinary sentences
scored 3, all four salads scored 0--1, and the deliberately compressed `Items
draw award` scored 1. Known charming material is therefore retained at tier 1;
new material needs both halves at tier 2 so tier-1 salad cannot enter with it.

No pair cleared that floor. Half-score counts were 100 at 0, 98 at 1, 10 at 2,
and 2 at 3; pair floors were 77 at 0 and 28 at 1. The earlier batched binary
judge was also rejected: it accepted salad controls in 12 of 14 batches.

Starting from natural corpus sentences does not evade the mirror cost. Of
6,066 eligible 3--7 word Brown sentences, zero had a reversed segmentation
that was sentence-shaped with every join attested. Without the join gate there
was one: `lives on welfare | era flew nos evil`, which fails by inspection.

**Decision:** POS planning and bigram attestation are useful proposal pruning,
not a quality mechanism. Do not enlarge or rerank this bank into production.
The next generator must decode both halves jointly with whole-clause state;
post-hoc filtering, one-good-side mining, and scalar promotion are exhausted.

## Joint clause-state search

A bidirectional Brown word n-gram model was added to carry up to three words
of history on both halves during center-out decoding. The right clause uses
ordinary reading order; the prepended left clause uses a model trained on
reversed sentences. Hard sentence-plan feasibility remains active throughout.

This did not rescue beam search. At 3,000 words and beam 96, both order 2 and
order 4 closed 0/16 attempts. Widening to beam 512 and 6,000 words still closed
0/4 for each arm, at about 163 seconds per arm. Because the order-2 control
also failed, this diagnoses loss of the rare closable lineage rather than the
extra history alone.

The same clause model was then used only to order siblings in the exhaustive
walk, never delete them. At the previously productive 1,200-word, 100,000-node
smoke, both orders found 1 pair rather than the unbiased plan's 105, and both
found the same salad: `it set no war of | for a won test i`. Language-priority
depth-first traversal drills a bad corner just as beam search does. No Polaris
allocation is justified for this branch.

## Honest quality ceiling: sentence refrain

When readable distinct mirror-pairs are absent, the remaining exact algebra
is a mirrored sequence of self-palindromic sentences: `A B ... C ... B A`.
`llm_palindrome.refrain` implements that form with hard properties: every unit
is a complete character palindrome, no adjacent sentence repeats, each
non-central sentence occurs exactly twice, and the whole passage is a
character palindrome. Thematic pools (`dark`, `journey`, `reflection`, and
`absurd`) reduce arbitrary topic seams.

At a 500-letter request the themed catalogued pools yield 359--491 letters and
15--21 sentences. This is meaningful constrained refrain poetry, not novel
prose, and the API says so: `/api/v3/refrain` reports `source=catalogue`, the
sentence list, maximum sentence uses, and `novel=false`. It is a quality
control/ceiling alongside the novel composition endpoint, not a replacement
for it.

## Exact finite sentence-language intersection (5 September 2026)

**Decision:** do not scale the tested template inventory. A complete local
intersection found no pairs, with an obstruction visible at just three boundary
letters. This is a finite-coverage result, not evidence that sentence grammars
in general are impossible.

The new `experiments/sentence_intersection.py` enumerates seven explicit clause
families with agreement and verb complements chosen in advance: past transitive,
past motion, plural and singular present, imperative, modal transitive, and
negative motion. Unlike the preceding word-level searches, it never walks,
ranks, or segments partial word sequences: it indexes full clauses by letters
and looks up exact reversals. Different spacings are retained. Self-palindromic
sentences are counted separately, and repetition is checked both within clauses
and across the two-clause seam. No material enters the production bank.

Reproduce from the repository root:

```
.venv-v3/bin/python -m experiments.sentence_intersection --out runs/sentence_intersection.json
.venv-v3/bin/python -m pytest -q tests/test_sentence_intersection.py
```

The preserved machine-readable result is
`experiments/sentence_intersection-results.json`.

| measure | result |
|---|---:|
| enumerated / distinct clauses | 60,576 / 60,576 |
| exact non-self mirror pairs | 0 |
| self-palindromic clauses | 0 |
| reversed clauses matching an inventory opening for 1 letter | 14,244 |
| same, 2 letters | 13,026 |
| same, 3 letters | 0 |
| local runtime | 0.55 seconds |

Inspected examples include `i saw the map`, `i waited at home`, `we wait at
home`, `he waits at home`, `find the map`, `i can find the map`, and `i did not
wait at home`. These are transparent clause proposals, not a semantic quality
assessment of every slot combination. The grammar is small and ordinary
sentence endings/openings dominate its coverage; 60,576 combinations must not
be mistaken for 60,576 independent linguistic opportunities.

Separate mechanical controls recover `Go hang a salami | I'm a lasagna hog`,
reject the one-letter mutation `I'm a lasagna dog`, retain the compressed
`Items draw award`, and reject `do do do`. The known pair is control-only and
never mixed into the generated inventory. These calibrate exact retrieval and
repetition, **not** a semantic judge. No semantic judge was run because there
were zero candidates. Tests additionally compare retrieval to brute-force
pair enumeration, exercise alternate spacing, exclude self-palindromes, and
check repetition across the seam.

**Next route and gates:** design a boundary-compatible clause grammar before
multiplying its interior slots. Require independently authored, non-control
clause families to survive at least the three-letter boundary diagnostic and
produce distinct exact pairs locally. A positive boundary count alone licenses
only a local experiment. Polaris still requires actual novel pairs with both
halves reading under calibrated whole-sentence evaluation, inspected examples,
and a plausible extension beyond one template. No model/POS/bigram/embedding
score, known control recovery, or grammar label can satisfy that gate. The
100-word coherent novel paragraph remains unachieved.
