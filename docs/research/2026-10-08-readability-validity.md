# Readability and search validity audit — 2026-10-08

## Decision

We are **not 100% sure** that Brown or the current automatic measures are
valid measures of readable long palindromes. The evidence supports a narrower
statement: several diagnostics are reproducible and useful for rejecting
obvious failure modes or comparing a fixed search intervention. None is a
validated proxy for reader-perceived grammar, meaning, or coherence.

Keep Brown word-order gain as a **secondary diagnostic**. Do not tune search
parameters to maximize Brown, GPT-2, Qwen, or Jev scores. Ask Jev only to
propose a small, interpretable candidate set after the objective and gates
below are frozen. Jev's output is a hypothesis menu, not evidence.

The decisive mismatch is population and endpoint: the Oct. 2 penalty study
compares exact 30–119-letter outputs from two search implementations, while
the current long construction frontier includes separate 568–752-letter
lineages. The former cannot establish how a parameter behaves on the latter;
none of the long constructions has human readability ratings. The separate
October8 cap probe produced804-letter exact outputs that fail the strict
screen; they do not resolve this gap. Exactness,
length, basic lexical admission, and local-order fit must remain separate
outcomes.

## What the existing evidence does establish

### Brown word-order gain

The current scorer is substantially better controlled than the old
all-Brown in-sample report:

- It fits add-0.1 word bigrams on a deterministic 80/20 **document split**
  (401 training and 99 held-out Brown files), so control text is not used to
  fit the model.
- Held-out intact controls are complete Brown sentence spans; candidate and
  control scores use the same ASCII-letter tokenizer, sentence-boundary
  treatment, word multiset, and deterministic 32-shuffle comparison.
- The score is the candidate's mean word-bigram log probability minus the
  mean for shuffles of the same words, repartitioned to the original sentence
  lengths. This removes much unigram-frequency advantage and tests whether
  the observed local order fits this Brown model better than those shuffles.
- Frozen inputs, hashes, split manifests, scorer source, and saved scores make
  the reported diagnostic reproducible. The 2026-09-25 audit correctly found
  that the earlier all-document fit was in-sample and excluded it from current
  claims.

These are real strengths for a **local-order control**. Positive gain means
only that this exact order beats its own sampled permutations under one
smoothed Brown word-bigram model. Held-out Brown spans separating from their
shuffles demonstrate scorer sensitivity to local order; they do not show that
the model can rank palindrome readability. The nine longer matched candidate
spans all scoring below Brown prose is descriptive evidence under this scorer,
not a causal penalty of palindromy or a population estimate.

### Other current measures

- Independent forward/reverse scans, normalized-letter counts, and hashes are
  appropriate for exactness and provenance. They do not assess language.
- Repetition, duplicate counts, distinct-content, mirrored-unit, and
  self-palindromic-span screens expose identifiable shortcut patterns. They
  are useful construction gates, but hard screens can reject unusual yet
  readable text and cannot certify the survivors.
- The fragment-penalty study is a valid **within-family intervention** for
  its frozen seeds, resource bundle, vocabulary, methods, and 30–119-letter
  bands. It found reduced target phrase recurrence and higher surface
  diversity, alongside a lower strict mechanical pass count. It does not show
  a readability gain. Its selection-conditioned diversity and two native
  search implementations limit causal interpretation.
- The Qwen3 control screen detects gross calibration failure for that rubric
  and model. The 0/0 candidate floor and 0/0 shuffle outcomes do not distinguish
  the candidates. Passing prose controls is not human validation; multi-
  sentence Brown controls and possible Brown pretraining exposure remain
  limitations. The earlier small local judge failed its own control gate.
- GPT-2 candidate-minus-shuffle scores are another model-specific local
  preference. Historical Brown/GPT-2 sign disagreements and broad shuffle
  ranges show that rankings depend on model and control realization. Neither
  is an independent reader outcome.
- Search closure, best score, beam survival, and count of exact outputs are
  valid efficiency/coverage outcomes only under fixed, reported resources.
  Equal nominal limits can still mean different expansion counts or search
  opportunities. Output-only scoring also creates success-conditioning bias.

## Confounds and failure modes to guard against

1. **Wrong target population.** Short 30–119-letter runs do not validate
   tuning for 568+ letter lineages. A setting can improve short local
   transitions while worsening long-distance coherence or repetition.
2. **Proxy gaming and objective mismatch.** Tuning to bigrams favors locally
   familiar transitions, not a globally interpretable scene. More frequent
   words, repeated names, formulaic frames, and easy phrase fragments can
   improve component scores without making a text more readable.
3. **Brown domain mismatch.** Brown's period, genres, editorial style, and
   tokenization differ from contemporary reader expectations. Its held-out
   status prevents document leakage, not corpus/domain bias. The current
   apostrophe split and ASCII-only tokenizer also discard distinctions.
4. **Shuffle-control mismatch.** Same-bag shuffling preserves vocabulary and
   sentence-length vector, but destroys syntax and discourse by design. It is
   a useful negative control, not a plausible quality-matched baseline. Only
   32 draws approximate the permutation distribution; their min/max are not
   confidence intervals, and deterministic seeds do not add independent
   samples.
5. **Length and sentence-boundary effects.** Mean per-transition scores can
   hide long-range failures. Candidate punctuation controls candidate
   segmentation, while Brown controls retain corpus segmentation. Sentence
   segmentation errors or different sentence counts can alter the score.
6. **Repeated and related outputs.** Many long candidates share parent text,
   events, names, or construction operators. Counting them as independent
   evidence overstates the effective sample size. Seed-level bootstrap does
   not generalize across lineages, grammars, topics, or readers.
7. **Retrospective selection.** Selected best outputs have passed through
   search and manual repair. Scoring only selected successes rewards
   survivorship and allows indirect tuning to the evaluation set.
8. **Judge calibration and floor/ceiling effects.** A small held-out prose
   screen tests gross sensitivity only. Model ratings are not interchangeable
   with humans, and an all-zero outcome cannot rank quality.
9. **Multiple tuning decisions.** Repeatedly changing weights, gates, lengths,
   seeds, controls, or rubric after seeing scores contaminates validation even
   if each individual run is deterministic.
10. **Length/readability tradeoff hidden by a scalar.** An unconstrained sum
    can prefer longer poor text or shorter fluent text depending on arbitrary
    units and weights. Length must be reported with quality and coverage, not
    folded into an unvalidated scalar reward.

## Frozen tuning and validation design

This is the recommended next study. It is a design, not a completed
reader evaluation. Keep the current canonical paper and
historical artifacts unchanged until a completed result supports a revision.

### 1. Freeze the decision and outcome hierarchy

Target the actual goal: **find the longest original exact palindrome that
readers judge to be intact, understandable prose**. Use a lexicographic
outcome, not one weighted score:

1. Exactness and provenance are hard gates.
2. Report how many tasks produce an exact output at each target length under
   identical compute/resource budgets.
3. Among exact outputs, report the proportion passing prespecified mechanical
   shortcut/novelty gates. Keep each gate visible; do not hide it in a score.
4. Primary quality outcome is blinded human rating of grammaticality and
   whole-text coherent meaning, reported separately. The prespecified joint
   pass threshold is set before generation and validated on control items.
5. Compare the longest target band that clears the human-quality threshold;
   if no band clears it, conclude no readable-length result was established.

Do not claim that an automatic score substitutes for outcome 4. Brown,
GPT-2, and an independently chosen language-model judge are secondary
diagnostics whose correlations with human ratings are estimated on development
data only.

### 2. Define split units to prevent leakage

Before parameter suggestions or candidate generation, commit a protocol that
assigns disjoint **construction lineages and semantic-frame families** to
development and held-out validation. Keep all descendants, paraphrases,
near-duplicates, and shared-parent outputs in one partition. Split seeds
within each permitted family as a second grouping factor. Publish the split
manifest and text hashes before rating.

Use development data to calibrate rubric wording, control sensitivity, and
choose parameters. Do not reuse it as final evidence. Reserve a fresh
confirmatory set of lineages, semantic frames, and seeds; freeze the selected
setting and baseline before generating or unblinding its texts.

### 3. Keep Jev's role bounded

Give Jev only the existing named parameters and the frozen development
objective. Ask it to return at most three materially distinct weight settings,
each with a short rationale and an expected failure mode. Include the current
default as a mandatory baseline. Do not expose validation identities, scores,
or texts. Reject suggestions that add features, alter hard gates, or optimize
Brown/GPT-2/Qwen scores directly. Freeze all arms before the development run;
then choose one challenger with development human results and test it once on
the untouched validation split.

Parameters available in `llm_palindrome/tunable.py` are `zipf`, `length`,
`reuse`, `adjacent_repeat`, and `bigram`; current defaults are
`[1.0, 0.3, -2.0, -4.0, 0.0]`. Make clear that these weights change proposal
ranking, not the exactness invariant. Record full configuration and code
hashes.

### 4. Match search opportunity and preserve failures

For every arm, use the same vocabulary, grammar/construction families,
target-length bands, seeds, beam/candidate caps, wall budget, workers, and
stop rules. Randomize arm order within each paired seed/lineage block. Retain
every attempt, including no-closure, malformed, duplicate, and low-scoring
outputs. Record actual expansions, scorer calls, closure status, strict-gate
status, runtime, and resource limits. If a resource limit is reached, compare
both coverage and quality conditional on success; never silently compare only
the successful arm's best items.

For tuning candidates, use a small predeclared grid or Jev's capped suggestions
with the current default. Do not expand the grid after seeing outcomes. Any
follow-up changes are new exploratory development, not confirmation.

### 5. Rate a frozen, balanced packet blind

Select items by a published hash-based rule that is balanced across arms,
length bands, and paired blocks; do not hand-pick attractive examples. Dedup
identical surfaces before rating and retain the mapping to generation arms.
Hide arm, score, lineage, and condition labels. Randomize presentation order.

Use at least three independent human readers per item; preferably five for
the long-text study. Ask separately for grammaticality, coherent meaning
across the whole text, naturalness, and whether the reader can paraphrase the
main event. Permit “cannot judge” and collect a short reason for low scores.
Publish anonymized raw ratings, rubric, reader instructions, missingness,
inter-rater agreement with uncertainty, and arm-level paired effects clustered
by lineage. Model ratings may be collected as separate outcomes, never merged
with human ratings.

Controls should include diverse intact prose spans matched by scorer-token
length and sentence count, same-word shuffles, and a small known-malformed
negative set. Brown may supply some intact spans, but not all: include modern
prose from a disjoint source if permitted, and do not call either corpus a
human gold label. Include repeated hidden items to estimate reader
consistency. Predeclare a control-calibration gate; a failed gate invalidates
interpretation rather than inviting threshold changes.

### 6. Treat Brown as a secondary check

Retain the existing document split, tokenizer, smoothing, sentence policy,
input hashes, and fixed shuffle generator for comparability. For new work:

- report candidate score and the full same-word shuffle distribution, with
  the observed score rank/empirical tail fraction; call its spread a
  permutation distribution, never a confidence interval;
- show sentence/token counts and a length-stratified view;
- analyze all frozen outputs, not only best-of-search winners;
- keep output-level paired differences clustered by lineage/seed;
- preregister any tokenizer or corpus change as a new diagnostic version;
- evaluate Brown-vs-human association only on development data, then report
  the locked association on validation. If it does not generalize, stop using
  Brown to order candidates.

The 32-shuffle estimate is adequate as a cheap descriptive baseline for
historical compatibility, not for fine ranking among near-ties. If greater
precision is needed, choose a larger fixed shuffle count before seeing the
candidate score comparisons and preserve the historical 32-draw analysis.

### 7. Acceptance and reporting rules

Promote a parameter setting only if it improves the preregistered human
quality endpoint on held-out lineages without a material loss in exact-output
coverage at the target length. Define a non-inferiority margin for coverage
and quality before the run. If no setting meets both gates, keep the current
default and report the negative result. A Brown-only gain, GPT-2-only gain,
higher distinct count, or longer exact output cannot independently promote a
readability claim.

Publish all arms, attempted-task denominators, splits, raw human ratings,
diagnostics, exclusions, intervals, code/configuration hashes, and deviations.
Label all estimates as finite-sample and family-specific unless the sampling
design supports broader inference.

## Conclusion

Brown is methodologically reasonable for what it measures: whether local word
order beats deterministic same-word permutations under a held-out Brown
bigram model. It is not validated for the target decision. The current
automatic stack can verify exactness, describe construction and repetition,
screen obvious controls, and compare narrow interventions. It cannot tell us
whether the longest palindrome is readable. The next parameter study should
therefore tune on development human judgments and decide only on a fresh,
lineage-held-out human validation set; Jev can nominate settings but cannot
select the winner.

## Sources inspected

- `paper/review-handoff/2026-09-26/READABILITY-SCORE-AUDIT.md`
- `paper/review-handoff/2026-09-26/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md`
- `paper/review-handoff/2026-09-26/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md`
- `experiments/score_length_stratified_readability.py`
- `experiments/score_directional_search_brown_20260925.py`
- `llm_palindrome/wordorder.py`
- `llm_palindrome/tunable.py`
- `paper/naacl2027-v1.tex`
- `docs/research/2026-10-08-search-frontier.md`
- `docs/READABLE-PALINDROME-GOAL.md`
