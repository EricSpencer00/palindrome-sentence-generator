# Training: what is worth learning, and in what order

The search enforces the palindrome. A model only chooses among branches that
already satisfy it, so nothing here is trying to teach a model to be correct —
only to be readable. That narrows what training can be for, and it makes each
step falsifiable on its own.

Four pieces, each answering a question the one before it raises.

## 1. A model that reads backwards

**The question.** Every palindrome has a half built by appending words and a
half built by prepending them. The prepended half reads worse — measurably,
and in both growth directions, which is what says the cost belongs to backward
construction rather than to position in the text (see `architecture.md`).

Choosing a word to put *before* a fixed suffix is `p(word | what follows)`. A
forward model can only answer that by rescoring the whole suffix once per
candidate, which is why in-loop scoring costs about 31x. A model fine-tuned on
reversed token order answers it in one pass — and that pass yields a
distribution over the neighbouring token, which scores every candidate at once.
The cost is one pass per beam state per step rather than one per candidate.

**The method.** `training/corpus.py` builds a word-aligned token stream:
every word is tokenized on its own as `" " + word`, so the stream is a
concatenation of whole-word blocks. That is not what BPE does to running text,
and it is the price of two things worth more — whole words can be scored
against a cached context, and reversal is exact.

One stream is stored and read both ways, because a window of the reversed
stream is exactly the reverse of a window of the forward one. The forward and
backward runs therefore see identical text.

**The control.** Word-aligned tokenization is itself a departure from GPT-2's
pretraining, so stock GPT-2 is not a fair baseline for a model fine-tuned this
way. The forward fine-tune — same corpus, same schedule, same step count — is
the baseline, and it is the arm that separates "a language model in the loop
helps" from "a *backward* language model helps".

**The measurement.** `experiments/backward_study.py` runs three arms at matched
budgets and judges all three with an unmodified forward GPT-2 on the finished
text in reading order. The judge is never one of the fine-tuned models, and no
arm can improve its score by changing how it is measured.

If the gap between the halves does not move, the premise of the dual-head
architecture is wrong and it should not be built.

**Result of the fine-tunes** (GPT-2 small, wikitext-103, 37.9M word-aligned
tokens, 3000 steps at batch 32 x 512 on four A100s, ~12 minutes each):

| Direction | Validation loss | Perplexity |
|---|---|---|
| forward | 3.139 | 23.1 |
| backward | 4.060 | 58.0 |

Identical corpus, schedule and step count, so the **0.921 nats between them is
the cost of reading English backwards** — not a difference in data or budget.

That number is worth carrying into the architecture. The dual-head design has
head B predicting characters of text that follows them, and this says head B
starts about a nat behind head F on the same corpus. Backward text is genuinely
harder to model, which is a reason the prepended half reads worse and a limit
on how much any backward model can fix it. It does not say the approach fails —
a scorer only has to rank candidates, and a model a nat worse at prediction can
still rank well — but it does mean the two heads should not be expected to
contribute equally.

### Result of the study

24 seeds, 200 letters minimum, beam 60, every arm judged by stock GPT-2 on the
finished text. `gap` is the appended half minus the prepended one.

| arm | closed | gap/letter | gap/token | score/letter | score/token | letters |
|---|---|---|---|---|---|---|
| zipf | 24/24 | +0.443 | −0.824 | −2.431 | −7.383 | 222 |
| fwd@0.1 | 24/24 | +0.302 | −0.537 | −2.462 | −7.423 | 217 |
| bwd@0.1 | 24/24 | +0.597 | −0.856 | −2.454 | −7.246 | 217 |
| fwd@0.25 | 24/24 | −0.120 | −0.169 | −2.539 | −7.173 | 221 |
| bwd@0.25 | 24/24 | +0.271 | −0.613 | −2.513 | −7.093 | 212 |
| fwd@0.5 | 24/24 | −0.562 | +0.667 | −2.562 | −7.117 | 212 |
| bwd@0.5 | 24/24 | +0.132 | −0.474 | −2.574 | **−6.960** | 208 |

**The premise does not survive.** The baseline reproduces the recorded gap per
letter (+0.443 against +0.368) and reverses it per token (−0.824). The
prepended half does not read worse; it has shorter words.

**The model works anyway.** Per token the backward arm beats its matched
forward control at every weight — +0.177, +0.080, +0.157 — and the best arm
beats the plain baseline by **+0.423**. Standardizing the term fixed the
closure problem: 24 of 24 everywhere, against 0 of 4 at weight 1.0 unstandardized.

**And per letter you would have thrown it away.** `bwd@0.5` is the best arm in
the table per token and the worst per letter. The metric invented the problem
and then hid the solution.

Two limits on this. The gain grows monotonically with the LM weight across the
range swept, so 0.5 is the edge of what was tested rather than an optimum, and
the text also gets shorter as the weight rises (222 → 208 letters). And at the
full 30k vocabulary, single-token scoring is exact for only 58.6% of words —
95% at 3k — so the backward term is approximate for two words in five.

### Two details that fail silently

- Reversing the stream reverses tokens *inside* a word as well as the words:
  `"diverged"` is `[" diver", "ged"]` forwards and `["ged", " diver"]`
  backwards. A backward model therefore reaches a word by its **last** token.
  Scoring the first asks it about a token it only ever sees in second place —
  and because 95% of the frequency-ranked vocabulary is a single token, that
  mistake would have cost a little accuracy rather than obviously breaking.
- A word is scored by one token, so the score is exact for single-token words
  and approximate for the rest. `single_token_fraction` reports the coverage.

## 2. A judge that answers in microseconds

GPT-2 is the metric this project trusts and it is far too slow to be a reward
signal: reinforcement learning wants a score per rollout, and a search produces
thousands. `llm_palindrome/instant_judge.py` is a linear model fit to predict
GPT-2's score from features that cost nothing — attested bigrams, frequency,
repetition, word shape.

Linear on purpose: the training set is thousands of examples, the features are
already the things known to drive the score, and `explain()` prints what the
judge believes, which is worth more than a point of correlation.

It is scored by **rank agreement**, not error. A judge used to choose among
candidates only has to order them the same way.

**What the first attempt got wrong.** Trained on palindromes from a single
scorer, the fit reported a strong correlation and was at chance on held-out
data. Those palindromes land within a few hundredths of each other under GPT-2
— there was almost no spread to learn a direction from. Collection now mixes
deliberately weakened scorers (frequency-only, and one that ignores the words
entirely) so the judge sees what bad looks like, and the held-out set is split
**by search seed**, since palindromes from one seed share words and phrasing.

The number that matters is reported separately: rank agreement *within* the arm
the search actually uses. Telling good palindromes from random ones is easy and
is not what the judge is for.

**Result** (899 palindromes, held out by seed, GPT-2 small as the target):

| Set | Spearman | Pairwise agreement | MAE | Target sd |
|---|---|---|---|---|
| All arms | 0.991 | 0.964 | 0.045 | 1.170 |
| The real scorer's arm only | 0.888 | 0.821 | 0.045 | 0.114 |

The second row is the honest one, and the gap between the two is the whole
reason for reporting both: across arms the judge is separating good from
random, and within one arm it is resolving differences of about a tenth of a
nat. It agrees with GPT-2 on 82% of those calls at **280 µs** a palindrome,
against roughly 40 ms for GPT-2 itself.

What it learned is mostly what a reader would expect and one thing they might
not: a low type-token ratio is the strongest single predictor of a good GPT-2
score, well ahead of word frequency. Repeating yourself reads as fluent to a
language model. That is precisely the bias a small preference set exists to
correct, and it is why the verifiable reward penalizes repetition directly
rather than leaving it to the judge.

## 3. Learning the search's policy against a mostly-verifiable reward

Most of what this task demands is decidable. `llm_palindrome/verify.py` splits
the reward accordingly:

- **Asserted**, never scored: the text is a palindrome, every word is in the
  dictionary. These are guarantees of the search, so `verify` raises on them.
  A policy cannot trade away a property that throws.
- **Computed exactly**: closure, length (saturating at the target, because
  rewarding length past it is how a search becomes a filler generator),
  adjacent repetition, short-word rate.
- **Judged**: readability, and only readability. This is the one term a policy
  can chase an artifact in, and it is bounded by the terms it cannot fake.

The policy is `llm_palindrome/tunable.py` — the scorer's hand-chosen constants
exposed as five parameters over the features already in use. Learning better
weights for known features is a claim that can be checked; inventing features
at the same time would leave nothing to attribute a gain to.

`training/rlvf.py` optimizes them with an evolution strategy. Five parameters
and a non-differentiable beam search is not a case where ES is a compromise —
it is the right tool, and it needs no backward pass through a search. Members
of a generation share their search seeds, because seed variance is larger than
the difference between nearby weight vectors, so scoring them on different
seeds would rank noise.

Every few generations the elite are re-scored with real GPT-2. The judge is an
imitation and can be gamed; anchoring puts the drift in the log instead of in
the result.

### What the first run actually found: the metric is gameable

The learned weights beat the hand-chosen ones by +0.298 `lm_score`, winning
24 of 24 paired seeds. They are worse.

`lm_score` is total token logprob divided by **letters**. Longer words cost
fewer tokens per letter, so text made of longer words scores better whether or
not it reads better. The RL run found this in ten generations: it raised the
length coefficient sevenfold and cut the frequency coefficient by two thirds,
which dropped tokens-per-letter from 0.329 to 0.265 — a 19.5% cut that accounts
for the entire gain.

Normalized per token instead, the same texts are **0.673 worse, losing 24 of
24**:

| Weights | per letter | per token | tokens/letter | short words |
|---|---|---|---|---|
| Hand-chosen | −2.4306 | −7.3834 | 0.3293 | 0.337 |
| Learned | **−2.1329** | **−8.0563** | 0.2650 | 0.256 |

The GPT-2 anchor did not catch it, because the anchor used the same per-letter
metric. A check against the real model is only a check if it is normalized
differently from the thing being optimized.

Two consequences, and the second is the larger one:

- Every experiment here now reports both normalizations and says so out loud
  when they disagree. Where they disagree, believe per token.
- **The half-asymmetry gap is measured in per-letter units too.** The prepended
  half is the one that leans on short filler words, which is exactly the
  condition that manufactures a per-letter gap. Some part of the +0.368 may be
  word length rather than readability, and `backward_study.py` reports the gap
  both ways so the question is settled rather than assumed.

This is what the verifiable half of the reward is for. Length is capped there
at a target, repetition is penalized directly, and palindromicity is asserted —
none of which a policy can talk its way around. The judged half is where the
policy went hunting, and it found something in one afternoon.

### Refitting the judge per token, and what that costs

The judge was fit to the per-letter score, so it inherited the exploit. It can
be refit against the per-token score without another GPT-2 pass: total logprob
is the stored score times the letter count, and dividing by the token count
needs only a tokenizer.

| Judge target | All arms | Real scorer's arm only |
|---|---|---|
| Per letter | 0.991 / 0.964 | 0.888 / **0.821** |
| Per token | 0.957 / 0.938 | 0.244 / **0.595** |

(Spearman / pairwise agreement.) `word_len_mean` flips sign between the two
fits, +0.717 to −0.421, which is the same finding from the other direction.

The honest reading is that **most of what the per-letter judge predicted so
well within the real arm was word-length variance**, and once that is removed
the remaining readability signal is close to invisible to these features:
59.5% pairwise is barely above chance. Across arms the judge still works — it
tells good palindromes from broken ones at 93.8% — so it remains useful as a
filter, and it is not yet a usable reward for improving text that is already
decent.

That is a result about the features, not about the method. The next thing to
try is a reward that uses GPT-2 per token directly at a smaller rollout count,
which is affordable precisely because the verifiable terms carry most of the
signal and the judged term only has to break ties.

### The human part, sized honestly

GPT-2 fluency is wrong in a specific way: it likes frequent words, so it will
take a string of common monosyllables over a sentence that says something. That
bias is consistent enough that tens of judgements can correct it.

`training/preferences.py` fits a logistic correction on the same features, from
pairs marked `"a"` or `"b"` in a JSON file. With that little data the only
honest report is held-out pair accuracy, and the fitter **refuses to save a
model that is at chance** rather than shipping a flattering number. Skipping a
pair you have no opinion about is more useful than guessing on it.

## 4. The full model

`architecture.md` describes it: one causal backbone over the palindrome's
spine, a forward head and a backward head, word-boundary bits per reading, and
the dictionary enforced by trie masks at decode time. Because only the spine is
emitted, every sample is a palindrome by construction — the constraint stops
being something a search maintains and becomes something the architecture
cannot violate.

It is fourth on this list, not first. Step 1 is the cheap experiment that can
falsify its premise, and it should be allowed to.

## What none of this can fix

Two cheap experiments bound the four pieces above. Both were run to falsify a
plan, and both did.

### Reranking the search's output is exhausted

`experiments/oracle_bound.py` generates 2000 palindromes from the plain search
and scores them with Qwen2.5-0.5B. Best-of-N is an upper bound on what *any*
reranker could achieve on a fixed candidate pool, so if the curve flattens, no
future judge — larger, better-calibrated, human-corrected — can help.

| N | 1 | 24 | 100 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| per token | −7.288 | −6.978 | −6.883 | −6.834 | −6.817 | −6.810 |

It flattens. Going from the 24 seeds the CLI uses to 2000 buys 0.168 nats per
token; the last thousand samples buy 0.007. The winner sits about three
standard deviations above the mean, which is what order statistics predict for
a roughly normal distribution — there is no tail of secretly-coherent
palindromes at larger N. Read side by side, the best of 2000 and the median of
2000 are the same kind of object. What the extra samples buy is cleaner noise:
the median is littered with junk tokens ("pp", "xo", "su", "ee") and the winner
is made of real words that still say nothing.

### But that bound was measured inside a corridor

Of those 2000 palindromes, **1975 opened with the same five words and 1983
closed with the same five**, and the whole sample used 1317 of the 30000 words
available. The search was not producing 2000 palindromes; it was producing 2000
variations on the interior of one.

The cause is the beam's only source of seed variation. `beam_search` adds
`rng.random() * diversity` to each candidate with `diversity=0.4`, while
ZipfScorer's own range is several units wide — word frequency alone spans 0 to
8. The jitter cannot reorder the leading candidates, so every seed walks into
the same opening. The same opening and ending appear in the README's example,
so this funnel predates every experiment recorded here.

### Escaping the corridor is easy, and does not buy coherence

`experiments/diversity_sweep.py` raises that one number. 200 seeds per arm,
same judge.

| diversity | closed | distinct openings | distinct words | mean/token | best/token |
|---|---|---|---|---|---|
| 0.4 | 200/200 | 3 | 708 | −7.295 | −6.912 |
| 1.0 | 200/200 | 63 | 1048 | −7.354 | **−6.790** |
| 2.0 | 200/200 | 176 | 1591 | −7.516 | −6.887 |
| 4.0 | 200/200 | 199 | 2328 | −7.703 | −7.229 |
| 6.0 | 200/200 | 200 | 2854 | −7.781 | −7.181 |

Exploration responds exactly as predicted — 3 distinct openings become 200, and
the vocabulary in use quadruples — at no cost in closure, which stays 200/200
across the whole range.

**Readability moves the other way.** The mean falls monotonically by 0.49 nats
per token from the narrowest arm to the widest, and the effect is far larger
than its standard error. Unanchored from word frequency, the search reaches for
"maserati", "merseyside" and "cabaret", and the text gets worse, not better.

One number argues the corridor still mattered: best-of-200 at diversity 1.0
(−6.790) beats best-of-**2000** in the corridor (−6.810). A slightly wider
search beat ten times as much sampling of the narrow one. Treat it as
suggestive rather than settled — a single arm maximum is a noisy statistic, and
the arms are not monotone in it (2.0 is worse than 1.0). Confirming it needs a
paired run at larger N.

### What that leaves

Coherence is not reachable by ranking the candidates this search produces, and
not by making the search produce more varied ones. Both levers are now measured
and both fail. What has not been tested is the *unit* the search consumes: the
trie holds single words, so coherence has to emerge from adjacent-word scoring.
A corpus-derived phrase inventory — attested n-grams consumed atomically —
would make local coherence a property of each unit rather than something the
scorer has to discover, and it is the one remaining cheap idea.

The other honest option is length. Coherence and length trade off directly
here, and 200 letters may simply not admit a coherent solution in a 30k
vocabulary. Nothing above has tested what the search can do at 80.

## Running it

```bash
python training/corpus.py --split train --limit 600000 --out data/tokens
python training/train_directional_lm.py --direction backward \
    --tokens data/tokens/train.bin --out runs/gpt2-backward
python experiments/backward_study.py --backward runs/gpt2-backward \
    --forward runs/gpt2-forward --seeds 24

python training/fit_instant_judge.py --seeds 900
python training/rlvf.py --generations 12
python training/preferences.py ask --n 30   # then fit

python experiments/oracle_bound.py             # ~13 min, caches its texts
python experiments/diversity_sweep.py          # ~7 min
```

Cluster job scripts are operator-specific and live outside this repository.

## The paragraph problem, decomposed (August 2026)

Everything above optimized free-running text against the mirror, and the
decisive number arrived late: reversing English letters and re-segmenting into
the vocabulary costs **3.3 bits per free letter** (stable across span lengths
and segmentation strategies), against English's ~1.6. Free-running palindromic
prose past ~30 letters is not sparse; the walk-in coherent set is effectively
empty, and every beam result in this file is that emptiness being reported by
a different instrument.

What survives the constraint:

- **Exhaustive enumeration of the short regime.** At ≤28 letters the space is
  walkable — 2.55M distinct palindromes in a 30-minute 32-way walk on Polaris
  — and best-of-ALL beats best-of-N over any beam pool, because there is no
  proposal distribution for the oracle bound to apply to. Yield includes
  novel classics-register sentences ("non academia aimed a canon", 22
  letters, grammatical, in no corpus).
- **Assembly, not search, for length.** Mirror-pairs and self-palindromic
  sentences pay the constraint internally; nested as L1..Lk C Rk..R1 they
  form palindromes of unbounded length by construction. Coherence becomes
  selection and ordering — the first subproblem here with no letter
  constraint at all.
- **The refrain form.** Judged strictly, mirror-pairs with two readable
  halves are ~1% of the bank (3 of 240); whole short palindromes pass as
  sentences far more often. A mirrored sequence A B C D C B A of them is the
  honest paragraph form, and it is the form long human palindromic poetry
  already uses.

Blind whole-text judgment of the resulting 286-letter paragraph: INTENTIONAL,
ranked above its own shuffle and below real prose — "constraint-based or
refrain poetry, semantically incoherent." That is not a failure to reach
prose; it is the measured ceiling of the form at this vocabulary, and the
open levers are unit-bank scale (deeper walks, 30-34 letters) and better
selection, not better search.

### The strict-judging result (the number that matters)

Every unit vetting in this project used one of two rubrics. The lenient one
("terse telegraphic register") passed 21/26 of the walk's top units — and also
passed 1/5 word-salad controls, so it was leaking. The strict one ("reads as
intentional AND says something", calibrated real 5/5, salad 0/5) was then run
across five independently-selected populations:

| population              | selector            | strict passes |
|-------------------------|---------------------|---------------|
| walk top 26             | GPT-2 rank          | 0/26          |
| walk ranks 26-62        | GPT-2 rank          | 0/36          |
| local walk top          | GPT-2 rank          | 0/16          |
| readable-vocab filter   | rule-based          | 0/20          |
| uniform random sample   | random over 2.55M   | 0/72          |
| **total**               |                     | **0/170**     |

Zero. Including the units the assembled paragraph is built from. The paragraph
is palindromic by construction and was judged INTENTIONAL as a whole text —
above its own shuffle, below real prose, identified independently as
"constraint-based or refrain poetry" — but its lines do not pass as English
sentences under a rubric that admits real prose and rejects salad.

Four surface proxies for readability were also killed against labeled judge
verdicts: GPT-2 weaker-half score (AUC 0.659), full bigram-join attestation
(rejects the best unit found, which has zero attested joins), rule-based
vocabulary filtering (0/20), and edge-join attestation (no separation). Direct
judging is the only instrument that works, which makes judge throughput — not
search, not scoring — the binding constraint on any further progress here.

## Coherent paragraphs: what actually worked (loop, August 2026)

Four requirements, each measured against controls.

**1. Units must be authored, not searched.** Exhaustive walks produced 12M
closures (2.55M at 18-28 letters, 9.29M at 24-34, 27k at 10-17) and zero units
that survived strict judging. One LLM call produces 30 valid palindromes. The
walks are also not exhaustive in any useful sense: with a 14k-word vocabulary
the tree branches ~14k wide at every closure, so a time-budgeted DFS returns a
deep prefix of one corner. Canon recall — how many of the 27 catalogued
palindromes a walk rediscovers — is the acceptance test that exposes this, and
it goes UP as vocabulary shrinks: 10/18 at 83 canon-seeded words and depth 4,
1/18 at 300 frequency-ranked words, 0/27 at 14k.

**2. Verification must be mechanical.** An authoring model claimed 20 novel
palindromes; all 20 were palindromic and all 20 were catalogued classics. A
71-entry reference missed it; the 120-entry one in `data/known_palindromes.json`
catches it. Every surface proxy for readability also failed against judge
verdicts: GPT-2 weaker-half score (AUC 0.659), bigram-join attestation (rejects
the best unit found, which has zero attested joins), vocabulary filtering
(0/20), edge joins (no separation).

**3. Length is free once units exist.** `paragraphs.assemble` nests mirror-pairs
around a centre; the result is palindromic by construction at any length. Given
canonical units it produces 305 letters judged INTENTIONAL.

**4. Each reversal must be a consequence, not an echo.** This is the step that
converts a shared setting into a held subject. A themed paragraph scored
`holds_subject: false`; one where each sentence's word-reversal advances the
story scored `holds_subject: true` — "Waves took sons" returning as "Sons took
waves".

### The letter/word split, and why the letter side won in the end

Word-ORDER palindromes mirror the sentence sequence and not the letters, so
they pay nothing per letter. For a stretch of this work they were the better
paragraph and `/api/v2/paragraph` served them, on the reading that the mirror
cost capped the letter-level form at refrain poetry.

That reading was wrong, and the error is worth stating precisely because it
survived roughly forty iterations. The cost does force units to be short. Short
units carry no subject. What does not follow — and what was assumed — is that
palindromic units must be short. Whole self-palindromic SENTENCES pay exactly
the same 3.296 bits per free letter and carry subjects perfectly well.

The evidence that settled it, all blinded against real-prose and word-salad
controls:

- Thematic selection over two-word halves fails absolutely: of every
  content-word pair across the 26 usable halves, **2** co-occur anywhere in
  3,932 corpus sentences.
- Thematic selection over whole sentences works: 8 judged centres share "saw"
  with a first-person narrator, and grouping them beat the same structure with
  mixed topics.
- Four-word halves — the length at which prose would start — yield **0**
  both-attested pairs from 34,688 attested 4-grams. The material for a
  half-level through-line does not exist at any corpus size tried.

`/api/v2/paragraph` now defaults to `mode=letter` and serves the sentence-level
form. The word mode remains at `?mode=word`, labelled `letterPalindrome:
false`, with the 17-unit threshold below applying to its own path.

### Proxy scoring: filter yes, ranker no

Every proxy tried here was eventually checked against blind judging rather than
against another proxy, and the result did not vary:

| proxy | as a filter | as a ranker |
|-------|-------------|-------------|
| GPT-2 mean logprob | sound — selects nothing a judge rejects | failed three times |
| `reads_as_attested` | sound — 79% accepted against 38% | not used |
| `themes.cohesion` | sound — finds the theme | failed — sizes the paragraph wrong |
| word frequency | no signal (4.34 accepted vs 4.06 rejected) | — |

The three GPT-2 ranker failures: a guarded 0.58 gain in its own score was
invisible to a judge; a larger unit pool it preferred was judged worse; a set
of 18 verified novel units it preferred made the paragraph worse. In each case
the proxy moved and the reader did not.

Two gaming modes were found and bounded on the way — word repetition
(`sequencing.repetition_rate`, 0.356 → 0.471 when optimised against) and
cadence concentration (`sequencing.cadence_concentration`, "Partner is. Sign
is. Warning is."). Each guard closed one door and the search found the next,
which is the argument against trusting any of them to rank.

### Storing beats inferring

`data/known_palindromes.json` stores the canon normalised, which is right for
the novelty check and fatal everywhere else: with no word boundaries, every
entry can only be a centre, and `harvest` returned 0 pairs from 145 entries.

Recovering the spacing by segmentation (`respace`) works for most entries and
breaks 14 of them — "siri demand i am a maid named iris", "a nut for ajar of
tuna", "sit on a potato pa not is". Blind judging accepted 28 of 61 centres and
those 14 were all in the rejected half, so the sentences were fine and recovery
had broken them.

No corpus statistic fixes this. Attested-join weighting recovers 3 of 10 and
the weight was swept, not picked (0 fixes nothing, 1–4 fixes three with no
regressions, 6+ breaks good readings). The rest cannot be recovered at all:
choosing "i slam" over "islam" requires knowing the sentence, and "islam" is a
common word attested beside its neighbours, so unigram, attested-count and
attested-fraction scoring all prefer it. Absolute count is actively perverse —
a shorter reading has fewer joins to miss.

`data/canon_spelled.json` stores the 56 spellings instead. Re-judged blind, the
corrected centres are accepted **13 of 16** against 0 of 14 before. `respace`
keeps its job on the mining path, where no true spelling exists to store.

### Where the word-order paragraph actually sits

Judged three times against real prose, with the texts reordered and the
instruction reworded each time to defeat position and phrasing bias:

| pass | first | second | third |
|------|-------|--------|-------|
| A | observatory | protagonist | prose |
| B | prose | observatory | protagonist |
| C | prose | observatory | protagonist |

Majority: **prose first, the palindromic paragraph second.** One pass ranked
the palindrome above prose; two did not, so that pass is noise — the same
single-verdict instability measured on unit judging, where borderline items
flip about half the time. Anything claimed from one pass in this project has
twice turned out to be wrong.

What survives repetition: a ~65-word word-order palindromic paragraph holds a
subject in 3/3 passes and ranks second of three in 2/3. The same bank at 291
words scores `holds_subject: false` — length, not subject matter, is what loses
the thread, which corrects an earlier reading of this that blamed the units'
topic (instruments rather than people). A protagonist-centred bank was written
to test that reading and ranked third, holding a subject in only 2/3.

### Arc-aware selection: a feature built on a single verdict, and refuted

A one-pass length sweep found `holds_subject: true` at 147 words and `false` at
33, 63 and 99, with the judge attributing the difference to the longer text
containing "doubt, fatigue, theories, dawn" — stakes rather than scenery. Unit
selection was changed accordingly: stakes-bearing units are kept first and
seated nearest the centre (`server/v2.select_units`, `ARC_WORDS`).

Judged twice afterwards, texts in opposite order: **the arc-aware paragraph
does not hold a subject** (0/2), and prose ranks first in both. The verdicts
were "formally brilliant but image-scattering" and "sacrifices semantic
coherence for palindromic form".

The sweep it was built on was a SINGLE verdict, and single verdicts flip about
half the time at the margin — a fact measured earlier in this same project and
then ignored. Two prior claims here died the same way ("No devil lived on" as
novel; "Deep spot top speed" as judged-good). The rule that survives: no
feature and no finding from one judge pass.

The selection change is kept — it is harmless, tested, and keeps stakes units
in short requests — but it is not evidence of anything. The standing position
is unchanged from the three-pass comparison: a word-order palindromic paragraph
reads as an intentional composition and ranks second to prose.

### How long a word-order palindromic paragraph can hold a subject

A length ladder from one bank — same units, same assembly, only the number of
units varying — judged for `holds_subject` at each rung. Pass A presented them
ascending, pass B descending:

| words | pass A | pass B | pass C | majority |
|-------|--------|--------|--------|----------|
|  33   | no     | no     | yes    | **no**   |
|  63   | no     | no     | yes    | **no**   |
| 105   | yes    | yes    | yes    | **yes**  |
| 147   | yes    | yes    | yes    | **yes**  |
| 207   | yes    | yes    | yes    | **yes**  |

A ascending, B descending, C shuffled. A and B agree exactly under reversed
presentation, so the threshold is not position bias. C answered "yes" to all
five and so discriminated nothing — a pass that separates no cases is a weak
vote, and it is recorded rather than dropped because dropping inconvenient
passes is how the earlier false positives in this file happened. The
through-line appears at ~105 words and holds above it, because that is where
the bank's storm arrives: the short cuts stop before anything happens to
anyone, leaving harbour scenery in mirror form.

This corrects two earlier readings recorded above, both taken from single
verdicts. "Instruments rather than people" was wrong — the same harbour units
fail at 63 words and hold at 105. "Length, not subject matter, loses the
thread" was backwards — short is what fails here; length is what allows an
arc to fit. The operative variable is whether the selected units contain
a completed event, and short paragraphs cannot.

Practical consequence: requests to the WORD mode below about 17 units return a
formally correct palindrome with no story in it, so `?mode=word` floors its own
length at 18 regardless of what was asked. The threshold is that mode's and
does not transfer — 17 whole sentences would exhaust a 40-centre inventory,
and the letter mode's paragraph is 7.
