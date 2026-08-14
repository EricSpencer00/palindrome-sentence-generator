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
```

Cluster job scripts are operator-specific and live outside this repository.
