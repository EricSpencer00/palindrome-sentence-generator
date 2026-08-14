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
