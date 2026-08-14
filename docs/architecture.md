# A transformer whose output space is exactly the palindromes

## The measurement that motivates this

A palindrome has two halves that share one letter sequence but segment into
different words. Any generation scheme builds one half by appending words
(left to right, the direction English is written) and the other by prepending
(against the grain). Measured across 24 candidates per arm with GPT-2 small:

| Growth direction | Natural-order half | Reversed half | Gap | Natural half wins |
|---|---|---|---|---|
| Outside-in (appends left) | −2.322 | −2.690 | **+0.368** | 22/24 |
| Center-out (appends right) | −2.518 | −2.650 | **+0.132** | 18/24 |

The gap follows the construction direction rather than the position in the
text: flip which side gets appended and the advantage flips with it. So the
half built backwards is reliably the worse half, in both schemes.

## Both rows above are measured in units that word length can move

`lm_score` is total token logprob divided by **letters**, so a half made of
longer words scores better whether or not it reads better. The two halves of a
palindrome segment the same letters into different words, and the prepended
half is the one that leans on short filler — exactly the condition that
manufactures a difference here.

Re-measured per token, on 24 seeds at the same budget, the baseline gap is
**−0.824**: the *prepended* half reads better. The sign reverses.

| Normalization | Gap (appended − prepended) |
|---|---|
| per letter | +0.443 (reproduces the +0.368 above) |
| per token | **−0.824** |

The outside-in row survives as a fact about word length. It does not survive as
evidence that backward construction produces less readable text, which is what
the rest of this document is built on.

The center-out row has a second problem, independent of this one: it was
measured with a scorer whose adjacency term assumed the outside-in convention,
so under center-out it compared each word against one from the far end of its
half and adjacent repeats went unpenalized. That is fixed (`scoring.adjacent`,
and the `growth` argument the searches now supply), and the row has not been
re-measured since.

**What this does to the case below.** A backward language model was built and
tested against this (`experiments/backward_study.py`, `docs/training.md`). It
works on its own terms — it improves the half it scores, by +0.32 to +1.14 per
token depending on weight. But the deficit it was built to repair is not there
in the units that matter. The dual-head architecture should not be built on
this measurement. If there is a case for it, it has to be made per token, or
from something other than the half-asymmetry.

This says the bottleneck is not *which* direction you grow. Both directions
build one half backwards, and we have no model that generates fluent English
backwards. **Choosing a different direction relocates the problem; it does not
remove it.**

## The degrees-of-freedom insight

A palindrome of 2k letters has only k free letters. Verified on a generated
sample: 217 letters, 108 free, with the left half's letters exactly equal to
the right half's letters reversed — while the two readings segment into
completely unrelated words:

```
letters (shared spine)   x i f r o m l i f e w i t h g i r l l a w o n e ...
left reading             Xi from life with girl law one ...
right reading (mirrored) ... right i we. Film or fix.
```

A model that emits 2k characters is therefore modeling k characters of real
choice plus k characters of bookkeeping. Emit the spine instead, and every
character lands in two places at once.

## Architecture

One causal backbone over the k-letter spine, two reading heads.

```
spine        c₁ c₂ c₃ ... c_t            (the k free letters, emitted once)
               ↓  ↓  ↓       ↓
        ┌──────────────────────────┐
        │  causal transformer      │     h_t = f(c≤t)
        └──────────────────────────┘
             ↓                ↓
      ┌────────────┐   ┌────────────┐
      │ head F     │   │ head B     │
      │ forward LM │   │ backward LM│
      └────────────┘   └────────────┘
       left reading     right reading
       p(c_t | c<t)     p(c_t | c>t in the mirrored reading)
             ↓                ↓
      boundary bit b^L   boundary bit b^R
```

**Head F** is an ordinary causal character LM: the left half read normally is
just `c₁…c_k`.

**Head B** is a backward LM. The right half, in reading order, is `c_k…c₁` —
so as generation proceeds, head B is being handed the right half's characters
in reverse. Predicting `c_t` under head B means predicting a character from
the text that *follows* it.

**Boundary heads** emit one bit per reading: is there a word break after this
character in the left reading, and in the right reading. Dictionary validity is
enforced at decode time by masking against a trie in both readings. This is the
overhang bookkeeping from `search.py` in another form — the dictionary
constraint does not disappear.

**What does disappear is the palindrome constraint.** Because only the spine is
emitted, every sample is a palindrome by construction. The constraint becomes
an architectural invariant rather than something a search has to maintain, and
the model spends its capacity on readability instead.

## Training

The two heads can be trained on ordinary English, alternating:

- head F on text in normal order,
- head B on the same corpus reversed,

with the backbone shared, so it learns representations of letters that are
plausible read either way. Decoding sums both heads' log-probabilities plus the
two trie masks.

There is no palindrome corpus large enough to train on directly, and none is
needed: the joint constraint is imposed at decode time, not learned.

## This changes the compute answer

The current pipeline in this repository needs no training — the LLM only scores.
This architecture does, which makes it the first part of the project with a real
claim on Argonne.

Two tiers, in order:

1. **Backward GPT-2 (small).** Fine-tune GPT-2 on reversed English and add it as
   a second scorer for the backward half. No new architecture, and it directly
   tests the mechanism: if the +0.368 gap narrows, the diagnosis is right. This
   is a debug-queue-sized job.
2. **Dual-head character model from scratch.** The full architecture above.
   Needs a real allocation, and is only worth requesting if (1) moves the gap.

Doing (1) first is the point. It is cheap, and it can falsify the premise of (2)
before any large allocation is spent.
