# Can an LLM judge coherence here?

Run 25 August 2026. `experiments/oss_judge.py` (pairwise),
`experiments/coherence_scale.py` (absolute), `experiments/presenter_cost.py`
(punctuation), scored by the matching `score_*.py`. Two arms throughout:
`gpt-oss:20b` local and `gpt-oss:120b` via Ollama cloud.

`docs/NORTH-STAR.md` records that four automated proxies in this project have
disagreed with blind human ranking and that none has ever agreed. All four were
*scorers* — perplexity, cohesion — which read a number off a model. A chat
model asked to judge is a different instrument, so the pattern was worth
re-testing rather than assumed.

The answer is that it depends entirely on the shape of the question, and that
the experiment turned up something larger than the question it was asked.

## 1. Pairwise: full power on the easy calls, inverted on the hard one

Ground truth is two batches where blind human judging already returned a
unanimous verdict, so the model's job is to reproduce a verdict rather than set
one. Every item is asked twice with the sides swapped: a model answering by
position agrees with itself 0/n, one reading the text agrees n/n. Both
orientations are scored, so a 7-item arm contributes 14 decisions.

| set | arm | 20b | 120b |
|---|---|---|---|
| seam | calibration (prose vs its own shuffle) | **12/12** | **12/12** |
| seam | k=2, one seam | 11/14 | 10/14 |
| seam | k=4 | 7/14 | 8/14 |
| seam | k=8 | **2/14** | 7/14 |
| grow | calibration | **12/12** | 11/12 |
| grow | seed vs greedily extended | **32/40** | **33/40** |

Both models have full power: prose against its own shuffle is 12/12, the same
as the human judges. Both reproduce the grow verdict at p < 0.001 — they can
see that `and and and and` padding is worse than what it padded.

Both then fail on seams, and 20b fails in the informative direction. At k=8 it
picks the eight-chunk nest 12 times out of 14, with 7/7 self-agreement across
flipped orientations. That is not noise; it is a confident, consistent
preference for the longer text, and the rubric says in as many words that
longer is not better. Blind human judging is 14/14 the other way.

**So: these models detect local degradation and miss discourse-level
incoherence across a join, which is exactly the judgement this project needs.**

## 2. Absolute: the ordering comes out right

Ranking candidates against each other is the wrong shape for search anyway —
a search loop needs a number per candidate. Asking one question per text
("is this real English that vaguely makes sense?", 0–3, with archaic phrasing
and `canon`-for-`cannon` spellings explicitly not penalised) gives a different
answer.

| arm | 20b | 120b |
|---|---:|---:|
| prose, as written | **0.80** | **1.10** |
| catalogued human palindromes | 0.55 | 1.00 |
| our single chunk | 0.35 | 0.65 |
| prose through our presenter | 0.20 | 0.45 |
| k=2 | 0.05 | 0.40 |
| k=4 | 0.00 | 0.35 |
| k=8 | 0.00 | 0.15 |
| shuffle | 0.05 | 0.10 |

Power (prose − shuffle) is +0.75 for 20b and +1.00 for 120b. The ordering is
monotone and identical across the two models, and **the seam effect is visible
here**: chunk > k2 > k4 > k8, running down to the shuffle floor. The same model
that preferred the eight-chunk nest in a pairwise comparison scores it at the
floor when asked about it alone.

That is the methodological result. Pairwise invites a length heuristic that the
instruction does not remove; absolute does not. It also means the scale
disagrees with humans about the *shape* of the seam effect — humans found a
step function at the first join, the scale finds a gradual decline — so the two
agree on direction and not on mechanism.

The first version of this experiment had only the presented-prose arm and
scored it 0.45 against a shuffle at 0.05, which is not a positive control. That
run has been discarded; the raw-prose arm exists because of it.

## 3. What the experiment actually found: our punctuation is the problem

The prose-versus-presented-prose gap above is large, and prose is out of
distribution for a presenter built to punctuate palindrome word runs. So it was
measured directly on the material that matters: the same 26 catalogued
palindromes, identical letters throughout, only the marks varying.

| variant | 20b | 120b |
|---|---:|---:|
| punctuated by hand | **1.81** | **2.23** |
| no punctuation at all | **0.96** | **1.58** |
| `present.py` with length-weighted segmentation | 0.88 | 1.31 |
| `present.py` as shipped | **0.50** | **0.88** |

Paired by text, hand punctuation beats the shipped presenter on 21 of 26 for
20b and 23 of 26 for 120b, ties on the rest, and loses on **none** for either.

**The shipped presenter is below plain spacing on both models** -- by 0.46 on
20b and 0.69 on 120b. It is worse than doing nothing.

The cause is in the segmentation objective. `present.segment` adds a fixed gain
per run it creates, so creating runs is free and the dynamic programme buys
fragments whenever the pieces score anything at all:

```
some men interpret nine memos            -> 4.0  as one run
some men interpret | nine | memos        -> 4.0 + 0.3 + 0.3 = 4.6   chosen
```

which prints as `Some men interpret. Nine, memos.` The same mechanism turns
`A man, a plan, a canal: Panama` into `A, man a plan. A canal panama`, scored 3
and 0 respectively.

Weighting the gain by run length instead removes the incentive and recovers
most of the sensible cuts (`Drab as a fool. Aloof as a bard.`,
`Eva can I see bees in a cave.`). It is worth +0.38 over the shipped objective
on 20b and +0.43 on 120b, and it still does not beat plain spacing on either:
0.88 against 0.96 on 20b, which is a tie, and 1.31 against 1.58 on 120b, which
is not. The safe statement is that **neither punctuation scheme we have beats
putting in no punctuation at all**, and that the shipped one is clearly worse
than both. The fix is real and does not clear the bar.

## 4. Punctuating afterwards, by inference, is what fixes it

The presenter entangles two jobs: it decides where the sentences are, and it
prints the marks. Only the second is required, and doing the first with a
dynamic programme is what buys fragments. So the search returns a bare word
run and a model is asked to add marks and nothing else. It has no constraint to
satisfy — the palindrome is already closed.

**No reply changed a letter. 26 of 26 for both punctuaters.** A model that adds
or drops one has broken the palindrome, and that reply would be discarded and
counted; none had to be.

Scores against bare spacing, four judges, the same 26 texts. mistral is a
different model family and passed the power gate below at +1.20; llama3.1:8b
failed it at +0.15 and is shown to make the failure visible.

| scheme | 120b | 20b | mistral | llama3.1:8b |
|---|---:|---:|---:|---:|
| punctuated by hand | +1.00 | +0.85 | +0.46 | +0.04 |
| **marks by gpt-oss:120b** | **+0.77** | **+0.88** | **+0.42** | +0.08 |
| **marks by gpt-oss:20b** | **+0.65** | **+0.54** | **+0.35** | +0.08 |
| `present.py`, length-weighted | −0.12 | −0.08 | +0.00 | +0.00 |
| `present.py` as shipped | −0.58 | −0.46 | −0.08 | +0.00 |

Absolute means, 120b judge: hand 2.31, llm_120b 2.08, llm_20b 1.96, bare 1.31,
perword 1.19, present 0.73.

**The ranking is identical on all three judges with power** — hand >
llm_120b > llm_20b > bare ≈ perword > present — across two model families.
mistral compresses the differences, which is what a judge that scores this
material between 1.81 and 2.35 has room to do, but it never reverses one.

### A fourth judge, from a fourth family, disagrees

Claude Haiku 4.5 was run blind over the same 196 texts, with the power-gate
items mixed into the same file so it was gated on the pass it was measured on.
It passes at +0.60. It then reverses both headline claims:

| scheme, against bare | 120b | 20b | mistral | haiku |
|---|---:|---:|---:|---:|
| hand | +1.00 | +0.85 | +0.46 | **−0.04** |
| llm_120b | +0.77 | +0.88 | +0.42 | +0.42 |
| llm_20b | +0.65 | +0.54 | +0.35 | **−0.04** |
| perword | −0.12 | −0.08 | +0.00 | +0.19 |
| present | −0.58 | −0.46 | −0.08 | **+0.15** |

Haiku puts the shipped presenter slightly *above* plain spacing and hand
punctuation slightly *below* it. Only the llm_120b result survives.

There is a reason to doubt its resolution, and it is post-hoc, so it is
labelled as such: Haiku rates **0.70** of shuffled word salad at 2 or above,
where both gpt-oss judges rate **0.00** of it that way. It is much more lenient
everywhere — 0.85 of prose at 2+ against gpt-oss's 0.20 — so its +0.60 gap is a
small shift near the top of a scale it is not using the bottom of. That is a
better gate criterion than the one the gate uses, and it was not specified in
advance.

Because explaining away a disconfirming judge after seeing it is exactly the
failure this document exists to name, the contested comparisons were sent to
blind pairwise judging — the protocol that returned 12/12 calibration and 14/14
on seams, with identical letters on both sides so length and vocabulary are
held exactly. See `## 5` below.

So post-hoc LLM punctuation is the first scheme here that beats doing nothing,
and it lands within 0.23 of hand-written. The local 20b is close enough to the
cloud 120b that this needs no cloud inference.

**No self-preference.** The 20b judge rates the 120b's punctuation highest
(1.85, above hand at 1.81) and its own lowest (1.50). If either model favoured
itself the cross would show it; the bias runs the other way.

The remaining gap to hand is legible and looks like prompt work rather than a
limit: the model over-uses em-dashes (`a jar of gum—nit a`), drops terminal
marks (`May a moody baby doom a yam.` wants a question mark) and misses a
proper noun (`bottoms up, mac`). Where it beats hand it does so on judgement,
not luck: `Are we not drawn onward? We few, drawn onward to new era.`

### Most models cannot do this job, and they fail silently

llama3.1:8b scores the six schemes at 2.00, 2.00, 2.00, 2.00, 2.04 and 2.08 —
a range of 0.08. Read at face value that is "no punctuation scheme differs from
any other", which is a finding. It is not what happened.

`experiments/judge_power.py` gates a candidate before its verdicts count: 20
WikiText spans against those same spans shuffled, raw, with a judge already in
use as the positive control.

| candidate | prose | shuffle | gap | |
|---|---:|---:|---:|---|
| mistral:latest | 2.20 | 1.00 | **+1.20** | usable |
| gpt-oss:20b | 0.85 | 0.00 | **+0.85** | usable — the positive control |
| deepseek-r1:8b | 0.70 | 0.20 | +0.50 | borderline, on the threshold |
| gemma3:4b | 2.00 | 1.75 | +0.25 | no power |
| llama3.1:8b | 2.00 | 1.85 | +0.15 | no power |

Three of five local models cannot separate English from word salad, and two of
them would have reported all six punctuation schemes as equal. The threshold is
a judgement call rather than a derived quantity, which is why deepseek-r1 is
called borderline instead of rounded to one side.

Ollama's cloud entries for other families are gone — kimi-k2.5 retired 31 July
2026, deepseek-v3.1 retired 15 July 2026, qwen3.5:397b needs a paid
subscription — so the cross-family judge is mistral, run locally.

## 5. Blind pairwise settles it, for the original claim

Two blind judges, sides flipped between them, over 26 present-vs-bare pairs, 26
hand-vs-present pairs and 8 calibration items. Both sides of every palindrome
pair carry identical letters in identical order, so the only thing varying is
the marks.

That identity is why pairwise is the right tool here and was the wrong one for
seams. The failure in §1 was a length heuristic; with both sides the same
length, same words, same order, there is no length for it to bite on.

| comparison | preferred | p | agree | left picks |
|---|---:|---:|---:|---:|
| calibration: ordered prose | **16/16** | 0.0000 | 8/8 | 8/16 |
| `present.py` over bare spacing | **3/52** | 1.0000 | 23/26 | 27/52 |
| hand over `present.py` | **52/52** | 0.0000 | 26/26 | 26/52 |

Calibration is 16/16, so the protocol has full power. Left-pick rates are 27/52
and 26/52 — nobody is answering by position.

**Bare spacing beats `present.py` 49 times in 52. Hand punctuation beats it 52
times in 52, unanimously, with the judges agreeing on every single item.**

So the gpt-oss and mistral verdict stands and Haiku's absolute scale is the
outlier on this comparison. The leniency explanation offered above was post-hoc
when it was made; it now has independent support, since a judge that rates 70%
of word salad as English is exactly the judge that would fail to see a
difference blind pairwise finds 52/52.

This does not make Haiku wrong in general, and it does not retract the gate.
What it retracts is the idea that passing a prose-versus-shuffle gate makes a
judge usable on *this* material. The better criterion — what fraction of word
salad does it call English — is now measured for every judge, and it should
have been the gate from the start.

## What this licenses

- An LLM judge is usable as an **absolute** coherence score for filtering
  search candidates. Its ordering is correct and stable across two model sizes.
  It is not usable as a pairwise ranker on this material, where it inverts.
- It still may not decide. It disagrees with blind human judging about the
  shape of the seam effect, and the standing rule in `docs/NORTH-STAR.md` —
  proxies may filter and propose, blind judging decides — survives this
  experiment rather than being overturned by it.
- **Punctuation belongs after the search, not inside it, and a model should do
  it.** The shipped presenter is worse than plain spacing; a model given the
  finished word run and told to add only marks beats plain spacing by +0.77 and
  never once broke the palindrome.
- **A share of this project's readability problem is its own punctuation, not
  its search.** Every judged experiment here ran text through `present.py`.
  Comparisons where both sides were presented are unaffected — the seam result
  is one of these and stands — but every absolute claim about how our output
  reads has been measuring the presenter as much as the search.

## What it does not license

The absolute scale is compressed: nothing scored 3 except in the hand-punctuated
arm, and raw prose only reaches 1.10 because it too is stripped to letters and
spaces before scoring. Differences near the floor (20b's k4 and k8 are both
exactly 0.00) are not resolvable.

The hand punctuation is ours. It was written to match how these palindromes are
normally printed, and every variant is asserted to have identical letters, but
a judge scoring "text a person punctuated" above "text a program punctuated" is
not a surprising outcome and part of the 1.35 gap may be fluency of intent
rather than correctness of marks.

26 items is small, and the three variants of each text are not independent.
