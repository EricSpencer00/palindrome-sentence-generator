# How to find a palindrome that reads

A long-form treatment of the task's search structure, its regimes, and what
bounds the output. Source material for the four-page paper in this directory;
longer than any venue will take, and kept because the conference version has to
throw most of it away.

The organising claim is that readability on this task is bounded by the
material rather than by the search, which is why §3 to §6 are about structure
and §7 is only the account of why the structure runs out. An earlier version of
this document led with §7 and it was the wrong emphasis: the bits figure is a
range with dependencies, and it explains a wall that seven independent routes
had already located without it.

---

## 1. The task


A character-level palindrome is text whose letters, once case, spacing and
punctuation are stripped, read identically in both directions. `A man, a plan,
a canal: Panama` is the canonical example. The constraint is trivially
checkable, exactly satisfiable, and orthogonal to whether the text means
anything.

That orthogonality makes the task an unusually clean instrument. In most
constrained-generation benchmarks, satisfying the constraint and writing well
are entangled: a lexically constrained decoder that is forced to use the word
*sledgehammer* may satisfy the constraint by writing an awkward sentence, and
the resulting score mixes the two. Here they separate completely. A dictionary
search over letter remainders guarantees validity; every candidate it emits is
a palindrome whether or not a language model is involved. Whatever the model
contributes, it contributes to readability alone.

Whether a system can produce palindromes was settled long ago: Hoey's 1984
program could, and Papadopoulos et al. (2015) later showed how to enumerate all
of them obtainable from an n-gram corpus. The question here is why the output
does not read, and whether that failure is a fact about the systems or a
fact about English.

## 2. Prior work on the task itself


Dan Hoey wrote a program in 1984 that extended `A man, a plan, a canal:
Panama` to 540 words, using the Unix spelling dictionary. Hoey's own note said
a better word list and a smarter program should get ten times as far. Peter
Norvig took that up in 2002, extracted a 126,342-word dictionary from the Moby
word list, and produced a palindrome of 21,012 words and 90,439 letters.
Norvig documented the algorithm in detail; this repository uses it, and the
description below is his.

The result is also, by its author's own description, a noun list rather than
prose. That is the observation the present work is about. It is repeated
everywhere in the recreational-linguistics literature and in the introduction
of the one peer-reviewed paper on the subject, and nobody has attached a
number to it.

Papadopoulos, Roy, Régin and Pachet (IJCAI 2015) give the combinatorial
treatment: a graph structure from which all palindromes obtainable from an
n-gram corpus can be generated in linear complexity, handling the coupling
between the character level (where the constraint lives) and the word level
(where the corpus lives), with word probabilities biased from an auxiliary
corpus to steer the semantics. Their framing, that long palindromes tend to be
less meaningful, is the thing measured here. Their contribution is feasibility
and completeness. This one takes feasibility as given and asks
about the shape of the feasible set.

A separate line generates Chinese palindrome poetry (CPPGM, ICONIP 2020), where
the constraint operates over characters in a language whose orthography makes
the problem different in kind. It is cited for coverage rather than as a
baseline.

## 3. The overhang formulation


Word-by-word left-to-right generation cannot enforce a palindrome, because the
last character of the text is determined by the first. Neither can an
autoregressive language model, for the same reason, and this is a specific
instance of a general limitation that the reversal-curse and factorization-curse
literature has since made precise: a model trained on one factorization of the
joint distribution does not thereby acquire the others.

The construction is therefore two-sided. Text is grown from both ends inward.
At every point, one side owes the other a run of letters, the **overhang**.
Writing `rats live on` on the left leaves the right side owing `no` at its
start, because those letters must appear reversed at the end. Adding `no evil
star` on the right pays that debt and creates a new one on the other side. The
palindrome closes when the overhang is itself a palindrome, which becomes the
centre.

```
left:  rats live on              overhang: "no"
right:            no evil star
                        closes when the overhang reads the same both ways
```

The overhang is a sufficient statistic for feasibility. Two partial palindromes with different text but the same
overhang have exactly the same set of legal continuations. This is what makes
the constraint checkable incrementally at all, and it is the property any
constrained decoder for this task must find, in the same sense that a
finite-state constraint's state is what makes constrained beam search possible
(Anderson et al., 2017; Post and Vilar, 2018).

It also answers the question of why the principled machinery for conditioning
on hard constraints (sequential Monte Carlo steering
(Lew et al., 2023; Loula et al., 2025), grammar-aligned decoding (Park et al.,
2024), MCMC-based constrained sampling (Anaya Gonzalez et al., 2025)) was not
used directly. Those methods condition on a constraint whose satisfiability is
decidable from a prefix through some tractable state. For a palindrome, the
prefix alone tells you nothing: any prefix is extendable. It is the overhang,
which is a function of both ends, that carries the state. Once the search is
formulated over overhangs rather than prefixes, those methods apply and would
be a sensible next system. Nothing in this paper's result depends on which of
them is used, because the result is about the search space, not the searcher.

## 4. Three regimes

The space behaves differently at different lengths, and choosing the wrong
method for a length is the most common way to waste a budget on this task.
Three regimes, with a boundary that is measured rather than assumed.

### 4.1 The short regime, and why "enumerable" was the wrong word

A 30-minute 32-way walk returns 2.55M distinct palindromes, and this is the
only regime that yields novel grammatical sentences: `non academia aimed a
canon`, 22 letters, appears in no corpus we checked.

Calling it *enumerable* was wrong and this document said so for a while.
Completeness is checkable — run a band at two node budgets and see whether the
count moves — and at a 60-word vocabulary only 12-16 letters finishes, while at
200 words nothing does. The ratio sits flat at 2.00 from 17 letters to 46, so
the walk is no closer to finishing at 20 letters than at 45. What separates the
bands is not completeness but the density of readable material: 4.4% of results
pass `sentence_like` at 12-16 letters, 1.7% at 17-21, 0.064% at 22-26, and zero
from 27 up. A seventy-fold collapse, then a floor. Full record in
`experiments/RESULTS-regime.md`.

A second correction came out of the same run. The enumerator defaults to at
most 12 units, and a short-word vocabulary cannot reach 40 letters within that,
so the long bands returned nearly nothing and the emptiness looked like a
property of the space. Lifting the cap turns 17 results into 87,693 at 32-36
letters.

Best-of-all over an enumeration is not the same object as best-of-N over a
beam's output, and the oracle bound in §6 says nothing about it, because that
bound is over a fixed proposal distribution and an enumeration has none. What
a language model does here is not steer the search but choose among everything
that exists — which is the one job in this project it has been unambiguously
good at.

The regime is also where the record lives. `A man, a plan, a canal: Panama` is
24 letters and `Sir, I demand, I am a maid named Iris` is 30. That is not a
coincidence about human patience; it is where the feasible set still has
readable members in it.

### 4.2 Not enumerable, above about 30 letters

With a 14k-unit inventory the tree branches roughly 14k wide at every closure.
A time-budgeted walk therefore returns a deep prefix of one corner of the space
and reports the count of what it found as though it were coverage. The failure
is silent: the walk produces millions of valid palindromes and no warning that
it has seen a vanishing fraction of the space.

That is why the regime needs an acceptance test rather than a candidate count.
Ours is **canon recall**: the fraction of catalogued palindromes a walk
rediscovers. A search that cannot find `a man a plan a canal panama` is not
enumerating a space that contains it.

| seeded vocabulary | depth | canon recall |
|---|---|---|
| 83 canon-seeded words | 4 | 10/18 |
| 14,000 units | — | 0/27 |

The test generalises to any constrained search with a catalogue of known
solutions, and it is cheap. We would rather have had it earlier.

### 4.3 Assembly, at any length

Units that pay the constraint internally compose, which converts the joint
search into a selection problem over independently verifiable pieces (§5). What remains is unit supply, and that is where the rest
of this document goes.

| regime | method | what limits it |
|---|---|---|
| ≤ 30 letters | enumeration | nothing; the space is walkable |
| above 30 letters | beam search | readability, not validity |
| any length | assembly | supply of readable units |

The practical advice is the table. Below 30, enumerate and check with canon
recall that you really are. Above it, do not expect a search to return readable
text, and assemble instead.

## 5. Composition


Length is separable from the constraint, and this is the one structural result
that changes what is possible.

Units that pay the constraint internally nest like brackets. Given units
L1…Lk with mirror-halves Rk…R1, the concatenation

```
L1 L2 … Lk  CENTRE  Rk … R2 R1
```

is a palindrome by construction, for any k. `paragraphs.py` asserts it in both
`assemble` and `render`.

What this buys is not length. Free-running search reaches length easily: the
deployed `/api/generate` endpoint closes a valid 958-letter, 309-word
palindrome with no unit boundaries in 14 seconds, roughly twice what a
paragraph needs, with an empty centre and only four incidentally
self-palindromic words in 309. An earlier version of this document said
composition "removes length from the problem", which was wrong in a way worth
recording: length was never the binding problem for free-running either.

What composition buys is the ability to **place verified text**. In a
free-running search every word is chosen by the search, so no externally
checked material can be inserted, and readability is whatever the scorer
happens to produce over a proxy that §9 shows cannot rank. Under composition
the mirror is discharged inside each unit, so units can be vetted one at a time
and the remaining problem — which units, in what order — carries no letter
constraint at all.

There are exactly three ways to pay a mirror across a multi-sentence text, and
two of them are closed.

**Free-running.** One search, one palindrome, sentence boundaries wherever they
fall. Structurally the right answer: nothing repeats, no unit is independently a
palindrome, nothing is quoted. It is also the route measured to fail. The 40–60
letter band was searched exhaustively enough to be conclusive: 880k closures,
112,523 of the first 300,000 using only dictionary words, and none within three
unattested joins of reading. Closure is not what limits this route; the
deployed system reaches 958 letters. A construction that cannot produce 40
readable letters will not produce 500, so the length it reaches is beside the
point.

**Self-palindromic units.** Any concatenation of units that are each palindromes
reverses into those same units in the opposite order. The whole is a palindrome
only when the unit *sequence* is itself a palindrome, so unit k must equal unit
n+1−k. Repetition is forced by the algebra, not chosen for effect, and it is
forced whether the unit is one sentence or five. This is why a refrain is the
form that long human palindromic poetry takes.

**Mirror-pairs.** The unit is two halves that spell each other backwards, placed
at mirrored positions. The sequence still mirrors and nothing repeats, because
what returns at position n+1−k is the other half. This is the only structure
that survives a specification forbidding repetition, and the entire cost lands
on the material: both halves have to read, and pairs where both halves read are
the scarcest thing in the project. 29 such pairs exist in this repository's
inventory. All 29 are catalogued palindromes.

## 6. Construction routes


A geometric thinning rate predicts a wall, not a slope: constructions should
work at short lengths, degrade fast, and stop entirely somewhere. Seven
independent construction routes were run, and all of them stop.

**Language-model-scored beam search.** The Norvig search with a Zipf scorer.
Two configurations were run at matched budgets: GPT-2 reranking the finished
candidates, and GPT-2 rescoring the beam during the search so that unfluent
branches die before consuming budget. In-loop is measurably more fluent
(−2.297 against −2.382 best score) at roughly 31× the cost. Both arms are 12/12
valid, because validity is the search's and not the model's, and both produce
text that is locally fluent and globally meaningless. They are one route in two
configurations, not two routes.

**Exhaustive enumeration.** At 28 letters or fewer the space is walkable: 2.55M
distinct palindromes in a 30-minute 32-way walk, and best-of-all beats
best-of-N over any beam pool because there is no proposal distribution for the
order statistics to apply to. This is the one regime that yields novel
grammatical sentences — `non academia aimed a canon`, 22 letters, in no corpus.
Above roughly 30 letters "exhaustive" stops meaning anything: with 14k units
the tree branches 14k wide at every closure, and a time-budgeted walk returns a
deep prefix of one corner. The acceptance test is canon recall — how many of
the catalogued palindromes a walk rediscovers — and it reaches 10/18 at 83
canon-seeded words and depth 4, and 0/27 at 14k.

**Mining attested corpus n-grams for mirror-pairs.** 272k attested bigrams
yield 3,894 mirror-pairs; 131 have both halves attested. At three-word halves,
27. At four-word halves, where prose would start, **0 out of 34,688 attested
4-grams**. That single row is the sharpest number in the project.

**Closed-form reversible-word chains.** 392 reversible words, walked
exhaustively: 57 joins attested on both sides of the mirror, 4 novel chains,
all instances of one template.

**LLM authoring.** 40 verified units, 37 novel, about 3 usable. A second round
added 18 more and made the assembled paragraph *worse* under blind judging.
Unit count is not the lever.

**A sharded vocabulary walk.** 1,915 pairs in 460 seconds, of which roughly
none read. Throughput scales; yield does not.

**Authoring one half and segmenting the other.** 148 authored sentences, 12
with any spellable mirror, **0** whose mirror reads. The failure has a cause
worth stating: the opening word of one half is the closing word of the other,
so a half written to start well ends the other one badly. Writing better
sentences cannot fix a constraint that runs the other way.

Two more levers were measured directly and also fail.

**Reranking is exhausted.** Best-of-N is an upper bound on what any reranker
can achieve over a fixed candidate pool. Over 2000 palindromes scored with
Qwen2.5-0.5B, the curve flattens: 24 → 2000 buys 0.168 nats per token and the
last thousand samples buy 0.007. The winner sits about three standard
deviations above the mean, which is what order statistics predict for a roughly
normal distribution. There is no tail of secretly coherent palindromes at
larger N.

**And that bound was measured inside a corridor, which does not rescue it.** Of
those 2000 palindromes, 1975 opened with the same five words and 1983 closed
with the same five, using 1317 of 30000 available words. The cause was the
beam's only source of seed variation: a jitter term too small to reorder the
leading candidates against a frequency scorer spanning several units. Raising
it takes 3 distinct openings to 200 and quadruples the vocabulary in use, at no
cost in closure. Readability falls monotonically, by 0.49 nats per token from
the narrowest arm to the widest. Unanchored from word frequency the search
reaches for `maserati` and `merseyside`, and the text gets worse.

One number cuts the other way and is reported because it does: best-of-200 at
diversity 1.0 (−6.790) beats best-of-2000 in the corridor (−6.810). A single
arm maximum is a noisy statistic and the arms are not monotone in it, so this
is suggestive rather than settled.

## 6a. What the punctuation can and cannot do

Punctuation and casing are invisible to `normalize`, so where the sentence
breaks fall is a free variable the constraint never prices. The project had
been spending it on one heuristic and never searched it.
`experiments/punctuation_search.py` searches it: a dynamic programme over cut
positions, each candidate run scored by the strongest Brown-tag test it passes
and given the mark that suits what it claims.

It renders better. The same 196-word palindrome, fixed-stride against searched:

    A  Fired now be way as to note. Filed a me by me by awe. Nodes us if
       it is also similar. To i honor of test is a.
    B  Fired now be? Way as to? Note filed a me. By me by? Awe nodes us?
       If -- It is also similar. To i honor of test is a.

262 runs across 8 texts reach the sentence tier, and some are real
constituents: `may be not so`, `it in use but not`, `use of sign in even
after`.

It finds nothing that was not there. Against the text's own words shuffled —
the control that holds vocabulary, word length and tag ambiguity fixed and
destroys only order — the word order buys +0.008 (t=+0.63, n=48). The identical
procedure detects real English's order at +0.115 (t=+7.69), and the result
holds across eight parameter cells.

Blind judging agrees, after two confounds. A batch scored 20/20 and was a false
positive: the search has a fingerprint — `for on is` opens 9 of 40 texts, `in
or of` closes 11 — and shuffling destroys it, so the judge was recognising the
generator rather than reading.

The final batch removes the last objection, which was that the judge and the
experimenter were the same model. Two annotators with no access to the key, the
repository or the hypothesis, the second shown every pair with its sides
exchanged so that agreement on the text cannot be manufactured by position:

| | calibration (n=8) | palindrome (n=20) |
|---|---|---|
| judge 1 / judge 2 accuracy | 8/8 · 8/8 | 10/20 · 10/20 |
| chose the left-hand passage | 4/8 · 4/8 | **20/20 · 20/20** |
| agreement on the text | 8/8 | **0/20** |
| Cohen's kappa | **+1.000** | **0.000** |

Perfect and unbiased on calibration; on the test items both defaulted to
position on every single one, which puts accuracy at the base rate and text
agreement at zero. Two annotators falling back entirely to position, having
just shown no position preference on controls, is what a pair carrying no
distinguishing signal looks like. Full record in `runs/punct/RESULTS.md`.

Two measurements with demonstrated power on controls, agreeing in the negative.
That is the rarest event in this document, and it is worth noting that it took
a self-shuffle control, a fingerprint control and a position-bias split to get
there; each of the three, alone, would have produced a publishable-looking
number that was wrong.

## 7. Why the ceiling is there

The routes in §6 stop in the same place, and this section is the account of
why. It has two halves. The first involves no language model and carries most
of the argument; the second prices the same fact in bits and is reported as a
range because that is what it is.

### 7.1 The model-free half

Take English prose, strip it to letters, and segment both those letters and
their reversal into one vocabulary by one algorithm. Then count the fraction of
letters that land inside a real word of three letters or more.

| direction | coverage |
|---|---|
| forward | 0.88 – 0.92 |
| reversed | 0.48 – 0.55 |

Stable across span lengths from 20 to 120 letters, across three segmentation
objectives and across four models, because no model is involved in it.

**About half the letters of reversed English cannot be placed in a dictionary
word at all.** Not rare words; no words. That single fact explains the shape of
everything in §6 without any information theory, and it is the number to quote
if only one is going to be.

The rest of this section is the same observation measured under a language
model, which adds precision and a comparison to English's own entropy, and
which is worth having provided its conditions travel with it.


### 7.2 What is being priced

A palindrome of 2k letters has k free letters. The other k are determined. Each
free letter is placed twice: once in the reading that runs left to right, once
in the reading that runs right to left. Both readings have to be English.

So take real English prose, strip it to letters, and score those letters two
ways under one model and one vocabulary:

- **forward** — the letters in their own order, re-segmented into words;
- **reversed** — the same letters in the opposite order, re-segmented into
  words.

The forward number is what English costs. The reversed number is what the same
letters cost when they are required to be English in the other direction too.
The difference is the price of the mirror, per free letter.

This is a descendant of the Shannon (1951) estimate of the entropy of printed
English and of the Cover and King (1978) gambling refinement of it, in method
as well as in units. Shannon's number is what one direction costs. The quantity
here is what the second direction adds.

### 7.3 The estimator

`experiments/mirror_cost.py`. Corpus: wikitext-2, headings removed, spans cut
at word boundaries to at least a target letter count, each span carrying its
own denominator. An exact count was tried first and abandoned: it silently
selects for word-length compositions that sum to the target, which favours
short words (mean word length 4.82 against 5.04 in an unfiltered sample at the
same target). Since the price is a difference between two per-letter figures
over one letter sequence, the denominator only has to match within a span.
Vocabulary: the repository's 52,927-headword
lexicon, filtered by `shortwords.is_real_short`, which removes the one- and
two-letter strings a frequency list contains but no reader accepts. One
vocabulary serves both directions, so nothing in the comparison can come from
one side having more words available.

Segmentation is by dynamic programme over letter positions under three
objectives, so the result can be checked against each:

- `unigram` — maximise summed Zipf frequency less a fixed per-word cost;
- `fewest` — minimise the number of units, ignoring frequency;
- `greedy` — longest match left to right, with backtracking.

Both directions are re-segmented, including the forward one. The obvious
comparison — original text with its own spacing against reversed text
re-segmented — confounds the direction with the segmenter, because optimal
re-segmentation under a word model is not free. Running the identical procedure
on both sides removes that. The natural-spacing score is reported as well, and
the gap between it and the re-segmented forward score is what re-segmentation
alone costs.

### 7.4 Segmentation fallback

The first version of the estimator required both directions to segment fully
into the vocabulary and discarded spans where either failed. That estimator has
no data: at twenty letters, **none** of a first sample of thirty reversed spans
segmented into the dictionary at all, against about two thirds of the forward
ones. Reversed English is not merely harder to segment; it usually cannot be
segmented.

Both directions are therefore segmented over the vocabulary plus the
twenty-six bare letters. Segmentation then always succeeds, and the penalty for
a bad one is set by the language model rather than by a constant chosen here.
The forward direction gets the same escape and rarely uses it, which is the
control.

Dictionary coverage — the fraction of letters landing inside a real word of
three letters or more — is reported alongside, because it separates the part of
the gap that is words the model dislikes from the part that is not words.

### 7.5 Result

Bits per free letter, 150 spans per cell, standard errors 0.04 to 0.10.

| model | objective | L=20 | L=30 | L=40 | L=60 | L=80 | L=120 |
|---|---|---|---|---|---|---|---|
| DistilGPT-2 | unigram | 3.38 | 3.20 | 3.03 | 2.90 | 2.88 | 2.79 |
| | fewest | 3.35 | 3.15 | 3.00 | 2.84 | 2.81 | 2.69 |
| | greedy | 2.67 | 2.52 | 2.36 | 2.26 | 2.24 | 2.16 |
| GPT-2 small | unigram | 3.44 | 3.27 | 3.16 | 3.04 | 3.03 | 2.96 |
| | fewest | 3.40 | 3.22 | 3.10 | 2.97 | 2.95 | 2.84 |
| | greedy | 2.71 | 2.57 | 2.47 | 2.39 | 2.37 | 2.29 |
| GPT-2 medium | unigram | 3.60 | 3.47 | 3.30 | 3.19 | 3.16 | 3.09 |
| | fewest | 3.56 | 3.42 | 3.28 | 3.15 | 3.10 | 2.99 |
| | greedy | 2.83 | 2.74 | 2.63 | 2.54 | 2.50 | 2.42 |
| GPT-2 large | unigram | 3.60 | 3.44 | 3.29 | — | — | — |

GPT-2 large was cut short at three lengths and Qwen2.5-0.5B was not run; the
sweep was stopped because it was heating the machine it ran on, not because of
anything it returned. Everything below is over the 54 complete cells.

**The price is around three bits per free letter**, and it declines gently and
monotonically with span length, by about 0.5 bits from L=20 to L=120 under each
objective. That is a boundary effect: short spans have proportionally more text
next to an edge the model has no context for.

**The segmentation objective moves the number by up to 0.7 bits, and it moves
it through the forward direction, not the reversed one.** Forward bits per
letter range from 2.43 (unigram) to 3.21 (greedy); reversed bits per letter
stay within 5.39–5.87 across every objective, length and model. The reversed
reading is already at the ceiling of what segmentation can do for it, which is
the same fact as the coverage figure.

**About half the letters of reversed English cannot be placed in a real word.**
Coverage is 0.88–0.92 forward and 0.48–0.55 reversed, stable across lengths,
objectives and models. Half the mirror's letters are not improbable English;
they are not words. This half of the effect is model-free, and it is why the
reversed column barely moves.

**The price grows with model capacity.** At the unigram objective and L=40 it
is 3.03 (DistilGPT-2), 3.16 (GPT-2 small), 3.30 (GPT-2 medium), 3.29 (GPT-2
large). The mechanism is in the forward column: a larger model finds forward
English cheaper (2.56 → 2.38 bits per letter at L=80) and does not find
reversed English cheaper, because there is no reading of reversed English for
it to find. The figure is therefore not an artifact of a small 2019 model, and
a stronger scorer would report a larger price rather than a smaller one. The
caveat is that all four models share a family lineage, so this is a
within-family observation until a second family is run.

Natural-spacing English scores 1.43–1.88 bits per letter over the same spans,
somewhat above the 0.6–1.3 bits per character Shannon obtained from human
predictors, which is what a small model should give.

The repository's previously recorded figures — 1.63 forward, 4.92 reversed,
3.296 for the difference — were prose in a README with no script behind them.
They reproduce at the short end under the unigram objective and drift to 2.8–3.1
at length. **The honest statement of the result is a range: 2.2 to 3.6 bits per
free letter depending on model and segmentation objective, and 2.8 to 3.1 under
the objective most favourable to the mirror at lengths where prose would live.**
The single number 3.296 should not be quoted again without its conditions.

### 7.6 What the price implies

If free letters paid independently at Δ bits, the probability mass an
unconstrained English model assigns to strings that also read backwards would
fall by a factor of 2^Δ for each additional free letter: about 8 at Δ = 3, and
about 12 at the top of the range.

**Independence is an assumption and it has not been tested here.** Letters in
English are not independent, and the palindrome constraint couples them in a
specific way, so the true rate could be either side of this. What follows is an
extrapolation that is consistent with what §6 found, not a second measurement.
The evidence for the wall is the seven routes and the zero four-grams; the
arithmetic is the account that fits them.

The repository's own documentation said the feasible set "thins by roughly 10×
every three letters." That is wrong by two orders of magnitude at three letters
and follows from misreading the exponent: 2^3.296 = 9.8 is per letter, not per
three. The corrected rate is what the rest of the argument uses, and it is far
more pessimistic. A hundred-letter palindrome is fifty free letters at roughly
three bits each: about 150 bits of constraint, against text that carries
perhaps 160 bits of English in the same span. That comparison is itself loose,
since the 160 is natural-spacing forward English and the 150 is a difference
between two re-segmented readings; it is an order-of-magnitude statement and
should not be quoted more precisely than that.

## 8. Directionality


Every construction builds one half by appending words and the other by
prepending them. The repository's original claim was that the prepended half
reads measurably worse, in both growth directions, which would locate the cost
in backward construction rather than in position.

**That claim does not survive re-measurement and is retracted.** The gap is
+0.443 per letter and −0.824 per token on the same 24 seeds. The prepended half
does not read worse; it has shorter words, and `lm_score` divides by letters.

What survives is a different result. A GPT-2 small fine-tuned on reversed
word-aligned tokens (wikitext-103, 37.9M tokens, 3000 steps) reaches validation
loss 4.060 against the forward fine-tune's 3.139 on identical corpus, schedule
and step count: **0.921 nats is the cost of modelling English backwards**, with
data and budget controlled. Used as a scorer, that model beats its matched
forward control at every mixing weight tried (+0.177, +0.080, +0.157 per token)
and beats the plain baseline by +0.423 at the best weight, closing 24 of 24
seeds throughout. Per letter, the same arm is the worst in the table.

This connects to the reverse-training and right-to-left LM literature
(Golovneva et al., 2024; the LEDOM reverse LM, 2025) and to fill-in-the-middle
training (Bavarian et al., 2022; Donahue et al., 2020), because a palindrome is
a two-sided infilling problem in which the two sides constrain each other
through a channel that is not the text.

## 9. Proxies


Four automatic proxies were checked against blind judging with real-prose and
word-salad controls in the same batch.

| proxy | as a filter | as a ranker |
|---|---|---|
| GPT-2 score | sound: selects nothing a judge rejects | failed, three times |
| `reads_as_attested` | sound: 79% accepted against 38% | not used |
| thematic cohesion | sound: finds the theme | failed: sizes the output wrong |
| word frequency | no signal (4.34 against 4.06) | — |

Four proxies, zero agreements on ranking finished text. Every result in the
repository that ranks anything was settled by a blinded batch rather than by a
number, and the recommendation that generalises is that papers should state
**which role a proxy plays**. Filter and ranker are different claims carrying
different validation burdens, and the field routinely validates one and uses
the other.

Two demonstrations of how a proxy gets gamed come with this, and they are the
part that transfers.

**The normalisation.** `lm_score` divides total token log-probability by
letters. Longer words cost fewer tokens per letter, so a policy that prefers
them scores better without writing better. An evolution-strategy run over five
scorer parameters found this in ten generations: it raised the length
coefficient sevenfold, cut the frequency coefficient by two thirds, dropped
tokens-per-letter from 0.329 to 0.265, and won 24 of 24 paired seeds by +0.298.
Per token the same texts are 0.673 *worse*, losing 24 of 24. A GPT-2 anchor was
in place and did not catch it, because the anchor used the same per-letter
normalisation. **A check against a stronger model is only a check if it is
normalised differently from the thing being optimised.**

**The judge inherited the exploit.** A linear judge fit to predict the
per-letter score reached 0.821 pairwise agreement within the real scorer's arm.
Refit against the per-token score, the same features reach 0.595 — barely above
chance — and `word_len_mean` flips sign from +0.717 to −0.421. Most of what the
per-letter judge predicted so well was word-length variance.

This belongs in the same conversation as length bias in reward models and in
LLM-as-judge evaluation, and the remedy is the same family as length-controlled
evaluation (Dubois et al., 2024): report the metric under a second
normalisation, and adversarially check a metric *before* pointing an optimiser
at it.

## 10. Evaluation


The judging design is sound and the reporting of it is not, and the second half
of that sentence is a limitation the paper has to carry.

Design: a batch is scored with the key withheld until afterwards
(`runs/blind_key.json`), with real-prose and word-salad calibration controls in
the same batch, so a judge that cannot separate prose from salad invalidates
its own verdicts on the items between them. That is the right shape.

What is missing is everything about the judge. The verdict files record one
verdict per item with no annotator identity, no annotator count, no agreement
statistic and no verbatim instructions, and no script produces them. **Every
"blind judging" result quoted in this repository is n=1 from an undocumented
judge**, and any venue that takes human evaluation seriously will say so. The
two honest paths are to run a multi-annotator batch and report agreement, or to
relabel every such result as single-judge blinded assessment with calibration
controls and cite the LLM-judge bias literature if the judge was a model. The
paper takes the second and says so.

The stricter judging rubric, applied to the unit bank, gives roughly 1% of
mirror-pairs with two readable halves (3 of 240), against whole short
palindromes passing as sentences far more often. That is the empirical form of
§5's conclusion: the cost bounds unit *length* rather than palindrome length.

## 11. The specification


`docs/NORTH-STAR.md` states the target as nine conjunctive criteria: at least
100 words; the whole text a letter-level palindrome; at most one sentence
independently a palindrome; no sentence repeated; the halves sharing no
sentence; every sentence grammatical; a nameable subject; judged coherent
against both real-prose and disparate-real-sentence controls; and novel.

The interesting part is not the list. It is that each criterion exists because
the corresponding shortcut was actually taken in this repository, with the
iteration recorded:

- **Word-order palindromes** (iteration 49). The sentence sequence mirrors, the
  letters do not. It pays nothing per letter, and it was the endpoint's default
  for about forty iterations.
- **Self-palindromic units** (iterations 105–121). Passes a palindrome check;
  reverse it and every sentence returns unchanged, so the mirror does no work.
- **Optimising a proxy and reporting it as quality** (iterations 96–97).
- **Borrowed material presented as generated.** Every readable unit in the
  assembled paragraph is a catalogued palindrome.
- **Preferring the version that scores better by not attempting the
  constraint.** The refrain beat the pair construction under blind judging, and
  it read better because it was not doing the hard thing.

Constrained-generation projects drift toward easier constraints, and the drift
is structural rather than careless: at every step the easier constraint scores
better on the metric in hand. A specification that names the shortcuts is a
cheap defence, and `tests/test_north_star.py` holds the criteria as failing
tests so they cannot be quietly dropped.

Against those criteria the shipped system passes six of nine. It fails
grammaticality, subject, and coherence. Those three are one problem: units long
enough to be sentences.

## 12. Provenance


The paragraph endpoint has two modes. `?source=catalogue` assembles catalogued
palindromes, and reads better. The default assembles 28 mirror-pairs walked out
of the vocabulary by this project's own search, and reads worse:

> War dog. Rob a log. No cotton. Fired now. Went on. Trade man. … Not new.
> Wonder if. Not to con. Go labor. Go draw.

101 words, every unit generated here, and a run of two- and three-word
fragments. The catalogued version is better text and is not the thing being
built. `is_novel_palindrome`, over 160 known entries, is what keeps the
distinction enforceable rather than rhetorical.

This is a small ethics point and a real one for any system that assembles text
from a corpus: the assembly can be the contribution while the sentences are
somebody else's, and a paper that does not separate them is claiming the wrong
thing.

## 13. Transfer


The estimator generalises directly, and this is the reason a reader who never
thinks about palindromes should care. For any formal textual constraint C,
price it as the difference in bits per unit between a corpus scored normally
and the same corpus mapped into the constrained form under a fixed model and
vocabulary:

- **lipograms** — cost per letter of re-expressing text without a letter, which
  Balasubramanian et al. (2025) approach empirically by showing translation
  fidelity decaying predictably as more letters are excluded;
- **rhyme and meter** — cost per line of the positions the scheme fixes;
- **acrostics** — cost per line of a fixed initial;
- **output-format grammars** — cost per token of the syntax the format
  requires, which is the quantity behind the observed degradation in reasoning
  under format restriction (Tam et al., 2024) and behind the distribution
  distortion that grammar-aligned decoding exists to remove (Park et al.,
  2024).

The currency is the same one linguistic steganography uses, pointed the other
way. That literature measures how many bits of arbitrary payload text can
absorb per word before it stops reading. This measures how many bits per letter
a constraint charges. A constraint that charges more per letter than English
carries is one that cannot be satisfied while writing.

By that arithmetic English carries about 1.6 bits per letter forward and the
mirror charges about 3. The constraint is roughly twice as expensive as the
text is rich, which is a statement of impossibility for long spans and not a
statement about any system.

## 14. What a solution would require


Stated as a constraint on the search space rather than a wish.

The cost bounds unit *length*, not palindrome length, and composition removes
the length problem. So a solution needs a supply of units that (i) pay the
mirror internally, (ii) are long enough to carry a subject, and (iii) are not
already in the record. Whole self-palindromic sentences satisfy (i) and (ii);
all the good ones satisfy (iii) only for somebody else. Mirror-pairs satisfy
(i) and (iii); at four-word halves they do not exist.

That is the gap, and it is narrow enough to state: **novel self-palindromic
sentences, generated rather than catalogued, at the length where a sentence
carries a subject.** Exhaustive enumeration produces them at 22 letters and
2.55M candidates. Whether it produces enough of them at 30–34 letters, and
whether selection over such a bank can be made to do better than the refrain
form, are the two open levers. Neither is a better search over free-running
text, and neither is a better reranker.

## 15. Limitations


The price is model-dependent and vocabulary-dependent; both are reported and
swept rather than fixed. It is English only. The corpus is one corpus.

Judging is uneven across this document. The word-order result in §6a used two
annotators with no access to the key or the hypothesis, with position handled
structurally and agreement reported; everything else judged here is
single-judge blinded assessment with calibration controls and no agreement
statistic.

The system does not produce coherent novel palindromic prose. The readable
output is assembled from catalogued material, and the generated-material mode
produces fragments. Nothing here should be read as a claim to have solved the
task; the claim is a measurement of why it is hard and evidence that the
measurement predicts where seven independent methods stop.

The retracted directional-asymmetry result is left in the record above rather
than deleted, because it is a second instance of the same normalisation error
and the pair is more instructive than either alone.
