# Palindrome Sentence Generator

Generates long, multi-sentence character-level palindromes: text that reads
identically forwards and backwards once case, spaces, and punctuation are
stripped. A dictionary search enforces the palindrome constraint; a language
model chooses among the branches that satisfy it.

```
$ python -m llm_palindrome.generate --min-letters 200 --seeds 24

Xi from life with girl law one. Most egan at eyes in go certified. Is nine
popular referred role to how. Type else my sat animal ha as. Air on a lot
in it fit is. So gone last as it in if. Trap arts ally as no oh no. Say
llas trap art fin if it. Sal tsa lengo si fit in it. Ol a no ri ash ala
mina stay melse. Pyt wohot el orde referral. Up open inside if it recognise
yet. An a get some now all right. I we film or fix.

[letters=201 sentences=14 lm_score=-2.382]
```

## How it works

Word-by-word generation cannot enforce a palindrome, and neither can an LLM
generating left to right: the last character is decided by the first. So the
text is built from both ends inward.

At every step one half owes the other a run of letters — the **overhang**. A
word added on the left must match the overhang forwards; a word added on the
right must match it reversed. The palindrome closes when the overhang is
itself a palindrome, and that becomes the center.

```
left:  "rats live on"          overhang: "no"   (right owes "on" reversed)
right:            "no evil star"
                        ^ closes when the overhang reads the same both ways
```

This overhang search is **Peter Norvig's palindrome algorithm**, building on
Dan Hoey's 1984 program — see Credit below. What this repository adds is the
scoring: at each step the
search faces hundreds of letter-valid continuations, nearly all of them
gibberish. A language model ranks them, so the search follows the branches
that read as English.

Two ways to apply the model, both implemented:

- **Rerank** (default): search with word-frequency scoring, then score the
  finished palindromes with GPT-2 and keep the most fluent.
- **In-loop** (`--lm-in-loop`): GPT-2 rescores the beam *during* the search, so
  unfluent branches die before they consume the budget.

## Results

Matched budgets, 12 seeds, min 200 letters, beam 60, GPT-2 small on an M-series
Mac. `lm_score` is mean token log-probability per letter (higher is better).

| Configuration | Best score | Mean score | Letters | Time | Valid |
|---|---|---|---|---|---|
| Zipf + GPT-2 rerank | −2.382 | −2.455 | 201 | 5.5 s | 12/12 |
| Zipf + GPT-2 in-loop | **−2.297** | **−2.405** | 216 | 170 s | 12/12 |

In-loop scoring produces measurably more fluent text at roughly 31× the cost.
Every candidate in both arms is a valid palindrome — validity comes from the
search, not the model, so the model can only affect readability.

`lm_score` divides total token logprob by **letters**, which is what makes it
comparable across palindromes of different lengths — and also what makes it
possible to raise without writing anything better. Longer words cost fewer
tokens per letter, so a search that prefers them scores higher for free; a
tuning run found that on its own and gained +0.30 by it while the same texts
scored 0.67 *worse* per token. Both arms above run the same scorer over the
same vocabulary and differ only in when the model is applied, so word length is
not what separates them — but the two have not been compared per token, and any
change that moves word length has to be read that way. `experiments/` reports
both normalizations and says so when they disagree.

Reproduce with:

```bash
python benchmark.py --seeds 12 --min-letters 200 --beam 60
```

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m llm_palindrome.generate --min-letters 200 --seeds 24
```

| Option | Description | Default |
|---|---|---|
| `--min-letters` | Minimum palindrome length in letters | 120 |
| `--seeds` | Independent search runs to rank against each other | 24 |
| `--beam` | Beam width per search | 60 |
| `--vocab` | Size of the frequency-ranked vocabulary | 30000 |
| `--model` | Hugging Face model for scoring (`''` disables) | `gpt2` |
| `--lm-in-loop` | Prune the beam with the LM during search | off |
| `--words-per-sentence` | Sentence length in the formatted output | 7 |
| `--out` | Write the result to a file | none |

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers overhang matching, trie candidate generation, end-to-end
search validity, and the formatting invariant that punctuation and casing
never alter the normalized letters.

The page has its own suite:

```bash
cd web && npm test
```

It runs the poster in jsdom against a fake stream and a fake composer, and
covers what the page does with the frames it is sent and when it draws them.

## Project structure

```
llm_palindrome/
  validator.py     normalize / is_palindrome — the only arbiter of validity
  search.py        overhang matching, tries, beam search (Norvig/Hoey)
  centerout.py     outward growth, one word at a time
  exhaustive.py    walk the short regime instead of sampling it
  scoring.py       frequency scorer
  bigram.py        bidirectional bigram model
  lm_scoring.py    GPT-2 fluency, whole-text and conditional
  coherence.py     long-range conditional gain, self-shuffled controls
  instant_judge.py learned fast judge
  directional.py   forward vs reversed-resegmented cost
  safe_vocab.py    what must never reach a generated public output
  shortwords.py    which 1-2 letter strings are words
  lexicon.py       which strings are words at all ("utc" and "ips" are not)
  textify.py       sentence formatting, letter-preserving
  present.py       case and punctuation for a short palindrome, by exhaustive
                   segmentation over Brown-tag tiers; asserts the letters
  generate.py      CLI

  -- paragraphs --
  paragraphs.py    harvest / assemble / render / refrain; asserts the mirror
  pairs.py         walk the vocabulary for mirror-pairs of our own
  syntax.py        tag shapes from Brown: could this half be a sentence?
  wordorder.py     a half against its own shuffles: what did the order buy?
  spelling.py      apostrophes and the capital I, which the mirror cannot see
  respace.py       recover a spelling by segmentation (mining path only)
  mining.py        mirror-pairs from attested English phrases
  reversibles.py   mirror-pairs in closed form, from reversible words
  themes.py        find the shared subject, seat it in the refrain
  sequencing.py    order units; repetition and cadence guards
  compose.py       POS templates over the mined inventory
  phrases.py       phrase inventory and unit construction
  overhang.py      cached trie lookahead for the debt
  reversal.py      how well a unit survives being mirrored (word mode)
  tunable.py       swept parameters, named rather than inlined
  verify.py        end-to-end validity checks

data/
  canon_spelled.json    71 catalogued palindromes WITH their spacing
  known_palindromes.json  160 normalised, for the novelty check
  centres.json          49 blind-judged self-palindromic sentences
  mirror_pairs.json     4,656 mined pairs, attestation flagged
  mirror_units.json     29 pairs whose two halves are DIFFERENT text, catalogued
  novel_pairs.json      28 pairs walked out of the vocabulary — ours
  fallback_texts.json   8 palindromes this search closed earlier, for when it
                        closes none now
  lexicon.txt           52,927 headwords (dictionary ∩ frequency)
  word_banks.json       word-ORDER mode material
  count_2w.txt, ngrams_wikitext2.json   corpora
  composed_sentences.json   compose.py output; no code path reads it
  authored_sentences.txt    148 authored halves; 12 mirror, none readably
  v3_bank.json          540 verified palindromes v3 composes from — 499 ours

server/
  app.py           v1 endpoints
  v2.py            v2 endpoints, incl. GET /api/v2/paragraph
  v3.py            v3 endpoints, incl. GET /api/v3/composition
training/          corpus, judge, inventory and lexicon builders
experiments/       measurements quoted in this README and docs/training.md
tests/             pytest suite
web/               the page at palindrome.ericspencer.us
```

## Known limits

Output is locally fluent — phrases parse, sentences scan — but not globally
coherent; it does not hold a topic across its full length. Four explanations
have now been tested and eliminated, all recorded in `docs/training.md`.

**A better judge will not fix it.** Best-of-2000 scores 0.168 nats per token
above best-of-24 and the curve is flat by then, bounding every reranker over
that candidate pool.

**Nor a wider search.** Raising the beam jitter takes 3 distinct openings to
200 and readability falls monotonically.

**Nor shorter text.** `experiments/length_sweep.py` puts long-range coherence
on the word-salad line at every length from 71 letters to 1197.

**Nor a phrase inventory.** Attested bigrams consumed atomically drop
bigram coverage from 0.70 to 0.48 — locking two words together costs more at
the seams than the internal join buys. Whole corpus SENTENCES do work, and are
quotation: `server/v2.py` places them intact and attributes them.

### Why: the mirror costs about 3 bits per letter

Reverse the letters of English, re-segment them into the vocabulary, and score
both readings under one model: the reversed one costs **2.2 to 3.6 bits per
free letter** more, against the 1.4 to 1.9 bits per letter the same spans score
read normally. `experiments/mirror_cost.py` sweeps four models, three
segmentation objectives and six span lengths and writes the table; the range is
the honest report, and the number is near 3 at the objective and lengths that
matter. About half the letters of the reversed reading cannot be placed inside
a real word at all, and no segmentation strategy improves that.

Every letter is placed twice and both placements must be English, so the
coherent feasible set thins by roughly **8 to 10x per letter added**. (Earlier
versions of this file said 10x every three letters. That misreads the exponent:
2^3.3 is the factor for one letter.) That is why the human record contains no
long palindrome that reads:
Norvig's 21,012-word one is a noun list its own author calls nonsense, and half
the canonical palindromes are 12 to 17 letters.

Two consequences this project measured the hard way. Exhaustive enumeration is
only exhaustive at small vocabularies — with 14k units the tree branches ~14k
wide at every closure, so a time-budgeted walk returns a deep prefix of one
corner. `canon recall` (how many of the 27 catalogued palindromes a walk
rediscovers) is the acceptance test: it reaches 10/18 at 83 canon-seeded words
and depth 4, and 0/27 at 14k. And no surface statistic predicts readability —
GPT-2 weaker-half score, bigram-join attestation, vocabulary filters and
edge-joins were each measured against judge verdicts and each failed.

## Paragraphs

> **The goal is [docs/NORTH-STAR.md](docs/NORTH-STAR.md).** A paragraph of
> coherent English prose, at least 100 words, whose letters read identically
> both ways, built from sentences that are not themselves palindromes and were
> not written by somebody else. Nine criteria, conjunctive. What ships passes
> four — the structural ones. `tests/test_north_star.py` holds the other five
> as failing targets so they cannot be quietly dropped.


`GET /api/v2/paragraph` returns a paragraph whose **letters** read the same
both ways, at any length, assembled from whole sentences that share a subject.

    Was it a car or a cat i saw. Was it a cat i saw. A santa dog lived as a
    devil god at nasa. A santa lived as a devil at nasa. Able was i ere i saw
    elba. Stressed was i ere i saw desserts. Delia saw i was ailed. Stressed
    was i ere i saw desserts. Able was i ere i saw elba. A santa lived as a
    devil at nasa. A santa dog lived as a devil god at nasa. Was it a cat i
    saw. Was it a car or a cat i saw.

### How

A palindrome of any length can be assembled rather than searched. Units that
pay the constraint internally nest like brackets

    L1 L2 ... Lk  CENTRE  Rk ... R2 R1

and the result is palindromic by construction (`llm_palindrome/paragraphs.py`,
where both `assemble` and `render` assert it). Length stops being the problem.
Choosing and ordering the units becomes it — and that subproblem carries no
letter constraint at all.

The shipping path is five steps, none of them a search:

| step | module | what it does |
|------|--------|--------------|
| spellings | `data/canon_spelled.json` | 71 catalogued palindromes stored with their spacing, each verified by `is_palindrome` |
| centres | `data/centres.json` | 49 self-palindromic sentences that survived blind judging |
| theme | `themes.best_cluster` | picks the centres that share content words |
| order | `themes.order_for_refrain` | questions outermost, firmest statement on the turn |
| assembly | `paragraphs.refrain` | mirrors the sequence — palindromic by construction |

A prompt steers the theme rather than filtering it: "devil" matches two
centres, and filtering would return two sentences and call that a paragraph.

### Why sentences, and not the units everything else uses

The mirror costs about 3 bits per free letter, which forces units to be short.
Short units carry no subject, so nothing can be about anything — and for a long
time that was read as the cost forbidding a through-line. It is not the cost.
It is the length.

Measured, at half level:

| half length | mined | both halves attested |
|-------------|-------|----------------------|
| 2 words | 3,894 | 131 |
| 3 words | 762 | 27 |
| 4 words | 106 | **0** |

Four-word halves are where prose would start, and 34,688 attested 4-grams
produced not one whose mirror reads. Thematic selection over the surviving
26 usable halves failed just as flatly: of all their content-word pairs,
exactly **2** co-occur anywhere in 3,932 sentences.

Whole self-palindromic sentences pay exactly the same 3.3 bits and **do** carry
subjects — 8 of the judged centres are a first-person narrator doubting what
they saw. Grouping those produced the first paragraph here that a blind judge
called coherent, ranked above the same structure with mixed topics.

### Where the units come from

Four sources were built and measured against each other:

| source | produced | both halves attested | usable after dedup |
|--------|----------|----------------------|--------------------|
| exhaustive hunt | 20,000 | — | ~0 readable |
| mining attested corpus (`mining.py`) | 3,894 | 131 | 26 |
| LLM authoring | 40 verified, 37 novel | — | ~3 |
| closed-form reversibles (`reversibles.py`) | 13,924 | 5 | 2 |
| authored halves, mirror segmented (`experiments/authored_mirrors.py`) | 148 tried | 12 spellable | 0 |
| reversible chains, exhaustive (`experiments/reversible_chains.py`) | 57 joins | — | 4, one template |
| sharded walk (`pairs.py`) | 1,915 in 460s | — | ~0 readable |

All of them converge on roughly the same ceiling, and none produces sentences.
The canon does. `is_novel_palindrome` (160 entries) keeps the project honest
about which material it borrowed: **the assembly is ours, the sentences are the
record's.**

One filter looked like it had changed that and had not. Constraining the walk
so every join inside a half is one the bigram counts record makes the output
stop looking like word salad — "name not felt a ward || draw at left one man"
rather than "marc i ma reconsider || red is no ceramic ram". Then the filter was
run against the 29 catalogued pairs it is supposed to be finding more of:
`count_2w.txt` covers every join in 27% of their halves, Brown's million words
29%, GPT-2's top-400 followers 27%. A pair needs both halves, so the constraint
discards something like 93% of the target. "Evil rats on || no star live" has
not one attested join anywhere. See `experiments/join_attestation.py`; the
constraint is still available and is no longer the default.

### What was tried and did not work

Recorded because each one cost iterations and each conclusion is load-bearing.

- **Word-order palindromes.** The sentence sequence mirrors, the letters do
  not. A different and much easier constraint — it pays nothing per letter —
  and it was the default here while the letter-level paragraph was still a list
  of fragments. Still reachable at `?mode=word`, labelled
  `letterPalindrome: false`.
- **Longer palindromes.** The 40–60 letter band was searched: 880k closures,
  112,523 of the first 300,000 using only dictionary words, and **none** within
  three unattested joins of reading.
- **More material by re-mining.** Mining each phrase as a right half returned
  nothing new — all 7,704 entries were flips of each other. k-best segmentation
  recovers 4 pairs out of 198.
- **More units.** A second authoring round added 18 verified novel units and
  made the paragraph *worse* blind. Unit count is not the lever.
- **Better inference for spellings.** A unigram model cannot tell "for ajar"
  from "for a jar"; adding attested-join weight recovers 3 of 10 and no
  weighting recovers the rest, because choosing "i slam" over "islam" requires
  knowing the sentence. Storing the spellings fixed all 14.
- **Removing the refrain's repetition.** A paragraph with zero repeated
  sentences ranked *below* the refrain. Sentence-hood dominates repetition.

### Proxies: what they are good for

Every scoring proxy tried here has been checked against blind judging with
real-prose and word-salad controls, and the pattern has not varied:

| proxy | as a filter | as a ranker |
|-------|-------------|-------------|
| GPT-2 score | sound — selects nothing a judge rejects | **failed 3 times** |
| `reads_as_attested` | sound — 79% accepted vs 38% | not used for ranking |
| `themes.cohesion` | sound — finds the theme | **failed** — sizes the paragraph wrong |
| word frequency | no signal (4.34 vs 4.06) | — |

Proxies are usable for exclusion and for finding candidates. They have never
once been usable for ranking finished text, which is why every result above
was settled by a blinded batch rather than by a number.

### Honest limits

The paragraph is not prose. Judged blind against consecutive sentences from one
document, it ranks below both that and disparate-but-real sentences. It reads
as a composition on a subject — a voice doubting what it saw — not as an
argument that develops. The sentences repeat by construction, since a mirrored
sequence is what makes the whole a palindrome. And they are catalogued
palindromes: the contribution here is the selection, ordering and assembly.

The default now serves generated units instead, from `data/novel_pairs.json` —
28 mirror-pairs walked out of the vocabulary by `pairs.py` and read one at a
time before being let in. That fixes provenance and length and does not fix
reading:

> War dog. Rob a log. No cotton. Fired now. Went on. Trade man. … Not new.
> Wonder if. Not to con. Go labor. Go draw.

101 words, every unit ours, and a run of two- and three-word fragments. The
catalogued version reads better and is still one query parameter away
(`?source=catalogue`), which is exactly the trade docs/NORTH-STAR.md refuses to
take: a paragraph that reads well because somebody else wrote the sentences is
not the thing being built.

## v3

`GET /api/v3/composition` returns one palindrome of a requested length, written
out the way a person would write it. Read it at
[palindrome.ericspencer.us](https://palindrome.ericspencer.us).

    Deep, nam, ottoman a pat. Path submit a. Pop path submit. Pool a
    estimates. Set is levels. Sam a rest. Estimates pet set. A knock sir.
    Busy a snow. Order parts a: estimates dam pet. Snow order a.

Three things separate it from what came before.

**It does not search on request.** The good material comes from walks of
millions of candidates that take minutes on 32 cores, so v3 composes from a
bank found offline — 540 verified palindromes, 499 of them walked out by this
project — and re-verifies every entry on load and again before answering.

**Length is a parameter, not an outcome.** Mirror-pairs nest around a centre as
`L1 L2 … C … R2 R1`, where each `Ri` is `Li`'s letters reversed and is
therefore different text. A run of self-palindromic units cannot do this: the
sequence of units would itself have to mirror, so every unit but the centre
would appear twice. The bank holds about 14,500 letters of material and any
length up to that is free.

**The punctuation is free and it is chosen, not guessed.** Every way of cutting
the word run into runs is scored, each run by the strongest Brown-tag test it
passes, and the best total wins (`llm_palindrome/present.py`). `normalize`
strips case, spaces and marks before the mirror is checked, which is the same
licence the record takes when it writes "A man, a plan, a canal: Panama". The
letters are asserted unchanged before the value leaves the process.

**The default composition is now hierarchical.** Words stay inside structural
mirror halves; each half must independently have a whole-sentence reading, and
those paired sentences are nested into the paragraph. Punctuation no longer
cuts across unrelated chunks. Hard guards reject adjacent duplicate words,
content-word loops, repeated bigrams, and repeated sentence templates; odd but
interpretable compression such as “Items draw award” remains eligible. See
`experiments/RESULTS-hierarchical-v3.md` for the measured capacity and limits.

`GET /api/v3/refrain?theme=dark&letters=500` exposes the honest sentence-scale
quality ceiling: catalogued human palindromic sentences arranged as
`A B … C … B A`. Every sentence is a palindrome, no sentence repeats
adjacently, and each non-central sentence returns exactly once as an explicit
literary refrain. The response reports `source=catalogue` and `novel=false`;
it is not presented as generated prose. Themes are `dark`, `journey`,
`reflection`, and `absurd`.

`?grow=N` applies the wrap operations from `experiments/RESULTS-extend.md`.
It is off by default: 750 of 750 seeds grew, mean +36 letters, and two blind
annotators then preferred the ungrown seed on 20 of 20 pairs.

### What v3 does not clear

The name was reserved by docs/NORTH-STAR.md for the goal, and the endpoint
having it does not mean the goal is met. Over 24 seeds a length
(`experiments/RESULTS-north-star-v3.md`), v3 holds all six mechanical criteria
at 400 letters bar sentence repetition (21/24) and disjoint halves (23/24), and
repetition fails 23/24 by 1,200 letters. **No chunk ever repeats — that
guarantee holds — but punctuation is applied to the assembled run rather than
per chunk, so two unrelated chunks containing the same short word run are cut
into the same sentence.** The assembly is sound and the presentation collides.

Criteria 6–8 — grammatical, has a subject, reads as prose — need blind judging
and are not claimed.

## Credit

The two-sided overhang search is **Peter Norvig's**, described in
[World's Longest Palindrome?](https://norvig.com/palindrome.html) and in detail
in [The Algorithm](https://norvig.com/pal-alg.html) (2002). It builds on **Dan
Hoey's** 1984 program, which produced the first long "A man, a plan, a canal"
palindrome by the same remainder-matching idea. This project contributes the
bidirectional-bigram and language-model scoring layer on top of that search.

Developed within the **AI4FM group**.

## License

MIT
