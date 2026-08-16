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
  generate.py      CLI

  -- paragraphs --
  paragraphs.py    harvest / assemble / render / refrain; asserts the mirror
  pairs.py         walk the vocabulary for mirror-pairs of our own
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
  mirror_units.json     29 pairs whose two halves are DIFFERENT text
  lexicon.txt           52,927 headwords (dictionary ∩ frequency)
  word_banks.json       word-ORDER mode material
  count_2w.txt, ngrams_wikitext2.json   corpora
  composed_sentences.json   compose.py output; no code path reads it

server/
  app.py           v1 endpoints
  v2.py            v2 endpoints, incl. GET /api/v2/paragraph
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

### Why: the mirror costs 3.3 bits per letter

Forward English scores 1.63 bits/letter. Reverse the letters of English, then
re-segment them optimally into the vocabulary, and the result scores 4.92 —
a **3.30 bits per free letter** cost, stable across span lengths and
segmentation strategies. Every letter is placed twice and both halves must be
English, so the coherent feasible set thins by roughly 10x every three letters
added. That is why the human record contains no long palindrome that reads:
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

The mirror costs 3.296 bits per free letter, which forces units to be short.
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

All four converge on roughly the same ceiling, and none of them produces
sentences. The canon does. `is_novel_palindrome` (160 entries) keeps the
project honest about which material it borrowed: **the assembly is ours, the
sentences are the record's.**

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
