# Palindrome Sentence Generator

[Results](RESULTS.md) · [Full results ledger](docs/RESULTS-LEDGER.md) · [Design](DESIGN.md) · [Product](PRODUCT.md)

## Current status: exact constructions, human evaluation pending

The **38-letter** Diana sentence is an inherited exact reference. The
[historical evidence audit](paper/BASELINE-PROVENANCE-AUDIT-V1.md) found no
primary human ratings or administered reader protocol for it. The separate
**568-letter** causal-scene row remains
the exact working-length incumbent because it has the clearest construction
lineage. The exact **666-letter** comparison parent (SHA-256
`cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297`) and its
naturalness child (SHA-256
`2dc6afd1545b52b24f2d53d04e6e2529929f52dd45b81c00495f5cf3b86b19ed`) are
retained unpromoted after whole-text review.

No audited candidate has recorded human-readability evidence. Exactness
is mechanical; readability, repetition, and discourse are separate debts, and
no candidate is a readable world record. The short version is on
[`RESULTS.md`](RESULTS.md), and the complete committed artifact index is in
[`docs/RESULTS-LEDGER.md`](docs/RESULTS-LEDGER.md).

The comparison parent is [`comparison-alternative-nora-sees-666`](runs/incumbent-666-comparison-alternative-20260922.json).

The 560, 558, and 556 children remain preserved as a diverse repair frontier.
The 594-, 600-, and 592-letter children remain available as exact construction
evidence for seam repair. Fresh POS searches,
productive-affix cycles, short witnesses, and reader-study packets are frozen.
The separate 54-letter NP candidate
([`seed artifact`](runs/seed-np-cross-role-intersection-20260922.json), SHA-256
`2f88268e3a920af5ceb67cfb20d1498ef5ce47e91d8800c937639cc8ce376268`) is the
meaning-bearing growth seed. Its 24-form reader packet remains frozen and
unrun. The 568 and 666 candidates remain separate exact-construction
frontiers; neither has human-readability evidence.

The active acceptance standard is documented in
[`docs/READABLE-PALINDROME-GOAL.md`](docs/READABLE-PALINDROME-GOAL.md): an
independently constructed exact palindrome must clear the shared mechanical
gate and then pass a blinded study with intact-prose and shuffled controls.

## Reader feedback — October 9

[Eric’s 183 development ratings](data/reader-feedback/2026-10-09/README.md)
are preserved. Most score 1 for meaning: word associations with unclear
relations. The export lacks passage texts, so exactness and length comparisons
remain unresolved. The new audit importer separates verified palindromes,
non-palindromes, and missing texts before training.
[The tuning explained in 30 lines](docs/TUNING-30-LINES.md).

The literal word **“association” occurs in 44 of the 52 frozen review texts**.
That is a separate cohort from the 183 ratings; the earlier rubric-only
interpretation missed this repetition. The revised feedback notes distinguish
the two and retain the original ratings.

[The credited Reddit collection](data/reddit-palindromes/2026-10-09/README.md)
contains **196 distinct posts**, including **189 distinct exact palindromes**.
Every title keeps its poster and permalink, including excluded titles.
The audit also maps each word onto the words its reversed letters constrain.
These are reference examples, not newly generated results or human-quality labels.

[The proposed search](docs/research/2026-10-09-readable-search-design.md)
chooses words by the quality of their forced opposite continuation, requires
expressed relationships, and grows short successes only while meaning survives.
It builds on the existing exact solvers; its readability gains remain to be tested.

The [October 9 Argonne evaluation](runs/argonne-length-readability-20261009/README.md)
completed 72 comparisons on one Polaris node in 99 seconds. It found 43 exact
selected outputs and a longest observed closure of 447 letters. A separate
blinded machine screen found no whole-text grammar/meaning result above the
association level. The [length and readability plan](docs/research/2026-10-09-argonne-evaluation-plan.md)
targets connected 1,000+ letter paragraphs with flexible phrases, boundaries
and overhangs. That readable-length goal remains unachieved.

The [meaning and reuse rules](docs/research/2026-10-09-meaning-and-reuse.md)
allow repeated words that serve a useful role, including references and a
question followed by an answer. Repeated assertions and stock phrases do not
count as progress. Search limits now support wider beams, longer overhangs and
fresh constructions without a frozen center. The
[additional Argonne experiments](runs/argonne-length-meaning-20261009/README.md)
retain failures and separate mechanical length from whole-text meaning.

## Recent generations

These are the generations I actually want a reader to see before opening the
experiment directory. The lengths are normalized letters; punctuation and
casing are presentation choices.

| Length | Generation | What it shows |
|---:|---|---|
| 44 | `Now, an aide rips nine memos; some men inspire. Diana won.` | exact seed reconstruction; discourse unresolved |
| 54 | `An aide rips nine memo-hero memos. Some more home men inspire Diana.` | meaning-bearing exact growth seed; frozen reader packet unrun |
| 106 | Diana reverse-word insertion | exact growth, but a useful word-salad control |
| 240 | paired seam growth | the first substantial exact center with visible formulaic debt |
| 568 | causal-scene outer shell | the working incumbent and its repair lineage |
| 650 | event-shell substitution | exact exploratory descendant; repeated frames and malformed clauses keep it out of the reader lane |
| 666 | connected central mini-scene | exact comparison evidence; no human evidence |

<details>
<summary>Show the 650-letter generation</summary>

```text
A ram saw Nadia. Leon won. Wolf spots Nora. Nadia stops, so Tara rewards Nadia. Nora spots a rat. Nadia spots a ram. Mara spots rats. Leon maps Nora. Draw no maps. Aidan spots a rat. Nora spots a ram. Mara stops rats. A tub? He maps Nora. Deliver no evil. Now, Noel, did I live? Nora saw Noel live. Noel, I sit. Pat notes. Mara saw God. Sara, did I live? Nora, I saw desserts. Noel, was I stressed? Nadia delivers maps. Leon. Ari delivers maps. Spam's reviled, Ira. Noel; spam's reviled, Aidan. Desserts I saw, Leon. Stressed was I, Aron. Evil I did, Aras. Dog was Aram. Seton, tap. 'Tis I, Leon. “Evil Leon” was Aron. Evil I did, Leon won. Live on, reviled. Aron, spam. Eh, but a star spots Aram. Mara stops Aron. Tara stops Nadia. Spam onward. Aron, spam Noel. Star stops Aram. Mara stops Aidan. Tara stops Aron. Aidan's drawer, Aratos, spots Aidan. Aron stops flow now, Noel. Aidan was Mara.
```

[`runs/incumbent-608-repeated-shell-repair-20261002.json`](runs/incumbent-608-repeated-shell-repair-20261002.json) · exact after independent normalized reversal

</details>

The [results page](RESULTS.md) includes a retained 666-letter comparison
generation,
the 650-letter generation above, hashes, status notes, and the full historical
length progression.

## Historical workbench — quarantined; do not run or cite the commands below

The remaining development notes are preserved only for auditability. Their
commands, endpoints, examples, and reported outputs are disabled or rejected
under the current acceptance standard.

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
  finished palindromes with GPT-2 and keep the highest-scoring outputs.
- **In-loop** (`--lm-in-loop`): GPT-2 rescores the beam *during* the search, so
  lower-scoring branches are removed during beam selection.

## Evaluation

The current [public v1 paper](paper/README.md) reports 66 saved output records
containing 52 distinct exact strings and evaluates defined word-order controls
and model rubrics. It contains no matched published-generator comparison or
human readability result. See the [baseline audit](paper/BASELINE-PROVENANCE-AUDIT-V1.md)
for the evidence behind that scope.

The legacy `benchmark.py` command remains a development diagnostic. Its
12-seed, 200-letter, beam-60 configuration was a project choice. The earlier
README table compared GPT-2 scores per letter and called the difference a
fluency improvement without a validated readability outcome. That claim is
withdrawn. Model scores, token normalization, selection rules, and resource
budgets must be specified before interpreting any comparison.

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
  clause_ngram.py  directional n-gram state for two-sided clause growth
  sentence_plan.py finite prefix/suffix feasibility over Brown sentence plans
  semantic.py      lightweight semantic feature extraction
  semantic_graph.py coherent ordering edges over candidate pairs
  semantic_plan.py typed semantic plans for diagnostic construction searches
  admission.py     shared fail-closed mechanical admission checks
  exact_editor.py  paired-span edits with independent surface audits
  frontier_macros.py anchor-preserving residual search macros
  reader_first.py  quarantined reader-first presentation experiment
  hierarchy.py     discourse-level connectors and topic hierarchy
  pareto.py        non-dominated candidate selection across hard diagnostics
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
  refrain.py       reflected refrain construction and its explicit constraints
  tunable.py       swept parameters, named rather than inlined
  verify.py        end-to-end validity checks

data/
  canon_spelled.json    71 catalogued palindromes WITH their spacing
  known_palindromes.json  162 normalised, for the novelty check
  catalogue_provenance.json  provenance for the quarantined catalogue controls
  readable_palindrome_centres.json  quarantined catalogue centres; never output
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
  semantic_plan_20260913.json  frozen semantic-plan diagnostic inventory

server/
  app.py           v1 endpoints
  v2.py            v2 endpoints, incl. GET /api/v2/paragraph
  v3.py            v3 endpoints, incl. GET /api/v3/composition
training/          corpus, judge, inventory and lexicon builders
experiments/       dated construction searches, controls, and measurements
tests/             pytest suite, kept separate from experiment modules
RESULTS.md         curated generations and the current exact-construction status
docs/RESULTS-LEDGER.md
                   full method-by-method artifact index
web/               the page at palindrome.ericspencer.us
```

## Known limits

Exactness is established by normalized reversal checks. The current study's
outputs have no human ratings, and automatic scores do not establish fluency
or coherence. Earlier search, reranking, length, and phrase-inventory trials
are preserved in [the historical training notes](docs/training.md). They do
not eliminate those approaches in general or bound all possible rerankers.

The [frozen reversal-cost study](runs/mirror-cost-2026-09-11/RESULTS.md)
compares forward and reversed WikiText-2 spans after applying the same
segmenter to both directions. Across two fixed models, three segmenters, and
six lengths, its saved score differences are 2.14–3.34 bits per normalized
letter. The offline audit checks text and arithmetic, not fresh model logits.
These measurements depend on the sampled text, vocabulary, segmentation, and
models. They establish neither a universal cost of palindrome construction
nor a count of feasible English strings.

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
| centres | `data/centres.json` | 49 self-palindromic sentences retained by a historical model screen |
| theme | `themes.best_cluster` | picks the centres that share content words |
| order | `themes.order_for_refrain` | questions outermost, firmest statement on the turn |
| assembly | `paragraphs.refrain` | mirrors the sequence — palindromic by construction |

A prompt steers the theme rather than filtering it: "devil" matches two
centres, and filtering would return two sentences and call that a paragraph.

### Historical assembly evaluation

The earlier mining, authored-half, catalogue, and proxy experiments are
recorded in [the historical training notes](docs/training.md) and the saved
experiment files. Their selected counts and automated judgments do not form
a common-budget generator comparison or a human readability study. Earlier
README claims about a shared quality ceiling, universally sound filters, and
fluency gains are withdrawn.

The assembly path reuses complete units, with repetition implied by its
construction. Catalogue units are borrowed material; generated mirror pairs
have separate provenance. These are distinct from the from-empty searches in
the v1 paper and the independently checked construction examples in
[the audit](paper/BASELINE-PROVENANCE-AUDIT-V1.md).

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
