# Abstract and introduction

Draft for the ARR October 2026 cycle (deadline 12 October 2026), committed to
NAACL 2027. Follows the long-paper outline in `docs/publication-plan.md` §5a.
Every number below is sourced from a file in this repository; the ones that are
not yet reproducible from a script are listed at the end.

---

## 1. Candidate titles

1. **3.30 Bits per Free Letter: Pricing a Character-Level Constraint and
   Testing the Ceiling It Predicts** — *recommended.* It states the
   measurement, the unit, and the falsifiable consequence, and it does not
   mention a system. The number in the title is what makes it citable by
   readers who are not studying palindromes.

2. What a Mirror Costs: Measuring a Formal Text Constraint in Bits per Free
   Letter

3. Cheap to Satisfy, Expensive to Satisfy Well: The Price of a Character-Level
   Constraint and Seven Routes to Its Ceiling

Title 2 is the safer choice if a reviewer objects that a single model-derived
number does not belong in a title before the scale replication is in (§5,
week of 24 August). Title 3 foregrounds the instrument argument and is the
weakest of the three at naming the measurement.

---

## 2. Abstract

*(193 words)*

> Constrained text generation is reported as a satisfaction rate and a fluency
> score. What the constraint costs is not reported. We price one constraint in
> bits. Under GPT-2 small, forward English scores 1.63 bits per letter over a
> frequency-ranked vocabulary; the same letters reversed and re-segmented
> optimally into that vocabulary score 4.92. The character-level palindrome
> constraint therefore costs 3.30 bits per free letter, stable across span
> lengths and segmentation strategies. Every letter is placed twice and both
> readings must be English, so the coherent feasible set thins by roughly an
> order of magnitude per free letter, which is a falsifiable prediction about
> where readable output stops. We test it with seven construction routes:
> beam search with language-model scoring, exhaustive enumeration of the short
> regime, mining attested corpus n-grams, closed-form reversible-word chains,
> LLM authoring, authoring one half and segmenting the other, and a sharded
> vocabulary walk. All seven stop in the same place: 34,688 attested four-grams
> yield no mirror pair whose two halves both read, and no route produces
> coherent novel palindromic prose. Four automatic proxies were each sound as
> filters and each failed as rankers under blind judging with real-prose and
> word-salad controls.

**Note on the thinning rate.** The repository states the consequence as "the
coherent feasible set thins by roughly 10× every three letters added"
(`README.md:204`, `docs/NORTH-STAR.md:124`). Taken at face value, 3.296 bits
per free letter is a factor of 2^3.296 = 9.8 per *free* letter, which is 10×
per free letter, not per three. The abstract above uses the arithmetic that
follows from the measured number and hedges it with "roughly". The discrepancy
is listed in §6.

---

## 3. Introduction

*(~1147 words)*

Everyone who has written about palindromes has observed that the long ones do
not read. Norvig's record-length palindrome is a list of nouns its own author
describes as nonsense, and half of the canonical short palindromes are between
12 and 17 letters. Papadopoulos et al. (2015) note in passing that long
palindromes are often less meaningful. The observation is repeated across the
recreational-linguistics literature and the engineering write-ups that define
the modern algorithm, and in none of them is it attached to a number. It is
treated as a fact about the genre rather than as a quantity that could be
measured, predicted from, or transferred to another constraint.

We measure it. A palindrome of 2k letters has k free letters: the second half
is the first half's letters reversed, so each letter is chosen once and placed
twice, and the two placements segment into unrelated words. That gives an
estimator. Score English forward under a language model and a vocabulary;
reverse the same letters, re-segment them optimally into the same vocabulary,
and score that. The difference is what the constraint charges per free letter.
Under GPT-2 small, forward English scores 1.63 bits per letter and the
reversed-and-re-segmented text scores 4.92, so the palindrome constraint costs
**3.30 bits per free letter**, stably across span lengths and segmentation
strategies. The estimate rests on one small 2019 model; §4 reports its
replication across scales and model families. Read as a rate, 3.30 bits is a
factor of about ten per free letter, which says the set of letter sequences
that read as English in both directions thins geometrically in length. That is
a falsifiable prediction, and the rest of the paper tests it.

The palindrome is a cleaner instrument than it first appears, because it is
cheap to satisfy and expensive to satisfy well. It is cheap to satisfy because
the constraint has a sufficient statistic that makes it checkable
incrementally. At any point in a two-sided construction, one half owes the
other a run of letters, the *overhang*; a word added on the left must match the
overhang forwards, and a word added on the right must match it reversed. When
the overhang is itself a palindrome, the text closes. This is Norvig's
algorithm, building on Hoey's 1984 program; the search is theirs, and what we
add is the scoring layer over it. What the search buys is that **validity comes
from the search, not from the model**. In every configuration we ran, every
candidate returned was a valid palindrome: 12/12 and 24/24 at the matched
budgets used for language-model ablations, and 200/200 across a five-point
sweep of the beam's exploration parameter. A model in this pipeline cannot make
the output invalid and cannot make it valid; it can only make it more or less
readable. Most constrained-generation benchmarks conflate those two axes,
because the model is the thing enforcing the constraint, so a fluency gain and
a satisfaction gain arrive together.

The prediction was tested from seven directions, chosen so that they would fail
for different reasons if they failed: beam search with language-model scoring,
in a rerank and an in-loop arm at matched budgets; exhaustive enumeration of
the short regime; mining mirror-pairs from attested corpus n-grams; closed-form
chains of reversible words; LLM authoring; authoring one half and recovering
the other by segmentation; and a sharded walk over the vocabulary. They stop in
the same place. Mining 272k attested bigrams yields 3,894 mirror-pairs of which
131 have both halves attested; at three-word halves, 27; at four-word halves,
where prose would start, **0 out of 34,688 attested four-grams**. Of 148
authored sentences, 12 have any spellable mirror and none whose mirror reads.
In the 40–60 letter band, 880k closures produced none within three unattested
joins of reading. Under a strict rubric calibrated to accept real prose 5/5 and
reject word salad 0/5, five independently selected populations drawn from an
exhaustive walk of 2.55M short palindromes passed **0 of 170**. The system does
not produce coherent novel palindromic prose. Its readable output is assembled
from catalogued palindromes; its fully generated mode produces two- and
three-word fragments.

Two further results come from the same measurements. Ranking is where the
automatic signals break: four proxies (a language model's mean log-probability,
attestation of word joins, thematic cohesion, and word frequency) were each
checked against blind judging with real-prose and word-salad controls, and each
was sound as a *filter* and failed as a *ranker*, the language-model proxy
three separate times. The proxies are also gameable in ways a check against a
stronger model does not catch: optimising a per-letter-normalised
language-model score gained +0.298 while the same texts scored 0.673 *worse*
per token, losing 24 of 24 paired seeds, because the policy had found longer
words rather than better text.

Contributions:

1. **A price for a formal textual constraint, in bits per free letter**, with
   an estimator reimplementable in a page of code, and its stability across
   span lengths and segmentation strategies (§4).
2. **A ceiling predicted from that price and tested from seven independent
   construction routes**, with the negative results reported in full (§6).
3. **A filter/ranker separation for proxy metrics**, with blind judging on both
   sides, two demonstrated gaming modes, and the recommendation that papers
   state which role a proxy plays (§8).
4. **Artifacts**: mined mirror-pairs with attestation flags, blind-judged
   self-palindromic centres, catalogued palindromes stored with their spelling,
   generated novel pairs, and a nine-criterion conjunctive task definition
   whose test suite currently fails on purpose (Appendix A).

Nobody has measured the price of a formal textual constraint in bits and used
it to predict a readability ceiling; nobody has published the negative results
that bound this task; and nobody has separated the filter role from the ranker
role of proxy metrics with blind judging on both sides.

A reader who never thinks about palindromes should care because the estimator
does not depend on anything specific to mirrors. Any formal constraint that
restricts which strings are admissible has a price in the same units, obtained
the same way: score the unconstrained text, score the constrained text under
the same model and vocabulary, and take the difference per unit of free choice.
A lipogram removes letters from the alphabet, and its price is the entropy of
what was removed. A rhyme scheme constrains line-final words, and its price is
per rhymed position. Meter, acrostics, and output-format grammars such as a
JSON schema or a regular expression all restrict the continuation set at every
step, and all admit the same arithmetic. Format restrictions are already known
to degrade model reasoning; what has not been available is a way to say by how
much, in units that compare across constraints and predict where a constrained
decoder stops producing usable text. The palindrome is the case where that can
be validated, because the constraint is severe enough that the ceiling is
reachable with a laptop and a dictionary.

---

## 4. Positioning against Papadopoulos, Roy, Régin & Pachet (IJCAI 2015)

*Draft of the paragraph that opens Related Work §2.1. This is the paragraph the
novelty claim depends on.*

> The closest prior work is Papadopoulos et al. (2015), which gives a graph
> structure that enumerates **all** palindromes obtainable from an n-gram
> corpus in time linear in the size of the structure, handles the simultaneous
> character-level and word-level coupling that makes the constraint awkward to
> express, and steers the output semantically by biasing word probabilities
> from an auxiliary corpus. Their contribution is completeness: given a corpus,
> the set of reachable palindromes is characterised and can be walked. We take
> that result as given and ask a question it does not address. Their own paper
> observes in passing that long palindromes are often less meaningful; that
> observation is the object of study here, and we answer it with a measurement
> rather than an algorithm. The distinction matters in three places. (i) Their
> completeness result makes our negative results stronger, not weaker: if the
> reachable set can be enumerated in full, then a search that finds nothing
> readable is reporting a property of the space rather than a failure of the
> search, and our exhaustive walk of 2.55M short palindromes with 0 of 170
> strict-rubric passes is the empirical form of that argument. (ii) Their
> semantic steering is a *ranking* method over the enumerated set, biased by
> corpus word probabilities. Our proxy audit finds that ranking is exactly
> where automatic signals fail on this material: four proxies, each sound as a
> filter and each failed as a ranker under blind judging, and an attestation
> constraint of the kind corpus biasing implements would discard roughly 93% of
> the 29 catalogued mirror-pairs it is meant to be finding more of, because a
> palindrome's English is unusual English and frequency lists record the usual.
> (iii) Their cost is measured in complexity; ours is measured in bits of the
> language, which is the unit that transfers to lipograms, rhyme, meter and
> output-format grammars. The two results are complementary. Theirs says what
> is reachable; ours says what reaching it costs and why almost none of it
> reads.

Three things to get right when this is compressed to ACL length:

- Do not imply they attempted a readability claim and failed. They did not
  attempt one.
- Do not claim our search is better than their enumeration. It is not; it is
  Norvig's, and it is a different algorithm aimed at a different regime.
- Keep the "completeness strengthens our negative result" move. It is the
  sentence that converts the nearest competitor into supporting evidence, and
  it is the single most load-bearing sentence in the related-work section.

---

## 5. NAACL 2027 submission plan

Working backward from the ARR October cycle deadline, **12 October 2026**.
Today is 19 August 2026: 54 days, or roughly 7.5 weeks.

Downstream dates: ARR reviews and meta-reviews return during the cycle
(dates to confirm on the ARR site); **commitment to NAACL 2027 closes 20
December 2026**; NAACL 2027 is in San Francisco, **1–5 June 2027**.

### 5.1 Load-bearing experiments

Two are named in `docs/publication-plan.md` §6 as things a reviewer will ask
for, and the paper is materially weaker without either.

**E1 — Model-scale replication of the bits-per-free-letter measurement.**
*Highest priority; start first because everything in §4 of the paper depends on
it and because it may change the abstract's headline number.*

Before it can be run, the estimator needs to exist as a script. No file in this
repository computes 1.63, 4.92, or 3.296; the numbers are recorded in prose in
`README.md`, `docs/NORTH-STAR.md`, `docs/training.md` and
`llm_palindrome/paragraphs.py`, and the closest runnable code is
`llm_palindrome/respace.py` (segmentation) and `llm_palindrome/lm_scoring.py`
(scoring). Writing `experiments/mirror_cost.py` is therefore step zero, and it
is also what makes the claim reproducible for the artifact appendix.

| date | step |
|---|---|
| Aug 19–23 | Write `experiments/mirror_cost.py`: forward score, reverse, optimal re-segmentation, per-free-letter difference. Reproduce 1.63 / 4.92 / 3.296 under GPT-2 small, or record what it actually produces. |
| Aug 24–30 | Sweep model scale within one family (gpt2, gpt2-medium, gpt2-large, gpt2-xl) and at least one non-GPT-2 family (Pythia and/or Qwen2.5-0.5B, which `experiments/oracle_bound.py` already loads). Sweep vocabulary size (3k / 14k / 30k / 52,927 lexicon) and span length. |
| Aug 31–Sep 3 | Table and figure for §4. If the number drifts across scales, report the drift as a range with its dependencies named; the paper survives either outcome, but the abstract has to be rewritten if it drifts. |

**E2 — Frontier-LLM baseline.**
*Second priority. It is the cleanest demonstration that the search, not the
model, is what makes the constraint hold.*

| date | step |
|---|---|
| Sep 4–10 | Prompt current frontier models for palindromes at several target lengths (e.g. 30, 60, 120, 200 letters), several prompt formulations, fixed sample count per cell. Score validity with `llm_palindrome/validator.is_palindrome` — the repository's only arbiter — and novelty with `is_novel_palindrome` against `data/known_palindromes.json` (160 entries), because an authoring model in this repository already claimed 20 novel palindromes of which all 20 were catalogued classics. |
| Sep 11–13 | Readability of whatever passes validity, scored through the same blind protocol as E3. Write §9. |

**Supporting runs, if time allows (drop first if it does not):** confirm the
model-independence of the prepend/append asymmetry (§7 of the paper), and
re-measure the center-out row in `docs/architecture.md`, which has not been
re-run since the adjacency contract in `scoring.adjacent` was fixed.

### 5.2 Human evaluation

`docs/publication-plan.md` §6 item 3 calls this a desk-reject risk if
unreported, and it is the largest single gap between what the repository has
and what an ACL venue requires. **What the repository calls "blind judging" is
a blinded batch scored by a language model with real-prose and word-salad
calibration controls** (`runs/blind_key.json`, `runs/top26_strict.json`,
`runs/centre_verdicts.json`), not human annotation. No annotator count,
agreement statistic, or verbatim instruction set exists in any file. A paper
that reports these verdicts as "blind judging" without saying who judged will
be caught.

Two options, in order of preference:

- **Run a real human batch (Sep 14–20).** Three or more annotators, the same
  item set, the same real-prose and word-salad controls, pre-registered
  instructions, and a reported agreement statistic (Krippendorff's α or
  pairwise agreement — report the raw agreement too, since the label
  distribution here is skewed). Items per condition sized so the paper can say
  something about the C3 filter/ranker claim specifically. An under-powered
  study that reports its power is acceptable; an unreported one is not.
- **If no human batch is feasible:** relabel every verdict in the paper as
  *LLM-judge* evaluation, report the calibration controls and their pass rates
  as the validity evidence for the judge, cite the LLM-judge-bias literature,
  and state the limitation in §11. This is defensible for a negative-results
  argument and is not defensible if the paper's language implies human raters.

Either way, the exact instruction text goes in an appendix verbatim, and the
single-verdict instability the repository has already measured (borderline
items flip about half the time; three prior findings in `docs/training.md` died
this way) is reported as a methodological result rather than buried.

### 5.3 Anonymity, preprint, and artifacts

- **Confirm the current ARR anonymity and preprint policy on the ARR site
  before doing anything else here** (`docs/publication-plan.md` §3 flags this
  too). The policy has changed between cycles and should not be recalled from
  memory. Deadline for the check: **26 August**.
- **This repository de-anonymises the submission.** It is public, the live
  service is at `palindrome.ericspencer.us`, `CITATION.cff` names an author,
  and the README credits the AI4FM group. A paper that cites its own artifacts
  by URL is not anonymous. Mirror the artifacts to an anonymised host
  (anonymous.4open.science or equivalent) and reference that in the submitted
  version. **Do this before drafting the appendix**, since every reproduction
  command in it carries a path.
- **Settle authorship first** (`docs/publication-plan.md` §7): the README
  credits the AI4FM group and `CITATION.cff` lists one author. Deadline:
  **23 August**, because it determines who reviews the draft and how the
  acknowledgements read.
- **Pin the artifacts (Oct 5–9).** Release-tag the datasets, add licences to
  `data/`, and record the provenance split between catalogued and generated
  material — the catalogued sentences are other people's text and the paper
  publishes them.
- **arXiv:** post only after the ARR policy check confirms it is permitted for
  this cycle, and post the same version that was submitted.

### 5.4 Writing schedule

| week | dates | writing | running |
|---|---|---|---|
| 0 | Aug 19–23 | Freeze notation and the definition of "a free letter". Draft §3 (problem and formalisation: the overhang as sufficient statistic; the nine-criterion conjunctive definition of "reads"). | Write `experiments/mirror_cost.py`. Settle authorship. |
| 1 | Aug 24–30 | Draft §1 introduction and §2 related work (six threads, one paragraph each, Papadopoulos paragraph first). | E1 sweeps. ARR policy check. |
| 2 | Aug 31–Sep 6 | Draft §4 (the price of a mirror) against E1's actual output. Build the convergence table for §6 — it is the paper's centrepiece and should be readable in ten seconds. | E1 tables. Start E2 prompting. |
| 3 | Sep 7–13 | Draft §5 (two systems) and §6 (the ceiling, one short subsection per route). | Finish E2. |
| 4 | Sep 14–20 | Draft §8 (evaluation methodology) and §9 (LLM baselines). | Human evaluation batch, or the relabelling decision. |
| 5 | Sep 21–27 | Draft §7 (directional asymmetry — see the caveat below), §10 (discussion and transfer), §11 limitations, §12 conclusion. | Agreement statistics. Anonymised artifact mirror. |
| 6 | Sep 28–Oct 4 | Full internal read. Cut to 8 pages. Appendices: judging instructions verbatim, negative-results log, per-experiment reproduction commands. | Re-run anything the draft revealed as missing. |
| 7 | Oct 5–11 | Two external readers, at least one outside the project. Limitations section rewritten last, against the final numbers. Anonymity sweep of the PDF (metadata, URLs, acknowledgements). | Artifact release tags. |
| — | **Oct 12** | **Submit to ARR.** | |
| — | Oct 13–Dec 19 | Reviews, author response, revision. | |
| — | **Dec 20** | **Commit to NAACL 2027.** | |

**Caveat carried into week 5.** §7 as scoped in `docs/publication-plan.md` (C5,
"the half built by prepending reads measurably worse") does not survive the
repository's own later measurement. `docs/architecture.md:27–37` and
`docs/training.md:83–94` record that the +0.368 per-letter gap reverses to
−0.824 per token, and that the prepended half does not read worse — it has
shorter words. §7 must be written from the per-token numbers and from the
backward-LM result that does survive (+0.423 per token over the plain baseline
at the best weight, 24/24 closure), or dropped. Writing it from the C5 framing
in the publication plan would put a refuted claim in the paper.

### 5.5 Fallbacks

- If E1 shows the price drifting badly across model families, the paper becomes
  "the price is model-dependent and here is its range", which is still
  publishable and is a more honest §4. Do not delay the submission for a
  cleaner number.
- If the October cycle slips, `docs/publication-plan.md` §3 lists INLG 2027
  (deadline around July 2027) and the CL journal route, and the short C3-only
  paper remains available for the Insights workshop.

---

## 6. Numbers to verify

Items marked **[TK]** are used above or are required before submission, and are
not currently recoverable from a script in this repository.

1. **[TK] The estimator behind 1.63 / 4.92 / 3.296 bits.** No file computes
   these. They appear as prose in `README.md:201–203`,
   `docs/NORTH-STAR.md:122–125`, `docs/training.md:386`, and
   `llm_palindrome/paragraphs.py:3`. Model, corpus, vocabulary size,
   segmentation algorithm and span lengths are all unrecorded. The abstract
   states GPT-2 small on the project lead's assertion; that needs a file.
2. **[TK] Vocabulary used for the measurement.** The abstract says
   "frequency-ranked vocabulary" without a size, because the repository does
   not state one. Candidates in use elsewhere: 30,000 (`--vocab` default),
   14,000 (the walk), 52,927 (`data/lexicon.txt`).
3. **[TK] "Stable across span lengths and segmentation strategies."** Asserted
   in three files, tabulated in none.
4. **Arithmetic tension in the thinning rate.** The repository says "roughly
   10× every three letters" (`README.md:204`, `docs/NORTH-STAR.md:124`).
   2^3.296 = 9.82, which is 10× per *one* free letter. Either the stated rate
   or the stated cost needs correcting before either appears in the paper.
5. **[TK] Human-evaluation reporting.** Annotator count, items per condition,
   agreement statistic, and verbatim instructions do not exist. What exists is
   LLM-judge blind batches with real-prose and word-salad calibration controls.
   See §5.2.
6. **Data-file counts disagree between documents.** `data/canon_spelled.json`
   holds 71 entries; `docs/training.md:556` says 56 spellings.
   `data/centres.json` holds 49; the commit message for 884e79e and
   `docs/training.md` say 40. `data/known_palindromes.json` holds 160;
   `docs/training.md:463` describes it as the 120-entry reference. Version
   drift, harmless in the repository, quotable-and-wrong in a paper.
7. **"Seven routes" is two different lists.** `docs/publication-plan.md` §C2
   enumerates beam search, exhaustive enumeration, mining, reversible chains,
   LLM authoring, authored-half segmentation, and the sharded walk. The
   README's source table has seven rows but splits reversibles into two
   (closed-form and exhaustive chains) and omits beam search. Pick one list and
   use it everywhere; the introduction above uses the publication-plan list.
8. **"Exactly the ceiling" overstates the convergence.** Yields differ by
   route: mining gives 26 usable halves, LLM authoring ~3, the walks ~0
   readable (`README.md`, "Where the units come from"). What the routes agree
   on is that four-word halves do not exist, not that their yields match. The
   introduction above says "stop in the same place", which is defensible; the
   publication plan's "hit exactly the ceiling" is not.
9. **Exhaustive-walk totals need one canonical figure.** The README's route
   table says 20,000 produced; `docs/training.md:453` and
   `llm_palindrome/paragraphs.py:24` report 2,550,331 at 18–28 letters and 12M
   closures across bands. These are different runs and the paper must say
   which is which.
10. **C5 is refuted by later measurements in the same repository.** See the
    caveat in §5.4. `docs/publication-plan.md` §C5 should be rewritten before
    §7 of the paper is drafted.
11. **[TK] ARR October 2026 review and rebuttal dates**, and the current ARR
    anonymity/preprint policy. Neither is recorded here; confirm on the ARR
    site by 26 August.
