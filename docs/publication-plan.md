# Publication plan

What this repository has that is publishable, where it can go, and what the
paper looks like at two lengths. Written 19 August 2026.

---

## 1. What the contribution actually is

The publishable finding is not "we built a palindrome generator." Generators
exist and one of them is already in the literature. The finding is:

> **A character-level palindrome constraint costs a measurable, stable number
> of bits per letter over English, and that cost — not model capacity, not
> search width, not reranker quality — is what bounds readability.** The cost
> is measured, its consequence is predicted, seven independent construction
> routes are shown to hit exactly the ceiling it predicts, and no automatic
> proxy is able to rank what survives.

That decomposes into three claims that can be defended separately, plus three
supporting ones.

### C1 — The constraint has a price, and it can be measured

Forward English scores one entropy rate under a model and vocabulary; the same
letters reversed and optimally re-segmented into the same vocabulary score
another; the difference is the constraint's price per free letter. It is stable
across span lengths and across segmentation strategies.

Why this is the lead claim: constrained-generation papers report *satisfaction
rate* and *fluency*, almost never *the price of the constraint*. This is a
Shannon-style measurement of a formal constraint, and it converts a widely
repeated anecdote ("long palindromes are nonsense") into a number with a
falsifiable consequence — the coherent feasible set thins geometrically in
length at a rate the number predicts.

The method generalises immediately to lipograms, rhyme, meter, acrostics,
syllable counts, and output-format grammars. That is the transfer.

### C2 — The predicted ceiling is real, and was hit from seven directions

Everything in `docs/NORTH-STAR.md` §"What the constraint actually costs" and
the `experiments/` tree. The routes: LM-scored beam search (rerank and
in-loop); exhaustive enumeration in the short regime; mining attested corpus
n-grams; closed-form reversible-word chains; LLM authoring; author-one-half-
and-segment-the-other; a sharded vocabulary walk. All converge on the same
wall, and the sharpest single number is that zero of tens of thousands of
attested four-grams has a mirror that reads.

This is what makes C1 more than an arithmetic exercise. The number predicted a
wall; seven methods found it.

### C3 — Automatic proxies can filter but cannot rank

Under blind judging with real-prose and word-salad controls, four proxies (LM
score, attestation, thematic cohesion, word frequency) were each *sound as
filters* — they reject nothing a human accepts — and each *failed as rankers*.
Four proxies, zero agreements on ranking finished text.

Two demonstrations of gaming come with it, and they are the memorable part:
a per-letter-normalised LM score rose substantially under a policy that only
found longer words, while the same texts scored worse per token; and a
pre-registered adversarial check found the coherence gain could be won several
times over by constructions that write nothing.

The generalisable recommendation: **report which role a proxy plays.** Filter
and ranker are different claims with different validation burdens, and the
field routinely validates one and uses the other.

### Supporting claims

- **C4 — Structure.** Palindromes compose: units that pay the constraint
  internally nest like brackets, so length stops being the binding problem and
  unit *selection* becomes it. With that comes a small taxonomy of the three
  ways to pay a mirror across a paragraph (free-running, self-palindromic
  units, mirror-pairs) and the argument that self-palindromic units *force*
  sequence repetition rather than merely inviting it.
- **C5 — Directional asymmetry. RETRACTED as originally stated (19 Aug).**
  I wrote that the prepended half reads measurably worse in both growth
  directions. `docs/architecture.md:19-37` refutes this: the per-letter gap
  (+0.443) reverses to −0.824 per token, and the prepended half simply has
  shorter words. The repository already says the row "does not survive as
  evidence that backward construction produces less readable text." What may
  survive is the per-token backward-LM result — the gain and the closure rate
  in `experiments/backward_study.py` — and §7 of the long paper has to be
  rebuilt on that or dropped. **Do not draft §7 until this is re-measured.**
  This is also the paper's own cautionary tale: the normalisation that gamed
  `lm_score` gamed this measurement too, in the same direction, and the repo
  caught it. That is worth reporting in §8 as a second instance rather than
  losing.
- **C6 — Artifacts.** Mined mirror-pairs with attestation flags; blind-judged
  self-palindromic centres; catalogued palindromes stored with spelling; novel
  generated pairs; and a nine-criterion conjunctive benchmark with a test suite
  that currently fails on purpose. The benchmark is a contribution on its own —
  it is a task definition that resists the shortcuts that were actually taken.

### The honest framing of what does not work

The paper must say plainly that the system does not produce coherent novel
palindromic prose, that the readable output is assembled from catalogued
material, and that the generated-material mode is fragments. This is the
paper's credibility, not its weakness — and it is why a negative-results venue
is a genuine fit rather than a consolation prize.

---

## 2. Literature review

### 2.1 Direct prior work on palindrome generation

- **Papadopoulos, Roy, Régin & Pachet (IJCAI 2015), "Generating all Possible
  Palindromes from Ngram Corpora."** The closest paper in existence and the one
  that must be positioned against first. It gives a graph structure that
  enumerates *all* palindromes obtainable from an n-gram corpus in linear
  complexity, handling the simultaneous character/word-level coupling, and
  biases word probabilities from an auxiliary corpus to steer semantics.
  **Positioning:** they solve feasibility and completeness. This work takes
  feasibility as given and asks the different question of why the output does
  not read — answering with a cost measurement rather than an algorithm. Their
  own framing ("long palindromes are often less meaningful") is precisely the
  observation being quantified here. Complementary, not competing, but the
  related-work section lives or dies on saying that convincingly.
- **Norvig (2002), "World's Longest Palindrome?" and "The Algorithm"**, building
  on **Hoey (1984)**. The two-sided overhang search this repository uses.
  Engineering write-ups rather than peer-reviewed work; cite as the algorithmic
  ancestor and be explicit that the search is theirs and the scoring layer is
  the contribution.
- **Chinese Palindrome Poetry Generation (CPPGM, 2020)** — seq2seq plus a
  constrained beam search for a different constraint at a different level, in a
  different language. Cite for coverage; not a baseline.

### 2.2 Constrained decoding — the audience the paper is written for

- NeuroLogic Decoding (predicate-logic lexical constraints via modified beam
  search) and NeuroLogic A\*esque (lookahead heuristics for future cost).
  The lookahead framing is the nearest analogue to the overhang trie here.
- COLD Decoding (energy-based, Langevin dynamics), grid beam search / dynamic
  beam allocation, and the classical lexically constrained decoding line.
- Grammar- and automaton-constrained decoding: efficient GCD, grammar-aligned
  decoding (ASAp) and the result that naive constrained decoding *distorts* the
  model distribution rather than conditioning on it.
- Sequential Monte Carlo steering (Lew et al., 2023) and the syntactic/semantic
  SMC control line — the principled way to condition on a hard constraint, and
  the obvious "why didn't you do it this way" reviewer question. Answer it in
  the paper: the palindrome constraint's feasibility check is not incremental
  in the usual sense, and the overhang is exactly the sufficient statistic that
  makes it so. That is worth a paragraph.
- **"Let Me Speak Freely?" (EMNLP Industry 2024)** — format restrictions degrade
  reasoning. The general form of the finding here. Cite as the evidence that
  the field already suspects constraints have a price, and offer the
  measurement as the way to state it in units.
- Garbacea & Mei, "Why is constrained neural language generation particularly
  challenging?" — the survey to anchor the framing paragraph.
- Evaluation of constrained generation: accuracy/coverage/PPL benchmarks, and
  the Oulipo-inspired benchmarks (OuLiBench; lipogram studies with frontier
  models; the word-play dimensions in recent multilingual benchmarks) which
  report frontier-model failure rates on exactly this family of constraints.

### 2.3 Information theory of English

- **Shannon (1951), "Prediction and Entropy of Printed English"** — the ~1.3
  bits/character estimate and the method. The paper's measurement is a direct
  descendant and should be presented that way.
- **Cover & King (1978)** gambling estimate (~1.25–1.35 bits/char), plus the
  modern neural-LM entropy estimates. Needed to justify why a model-derived
  bits-per-letter figure is a meaningful quantity and how to report the
  model-dependence honestly.
- **Linguistic steganography** as the surprising neighbour: that literature
  measures exactly "how many bits of arbitrary constraint can text absorb per
  word before it stops reading," in bits-per-word/bits-per-token. The
  palindrome cost is the same currency pointed the other way. A reviewer who
  knows this literature will be pleased to see it cited; nobody in the
  palindrome line has.

### 2.4 Directionality and backward construction

- The reversal curse and the factorization-curse analysis; reverse training;
  reverse/right-to-left LMs (e.g. LEDOM) and noisy-channel forward+reverse
  scoring. This is the literature C5 speaks to, and the one that makes the
  prepend/append asymmetry a result rather than a curiosity.
- Infilling and non-left-to-right generation: fill-in-the-middle training,
  blank language models, insertion/edit-based generation. Relevant because the
  palindrome is a two-sided infilling problem where the two sides constrain
  each other through a channel that is not the text.
- Theory: limitations of autoregressive models, and the pitfalls of
  next-token prediction. Supports the claim that a left-to-right model cannot
  enforce a constraint whose last character is decided by its first.

### 2.5 Character-level competence

- CUTE and its successors (multilingual token-understanding benchmarks,
  character-reasoning benchmarks, sub-token benchmarks), the counting-ability
  work, and byte-level / stochastic-tokenisation remedies.
- Purpose in the paper: explains why "just prompt a frontier model" is not a
  solution and grounds the LLM baseline that must be run. It also lets the
  paper make the sharper point that **the search, not the model, is what
  guarantees validity here** — which cleanly separates constraint satisfaction
  from generation quality in a way most constrained-decoding benchmarks
  conflate.

### 2.6 Evaluation of creative and constrained text

- The standing result that BLEU/ROUGE/perplexity correlate weakly with human
  judgement on creative text, especially poetry; the rap-lyric ghostwriting
  evaluation work as an early careful treatment.
- Reward hacking and Goodhart effects: length bias in reward models and in
  LLM-judge evaluation, length-controlled evaluation as the remedy, and the
  general specification-gaming framing. The per-letter-normalisation gaming
  result belongs directly in this conversation.
- Computational creativity evaluation frameworks, for the ICCC-facing version.

### 2.7 Where the gap is

Nobody has measured the price of a formal textual constraint in bits and used
it to predict a readability ceiling; nobody has published the negative results
that bound this task; and nobody has separated the filter role from the ranker
role of proxy metrics with blind judging on both sides. Those three sentences
are the paper's novelty claim and should appear nearly verbatim in the
introduction.

---

## 3. Venues

Assessed as of 19 August 2026.

### Closed for this cycle

| Venue | Why it's closed |
|---|---|
| EMNLP 2026 main (Budapest, Oct 24–29) | ARR deadline 25 May 2026; commitment 2 Aug |
| AACL-IJCNLP 2026 (Hengqin, Nov 6–10) | ARR 25 May; commitment 26 Aug — no new submissions |
| INLG 2026 (Utrecht, Oct 17–21) | Direct 18 Jul; ARR commitment 5 Aug |
| EACL 2027 main (Athens, Mar 9–14) | ARR 3 Aug 2026 was the *only* viable cycle |
| ARR August cycle | Closed 3 Aug 2026 |
| NeurIPS 2026 Creative AI track | Extended deadline 10 Aug 2026; non-archival anyway |
| CoNLL 2026, ICCC'26 | Both already held |
| Word Ways (recreational linguistics) | Ceased publication in 2020 |

### Open, ranked

**1. Insights from Negative Results in NLP @ EMNLP 2026** — Budapest, co-located
with EMNLP, week of 22–29 Oct. Short papers plus non-archival abstracts;
explicitly wants results that "highlight methodological issues with existing
approaches" and "point out pervasive misunderstandings or bad practices."
C3 alone is a textbook fit and C2 is the supporting evidence. Archival, in the
ACL Anthology, and the fastest route to a citable paper.

*Caveat I could not resolve:* the 2026 CFP page is not yet reachable at the
usual URLs, so the submission deadline is unconfirmed. Historically this
workshop's direct deadline lands a few weeks after the main-conference
notification. **Action: email insights-workshop-organizers@googlegroups.com
this week and ask.** If it has passed, everything below still stands.

**2. ARR October cycle — deadline 12 October 2026.** Feeds NAACL 2027 (San
Francisco, 1–5 Jun 2027), COLING 2027, and ACL 2027; commitment 20 Dec 2026.
This is the main-conference route and the target for the long paper. Roughly
eight weeks of runway from today, which is enough to run the two experiments
listed in §6 and write properly.

**3. Computational Linguistics (MIT Press) or TACL** — rolling submission, no
deadline pressure, no page limit fight. The right home for the longest version:
a measurement paper with an extensive negative-results log is exactly what CL
publishes well and what a 4-page venue mutilates. Slower, but the version of
this work with all seven routes reported in full belongs here.

**4. LaTeCH-CLfL 2027 @ EACL 2027** (Athens, March 2027). CFP will open around
Dec 2026/Jan 2027. The humanities-and-creative-language framing: constrained
writing, Oulipo, the recreational-linguistics tradition, provenance and
attribution of catalogued material. A good home for a version that leads with
C4 and the north-star benchmark.

**5. INLG 2027** — deadline around July 2027. The natural generation venue, and
it gives a Best Short Paper award. A fallback with a full year of runway, and
the place to send the follow-up if the October ARR round doesn't land.

**6. ICCC'27** (Association for Computational Creativity; CFP typically ~Feb).
The computational-creativity audience takes negative results and task
definitions seriously, and the nine-criterion spec with its "forbidden
shortcuts" section is unusually well suited to that room. Be careful to submit
to the ACC conference and not one of the similarly named commercial listings.

**7. arXiv, now.** Stake the claim while ARR runs. Costs nothing and the ARR
cycles do not require anonymity in the way that would forbid it — check the
current ARR preprint policy before posting.

### Recommendation

Two papers, not one.

- **Short → Insights @ EMNLP 2026** if the deadline permits; otherwise a short
  paper in the 12 Oct ARR cycle. Carries C3 (with C1 as setup).
- **Long → ARR 12 Oct 2026 → NAACL 2027.** Carries C1 + C2 as the primary
  result, with C3, C4, C5 as supporting sections.
- **Longest → CL journal**, after the conference version lands, with the full
  negative-results log and all seven routes.

Do not try to fit C1, C2 and C3 into four pages. Each is a paper's worth of
argument and the short version needs exactly one.

---

## 4. Short paper outline (4 pages + references)

One claim, one table, one figure. ACL style.

**Title** — name the measurement, not the artifact. Something in the shape of
"what a mirror costs" rather than "a system for generating X."

**1. Introduction** (~¾ page)
- The observation everyone repeats and nobody has priced.
- The claim in one sentence, in units.
- Three bullets of contribution: the measurement, the ceiling it predicts and
  the routes that hit it, the artifacts.
- One sentence on why a reader who never thinks about palindromes should care:
  the method transfers to any formal textual constraint.

**2. Background** (~⅓ page)
- Constrained decoding, one paragraph, ending at the observation that the price
  of a constraint is not usually reported.
- Prior palindrome generation: the combinatorial completeness result, and the
  one sentence that separates it from this work.
- Entropy of English, one paragraph, as the method's ancestor.

**3. The price of a mirror** (~1 page)
- Definition: what "a free letter" is, and why the forward/reversed-and-
  re-segmented pair is the right comparison.
- The estimator, stated so someone else can reimplement it in a page of code.
- Result, with the robustness checks that matter most (span length,
  segmentation strategy, and — see §6 — model scale).
- The derived consequence: the thinning rate of the feasible set.

**4. What the price predicts** (~1 page)
- One table: every construction route, what it produced, and where it stopped.
  This is the paper's centrepiece and it should be readable in ten seconds.
- The single sharpest row called out in prose.
- Half a paragraph on the one thing that does move: the cost bounds unit
  length, not palindrome length, and whole self-palindromic sentences pay the
  same price while carrying subjects.

**5. Limitations** (~¼ page, unnumbered per ACL convention)
- Model-dependence of the estimate. Vocabulary-dependence. English only.
- The system does not produce coherent novel palindromic prose; the readable
  output is assembled from catalogued material.
- Human judging sample sizes.

**6. Conclusion** (~¼ page)
- Restate in units; name the two other constraints the method should be pointed
  at next.

**Appendix** — reproduction commands, dataset descriptions, judging protocol.

### Alternative short paper, if aimed squarely at Insights

Same skeleton, different core. §3 becomes *the proxy audit* — four proxies,
each sound as a filter and each failed as a ranker, against blind judging with
prose and salad controls. §4 becomes *how a proxy gets gamed* — the
normalisation result and the pre-registered adversarial check — closing on the
recommendation that papers state which role a proxy is playing. §2 shrinks to
half a paragraph and the cost measurement becomes one sentence of setup.

---

## 5. Long paper outlines

### 5a. Conference long paper (8 pages + references + appendix)

**1. Introduction** — as the short version, wider. Add the framing that the
constraint here is *cheap to satisfy and expensive to satisfy well*, which is
what makes the task a clean instrument: validity comes from the search, so the
model can only affect readability. Most constrained-generation benchmarks
cannot separate those two.

**2. Related work** — the six threads of §2 above, in that order, each one
paragraph, each ending in a sentence that says what this paper does that the
thread does not.

**3. Problem and formalisation**
- The overhang formulation: the sufficient statistic that makes the constraint
  checkable incrementally, and why that matters for anyone building a
  constrained decoder.
- What "reads" means here, operationally, and the nine-criterion conjunctive
  definition. Emphasise conjunctive: not a scorecard to average.

**4. The price of a mirror**
- Method, estimator, controls.
- Robustness: span length, segmentation strategy, vocabulary size, model scale.
- The thinning rate and its prediction.

**5. Two systems, and what they establish**
- Search with LM scoring: rerank versus in-loop at matched budgets, with the
  cost multiplier. The result that matters is not which wins but that both are
  valid by construction.
- Compositional assembly: units nest like brackets; length stops being the
  binding constraint; the taxonomy of three ways to pay a mirror across a
  paragraph and the forced-repetition argument for one of them.

**6. The ceiling**
- Each construction route as a subsection of two to four sentences: what it
  assumed, what it produced, where it stopped.
- The convergence table.
- The attestation-filter result — that the filter which makes output *look*
  better would reject most of the catalogue it is meant to be finding more of —
  as the cautionary sub-result. It is a good story and it generalises.

**7. Directional asymmetry**
- The prepend/append gap, measured in both growth directions to isolate cause.
- The backward-LM test and its outcome, positioned against the reverse-LM
  literature.

**8. Evaluation methodology**
- The blind judging protocol, controls, annotator count, agreement.
- The proxy audit: filter versus ranker, four proxies.
- Metric gaming: the normalisation result, and the practice of adversarially
  checking a metric *before* pointing an optimiser at it.

**9. LLM baselines** — frontier models prompted directly, scored for validity
and for readability. (See §6; this section does not exist yet.)

**10. Discussion**
- Transfer: how to price a lipogram, a rhyme scheme, a meter, an output format.
- What a solution would have to look like, stated as a constraint on the search
  space rather than as a wish.
- What the field should take from the filter/ranker distinction.

**11. Limitations. 12. Conclusion.**

**Appendices** — artifacts and licences; provenance and attribution of
catalogued material; judging instructions verbatim; the full negative-results
log; per-experiment reproduction commands.

### 5b. The longest version (journal / thesis chapter, ~30–40 pages)

Same spine, with these expansions — each is material the repository already has
and a conference page limit would destroy.

- **A full history of the task.** Hoey 1984, the record-length lineage, Norvig,
  the combinatorial work, and the recreational-linguistics tradition. Nobody
  has written this and it would be cited.
- **The measurement chapter proper.** Multiple estimators, multiple models
  across scales, multiple vocabularies, multiple languages if reachable. Report
  the estimate as a range with its dependencies made explicit rather than as a
  single number. This is what makes the number citable by people who are not
  studying palindromes.
- **Every construction route in full**, one section each, including the ones
  that are three lines in the conference version: the exhaustive walk and what
  "exhaustive" stops meaning at scale; canon recall as an acceptance test for a
  search; k-best segmentation; the reversible-word closed form; the authoring
  experiments and why writing one half well ends the other badly.
- **The complete evaluation methodology**, as a contribution in its own right:
  the control design, why salad and real-prose controls are both needed, how to
  size a blind batch, and the pre-registration discipline for metrics.
- **The specification chapter.** The nine criteria, the taxonomy of shortcuts,
  and the argument that each shortcut is *structurally* tempting rather than
  merely careless — with the iteration numbers, which is the honest and unusual
  part. A methodological contribution about how constrained-generation projects
  drift toward easier constraints.
- **Provenance and attribution.** The catalogued-versus-generated distinction,
  the novelty check, and what it means to publish text a search assembled from
  other people's sentences. Small ethics section; genuinely relevant here.
- **Negative results appendix**, complete, with reproduction commands per
  experiment.

---

## 6. What to run before submitting

Two of these are load-bearing. A reviewer will ask both, and the paper is much
weaker without them.

0. **Write the estimator. The headline number has no code behind it.** Nothing
   in the repository computes 1.63, 4.92, or 3.296 — they appear as prose in
   `README.md:201`, `docs/NORTH-STAR.md:122`, `docs/training.md:491` and the
   `paragraphs.py` docstring, and the model, corpus, vocabulary size,
   segmentation algorithm and the span lengths over which stability was checked
   are unrecorded. The paper's central claim is currently unreproducible,
   including by its own author. `experiments/mirror_cost.py` is step zero of
   everything below, and if it does not reproduce the stated figures the paper
   reports what it actually finds.

   While writing it, fix the derived rate. README and NORTH-STAR both say the
   feasible set "thins by roughly 10× every three letters"; 2^3.296 = 9.8,
   which is 10× per *one* free letter. The two statements are not compatible
   and the second follows from the measurement.

1. **Replicate the cost measurement across model scales and at least one
   non-GPT-2 family.** The claim is that the price is a property of the language
   and the vocabulary rather than of one small 2019 model. Right now that is an
   assertion. If the number holds across scales, it becomes the paper's
   strongest asset; if it drifts, the paper reports the drift and is still
   publishable. Highest priority.
2. **A frontier-LLM baseline.** Prompt current models to produce long
   palindromes; score validity with the repository's own validator and
   readability with the blind protocol. The character-level-competence
   literature predicts they fail validity outright, which is the cleanest
   possible demonstration that the search — not the model — is what makes the
   constraint hold. Second priority.
3. **Resolve what "blind judging" means here, then report it properly.** This
   is the largest gap between what the repository has and what an ACL venue
   requires, and I understated it in the first draft. The design is sound —
   `runs/blind_key.json` with the batch revealed only after scoring, and
   real-prose and word-salad calibration controls in the same batch. What is
   missing is everything about the judge: the verdict files record one verdict
   per item with no annotator identity, no annotator count, no agreement
   statistic, and no verbatim instructions, and no script produces them. As it
   stands every "blind judging" result in the repository is n=1 from an
   undocumented judge.

   Two honest paths, and the choice drives the timeline more than anything
   else on this list. Either (a) run a real batch with multiple annotators,
   report agreement, and keep the strong claims; or (b) relabel every such
   result as single-judge blinded assessment with calibration controls, say so
   in the limitations, and cite the LLM-judge bias literature if the judge was
   a model. Path (b) is survivable for a workshop paper and risky for NAACL.
   Decide before drafting §8, not after.
4. **Confirm the model-independence of the prepend/append asymmetry.** Same
   argument as (1), smaller stakes.
5. **Pin the artifacts.** Release-tag the datasets, add licences, and update
   `CITATION.cff` with the paper once there is one. The repository is already
   unusually reproducible — make that legible to a reviewer in the first
   paragraph of the appendix.

## 7. Open questions for you

- **Authorship and the AI4FM affiliation.** The README credits the group; the
  `CITATION.cff` lists a single author. Settle who is on the paper before
  drafting, because it changes which venue is natural and who reviews the draft.
- **Which short paper.** C3 (proxies) is the better Insights fit; C1 (the
  measurement) is the better standalone. If both papers are going out, C3 goes
  to the workshop and C1+C2 goes to ARR.
- **How much of the paragraph work belongs in the first paper at all.** It may
  be a cleaner second paper — the assembly result, the north-star spec and the
  provenance argument together — rather than three sections of the first.
