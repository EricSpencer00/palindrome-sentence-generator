# Quarantined historical NLP positioning

This document describes retired mirror-cost, POS-throughput, and live-service
framing. It is not the current research position and must not guide a paper or
release. See [the target-paper specification](../paper/TARGET-PAPER-SPEC.md):
the only target is a long, independently generated readable exact English
palindrome backed by blinded human-reader evidence.

## Historical positioning notes

## Decision

This belongs in NLP as a study of **exact constrained natural-language
generation**, not as a record-length claim and not, by itself, as a new theorem
about constraint propagation.  The strongest central question already answered
by the repository is:

> What does an exact character-level reversal constraint do to the linguistic
> feasibility of English, and how can a decoder preserve that constraint while
> still searching for lexical and structural quality?

An exact palindrome forces one letter sequence to support two readings: its
ordinary direction and its reversed, re-segmented direction.  That is a
language modelling, lexical segmentation, constrained decoding, and evaluation
problem.  It is much stronger than a generic string-puzzle framing.

The present short paper, `paper/naacl2027.tex`, establishes a narrower,
structural result: a sound POS-shape viability gate improves accepted candidate
yield under a fixed search budget.  It should not be asked to prove that the
project generates grammatical or meaningful prose.  Those are different claims
with different evidence.

## The language result already in hand

`experiments/mirror_cost.py` measures the *mirror cost* without using a
generated palindrome as the object of study.  It takes ordinary English spans,
removes formatting, and re-segments both the forward and reversed letter
sequences with the same lexicon and procedure.  A causal language model scores
both readings per letter.  Re-segmenting both sides matters: comparing natural
spacing with reverse re-segmentation would confuse reversal with the cost of
changing word boundaries.

The retained run contains 57 model/segmentation/length cells, each based on
150 WikiText-2 spans.  It reports a reverse-reading penalty of **2.16–3.60
bits per free letter** (mean 2.94).  In the representative 80-letter unigram
segmentation condition, GPT-2, DistilGPT-2, and GPT-2-medium report
2.88–3.16 bits/letter; about 91% of forward letters but only 52% of reversed
letters are contained in dictionary words of at least three letters.  The
current models are related GPT-2-family models, so this is robustness across
sizes, not an architecture-independent conclusion.

This result explains two otherwise disconnected observations:

1. A deterministic two-ended decoder can guarantee exact character validity,
   yet most long candidates remain poor English.
2. Allowing obscure reversible material makes length cheap while making the
   result less like ordinary English.

The last step of the argument must be phrased carefully.  `2^mirror_cost` is a
likelihood-derived intuition for how quickly usable material thins; it is not a
direct count of grammatical strings or a theorem about English.

## Evidence map

| Evidence | What it supports | What it does not support |
|---|---|---|
| `experiments/mirror_cost.json` | Reverse-readable English has a large, stable-within-this-study model penalty; reverse lexical coverage is much lower. | A universal information-theoretic constant, a human quality score, or a count of all feasible texts. |
| `llm_palindrome/search.py`, `centerout.py`, and `validator.py` | A decoder can make exact normalized-letter palindrome validity a hard invariant rather than an LLM preference. | Readability or coherent meaning. |
| `runs/controlled-pos-pruning-openings-2026-09-11/RESULTS.md` | On 50 paired openings, the POS-shape gate gives 5.469x accepted candidates per popped state and 4.364x per CPU second, while changing the searched region. | Grammar: the accepted POS patterns admit clear false positives. |
| `experiments/RESULTS-graph.md` | Compiling overhang states makes short structural candidate supply much faster than the prior re-walk in the measured regimes. | That the additional candidates are better language; the document reports no high coherence scores for the long samples. |
| `artifacts/norvig-v3/audit.json` | A reproducible 90,937-letter result exceeds the published Norvig v3 text by 498 letters under its stated inventory and repetition restrictions. | A language-quality or world-record claim. |
| `web/` plus the live `palindrome.ericspencer.us` service | A public, interactive system exposes exact palindromic composition and generation rather than a static example. | Scientific usefulness, adoption, or human preference. |
| `experiments/RESULTS-revision-2026-09-07.md` | Automated evaluators, including a deliberately calibrated protocol, have important failure modes on this task. | Human evaluation: the repository explicitly records that historical subagent verdicts are not human judgments and its human-rating CSV is blank. |

The last row is important.  The honest NLP story is not “we solved semantic
generation.”  It is “we measured why exact bidirectional text is difficult,
and we built a verifiable decoder whose language constraints and evaluation
limits are explicit.”

## Recommended paper split

### 1. Primary research paper: the mirror-cost study

**Working title:** *The Mirror Cost of English: Exact Palindromic Generation as
Bidirectional Re-segmentation*

This paper should lead with the linguistic phenomenon, then use the generator
as a concrete constrained-decoding case study.

1. Define exact letter palindromes and the two-reading requirement.
2. Introduce the symmetric re-segmentation measurement; explain why natural
   spacing is not the control.
3. Report reverse lexical coverage and per-letter LM cost across corpora,
   lexica, unrelated language-model families, span lengths, and segmentation
   strategies.
4. Show the decoder: exact validator, overhang search/graph, and the POS-shape
   gate as feasibility constraints.
5. Use generation results to show the practical consequence: validity and
   candidate supply can improve without implying prose quality.
6. State a negative result clearly: current automatic selection is insufficient
   for coherent long palindromes.

This is an NLP paper because its unit of analysis is English under an unusual
but exact formal constraint.  It contributes a measurement protocol and an
empirical account of the constraint's interaction with lexical segmentation
and language-model likelihoods.

### 2. Separate system-demonstration paper: the live generator

**Working title:** *Palindrome: An Interactive, Verifiable System for Exact
Constrained Text Generation*

The deployed site is valuable here.  A system paper can demonstrate:

- exactness is checked server-side and independently testable;
- a visitor can request a length and theme, see the construction visualized,
  and inspect the mirror relation;
- the v3 bank moves expensive constrained search offline, giving responsive
  generation; and
- the UI preserves provenance and distinguishes generated material from
  catalogue material.

It should include task success/latency/diversity/property-test figures and a
small usability evaluation.  It must label composition as composition and must
not market the output as coherent prose.

### 3. Keep the current POS-pruning paper narrow

The current paper can remain a workshop or Findings-oriented constrained-search
study if it is reframed around the fixed-work observation: sound pruning can
increase useful-output throughput while also increasing generated candidates,
because it reaches a different region of a non-uniform search tree.  It needs
the requested budget/opening-order robustness and a direct distinction from
[Papadopoulos et al. (2015)](https://www.ijcai.org/Proceedings/15/Papers/353.pdf),
who already combine syntactic patterns and palindrome generation.  Call the
constraint **POS-shape feasibility**, not grammar or syntax in the everyday
linguistic sense.

## Where the Norvig result belongs

The 90,937-letter artifact is real and useful, but it is evidence about the
other end of the trade-off.  The original Norvig page describes its long output
as losing plot and character development, and notes that readily reversible
dictionary material becomes scarce.  Our audited extension shows that
inventory-aware search can push this structural objective further, even with
tighter repetition guards.  It therefore belongs as:

- a reproducibility/scalability stress test for the exact decoder;
- a visual artifact in the system demonstration; and
- a motivating contrast in the mirror-cost paper.

It should not be the headline NLP result, benchmark for meaning, or evidence
that the generator writes English prose.  [Norvig's reference page](https://www.norvig.com/palindrome.html)
reports the 90,439-letter, 21,012-word v3 baseline and makes the same
semantic distinction.

## Minimum work before making a language-quality claim

Do not add “readable,” “grammatical,” “coherent,” or “meaningful” to an
abstract until a separate evaluation has passed all of these:

1. Freeze a generation configuration and comparable controls: real prose,
   shuffled-word salad, short catalogue palindromes, and the system output.
2. Recruit at least three independent human raters, blind the source, randomize
   presentation side/order, retain the written rubric, and report agreement.
3. Score grammaticality, identifiable subject/intent, and whole-text coherence
   separately.  Exactness is a fourth, deterministic property—not a proxy for
   any of the first three.
4. Treat every automated judge only as a prefilter or diagnostic until it is
   validated against this held-out human batch.  The repository's own
   pairwise/absolute disagreement is a reason for this guardrail.

## One paragraph for a proposal or cover note

> We study exact palindromic text as a test case for constrained natural
> language generation.  The constraint is unusually severe: a single letter
> sequence must support English readings in both its forward and reverse
> directions, with word boundaries recovered after reversal.  We introduce a
> symmetric re-segmentation protocol to measure this burden, and find a large
> reverse-reading language-model penalty alongside a sharp drop in reverse
> lexical coverage.  We then build a verifiable two-ended decoder that enforces
> character-level validity exactly while incorporating lexical and POS-shape
> feasibility constraints.  Our results separate structural search progress
> from language quality: exact candidate yield can improve substantially even
> when the current generator has not achieved coherent long prose.  This makes
> palindromic generation a controlled setting for studying how formal output
> constraints interact with segmentation, language modelling, and evaluation.

## Next experiment

The highest-value, reversible next step is a fixed mirror-cost replication:
two held-out English corpora, at least one non-GPT-2 causal LM family, the
current three segmentation strategies, and saved spans/lexicon hashes.  That
turns the strongest existing observation into a publication-grade result
without pretending that a new text generator has already solved coherence.

If the goal is a system-demonstration submission instead, run a modest,
separate usability study of the live site and report it as product evaluation;
do not use it to answer the language-quality question.
