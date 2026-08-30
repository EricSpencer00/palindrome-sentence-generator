# How to Find a Palindrome That Reads

Target: ARR October cycle, deadline 12 October 2026. Commitment to NAACL 2027
closes 20 December 2026. Conference is San Francisco, 1–5 June 2027.

## Files

| file | what it is |
|---|---|
| `naacl2027.tex` | the paper: four pages plus references |
| `refs.bib` | 29 entries, each checked against the publisher's own record |
| `one-paragraph.md` | the whole result in one paragraph, which the abstract is cut from |
| `long-form.md` | the full treatment, 15 sections; source material the four-page version discards |
| `abstract-and-intro.md` | superseded earlier draft, kept for its framing notes |
| `superseded-long-draft.tex` | superseded; `long-form.md` replaces it |
| `../docs/publication-plan.md` | contribution framing, venue analysis, literature review |

## The claim

Readability on this task is bounded by the material, not by the search.

The paper gets there through structure rather than through a number. The
overhang is a sufficient statistic for feasibility, which is what makes the
constraint checkable at all and what lets the constrained-decoding literature
apply once the search is posed over overhangs rather than prefixes. Over that
state the space has three regimes with a boundary at ~30 letters, located by
canon recall. Composition converts the joint search into a selection problem
over independently verifiable units, and closes one of the three nesting
schemes by algebra. Seven construction routes stop in the same
place, and the two levers usually assumed to raise the ceiling — more
candidates, more exploration — are measured and fail.

The bits figure (2.2–3.6 per free letter) is section 6, not the thesis. Its
model-free component — about half the letters of reversed English cannot be
placed in any dictionary word — carries most of the argument on its own.

### Why the framing changed

The first draft led with the bits measurement. It should not have. The figure
is a range rather than a constant, its comparison against "the 1.6 bits English
carries" puts two different denominators side by side, and the thinning-rate
consequence assumes free letters pay independently, which is untested. The
one-line reviewer dismissal writes itself: *you measured that English is not
reversible under an English language model.* The structural results do not have
that problem, and the repository's hardest-won material is method — the
overhang formulation, the regime boundary, canon recall, the composition
algebra, the corridor bug, the filter/ranker split.

## What changed from the previous draft

The previous draft was scoped around three claims (the measurement, the
ceiling, and the filter/ranker distinction for proxies) and carried 13 `\tk{}`
markers for numbers that did not exist. This one carries the measurement and
the ceiling; the proxy audit is compressed into the limitations, where it
belongs in four pages, and is the natural core of a separate short paper.

**The estimator now exists.** `experiments/mirror_cost.py` computes the
headline figure, which previously appeared only as prose in `README.md`,
`docs/NORTH-STAR.md`, `docs/training.md` and a docstring, with no script behind
it and with the model, corpus, vocabulary, segmentation algorithm and stability
conditions unrecorded. It sweeps four models, three segmentation objectives and
six span lengths, and writes `experiments/mirror_cost.json`.

Two things the estimator changed about the claim:

- **The single number 3.296 does not survive as a single number.** It
  reproduces at short spans under the unigram objective and drifts to 2.8–3.1
  at lengths where prose would live; a longest-match objective gives 2.2–2.4.
  The paper reports the range with its conditions rather than a point estimate.
- **The first estimator design had no data.** Requiring both directions to
  segment fully into the dictionary discards essentially every span: at 20
  letters, none of a first sample of thirty reversed spans segmented at all.
  Admitting the 26 bare letters as fallback units makes segmentation total and
  moves the penalty from a constant we would have chosen to the language model.
  The failure rate is itself reported, as dictionary coverage: 0.88–0.92
  forward against 0.48–0.55 reversed.

**The thinning rate is corrected.** `README.md` and `docs/NORTH-STAR.md` said
the feasible set thins by 10× every three letters. At ~3.3 bits per free letter
the factor is ~9.8 per *one* letter. Both files now carry the arithmetic that
follows from the measurement, as does the paper.

**The sweep is incomplete and the paper says so.** DistilGPT-2 and GPT-2 small
and medium are complete at all three objectives and all six lengths. GPT-2
large has the unigram objective at $L \le 40$ only; Qwen2.5-0.5B, which would
have been the one non-GPT-2 family, was not run. Finishing both is cheap on any
machine that is not the author's laptop, and the direction of the incomplete
rows agrees with the complete ones.

**The directional-asymmetry section is gone.** It rested on a claim that does
not survive re-measurement (`docs/architecture.md:19-37`): the per-letter gap
reverses per token, with word length as the cause. What survives — the backward
fine-tune's 0.921-nat modelling cost and its per-token gain as a scorer —
is a good result and does not fit in four pages.

## Building

Compiles with or without the ACL style files. Without them it falls back to a
one-column `article` layout and prints a notice under the title, which is fine
for reading and useless for judging page count.

```bash
curl -O https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty
curl -O https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_natbib.bst
latexmk -pdf naacl2027.tex
```

On Overleaf, start from the official ACL template and drop `naacl2027.tex` and
`refs.bib` into it.

No LaTeX toolchain is installed on the machine this draft was written on, so
neither path has been compiled here.

## What must be resolved before submission

1. **The judging protocol.** The verdict files in `runs/` record one verdict
   per item with no annotator identity, count, agreement statistic or verbatim
   instructions, and no script produces them. The paper takes the survivable
   path — every judgment-dependent claim is labelled single-judge blinded
   assessment with calibration controls, in the limitations. Running a real
   multi-annotator batch would let those claims be stated more strongly, and
   nothing else in the paper depends on the choice.
2. **A frontier-LLM baseline.** Prompt current models for long palindromes,
   score validity with `llm_palindrome/validator.py` and readability with the
   blind protocol. `calderaro2025oulibench` already reports frontier failure on
   palindromes in Italian, so the section is defensible without it; running it
   in English would make the point directly, and it is the cleanest available
   demonstration that the search rather than the model is what makes the
   constraint hold.
3. **Reproduce Table 1's spread on a second corpus.** One corpus is one corpus.

## Anonymity

ARR review is anonymous and this repository is not. Build the scrubbed artifact
with `./tools/make_anon_mirror.sh`, point the submission's artifact link at an
anonymous host serving that tree, and swap it for the real URL at camera-ready.

Two items carry identity independently of the artifact: the acknowledgements,
which should be written last, and authorship itself, unsettled between the
single author in `CITATION.cff` and the group credited in `README.md`.
