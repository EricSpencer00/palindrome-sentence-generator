# Superseded research strategy

> **Do not execute this plan as written.** An implementation audit on 4
> September found that the candidate menu was shortest-word biased, the phrase
> screen automatically rejected every phrase it was meant to measure, the graph
> used narrower closure semantics, and directional scores depended on batch
> padding. The conclusions below about closed phrase/assembly routes, exhausted
> reranking, and a promising backward intervention are therefore withdrawn.
> The replacement sequence is: repair candidate and beam mechanics, repair the
> chunk/graph measurements, then rerun a corrected directional ablation.

Updated 4 September 2026. This is a decision document, not a claim that the
north-star paragraph has been achieved.

## Verdict

Stop optimising length supply and assembly. Keep the graph enumerator as the
validity engine, repair presentation after search, and spend the next effort on
a meaning-first proposal distribution with a blind-quality gate. If that gate
does not move, the honest outcome is a strong negative result and a useful
palindromic-poetry/assembly tool -- not generated coherent prose.

## What is established

| Question | Evidence | Decision |
|---|---|---|
| Can we make a valid character-level palindrome? | Every emitted candidate in the Polaris runs verified; v3 clears mechanical criteria 1, 2, 4, 5 and 9 through 4,000 letters. | Solved mechanically. Keep `validator.py` as a hard gate. |
| Is length a search-supply problem? | The graph reaches 37--51-letter single chunks locally; its long batch scored 225/256 at 0 and 31/256 at 1 on the gated absolute scale. | Supply is solved; quality is not. |
| Can several chunks make long text coherent? | A single added seam lost every blinded comparison, 14/14 (`RESULTS-seams.md`). | Do not assemble chunks to pursue coherence. |
| Can automatic wrapping grow a good chunk? | It grew 750/750 texts but the original seed won all 20 paired comparisons for both judges (`RESULTS-extend.md`). | Do not automate growth with current proxies. |
| Can phrases repair local fluency? | The replicated chunk screen found no core-yield gain and paid about 4.7x throughput. | Do not promote phrase inventory search. |
| Can a better ranker rescue the current beam pool? | Best-of-2,000 gained only 0.168 nats/token over best-of-24 and flattened (`runs/oracle_bound.json`). | More samples or reranking of that pool is exhausted. |
| Does direction-aware scoring help at all? | A backward GPT-2 scorer improves the held-out per-token proxy by +0.423 over Zipf while retaining 24/24 closures (`docs/training.md`). | Promising only as a proposal component; it has no blind-quality win yet. |
| Does presentation matter? | Post-search marking by a capable model beats bare spacing while preserving every letter; the shipped presenter loses to bare spacing. | Punctuate only after validity is frozen. |

The Polaris scaling run (`RESULTS-polaris-scaling.md`) is a final control for
the old redraw walk, not a reason to allocate more CPU: its best 27--31 core
rate was at 2,400 words, while the graph already makes that walk obsolete.

## What not to do

- Do not launch further CPU yield sweeps or a larger phrase inventory run.
  They address a supply bottleneck the graph has removed.
- Do not use nested mirror-pairs, self-palindromic refrains, or automatic wraps
  as a route to the nine-criterion prose target. They can make text long, but
  they take shortcuts the target explicitly excludes or lose at the first
  seam.
- Do not promote an LLM/GPT score as a quality result. It may filter candidates
  but previous ranker and normalisation failures rule out treating it as a
  decision-maker.
- Do not train the full dual-head character architecture yet. The word-level
  backward intervention has a proxy improvement, but has not demonstrated a
  quality improvement to a blinded evaluator.

## The next experiment

Run a pre-registered, paired **proposal test**, not another quality proxy
sweep.

1. Generate matched 80--120-letter single-chunk candidates from the current
   beam with Zipf and the best backward-scoring arm. Preserve seed, model,
   vocabulary, beam width and closure rate for every pair.
2. Freeze letters, then apply the already-validated post-search punctuation
   path identically to each arm. Reject any reply that changes a letter.
3. Mix each arm with real-prose and shuffled-word calibration controls. Blind
   the source labels and have at least three independent human annotators rate
   grammaticality, a nameable subject, and whole-text coherence. Report raw
   agreement and the annotator instructions.

### Gates

- **Promote the directional proposal path** only if it keeps at least 95%
  closure, beats Zipf on the preregistered coherence measure, and the
  calibration controls separate as expected. A proxy-only improvement does not
  count.
- **Stop the word-level directional path** if the paired result is null or
  negative. Do not respond by increasing the model size or running a dual-head
  model; that would be optimising an unvalidated proxy.
- **Escalate to a dual-head character model** only after a positive paired
  result *and* after a small ablation shows that dual boundary heads retain
  valid trie-constrained decoding. That is a dedicated accelerator job, not a
  CPU-yield job.

## Where a Sophia key is useful

It is optional for the validity engine. A frontier model cannot be trusted to
emit the palindrome; the graph/trie must retain that responsibility. A key is
useful for two bounded tasks after the proposal test is specified:

1. a frontier-model baseline at fixed lengths and prompts, measured by the
   repository validator and novelty checker; and
2. proposing semantic intents or candidate word material which the exact
   decoder must then accept or reject.

Neither task justifies an unbounded model-search loop. The key should not be
used to judge its own outputs.

## Two legitimate end states

### Positive result

A single non-assembled chunk passes the paired human gate. Then invest in the
proposal model, scale it cautiously, and treat the graph as a verifier/sampler
rather than the source of language. The immediate research question becomes
which semantic representation survives both readings.

### Negative result

The directional proposal test fails. Then the project has a coherent research
result: exact palindrome generation and long valid supply are cheap with a
graph, while novel coherent prose collapses under the double-reading
constraint; seams and automated growth make the failure worse. Package this
with the proxy-audit and evaluator limitations honestly. The product remains a
palindrome explorer/poetry assembler with provenance labels, rather than
claiming generated prose.

## Evidence and reproduction

- `experiments/RESULTS-graph.md` -- graph versus redraw walk and long-candidate
  quality screen.
- `experiments/RESULTS-seams.md` -- the seam experiment.
- `experiments/RESULTS-extend.md` -- mechanical growth versus paired quality.
- `experiments/RESULTS-llm-judge.md` -- what automated evaluation can and
  cannot decide, including post-search punctuation.
- `docs/training.md` -- direction-aware scorer, oracle bound, and known proxy
  failures.
- `experiments/RESULTS-polaris-scaling.md` and
  `runs/polaris/scale_20260904_030603/` -- job `7590268` artifacts.
