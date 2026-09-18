# Quarantined historical evidence

This is not current primary evidence for the readable-palindrome project. It
records legacy search/proxy experiments and must not be released, cited as a
readability result, or used to support a submission.

On 11 September 2026, the paired reversal-cost experiment froze 150
WikiText-2 spans at each of six target lengths and reused them across GPT-2,
SmolLM2-135M, and three segmentation strategies. All 36 planned cells were
scored. Reversal adds 2.14--3.34 bits per normalized letter; forward lexical
coverage is 86.9--91.5% and reversed coverage is 49.6--54.6%. Within-cell
paired standard errors are 0.04--0.11. The largest matched-cell difference
between model families is 0.136 bits per letter. The natural-formatting to
forward-resegmentation control costs 0.79--1.73 bits per letter.

Exact samples, model and corpus revisions, software versions, aggregate rows,
and the audit are in `runs/mirror-cost-2026-09-11/`. The audit recomputes the
factorial design, normalization, segmentations, coverage, and arithmetic; it
does not independently recompute model logits.

The long-form examples are frozen under
`runs/long-form-examples-2026-09-11/`. The generated construction has 101
words, 342 letters, and 24 mirror pairs with no borrowed centre. The catalogue
construction has 72 words and 237 letters and uses nine borrowed pairs plus a
borrowed centre. Both are exact. These facts establish form, scale, and
provenance, not grammaticality or discourse coherence.

# Current controlled decoder evidence

On 11 September 2026, 50 opening subtrees were selected from the 85 with at
least 20 immediate letter-compatible extensions. Selection used a fixed hash
rank without POS tags or outcome data. Terminal and incremental arms used the
same deterministic sibling ordering and a maximum of 50,000 popped states.
The raw run, accepted pairs, provenance, summary, and audit are in
`runs/controlled-pos-pruning-openings-2026-09-11/`.

The terminal arm returned 407 accepted pairs from 2,500,000 popped states; the
incremental arm returned 2,137 from 2,400,032. The corresponding rate ratio is
5.47 (paired bootstrap 95% interval 2.89--17.65). Accepted pairs per CPU second
increase by 4.36 times (2.30--13.93). The accepted-pair rate per generated
state shows no resolved improvement: the ratio is 0.95 (0.46--3.20). The gate rejects 83.28% of generated states
before insertion. All 407 terminal pairs occur in their matched incremental
arm. The audit rechecks 2,544 stored pair rows.

These results support a fixed-opening search-throughput claim, not a general
hardware, corpus, grammaticality, or readability claim.

# Historical evidence ledger

Supporting research record through 10 September 2026. The current manuscript
is `paper/naacl2027.tex`; this ledger preserves measurements and corrections
from the earlier Markdown draft. It is not a second manuscript.

This section is the release ledger for the claims above. It records the evidence that survived review, the part that was corrected or withdrawn, and the exact artifact location. A file in this list can support its stated fact; it should not be read as support for the stronger claim in the fourth column.

| Claim or question | Retained evidence | Release location | What the evidence does not show |
| --- | --- | --- | --- |
| Saved length artifact | 90,937 letters; 16,168 unique normalized phrases; no adjacent repeated words; max nonexception-word count 3; 498 letters over the identically parsed Norvig v3 reference | `artifacts/norvig-v3/`; `experiments/audit_norvig_result.py` | Readability, global novelty, an optimal solution, or a matched-runtime speedup |
| Structural filtering | Terminal: 20,989 pairs; incremental: 86,511; shared limits and aggregate counters are preserved | `runs/polaris/sentence_plan_20260904_204815/aggregate.json`; `paper/verify_structural_draft.py` | Equal CPU work, per-rank variation, a causal runtime speedup, or grammaticality |
| Pair identity and tag test | Ordered token-pair deduplication, full reversal, word uniqueness, and complete tag-pattern admission can be replayed from saved candidates | `paper/verify_structural_draft.py`; `tools/polaris/payload/brown.json.gz`; `tools/polaris/payload/vocab30k.txt` | A human-readable sentence or a general grammar |
| Composition | Mirrored unit nesting preserves the letter palindrome; greedy selection applies the published target and repetition caps | `llm_palindrome/hierarchy.py`; `server/v3.py` | A quality comparison of composition policies or paragraph coherence |
| Inventory-policy observations | The six saved letter/budget cells, including 68,286 static at 45 s and 90,937 feasible at 300 s | `artifacts/norvig-v3/`; `experiments/RESULTS-*.md` | An ablation of objective, inventory policy, budget, and hardware |
| Signed overhang reduction | The telescoping identity and sixteen traversal/vocabulary measurements replace the former stable 1.09 value | `runs/revision-2026-09-07/`; `experiments/revision_conservation.py` | Independent edge observations, a policy result, or a positive conservation constant |
| Vocabulary scaling | Five compatible points yield slope about -1.031 for worker-local distinct output per aggregate core-second | `runs/polaris/scale_20260823/summaries.jsonl` | A trie complexity law, global distinct-output rate, or a compute limit on quality |
| Length extrapolation | The earlier per-letter multiplier and 2,900-fold estimate are not retained | No replacement result; correction documented in `paper/SOURCE-AUDIT.md` | A length threshold, high-budget yield curve, or a 42--43-letter claim |
| Historical seams | The archived forced choices prefer singles in the small historical item set | Historical totals summarized in `paper/SOURCE-AUDIT.md` | A first-seam effect, a per-seam coefficient, independent human judgments, or a binomial test based on doubled ratings |
| Fixed-material seam order | Protocol, digest, component sets, random and optimized orders, raw model replies, missingness, and summaries are frozen | `runs/revision-2026-09-07/`; `experiments/revision_seams.py`; `experiments/verify_revision.py` | A causal seam-count result, a readability gain, or evidence that better ordering cannot help another bank |
| Model instrument | Exact 12/12 calibration gate; only `gpt-oss:20b` passed; one rater leaves ordinal inter-rater agreement unestimable | `runs/revision-2026-09-07/`; `experiments/verify_revision.py` | Human validity, several independent raters, or sensitivity above the nest floor |
| Punctuation | All six variants of 26 texts; saved means including the 120B +0.77 post-hoc-minus-bare difference; 52/52 model outputs preserve letters and tokens | `runs/punct/after_20b.json`; `runs/punct/after_120b.json` | A blind human punctuation effect, a general punctuation benefit, or new semantic content |
| Sentence-planning quality check | Join-restricted counts and 210 archived half ratings, including no pair with both halves at least 2 | `runs/polaris/sentence_quality_20260905_011235/aggregate.json`; `runs/sentence_quality*120b.json` | A human rejection of every candidate or impossibility of readable mirror pairs |
| Candidate-menu, beam, and finite-language diagnostics | The corrected menu, closure smokes, and 60,576-clause exact inventory test | `experiments/RESULTS-*.md`; `experiments/sentence_intersection-results.json` | A general claim about beams, language-model guidance, or English as a whole |
| Information-style arguments | Dictionary segmentation and forward/reverse coverage remain model-dependent diagnostics only | `paper/SOURCE-AUDIT.md`; `experiments/mirror_cost.py` | An information-theoretic lower bound or a product survival probability |

The release deliberately does not contain a human panel, independent human punctuation ratings, a high-budget length sweep, a controlled beam-width/readability curve, a best-of-N curve, raw sentence-plan per-rank traces, or the historical seam per-item files. These are missing evidence, not hidden negative results. No prose change can supply them.

### Corrected measurements

#### Overhang reduction

For an accepted placement, let `d_t` be the overhang length and `m_t` the new unit length. Prefix compatibility gives `d_(t+1) = |d_t - m_t|`. Signed overhang reduction is therefore `d_t - d_(t+1)`. Along a complete path, those signed reductions telescope to the initial overhang minus the final overhang. They cannot have a universal positive average as paths grow with bounded endpoints.

We reran the measurement with a corrected candidate menu, a 400-candidate limit, maximum overhang 24, and a 60,000-edge budget. At small vocabularies the traversal exhausted the reachable edges; at larger vocabularies it sampled different truncated sets.

| Requested vocabulary | Realized vocabulary | Edges per traversal | BFS mean signed reduction | DFS mean signed reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 926 | 11,918 | 0.51 | 0.51 |
| 2,000 | 1,899 | 26,641 | 0.65 | 0.65 |
| 4,000 | 3,820 | 57,619 | 0.80 | 0.80 |
| 6,000 | 5,741 | 60,000 | 0.17 | 0.95 |
| 8,000 | 7,655 | 60,000 | -0.50 | 1.06 |
| 16,000 | 15,223 | 60,000 | -1.77 | 0.87 |
| 32,000 | 30,262 | 60,000 | -2.33 | 0.73 |
| 47,000 | 44,232 | 60,000 | -2.36 | 0.46 |

The old stable `1.09` claim is withdrawn. The table is a finite, traversal-dependent enumeration, not a sample of independent successful paths. Negative overhang reduction can still be useful in a long search because a placement can add material while opening a larger overhang.

#### Scaling and length extrapolation

The historical vocabulary sweep measures worker-local distinct outputs per aggregate core-second. It does not measure state expansions, globally deduplicated outputs, or an abstract trie complexity. The five compatible settings support a fitted slope of about `-1.031` across a 32-fold requested vocabulary range. One 6,000-word cell used a different compute allocation and is not part of that fit. A high fit over five aggregated points is descriptive of that pipeline, not a universal inverse-vocabulary law.

The earlier per-letter multiplier and 2,900-fold length estimate are removed. They came from sparse cells, one of them with one event. No new high-budget sweep establishes a 42- or 43-letter threshold, and no best-of-N readability curve has been measured.

#### Seams, punctuation, and model instruments

A historical seam comparison put one chunk beside nests with two, four, or eight chunks. Two blind subagent judges preferred the single chunk for all seven items at every nest size. This supports a preference for those singles in that item set. It does not identify a first-seam collapse: nests also changed length, words, and topics, and the binary endpoint had no room below zero. The two ratings of one item are not two independent items. The claims that coherence dies at the first seam, later seams impose no further cost, and no ordering can help are withdrawn.

The new seam-order intervention draws twelve fixed sets of eight mirror pairs, then compares a random order with the order that maximizes both forced-join corpus-bigram scores. Each pair of nests has the same component words, multiplicities, and letter count. The center join remains outside this objective. Only `gpt-oss:20b` passed the exact 12-of-12 prose-versus-shuffle calibration gate. It rated random and optimized nests at 0.00 wherever both ratings were present: five complete pairs, 15 of 15 scored nests at the floor, and 10 experimental answers missing. Across all twelve sets, allowing missing scores anywhere from 0 to 3 bounds the mean order effect between -1.75 and +0.50. This is an order intervention and a model instrument result, not a seam-count study or human finding.

The punctuation files contain 26 underlying texts in six presentations. Under the saved 120B evaluator, post-hoc 120B punctuation averaged 2.08 on the 0--3 scale and bare spacing averaged 1.31: a descriptive within-model difference of +0.77. Hand or catalogue punctuation averaged 2.31; search-time punctuation averaged 0.73. The saved 20B evaluator gives different values. All 52 model-punctuated outputs preserve both normalized letters and word tokens. This does not make the evaluation blind, independent, or human; a model may be rating its own punctuation.

#### Sentence-quality and finite-language diagnostics

Sentence planning increases structural supply without rescuing the quality claim. Under an additional every-internal-join-attested restriction, the run found 105 pairs; allowing one unattested join found 227. An archived 120B half-by-half pass gave 100 of 210 halves a 0, 98 a 1, 10 a 2, and 2 a 3. No pair had both halves at least 2. Four ordinary-sentence controls scored 3, while four salad controls scored 0 or 1. This is a negative result for that bank and instrument. It is not a human rejection of all future search outputs.

Other negative runs remain diagnostic, not general laws. A corrected candidate menu produced 112 root words of at least five letters where the old menu had none, and all 256 attempts in one 32-rank debug run closed as valid 80--95-letter palindromes. Small beam and clause-score smokes closed no pairs under their tested settings; they do not show that wider beams or language guidance reduce readability. A finite seven-template grammar contained 60,576 clauses and no exact mirror pair. That zero is exhaustive for the specified inventory, not for English.

### Reproduction

From the repository root, run:

```sh
python3 paper/verify_structural_draft.py
python3 experiments/verify_revision.py
python3 paper/build_release.py
```

The first command audits the saved structural candidates and long artifact with a separate implementation in this project. The second checks the frozen revision packet. The third regenerates the source and evidence archives with a SHA-256 manifest. External corpora are intentionally not bundled; their input hashes and version notes are in `paper/SOURCE-AUDIT.md` and the archive manifest.
