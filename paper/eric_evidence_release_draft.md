# Revolt, Academia: Aimed a Cat Lover

## Early Structural Pruning in Two-Ended Search

### Abstract

Exact reversal and readable English are different requirements. This paper asks one narrow structural question: while building a palindrome from both ends, can search discard a partial half as soon as its word classes can no longer fit any permitted sentence pattern? It can. In the recorded bounded experiment, incremental filtering returned 86,511 distinct structurally admitted pairs, about four times the 20,989 returned when the same test ran only after closure. The result is about candidate supply, not readable sentences.

Composition explains how such pairs can be assembled without losing exact reversal. A separate adaptation of Norvig's dictionary search tests a different boundary: whether a heavily restricted search can still make a long exact object. Its saved output has 90,937 normalized letters, 498 more than the published version-3 reference when both are parsed the same way. The final section is an evidence release: it records the source for each retained result, the measurements that were corrected, and the evidence that does not yet exist.

## 1. The construction

Let `n(x)` lowercase a text and remove every character except ASCII letters. A text is a palindrome when `n(x)` equals its reverse. Spaces make the words legible; `n` ignores them, along with punctuation.

The search grows two word sequences toward a meeting point. Start with the visible pieces below.

```text
left piece:   step on
right piece:      pets
unmatched overhang after cancellation: on
add before the right piece: no pets
whole text:   step on no pets
```

`step` cancels `pets` in reverse, leaving `on` as the overhang. The search may consume that overhang or extend beyond it, in which case the next overhang belongs to the other side. In the pair search, future words are prepended to the left half and appended to the right half. That direction is why a left partial half must later fit a sentence-pattern suffix, while a right partial half must fit a prefix.

A trie retrieves words compatible with the overhang. The overhang is enough to prove this local letter step. It is not enough to merge two partial paths. Used phrases, word counts, sentence-pattern state, and language scores can make identical overhangs have different legal continuations. The implementation therefore stores the full partial word sequences when those restrictions are active.

A closed pair is accepted only after a second check. The pair must split at the letter midpoint between words, each half must have at least three words, no word may occur twice anywhere in the pair, and both halves must pass the sentence-pattern test below. The output identity is the ordered pair of space-joined halves. A different segmentation is a different output even when its normalized letters are the same.

## 2. Structural feasibility

The pattern inventory comes from Brown Corpus tags under NLTK's universal mapping. Each word keeps the tags observed for it in that corpus. A word can therefore have more than one reading. In the saved example below, `credits` is allowed as a `NOUN` or a `VERB`; the accepted tag assignment uses the verb reading. After punctuation and `X` tags are removed, the frozen inventory has 8,982 patterns of three to nine tags. We retain the 5,649 patterns that contain a `VERB` and begin with `PRON`, `DET`, `NOUN`, `ADJ`, `NUM`, or `ADV`.

For a completed word sequence `W`, the test `F(W)` asks whether one assignment of its observed word tags is exactly one of those retained patterns. A partial left half must match the suffix of at least one pattern, because future words are prepended to it. A partial right half must match a pattern prefix, because future words are appended to it. If either test fails, later growth in that direction cannot repair it.

This is a proposal filter. It proves only that a tag sequence remains possible. It does not prove agreement, a sensible argument structure, or a recoverable proposition. The saved accepted pair makes the distinction plain:

> a aaron ama credits erased / des ares tide rca manor aaa

Its left tag reading is `DET NOUN NOUN VERB VERB`; its right reading is `NOUN NOUN VERB NOUN NOUN NOUN`. The result is structurally admitted and visibly awkward. Conversely, prepending `she` to `aaron ama credits erased` gives `PRON NOUN NOUN VERB VERB`, which matches neither a permitted prefix nor suffix at that point. The filter rejects it before another search step.

The recorded comparison used the same Brown-known vocabulary subset, the same opening partition, 32 processes, and the same per-process stopping limits: 600 elapsed seconds, 20 million popped states, 10,000 accepted pairs, or exhaustion. Terminal filtering ran before incremental filtering within each process. A shared seed controls the opening shuffle, but pruning changes later random draws, so it does not make the walks identical.

| Measure | Terminal filter | Incremental filter |
| --- | ---: | ---: |
| Popped states | 84,815,872 | 115,236,212 |
| Closed states before final filtering | 1,117,411 | 536,805 |
| Children rejected by partial-state filter | 0 | 138,210,789 |
| Distinct admitted pairs | 20,989 | 86,511 |
| Sum of elapsed process time | 19,214.130 s | 19,050.983 s |

The incremental arm returned 4.12 times as many distinct admitted pairs. It also visited more popped states. That is not a contradiction: rejected children are counted before insertion and are excluded from the popped-state count, so pruning changes which states survive and which subtrees are explored. The comparison measures output yield under the recorded limits. It does not measure equal CPU work, isolate the cost of each operation, or establish a general speedup. Per-process traces are missing, so stopping causes and workload variation cannot be reconstructed.

## 3. What the other two parts contribute

Mirror pairs can be nested without another letter search. If `n(L_i)` is the reverse of `n(R_i)`, then

`L_1 L_2 ... L_k C R_k ... R_2 R_1`

is palindromic whenever the center `C` is palindromic. One nested adjacency creates two visible joins. For example, `rats live`, `on no`, and `evil star` compose to `rats live on no evil star`. Its normalized letters read the same in reverse. This small fact is algebra, not evidence that the resulting text reads well.

The composition kernel gives the structural pairs a precise downstream use: it preserves their mirror relation while assembling a longer object. It is not a second quality result. Given a center, a seed, and an inclusive maximum of `T` normalized letters, it makes one seeded greedy pass through a finite bank. It may stop below `T`; `T` is a ceiling, not an exact target. It rejects additions that would exceed that ceiling, repeat an adjacent word inside one component, exceed the bigram cap, repeat a word-length template too often, or exceed the token cap outside its stated exception set. A word-length template is the sequence of word lengths in one component, such as `(4, 4)` for `rats live`. An internal bigram is an adjacent word pair inside the center or one component; it excludes pairs created at component joins. The selector does not backtrack. Rendering gives each component a sentence boundary, but it does not make the output a readability evaluation.

The long dictionary artifact serves a different supporting role. It checks that an exact construction with phrase and repetition restrictions can still reach a large saved endpoint. It does not test the structural filter. The search uses the same dictionary as Norvig's published version-3 palindrome, records letters rather than completed phrase count, and adds phrase and repetition restrictions. The baseline and saved artifact are evaluated under lowercase-ASCII normalization.

| Property | Published Norvig v3 | Saved artifact |
| --- | ---: | ---: |
| Normalized letters | 90,439 | 90,937 |
| Distinct normalized phrases | 16,111 | 16,168 |
| Adjacent repeated word pairs | 9 | 0 |
| Maximum nonexception-word count | 14 | 3 |

The saved artifact is 498 letters longer under this evaluation. Its independent project-local auditor reconstructs the text from the saved phrases, checks dictionary membership, phrase uniqueness, the word restrictions, and complete reversal. The result is not a matched-runtime contest with the original program: the objective and feasible set changed. It is not a global record or a readability result.

The inventory-policy observations are narrower still. At 45 seconds, a dynamic unused-inventory estimate reached 76,979 letters and the static estimate reached 68,286; at 120 seconds, unused and feasible estimates reached 88,101 and 88,095. The final feasible, three-occurrence-cap run reached 90,937 at 300 seconds. These are single wall-clock runs under the letter objective. They do not isolate the inventory change from the new objective or prove an implementation-wide speedup.

## 4. Measurements that needed correction

The correction work matters because an exact palindrome can make nearby quantitative claims look firmer than they are. The following results stay in the record, but with their actual scope.

### 4.1 Overhang reduction

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

### 4.2 Vocabulary scaling and the withdrawn extrapolation

The historical vocabulary sweep measures worker-local distinct outputs per aggregate core-second. It does not measure state expansions, globally deduplicated outputs, or an abstract trie complexity. The five compatible settings support a fitted slope of about `-1.031` across a 32-fold requested vocabulary range. One 6,000-word cell used a different compute allocation and is not part of that fit. A high fit over five aggregated points is descriptive of that pipeline, not a universal inverse-vocabulary law.

The earlier per-letter multiplier and 2,900-fold length estimate are removed. They came from sparse cells, one of them with one event. No new high-budget sweep establishes a 42- or 43-letter threshold, and no best-of-N readability curve has been measured.

### 4.3 Seams, ordering, and the missing causal comparison

A historical seam comparison put one chunk beside nests with two, four, or eight chunks. Two blind subagent judges preferred the single chunk for all seven items at every nest size. This supports a preference for those singles in that item set. It does not identify a first-seam collapse: nests also changed length, words, and topics, and the binary endpoint had no room below zero. The two ratings of one item are not two independent items. The claims that coherence dies at the first seam, later seams impose no further cost, and no ordering can help are withdrawn.

The new seam-order intervention asks a smaller question. It draws twelve fixed sets of eight mirror pairs, then compares a random order with the order that maximizes both forced-join corpus-bigram scores. Each pair of nests has the same component words, multiplicities, and letter count. The center join remains outside this objective. This is an order intervention, not a seam-count study.

Only `gpt-oss:20b` passed the exact 12-of-12 prose-versus-shuffle calibration gate. It rated random and optimized nests at 0.00 wherever both ratings were present: five complete pairs, 15 of 15 scored nests at the floor, and 10 experimental answers missing. Across all twelve sets, allowing missing scores anywhere from 0 to 3 bounds the mean order effect between -1.75 and +0.50. The score is a model instrument result, not a human result or proof that order is irrelevant.

### 4.4 Punctuation and sentence-quality evidence

The punctuation files contain 26 underlying texts in six presentations. Under the saved 120B evaluator, post-hoc 120B punctuation averaged 2.08 on the 0--3 scale and bare spacing averaged 1.31: a descriptive within-model difference of +0.77. Hand or catalogue punctuation averaged 2.31; search-time punctuation averaged 0.73. The saved 20B evaluator gives different values. All 52 model-punctuated outputs preserve both normalized letters and word tokens. This does not make the evaluation blind, independent, or human; a model may be rating its own punctuation.

Sentence planning increases structural supply without rescuing the quality claim. Under an additional every-internal-join-attested restriction, the run found 105 pairs; allowing one unattested join found 227. An archived 120B half-by-half pass gave 100 of 210 halves a 0, 98 a 1, 10 a 2, and 2 a 3. No pair had both halves at least 2. Four ordinary-sentence controls scored 3, while four salad controls scored 0 or 1. This is a negative result for that bank and instrument. It is not a human rejection of all future search outputs.

Other negative runs remain diagnostic, not general laws. A corrected candidate menu produced 112 root words of at least five letters where the old menu had none, and all 256 attempts in one 32-rank debug run closed as valid 80--95-letter palindromes. Small beam and clause-score smokes closed no pairs under their tested settings; they do not show that wider beams or language guidance reduce readability. A finite seven-template grammar contained 60,576 clauses and no exact mirror pair. That zero is exhaustive for the specified inventory, not for English.

## 5. Conclusion

The central result is structural: early tag-pattern feasibility returned more candidates under the recorded stopping limits. The saved 90,937-letter artifact separately passes checks of its letter count, dictionary membership, phrase uniqueness, repetition restrictions, and complete reversal. Neither fact supplies readable prose. Composition preserves reversal once suitable units exist; it does not create suitable units. The next decisive evidence is independent human assessment of fixed, valid, consistently presented texts.

## Evidence release

This section is the release ledger for the claims above. It records the evidence that survived review, the part that was corrected or withdrawn, and the exact artifact location. A file in this list can support its stated fact; it should not be read as support for the stronger claim in the fourth column.

| Claim or question | Retained evidence | Release location | What the evidence does not show |
| --- | --- | --- | --- |
| Saved length artifact | 90,937 letters; 16,168 unique normalized phrases; no adjacent repeated words; max nonexception-word count 3; 498 letters over the identically parsed Norvig v3 reference | `artifacts/norvig-v3/`; `experiments/audit_norvig_result.py` | Readability, global novelty, an optimal solution, or a matched-runtime speedup |
| Structural filtering | Terminal: 20,989 pairs; incremental: 86,511; shared limits and aggregate counters are preserved | `runs/polaris/sentence_plan_20260904_204815/aggregate.json`; `paper/verify_structural_draft.py` | Equal CPU work, per-rank variation, a causal runtime speedup, or grammaticality |
| Pair identity and tag test | Ordered token-pair deduplication, full reversal, word uniqueness, and complete tag-pattern admission can be replayed from saved candidates | `paper/verify_structural_draft.py`; frozen Brown/vocabulary payloads named in `paper/structural-evidence.json` | A human-readable sentence or a general grammar |
| Composition | Mirrored unit nesting preserves the letter palindrome; greedy selection applies the published target and repetition caps | `llm_palindrome/hierarchy.py`; `paper/eric_short_working_draft.md` | A quality comparison of composition policies or paragraph coherence |
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
| Information-style arguments | Dictionary segmentation and forward/reverse coverage remain model-dependent diagnostics only | `paper/SOURCE-AUDIT.md`; `paper/naacl2027.tex` | An information-theoretic lower bound or a product survival probability |

The release deliberately does not contain a human panel, independent human punctuation ratings, a high-budget length sweep, a controlled beam-width/readability curve, a best-of-N curve, raw sentence-plan per-rank traces, or the historical seam per-item files. These are missing evidence, not hidden negative results. No prose change can supply them.

### Reproduction

From the repository root, run:

```sh
python3 paper/verify_structural_draft.py
python3 experiments/verify_revision.py
python3 paper/build_release.py
```

The first command audits the saved structural candidates and long artifact with a separate implementation in this project. The second checks the frozen revision packet. The third regenerates the source and evidence archives with a SHA-256 manifest. External corpora are intentionally not bundled; their input hashes and version notes are in `paper/SOURCE-AUDIT.md` and the archive manifest.

## References

- Francis, W. N., and H. Kučera. 1979. _Manual of Information to Accompany a Standard Corpus of Present-Day Edited American English_.
- Norvig, Peter. 2016. [World's Longest Palindrome? 21,012 Words](https://www.norvig.com/palindrome.html).
- Papadopoulos, Alexandre, Pierre Roy, Jean-Charles Régin, and François Pachet. 2015. [Generating all Possible Palindromes from Ngram Corpora](https://www.ijcai.org/Proceedings/15/Papers/353.pdf).
- Hokamp, Chris, and Qun Liu. 2017. [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](https://aclanthology.org/P17-1141/).
- Lu, Ximing, et al. 2022. [NeuroLogic A*esque Decoding: Constrained Text Generation Using Lookahead Heuristics](https://aclanthology.org/2022.naacl-main.57/).
