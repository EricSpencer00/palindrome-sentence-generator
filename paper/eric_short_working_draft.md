# Revolt, Academia: Aimed a Cat Lover

## Palindrome Construction under Lexical and Structural Constraints

### Abstract

We study two-ended palindrome construction with vocabulary, repetition, and sentence-structure restrictions. A partial-state filter checks whether the growing halves can still match prefixes and suffixes of permitted part-of-speech sequences. We compare this filter with checking the same condition only on completed candidates. In one experiment with the same per-process stopping rules, the configurations return 86,511 and 20,989 distinct structurally admissible pairs, respectively. We define the output identity, counters, and limits underlying this comparison. We also describe composition from mirrored sentence units and an adaptation of Norvig’s dictionary-based length search. The latter produces a separately verified 90,937-letter output with additional repetition restrictions. Existing runs provide partial comparisons of inventory bookkeeping but do not isolate the effect of changing the objective. We evaluate candidate yield and exact output properties, not readability.

### 1. Introduction

A palindrome retains its letter sequence when read backwards. Constructing one from words requires coordinating choices on both sides: the letters added on one side constrain the words available on the other. Vocabulary restrictions, repetition limits, and sentence patterns further restrict the continuations available to a partial construction.

We study how these restrictions can be incorporated into search and composition. The first experiment places a sentence-pattern test at two different points in the same bounded search: after closure, or during construction as well as after closure. The second line of work changes the objective and inventory bookkeeping of an existing dictionary-based length search. These experiments have separate vocabularies, acceptance conditions, and outcomes.

We use larger linguistic units because the eventual aim is readable palindromic prose. This paper measures lexical and structural admissibility. Lexically admissible means membership in a specified inventory; structurally admissible means passing the tag-pattern test. Readability requires evaluating the resulting language.

Our contributions are:

1. A prefix/suffix feasibility filter for two-ended palindrome search.
2. A comparison with terminal filtering under common stopping limits and explicit output deduplication.
3. A composition procedure preserving mirrored units and enforcing repetition restrictions.
4. A verified length artifact with a direct audit of the reference and partial comparisons of inventory policies.

The remainder construction and mirror identity are established techniques.

### 2. Search and structural feasibility

#### 2.1. State and output identity

Let n(x) lowercase a text and retain only ASCII letters a–z. A text x is palindromic when n(x) equals its reverse. A mirror pair is an ordered pair of word sequences (L,R) satisfying n(L) = reverse(n(R)). For example, “lived on decaf” and “faced no devil” demonstrate this letter relation; neither half needs to be a complete sentence. The spaces make the phrases readable but do not enter the palindrome definition.

The search state is S = (L,R,o,d), storing the left and right word sequences, the unmatched remainder o, and its orientation d. The unmatched remainder—the overhang, or debt—records the letters one side still owes the other. Norvig describes this state in his account of palindrome construction. A compatible addition can consume the remainder or extend beyond it, leaving a new remainder on the opposite side. [Norvig, 2016](https://www.norvig.com/palindrome.html)

Our pair-search experiment grows outwards: words are prepended to L and appended to R. A trie supplies compatible additions. Empty-overhang states are candidates for closure. A candidate is accepted only if it splits at the letter midpoint between words, both halves contain at least three words, no word occurs twice anywhere in the pair, and both halves pass the complete structural test below. Word uniqueness is checked on completed candidates.

The primary output key is the ordered tuple of space-joined word sequences (L,R). Deduplication uses this key within each process and again across processes. Different word segmentations remain distinct even when their normalized letters coincide. Swapping the halves is not an equivalence used by the deduplicator. Outputs are lowercase word sequences without punctuation.

#### 2.2. Sentence-pattern inventory

For each lowercased word w in the Brown Corpus (Francis and Kučera, 1979), T(w) contains its observed tags under NLTK’s universal tag mapping. Unknown words have no permitted tags. There is no separate morphological analysis. Corpus sentence patterns are exact tag sequences after removing punctuation tags (.) and X tags. The frozen set contains 49,815 word-tag entries and 8,982 distinct patterns of three to nine tags.

We retain patterns that contain VERB and begin with PRON, DET, NOUN, ADJ, NUM, or ADV. This leaves 5,649 patterns, denoted P. The opening-tag requirement is a structural proxy; it does not identify a grammatical subject.

A word sequence W passes the complete test F(W) if at least one assignment from the Cartesian product of its word-tag sets equals a pattern in P. The same assignment has to satisfy the pattern, opening-tag, and verb requirements. Ambiguous words may therefore pass through any permitted reading.

For a partial sequence W of m words, prefix feasibility requires an assignment matching the first m tags of some pattern in P; suffix feasibility uses the last m tags. Empty sequences pass when P is nonempty, and sequences longer than nine words fail. The partial-state condition is suffix-feasible(L) and prefix-feasible(R).

The orientation follows the growth directions: future words extend the beginning of L and the end of R. Once L cannot match any permitted suffix, prepending words cannot repair it; likewise, appending words cannot repair R if it cannot match a permitted prefix. The filter preserves completions satisfying the tag constraint. With finite limits, however, it can still change which completions the traversal reaches.

#### 2.3. Bounded traversal

Both configurations use a depth-first stack and the same complete acceptance test. Terminal filtering omits the partial-state condition. Incremental filtering applies it to opening states and candidate children before pushing them onto the stack.

Algorithm 1: structural pair search

1. Construct compatible openings, assign this process its shard, and shuffle its openings using the process seed.
2. Push openings satisfying the length and overhang limits; in the incremental configuration, also require partial-state feasibility.
3. Pop a state and increment the popped-state counter. If it closes, increment the closure counter and apply the complete acceptance and deduplication tests.
4. Generate and shuffle compatible children. Reject children exceeding a structural size limit; in the incremental configuration, also reject those failing partial-state feasibility. Push the remaining children.
5. Continue until the stack is empty or a stopping limit is reached.

The algorithm stores full partial word sequences, so the structural test can inspect their tag possibilities. The separate phrase-based length search additionally maintains used-phrase membership and word counts, restoring both during backtracking.

### 3. Structural-search experiment

The vocabulary payload has 28,402 entries. Restricting it to words present in the Brown tag table leaves 21,073 words for both configurations. Accepted outputs contain 20–44 letters. Intermediate states are limited to 18 word units and an overhang of at most 16 letters.

The launcher requests one Polaris node and 32 CPU processes, bound one per core, using Python 3.11. Polaris nodes provide a 32-core AMD EPYC 7543P processor and 512 GiB of memory; this search does not use their GPUs. [ALCF machine documentation](https://docs.alcf.anl.gov/polaris/)

Opening i belongs to process i modulo 32. Each configuration receives the same opening partition and rank seeds 0–31. Shards partition opening subtrees; their workloads need not be equal. Within each process, terminal filtering runs before incremental filtering. Pruning changes subsequent random-number consumption, so equal seeds do not imply identical sibling orders throughout the walk.

Each configuration stops at the first applicable limit: approximately 600 elapsed seconds per process, 20 million popped states, 10,000 distinct accepted pairs, or exhaustion. The clock is checked every 4,096 popped states. These are common stopping limits rather than equal measured computational work.

| Measure                              |   Terminal | Incremental |
| ------------------------------------ | ---------: | ----------: |
| Popped states                        | 84,815,872 | 115,236,212 |
| Closed states before final filtering |  1,117,411 |     536,805 |
| Rejections by partial-state filter   |          0 | 138,210,789 |
| Distinct accepted pairs              |     20,989 |      86,511 |
| Sum of process elapsed seconds       |  19,214.13 |  19,050.983 |
| Pairs per million popped states      |     247.47 |      750.73 |
| Pairs per summed process-second      |      1.092 |       4.541 |

In this experiment, incremental filtering produced 4.12 times as many distinct accepted pairs. The saved outputs were rechecked for exact reversal, word uniqueness, output-key uniqueness, and complete pattern acceptance.

The larger popped-state count is compatible with pruning because rejected children are counted before insertion and excluded from the popped-state counter. Incremental filtering avoids exploring their descendants and changes the distribution of retained states. The runs report the resulting yield, but not the time spent generating children, checking patterns, or processing closures. Pairs per popped state is therefore an output-density measure, not a measure of equal work. Summed process time is wall-clock time, not measured CPU time.

There is one recorded configuration per arm, with fixed arm order. Per-process summaries are absent from the copied run directory, so we cannot determine each process’s stopping cause or estimate workload dispersion. We report no confidence interval or general speedup claim from these aggregate results.

### 4. Composition with preserved units

Given mirror pairs (Lᵢ,Rᵢ) and a palindromic centre C, the sequence L₁…Lₖ C Rₖ…R₁ is palindromic. Each nested adjacency creates two surface joins, Lᵢ|Lⱼ and Rⱼ|Rᵢ. We call this paired adjacency a nesting seam. The identity lets us assemble a longer palindrome without searching again for letter symmetry.

The composition bank keeps pairs whose halves pass the sentence-pattern gate. The implementation excludes self-mirroring halves and repeated normalized half strings, retaining the first occurrence in bank order. Its deduplication is thus stricter than the ordered token-pair identity used for the search-yield experiment.

For a specified centre C, target T, and seed s, the selection kernel performs one seeded shuffle and one greedy pass. Counts are initialized from C. A pair is accepted if its addition fits T and the accumulated repetition limits: no adjacent duplicate words inside a component, at most two occurrences of an internal bigram, at most two occurrences of a word-length template, and at most three occurrences of a token outside the composition exception set. Accepted pairs update the counts; rejected pairs are skipped without backtracking. The rules only become tighter as material is added.

Rendering gives each component its own sentence boundary. An optional length preference sorts pairs by decreasing half-length after shuffling. It first runs ordinary selection with the same centre and shuffled bank, then uses that selected-pair count as a ceiling for length-first selection. The ceiling prevents this option from increasing the number of pairs; the selected total may fall short of T.

This section specifies a construction procedure. The search-yield table evaluates the pair-search configurations; it does not compare composition quality.

### 5. Dictionary-constrained length search

#### 5.1. Reference and restrictions

We use Norvig’s published version-3 output as a named historical reference. His paired-letter search records completed phrase count. Our adaptation records normalized letter count and adds repetition restrictions. The input dictionary is shared, but the objectives and feasible output sets differ. [Norvig, 2016](https://www.norvig.com/palindrome.html)

For both saved texts, evaluation retains lowercase ASCII letters and tokenizes words as runs matching [a-z]+. Reference phrases are recovered from its comma-separated text after removing the webpage heading and footer. Every recovered phrase is verified against the source dictionary.

| Property                            | Published Norvig v3 | Our saved output |
| ----------------------------------- | ------------------: | ---------------: |
| Source dictionary                   |                Same |             Same |
| Evaluation normalization            |       ASCII letters |    ASCII letters |
| Search recording objective          |        Phrase count |     Letter count |
| Letters                             |              90,439 |           90,937 |
| Distinct / total normalized phrases |     16,111 / 16,111 |  16,168 / 16,168 |
| Adjacent repeated token pairs       |                   9 |                0 |
| Maximum nonexception-token count    |                  14 |                3 |

The exception set for this comparison is {a, an, the, and, or, of, to, in, on, at, for, with, by}. Token counts follow the stated tokenizer, including fragments produced by punctuation. The original algorithm forbids phrase reuse but does not impose our token cap or adjacency restriction. The published output violates both added restrictions. Fourteen dictionary keys containing non-ASCII-letter characters are also excluded from our emitted material.

The saved output is 498 letters longer than this reference and has additional exclusions. This is neither a matched-runtime comparison nor a claim about the best result across palindrome methods.

#### 5.2. Inventory policies and available comparisons

Norvig orders candidate letters using counts of compatible phrase prefixes and suffixes. We compare three policies: static leaves these estimates unchanged; unused removes a completed phrase from the estimates and restores it on undo; feasible additionally removes phrases that fail the current used-phrase, character, within-phrase adjacency, or token-cap conditions. Updates use an index from affected words to phrases. Boundary-specific adjacency is checked separately when admitting a phrase.

All adapted runs below use the letter objective, phrase uniqueness, and the adjacency restriction. The token cap is shown explicitly.

| Inventory | Nonexception-token cap | Budget (s) | Best letters |
| --------- | ---------------------: | ---------: | -----------: |
| Static    |                      3 |         45 |       68,286 |
| Unused    |                      3 |         45 |       76,979 |
| Unused    |                      3 |        120 |       88,101 |
| Feasible  |                      3 |        120 |       88,095 |
| Unused    |                   None |        120 |       88,455 |
| Feasible  |                      3 |        300 |       90,937 |

At 45 seconds, the unused-inventory run is 12.7% longer than the static run. At 120 seconds, the unused and feasible runs differ by six letters. These are single runs, and the 45-second pair ran concurrently on the same machine. They are partial comparisons under the adapted objective. There is no matched original-objective arm, so the effect of changing the objective is not isolated. Nor can the 300-second result be attributed to infeasibility removal independently of its longer budget.

### 6. Related work

Hoey–Norvig construction supplies the unmatched-remainder and paired-letter mechanisms used here. Papadopoulos et al. construct a palindrome graph from forward and backward n-gram graphs and support further constraints on the generated sequences. Our partial pattern test specializes compatibility checking to the two growth directions and a finite set of corpus-derived tag patterns. [Norvig, 2016](https://www.norvig.com/palindrome.html); [Papadopoulos et al., 2015](https://www.ijcai.org/Proceedings/15/Papers/353.pdf)

Constrained generation also studies where to enforce restrictions and how to allocate search. Grid Beam Search retains hypotheses under lexical constraints; NeuroLogic A\*esque uses lookahead heuristics for future constraint satisfaction; DOMINO aligns formal-language constraints with subword generation. These methods motivate comparisons of constraint placement and overhead. They are not palindrome-length baselines, and we report no superiority over them. [Hokamp and Liu, 2017](https://aclanthology.org/P17-1141/); [Lu et al., 2022](https://aclanthology.org/2022.naacl-main.57/); [Beurer-Kellner et al., 2024](https://proceedings.mlr.press/v235/beurer-kellner24a.html)

### 7. Reproducibility and conclusion

The local artifact package contains frozen vocabulary and Brown payloads, the structural-pair aggregate, the length output and phrase sequence, and separate verification programs. A current-file audit checks all 107,500 stored pair records across the two arms against their acceptance conditions and reproduces the reference/output comparison. Full input and output hashes, commands, and provenance gaps appear in Appendix A. This is verification by a separate program in the same project, not an external audit.

Under its recorded stopping limits, the structural experiment produced more accepted pairs with incremental filtering. The length experiment produced a longer saved output from the named dictionary while applying added repetition restrictions. Repeated, matched experiments are needed to establish general performance effects for partial-state constraints and inventory-aware ordering.

### Limitations

The tag inventory defines an admissible pattern language and provides no human readability measure. This paper does not evaluate whether structural acceptance improves grammaticality or coherence. Composition also depends on a finite bank and may exhaust acceptable material before reaching its target.

The search comparison has fixed arm order, state-dependent traversal changes, and missing per-process traces. The length runs do not form a complete factorial ablation and do not include a matched contemporary-baseline study. Neither set of measurements supports a general runtime claim. Historical environment provenance is incomplete, although the saved output properties remain directly checkable.

### Appendix A. Reproduction details

#### Structural inputs and preprocessing

The frozen inputs are `tools/polaris/payload/brown.json.gz` and `vocab30k.txt`. The Brown construction uses lowercased observed words and universal tags, removes . and X only when forming sentence patterns, and retains patterns of length 3–9. The partial planner tests all possible tag assignments without the 20,000-reading cutoff used by the composition parser. It imposes no additional morphology or agreement model. The frozen payload, rather than a newly downloaded corpus, defines the experiment’s input.

#### Search configuration

The launcher is `tools/polaris/sentence_plan_debug.pbs`, invoking `sentence_plan_debug.py` with vocabulary request 30,000, letter bounds 20–44, at most 18 units, overhang bound 16, 20,000,000 popped states, 600 seconds, and 10,000 accepted pairs per process. It launches 32 ranks with seeds equal to rank IDs. The default arms run in the order terminal, planned. No bigram model participates in this comparison.

#### Composition

Reproduction requires an ordered bank, explicit centre, target, and seed. When the API seed is omitted, the implementation uses the current time; that request is not reproducible. The composition exception set is {a, an, the, of, to, in, on, at, as, is, it, i, for, and, or, if, no, not, was, are, be, by, my, we, me, its, this, that, with, from}. It differs from the thirteen-token length-search set. The selection kernel and rendering are implemented in `llm_palindrome/hierarchy.py`.

#### Verification

From the repository root, `python3 paper/verify_structural_draft.py` checks saved pair identities, reversal, word uniqueness, complete pattern acceptance, the normalized reference, and the 90,937-letter artifact. It also computes the manifest in `paper/structural-evidence.json`. Length-search reproduction uses `python3 -m experiments.norvig_letters --seconds 300 --dynamic --feasible --out <new-output-directory>`; its elapsed-time stopping condition need not reproduce the identical winning text.

#### Hashes

The manifest retains the full SHA-256 values. Identifying prefixes are: Brown payload `14a81b69751dbb9e`; vocabulary `6e24f122e2857271`; planning aggregate `ed584869a2fb54cf`; Norvig code `c21a79f77e3021c0`; dictionary `3f28b8a95d92be6c`; winning text `95722a8a12516c44`.

#### Provenance scope

The present evidence checkout is `d24517d6202a7aee4a0a796ac825f6c6bef020e9`. It identifies the source inspected for this revision, not a proven historical job checkout. The launcher specifies Python 3.11, but the historical aggregate does not freeze its patch version, operating-system image, MPI version, dependency versions, or source revision. Per-rank traces and a runtime profile are absent from the copied run directory. These omissions limit search replay and timing analysis; they do not prevent verification of the saved candidates.

### References

Beurer-Kellner, Luca, Marc Fischer, and Martin Vechev. 2024. [Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation](https://proceedings.mlr.press/v235/beurer-kellner24a.html). ICML, 3658–3673.

Francis, W. N., and H. Kučera. 1979. *Manual of Information to Accompany a Standard Corpus of Present-Day Edited American English, for Use with Digital Computers*. Revised edition. Brown University.

Hokamp, Chris, and Qun Liu. 2017. [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](https://aclanthology.org/P17-1141/). ACL, 1535–1546.

Lu, Ximing, et al. 2022. [NeuroLogic A\*esque Decoding: Constrained Text Generation with Lookahead Heuristics](https://aclanthology.org/2022.naacl-main.57/). NAACL, 780–799.

Norvig, Peter. 2016. [World’s Longest Palindrome? 21,012 Words](https://www.norvig.com/palindrome.html). Version 3.

Papadopoulos, Alexandre, Pierre Roy, Jean-Charles Régin, and François Pachet. 2015. [Generating all Possible Palindromes from Ngram Corpora](https://www.ijcai.org/Proceedings/15/Papers/353.pdf). IJCAI, 2489–2495.
