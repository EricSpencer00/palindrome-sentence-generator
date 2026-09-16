# Experiment novelty registry

This registry prevents a new filename from disguising a repeat search.  An
experiment is retained only when its state-space signature changes a
construction dimension (lexical roles, dependency structure, attachment
depth, or repair policy).  A larger pool, larger beam, or new random seed is
not a new operator.

The validator checks that every retained signature is unique and that every
artifact exists.  The entries below are diagnostic history; none is reader
evidence unless its own exact and blinded-reader gates are satisfied.

| id | state-space signature | changed dimension | result |
|---|---|---|---|
| `reversible-grammar-insertion-20260916` | seed-preserving, non-repeating insertion of independently annotated reverse lexical units at a grammatical seam | scalable context-free wrapper growth | base: 5 exact renderings (50--106 letters), all seam-incoherent; adjunct-insertion repair: 12 near misses, 0 exact; 0 reader-eligible |
| `typed-semordnilap` | PERSON/THING/VERB clauses with independent reverse segmentation | typed lexical roles | 0 closures >=39 |
| `global-brown-pos` | Brown POS shapes, 8--15 words, live center-out syntax | global shape lattice | short controls only |
| `typed-lexical-graph` | number/tense/determiner/transitivity in residual state | feature-carrying lexical graph | 0 closures >=39 |
| `brown-attested-svo` | independently attested DET? NOUN VERB DET? NOUN relations | attested valency edges | 0 residual matches |
| `brown-two-svo-attachments` | two SVO relations with adjective/PP attachments | adjacent-clause composition | 0 residual matches |
| `brown-cross-clause` | relative `who` and controlled conjunction dependency frames | cross-clause dependency | 0 residual matches |
| `brown-shared-relative` | shared subject/object co-reference with agreement | co-reference state | 0 residual matches |
| `brown-two-relative-chain` | two relative edges with shared agreement variables | dependency-chain depth | 0 residual matches |
| `brown-two-relative-attachment` | independently attested attachment on either relative edge | chain attachment depth | 0 residual matches |
| `brown-shared-coref-attachment` | attachment inside shared-coreference relative clause | inner relative attachment | 0 residual matches |
| `brown-bounded-repair` | one tense-preserving verb/attachment lexical edit | bounded lexical repair | 0 residual matches |
| `brown-joint-relation-repair` | complete attested SVO replacement with >=2 changed content slots | joint subject/object repair | 0 residual matches |
| `brown-relative-topology` | new PP positions plus determiner boundary variants; prior 497 keys excluded | attachment topology/boundary | 0 residual matches |
| `brown-coreferent-variable-pp` | variable-length co-referent PPs at every chain boundary; prior 8,695 keys excluded | co-referent PP length and boundary position | 0 residual matches |
| `event-frame-independent-relexicalization` | semantic event frame with temporal + intransitive/copular state, independently lexicalized reverse event phrase | semantic event structure and phrase-unit residual parsing | 0 reverse parses |
| `fresh-paired-clause-ledger` | four authored left/right clause proposals with repository-wide tape fingerprint | paired authoring provenance | 2 exact closures, both rejected |
| `character-ledger-promptbank` | cross-product of four fresh left clauses and four independently authored right guesses under a reverse-tape constraint | prompt-bank pairing policy | 0 exact closures |
| `two-event-discourse-frame` | ordered intransitive event plus result state with strict temporal rank and independent reverse discourse lexicalization | two-unit discourse state | 0 reverse parses |
| `connective-bearing-event-pair` | the same ordered event/result state with an explicit independently lexicalized connective slot | connective-bearing discourse state | 0 reverse parses |
| `constrained-reverse-lexical-v2` | memoized, frequency-ranked complete lexicon segmentations with bounded reverse-compatible prefixes | constrained lexical-prefix decoding | 0 complete parses; 4 partial probes |
| `connective-semantic-class-event-pair` | contrast/cause/consequence class state carried through the ordered event/result residual | semantic relation class | 0 reverse parses |
| `relation-graph-event-pair` | explicit directed cause/consequence or symmetric contrast edge with polarity and strict temporal order | relation-graph state | 0 reverse parses |
| `two-edge-event-micrograph` | event → intermediate state → result with two explicit edges and three-rank temporal order | two-edge semantic topology | 0 reverse parses; ladder stopped |
| `dialogue-acknowledgment-residual-inventory` | hand-authored question→answer and instruction→acknowledgment act cross-product with independent residual indexing | dialogue-act semantics | 0 residual matches |
| `dialogue-elliptical-ack-residual-inventory` | paired `can we`/`what about` prompts with elliptical answers and imperative acknowledgments | elliptical dialogue-act semantics | 0 residual matches |
| `dialogue-shared-topic-elliptical-residual` | paired prompts and paired elliptical/acknowledgment responses constrained to one shared discourse topic | shared-topic discourse state | 0 residual matches; dialogue family closed |
| `multiword-unit-transducer-local-repair` | independent idiomatic fragments with one reversible affix/compound rewrite and boundary resegmentation | lexical-unit transformation operator | 0 exact closures; 396 surfaces evaluated |
| `morphological-derivational-seam` | independent inflectional/derivational forms with reverse character seams crossing lexical boundaries | morphology and derivational seam state | 0 exact closures; 113,250 left surfaces |
| `cp-semantic-grammar-palindrome` | one-hot semantic grammar and lexical slots with global character-domain equality and exact-N satisfiability | constraint-programming grammar intersection | 0 closures; 44,071 states |
| `clause-lattice-joint-dp` | independent complete clauses solved by length-indexed character equations with explicit subject/tense/argument constraints | clause-lattice dynamic programming | 0 exact closures; 49 frame pairs |
| `character-clause-fst-joint-emission` | arithmetic/measurement clause banks compiled into a character-trie product that emits mirrored characters jointly | finite-state character transduction | 0 exact closures; 4 product states |
| `scene-slot-graph-residual` | coordinated locative/possessive/attributive scene clauses with `located_at`, `possesses`, and `describes` edges; independent right-side typed slot graph consumes the reverse character residual | bidirectional semantic slot graph with joint English slot order and reverse lexical constraints | 0 reverse closures; 6 retained residual probes |
| `two-bank-word-equation-seam-dp` | two independently authored, content-disjoint complete clause banks solved by a memoized character prefix/suffix equation with deliberate outer-letter compatibility | seam-aware word-equation DP across independent clause banks | 0 closures; 100 probes |
| `seam-first-complete-clause-authoring` | seam inventory and terminal widths selected before independently authored complete SVO clauses; exact joint enumeration crosses ordinary word boundaries | outside-in seam-first authoring order | 0 closures; 13 retained probes (3 shortcut rejections) |
| `semantic-dependency-outside-in` | fresh cause/preparation narrative with integer span boundaries, variable-length noun phrases, dependency-state obligations, and paired character domains | outside-in semantic arc consistency over spans | 0 closures; 5 partial probes across 39--87 targets |
| `synchronous-semantic-parse-equations` | two independent semantic parses expanded in lockstep against a shared character-equation frontier with typed role/discourse state | synchronous semantic parse coupling | 0 closures; 6,561 lockstep states; 18 probes |
| `internal-center-window-repair` | complete authored clause pairs with one lexical center rewritten from a finite semantic inventory while the exterior stays fixed | internal lexical-center window repair | 0 exact closures; 56 bounded repairs |
| `discourse-plan-coupled-expansion` | typed narrative-plan graph expands two independently authored two-sentence narratives with semantic role agreement before mirrored-character emission | discourse-plan coupling with online character equality | 0 closures; 12 early rejects |
| `collocation-synchronous-grammar` | role-typed natural collocation frames solved in lockstep against a mirrored character equation while preserving complete clauses | collocation-level synchronous grammar | 0 exact closures; 25 pairs |
| `human-compositional-center-window` | independently authored two-sentence mini-scenes with finite natural continuation substitutions at a content-word center | human-compositional center-window repair | 0 exact closures; 24 substitutions |
| `global-semantic-paraphrase-rewrite` | complete two-clause narratives jointly relexicalized across agent/verb/object slots with online mirrored equations | global typed semantic paraphrase | 0 exact closures; 1,296 clause pairs |
| `collocation-graph-path` | connected role-typed collocation graph with shared-node overlap and bounded local edge repair | overlap-aware collocation graph walk | 0 exact closures; 36 paths |
| `semantic-sentence-pair-alignment` | independently authored complementary complete sentences selected from semantic paraphrase alternatives and aligned over the whole normalized tape | cross-boundary semantic sentence alignment | 0 exact closures; 20 combinations |
| `template-analogy-semantic-lexicalization` | abstract role-shape templates independently lexicalized with fresh semantic words and paired seam-width constraints | template-analogy authoring | 0 exact closures; 499 clause pairs |
| `neural-dual-prefix-beam-v2` | left roles emit forward while right roles emit from the right edge as reversed characters; GPT-2 ranks only prefixes that satisfy the shared character equation | neural dual-prefix proposal ordering | 0 closures; 25 bounded expansions (v1 invalidated) |
| `dependency-attribute-grammar-chart` | two independently lexicalized recursive dependency forests carry number/tense/valency attributes; a bilateral chart joins complete derivations while consuming opposite character edges | head-driven attribute-grammar derivation and dependent attachment | 0 closures; 5,184 complete paired derivations |
| `evolutionary-prose-genome` | complete typed sentence pairs evolve through constituent-preserving two-point crossover and seam-biased terminal mutation; mismatch and word-order signals jointly rank the population | population repair over intact prose genomes | 0 closures; 5,760 genomes; best rendered mismatch 26 pairs |
| `grammar-intersection-chart` | recursive context-free grammar chart with independently derived complete sentences intersected against the reversed normalized-character language; relative-clause depth is the growth variable | recursive derivation language intersection | 0 closures; 644 unique chart items; 0 reverse intersections |
| `lexicalized-tag-yield-equation` | recursive lexicalized TAG derivations use typed auxiliary-tree stacks; independently lexicalized yields are joined from opposite terminal edges | recursive adjunction/pushdown yield equation | 0 exact joins; 21,158 structural states; 40 rendered probes; 20,914 repair states |
| `bpe-dual-continuation` | independently authored SVO clauses emit ordinary-order GPT-2 BPE pieces; each right piece consumes the next character residual while agreement-filtered grammar stays live | subword token lattice with held-out seam repair | 0 closures; 900 left clauses; 1,800 base/repair targets; 40 rendered probes |
| `mined-phrase-chunk-clause-composition` | mined 2--4-word phrase chunks are independently composed as complete determiner-led NP--finite VP--headed-NP clauses, then joined by a reverse index with disjoint content words and no mirrored chunks | multiword phrase atoms plus strict transitive-clause cross-product composition | 0 closures; 541,598 unique clauses; 1,826 chart items; 40 independently audited probes |
| `variable-boundary-tape-ilp` | one lexicalized grammar path with binary word-start arcs; word boundaries, semantic features, and mirrored character equations are solved together by a bounded MILP | variable-boundary character-tape flow | 0 exact closures at 39 letters within bounded HiGHS run; 57,166 lexical arcs; 1 intact grammar probe |
| `reverse-complement-eulerian` | character-context overlap edges from corpus text are assembled into edge-disjoint trails with reverse-complement balance; English word segmentation and conservative dependency parsing happen only after the trail | reverse-complement Eulerian coverage over character-context edges | 0 reader-eligible trails; 40 exact rendered probes; 4,452 eligible context edges; singleton-context repair yielded 0 balanced trails |
| `prosodic-foot-surface-realizer` | complete clauses are realized through CMU pronunciation entries carrying syllable count, lexical stress, and phrase-boundary obligations before independent reverse-tape joining | prosodic-foot lattice and stress-constrained surface realization | 0 exact joins; 9,000 realizations per side; 4 preserved runs including two concrete repairs; 0 reader-eligible outputs |
| `global-tied-masked-denoising` | every character position is a tied variable; parallel masked-word assignments are propagated across the whole mirrored tape before scoring | global tied-character denoising with a bidirectional position ledger | 6 rendered proposals (3 ordinary seeds, 3 catalogue controls); 0 admitted at 39+ letters; no reader-eligible output |
| `character-lm-half-tape` | a character-level beam generates the left half under joint forward/reverse n-gram scores; the reflected tape is segmented independently by Viterbi | character material generation plus independent lexical boundary recovery | 40 exact tapes per retained run; repeated-short-word collapse persisted through two repairs; 0 mechanically admitted and 0 reader-eligible outputs |
| `corpus-sentence-gram-fst` | rank-partitioned corpus sentence/n-gram phrase lattices generate the left side while a disjoint held-out phrase lattice consumes the synchronized reverse character residual around a single-letter center | phrase-token FST intersection with rank-held-out reverse segmentation | 0 exact closures; 120 dead-end/partial rendered probes across base and two phrase-bank repairs |
| `semordnilap-template-inventory` | finite semordnilap pairs fill typed syntactic templates; reflected character seams are independently re-segmented and a seam swap is the held-out repair | semordnilap seam inventory with template-level composition | 350 probes; 0 exact closures; 0 mechanically admitted and 0 reader-eligible outputs |
| `semantic-proposal-verifier` | intact human/external prose proposals are independently audited for exactness and semantic plausibility; mismatch diagnostics direct the next proposal without editing or mirroring input | proposal contract and independent acceptance gate | 3 proposals; longest 49 letters; 0 exact/admitted |
| `lexical-admission-centerout` | wordfreq/count-2w center-out search keeps whole-word uniqueness live and applies the complete mechanical admission gate at closure while sweeping target lengths | live lexical anti-filler state plus closure admission | 4 exact mechanically admitted surfaces (48--116 letters); 0 reader-eligible outputs |
| `grammar-boundary-resegmentation-repair` | a frozen exact tape is held fixed while a Brown-POS weighted dictionary chart searches independent word boundaries and reports a finite-verb diagnostic | fixed-tape grammatical boundary repair | v1/v2 had no lexical coverage; v3 produced 2 exact mechanically admitted segmentations; 0 reader-eligible outputs |
| `fixed-tape-valency-chart-repair` | the same frozen tape is parsed by a subject--finite-verb/object chart with clause-boundary transitions after POS resegmentation | argument-role/valency state over fixed boundaries | 2 exact mechanically admitted segmentations; 0 complete-clause parses; 0 reader-eligible outputs |
| `proper-name-caption-crossword` | typed proper-name/appositive incident records joined as complete captions with a crossword-style character compatibility filter | proper-name caption role grammar | 25 rendered probes; 0 exact closures; 0 reader-eligible outputs |
| `information-structure-focus-scope` | independently authored negative cause/result clauses carry focus, presupposition, and polarity state before the whole rendered tape is audited | information-structure and polarity state | 16 intact probes plus a terminal-seam repair; 0 exact closures; 0 reader-eligible outputs |
| `anaphoric-scene-chain-composition` | complete three-sentence scenes carry a singular object antecedent through pronoun and definite-description continuations | typed anaphora and discourse continuity | 4 intact scenes; 0 exact closures; 0 reader-eligible outputs |
| `multiset-balanced-pair-sampling` | two complete typed clauses are independently sampled and retained only when their 26-dimensional letter-count parity can support an exact palindrome | letter-multiset arithmetic frontier | 50,000 random pairs, an indexed 2,304-clause repair, and a 98,304-clause agreement repair; 2,720 parity joins; 0 exact closures |
| `proper-name-reverse-grammar` | a 140,368-clause grammar of typed names, objects, and locatives is indexed by normalized tape and joined against exact reversed tapes | proper-name reverse grammar | 0 exact pairs; 25 ordinary probes; 0 reader-eligible outputs |
| `lexical-chain-walk` | one forward typed lexical graph walk emits collocation-constrained word paths without reverse emission or paired-clause composition | single-clause lexical graph walk | 6,375 complete probes; 0 exact closures; 0 reader-eligible outputs |
| `adaptive-crossword-span-cover` | a whole normalized tape is solved as a variable-boundary span lattice; context-conditioned phrase proposals propagate mirrored character variables with reversible backtracking, then a non-mirrored lexical chart recovers prose | adaptive variable-boundary span cover | 10,785 exact tapes; all 10,785 failed the no-fragment/admission gate because lexical recovery collapsed to one-letter fragments; 0 reader-eligible outputs |
| `context-template-crossword-repair` | authored contextual phrase templates are held intact while a minimum-three-letter reflected segmentation and non-echo constraint reject the adaptive route's one-letter collapse | minimum-word-length prose repair | 18 probes; 0 exact closures; 0 admitted or reader-eligible outputs |
| `syntactic-mirror-template-repair` | independent typed subject/event/continuation slots enforce a minimum-two-letter surface and whole-tape audit | semantic-role-bound template repair | 2 exact surfaces at 43 and 51 letters, both rejected as known catalogue/duplicate-span material; 0 admitted |
| `semantic-selectional-prefix-automaton` | corpus-derived subject--verb--object preferences remain live while two independently ordered slot grammars synchronize characters through a prefix trie | learned selectional preference state | 441 frame/shape runs, 15,477 states after the subordinate-clause repair, 0 exact closures; best matched frontier preserved; 0 reader-eligible outputs |
| `model-authored-clause-bank-index` | a fresh model-authored bank of de-duplicated complete clauses is partitioned into independent banks; one- and two-clause concatenations are joined only through a character reverse index | independent complete-clause proposal bank and composition depth | 172 clauses; 75x97 single-clause probes plus 5,550 two-clause probes; 0 exact closures; 0 reader-eligible outputs |
| `semantic-scene-seam-growth` | complete model-authored events are grown into topic-linked three-event scenes; each event boundary carries a local two-character seam before an independent reverse lookup over full scenes | incremental semantic-scene growth and seam state | 333,306 coherent scene states plus 42,070 terminal-event repairs; best repair matched 2 reflected characters; 0 exact closures; 0 reader-eligible outputs |
| `live-seam-intent-continuation` | an intent-conditioned model continuation receives ordinary left prose plus the exact required reversed tape at a live seam; no proposal bank or reverse index is used | direct constrained continuation at a live character seam | 6 complete-sentence seam trials; all timed out before producing a candidate; 0 exact closures; bounded short-continuation repair recorded |
| `role-aware-reversible-reservoir-centerout-20260915` | Brown/word-frequency reversible pairs are derived and assigned to attested POS roles before center-out debt solving | corpus-derived reversible lexical reservoir | 78,933 states across three templates; 0 exact terminals; 0 reader-eligible outputs |
| `variable-length-role-reservoir-centerout-20260915` | six complete 5--10-slot templates are solved independently against the same role-aware reservoir | variable-length grammar family | 22,261 states; 0 exact terminals; 0 reader-eligible outputs |
| `asymmetric-template-reservoir-centerout-20260915` | four distinct left/right semantic templates carry an agreement state through bilateral character debt | asymmetric paired-template state product | 552 states; 0 exact terminals; 0 reader-eligible outputs |
| `attested-phrase-pair-wrapper-20260915` | Brown phrase spans are indexed as left chunks and their exact reverse tapes are segmented independently from a held-out lexical trie | phrase-level span lattice and reverse segmentation | 753 spans, 0 reverse segmentations, 0 exact closures; 0 reader-eligible outputs |
| `homograph-sense-lattice-20260915` | one orthographic tape is required to support two independent homograph sense/POS parses with boundary choices made in the exact search | dual-sense semantic parse state | 25 frame pairs, 1,912 explored states, 0 exact terminals; 0 reader-eligible outputs |
| `wordnet-synonym-frame-csp-20260915` | WordNet lemma alternatives are attached to semantic dependency slots; independently lexicalized complete frames are joined by a character equation | semantic-preserving synonym lattice | 6 frames, 6,834 reverse-frame calls, 0 reverse hits, 0 exact/admitted/reader rows |
| `wordnet-featured-frame-repair-20260915` | subject number, determiner agreement, transitivity, inflection, and optional adjunct attachment are carried through a finite WordNet frame automaton | agreement/subcategorization state | 90,000 left assignments, 60,012 reverse-frame calls, 0 hits, 0 exact/admitted/reader rows |
| `gpt2-topic-half-decoder-20260915` | GPT-2 samples topic-conditioned ordinary half-clauses; an independent lexical DP decodes each required reverse tape | model-conditioned proposal plus reverse lexical decoding | 288 samples, 104 reverse decodes, 73 exact but fragmentary surfaces, 0 mechanically admitted/reader rows |
| `gpt2-center-letter-bridge-20260915` | GPT-2 half clauses are joined through an explicit odd-length center-letter insertion before independent right-tape decoding | odd-center bridge state | 384 samples, 3,848 center decodes, 4 exact 39-letter surfaces, 0 mechanically admitted/reader rows; orientation repair also yielded 0 exact rows |
| `gpt2-center-fsm-bridge-20260915` | Brown-backed POS/constituency states are carried into the odd-center reverse decode; fragments are rejected before scoring | right-side clause automaton | 384 samples, 3,354 center trials, 0 complete POS-frame hits, 0 exact/admitted/reader rows |
| `interrogative-quantifier-fsm-20260915` | explicit auxiliary-inversion question states are joined to quantified-answer dependency states under an odd-center character equation | question–answer discourse automaton | base: 420,000 left assignments and 52,197,516 right-frame yields; indexed repair: 62,901,860 slot-prefix trials, 1 hit (known catalogue), 0 novel exact/admitted/reader rows |
| `morphology-first-dependency-lattice-20260916` | dependency topology is selected before node-local lemma, derivation, and inflection paths are unified and rendered in ordinary order | morphology-first dependency yield | base: 1,728 yields with 3 catalogue/symmetry diagnostics; derivational repair: 24,570 yields and 0 exact; 0 mechanically admitted/reader rows |
| `semantic-relation-alignment-20260916` | directed event-edge relations choose independent role lexicalizations while a deterministic boundary-synchronous ledger checks ordinary-order yields | semantic relation-edge alignment | base: 36,662 states, deepest match 6; terminal-span/odd-center repair: 646 states; 0 exact/admitted/reader rows |
| `weighted-cfg-sync-dp-20260916` | independent weighted CFG parse forests intersected by character position with typed adjunct repair | synchronous parse-forest intersection | base: 648 derivations/side and 13,392 chart states; adjunct repair: 3,240 derivations/side and 82,512 states; 0 exact/admitted/reader rows |
| `reversible-grammar-insertion-20260916` | five distinct annotated reverse lexical units inserted at a preserved seed seam | non-repeating scalable seam growth | 5 exact renderings (50--106 letters), all seam-incoherent and withheld from readers; 0 reader-eligible |

The abandoned local `brown-attached-two-clause-residual` probe is deliberately
absent: it duplicated the adjacent-clause space and was invalidated by an
empty composition frontier.  Its ignored run files were removed rather than
counted as evidence.

## Preflight exclusions (checked, not new families)

These artifacts are retained because they document concrete failed repair
attempts, but they are not registered as new construction families.  The
novelty audit compared their operators with the registered signatures before
the next frontier was chosen:

| artifact | exclusion reason |
|---|---|
| `experiments/seed_symmetric_mutation_search_20260915.py` | mirrored character substitution/insertion is a seed-local instance of the registered internal center-window repair dimension; changing the seed or edit count would be a replay, not a new method |
| `experiments/seed_boundary_shift_typed_resegmentation_20260915.py` | typed boundary shifting plus lexical replacement is the held-out repair of that same seed-local window and does not introduce a distinct state-space dimension |
| `experiments/semantic_involution_frame_20260915.py` | the probes repeat a frame in reverse lexical order and include a known catalogue palindrome, so they are explicitly rejected as a forbidden word-order shortcut rather than counted as a generator family |
| `experiments/thematic_grid_clause_composition_20260915.py` | its 4x4 authored-clause cross-product and whole-tape audit replay the existing complete-prose pairing space; seam filtering changes ordering, not the construction dimension. The 16 probes and zero closures remain failure evidence, not a retained family |
| `experiments/semantic_pairing_typed_clauses_20260915.py` | typed SVO/PP clause cross-product plus reverse-tape pairing overlaps the existing typed-semordnilap and complete-prose pairing families. The 17,280-frame run and bounded repair queue remain preserved, but are not counted as a new family |
| `experiments/multiset_balanced_grammar_20260915.py` | the implementation never enforces its declared multiset/count state; it only samples typed role productions and scores mirrored matches, overlapping existing grammar-production probes. Its 5,000 samples remain preserved as invalid-route evidence |

Their run records remain available for failure analysis (`runs/seed-symmetric-
mutation-20260915.json` and `runs/seed-boundary-shift-typed-resegmentation-
20260915*.json`). A future route may use their residuals only after a new
signature is pre-registered and its state space is disjoint from all previously
registered families.

## Novelty audit policy (2026-09-15)

The exact-collision preflight is now paired with a deterministic lexical-overlap
screen in `tools/audit_experiment_novelty_20260915.py`. It reports prior
families sharing distinctive signature atoms and marks those pairs for human
review; changing a seed, beam width, lexical bank, or filename is not a new
family. The current audit covers all 97 retained families and 6 explicit
exclusions, finds no exact signature collision, and flags historical near
pairs for review. The latest routes are below the review threshold:

| route | nearest prior family | atom Jaccard | disposition |
|---|---|---:|---|
| `semordnilap-template-inventory` | `internal-center-window-repair` | 0.167 | distinct seam inventory and typed templates |
| `corpus-sentence-gram-fst` | `event-frame-independent-relexicalization` | 0.056 | distinct rank-partitioned phrase FST |
| `semantic-proposal-verifier` | none (0.000) | 0.000 | external proposal contract, not a generator replay |
| `lexical-admission-centerout` | none (0.000) | 0.000 | live uniqueness and admission-at-closure state; distinct from half-tape character recovery |
| `grammar-boundary-resegmentation-repair` | none (0.000) | 0.000 | fixed-tape POS-weighted boundary repair; it emits no new palindrome letters |
| `fixed-tape-valency-chart-repair` | none (0.000) | 0.000 | fixed-tape subject/verb/object chart; stricter successor to POS-only segmentation |
| `proper-name-caption-crossword` | none (0.000) | 0.000 | retained because the proper-name/appositive record grammar is a distinct lexical-role state; no exact output |
| `information-structure-focus-scope` | none (0.000) | 0.000 | retained because focus/presupposition/polarity is an explicit information-structure state; no exact output |
| `anaphoric-scene-chain-composition` | none (0.000) | 0.000 | retained because antecedent-number continuity spans a complete three-sentence scene; no exact output |
| `multiset-balanced-pair-sampling` | none (0.000) | 0.000 | retained because letter-count parity is a new necessary-state filter; no parity survivor or exact output |
| `proper-name-reverse-grammar` | none (0.000) | 0.000 | retained because typed proper-name/object grammar is independently indexed against the reverse tape; no exact output |
| `lexical-chain-walk` | none (0.000) | 0.000 | retained because a single forward collocation graph walk with no reverse emission or paired clause is a distinct lexical transition space; no exact output |
| `adaptive-crossword-span-cover` | none (0.000) | 0.000 | retained because whole-tape variable-boundary span propagation with reversible backtracking and context-conditioned proposals is a distinct state space; exact but unreadable fragment outputs only |
| `context-template-crossword-repair` | none (0.000) | 0.000 | retained as a concrete minimum-word-length/non-echo repair operator for the adaptive collapse; it produced no exact closure |
| `syntactic-mirror-template-repair` | none (0.000) | 0.000 | retained only as a documented typed-slot repair; its two exact outputs were known catalogue/duplicate-span material and are not candidates |
| `semantic-selectional-prefix-automaton` | none (0.000) | 0.000 | retained because corpus-derived selectional preferences and live character-synchronous slot states change the construction dimension; no exact output |
| `model-authored-clause-bank-index` | none (0.000) | 0.000 | retained because the clauses are independently model-authored complete prose and the search indexes intact one- and two-clause compositions rather than emitting reflected units; no exact output |
| `semantic-scene-seam-growth` | none (0.000) | 0.000 | retained because scene states grow incrementally with topic continuity and local seam constraints before joining, rather than indexing independent clauses directly; no exact output |
| `live-seam-intent-continuation` | none (0.000) | 0.000 | retained because the model is queried directly at a live exact character seam, without a proposal bank or reverse index; all six bounded trials timed out |
| `role-aware-reversible-reservoir-centerout-20260915` | none (0.000) | 0.000 | retained because reversible lexical pairs are derived from independent corpus/POS attestation before debt solving; no exact output |
| `variable-length-role-reservoir-centerout-20260915` | none (0.000) | 0.000 | retained because a variable-length complete-template family changes the grammar dimension; no exact output |
| `asymmetric-template-reservoir-centerout-20260915` | none (0.000) | 0.000 | retained because distinct left/right template products carry agreement state through bilateral debt; no exact output |
| `attested-phrase-pair-wrapper-20260915` | none (0.000) | 0.000 | retained because phrase spans are indexed as construction units and reversed tapes are independently segmented; no exact output |
| `homograph-sense-lattice-20260915` | none (0.000) | 0.000 | retained because dual-sense orthographic parses are carried in the construction state; no exact output |
| `terminal-aware-grammar-intersection-20260916` | none (0.000) | 0.000 | retained because independent typed clause grammars explicitly take lexical-boundary epsilon transitions during character intersection; 9 short closures, 0 admitted/reader rows |
| `semantic-mcts-derivation-20260916` | none (0.000) | 0.000 | retained because UCT allocates rollouts over typed semantic derivation actions with a live reflected-character ledger; base: 30,000 rollouts, longest probe 29 letters; reverse-prior repair: 30,000 rollouts, longest probe 30; 0 admitted/reader rows |
| `morphosemantic-product-delay-20260916` | none (0.000) | 0.000 | retained because looping feature automata realize morphology on demand through a persistent output-delay monoid without complete-clause materialization; 0 closures |
| `morphology-first-dependency-lattice-20260916` | none (0.000) | 0.000 | retained because lemma/derivation/inflection paths are unified on typed dependency nodes before ordinary-order yield; exact diagnostics were catalogue/symmetry rejected |
| `semantic-relation-alignment-20260916` | none (0.000) | 0.000 | retained because directed event-edge topology and boundary-synchronous independent role lexicalization are a distinct state; deepest match 6 characters and no exact closure |
| `weighted-cfg-sync-dp-20260916` | none (0.000) | 0.000 | retained because independent weighted parse forests are intersected by character position; base and adjunct repair produced no exact closure |
| `reversible-grammar-insertion-20260916` | none (0.000) | 0.000 | retained because five distinct reverse lexical units grow the seed without repetition; exact outputs were seam-incoherent and withheld from readers |

The JSON report is retained at `runs/novelty-audit-20260915.json`. Near-pair
flags are not claims of equivalence; they are a stop-and-review gate before
another construction run.
