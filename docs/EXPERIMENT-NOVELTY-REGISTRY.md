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

The abandoned local `brown-attached-two-clause-residual` probe is deliberately
absent: it duplicated the adjacent-clause space and was invalidated by an
empty composition frontier.  Its ignored run files were removed rather than
counted as evidence.
