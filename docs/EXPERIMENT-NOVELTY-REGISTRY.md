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

The abandoned local `brown-attached-two-clause-residual` probe is deliberately
absent: it duplicated the adjacent-clause space and was invalidated by an
empty composition frontier.  Its ignored run files were removed rather than
counted as evidence.
