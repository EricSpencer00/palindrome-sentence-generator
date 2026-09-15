# Readable palindrome finder: goal record

## Objective and acceptance gate

Build a reproducible exact-English-palindrome finder and an ACL/NAACL paper
whose central claim is supported by evidence. The objective is a long readable
palindrome, and longer is better after an output is exact, independently
reproducible, and judged by
independent readers as grammatical, with a recoverable subject or intent and
coherent meaning. The paper must distinguish those reader outcomes from
mechanical exactness and automatic diagnostics.

## Reader-first directive

The deliverable is a reader-worthy construction, not a search-speed result and
not a negative-result paper. Every search run is an instrument for producing a
new intact-English candidate. A candidate is shown with its rendered text,
letter count, provenance, and two independent exact audits before any claim of
progress. Programmatic measures may reject obvious debris or prioritize work;
only blinded readers can establish readability. Failed runs stay in the
ledger, but immediately hand their residual frontier to a genuinely different
construction or repair operator.

The current benchmark remains the independently verified 38-letter sentence:

> *An aide rips nine memos; some men inspire Diana.*

The active target is an original, coherent, intact-prose palindrome longer than
that benchmark (with the paper's promotion floor at 100 letters), followed by
the blinded intact-prose versus shuffled-control study. No wrapper, catalogue
relexicalization, repeated unit, fragment, or filler output can satisfy it.

## Current frontier (2026-09-15)

- **Acceptance gate remains unchanged.** A result must be an original,
  exact letter-level palindrome, rendered as intact English prose, mechanically
  verified independently, and later supported by blinded human reading.  No
  automatic score or search statistic can promote an item.

- **Population repair over intact prose (2026-09-15).** A materially new
  construction tested a deterministic population of complete typed sentence
  pairs. Typed two-point constituent crossover and seam-biased terminal
  mutation preserved intact prose while optimizing mirrored character
  mismatch, rather than emitting one side from the other's reverse tape. It
  evaluated 5,760 genomes and produced no exact closure; the best rendered
  pair had 26 mismatched character pairs. The concrete successor is to retain
  this frontier while adding held-out lexical banks and a two-constituent seam
  repair, not to rerun the same population with a new seed or a larger beam.

- **Lexicalized TAG yield-equation search (2026-09-15).** A distinct
  construction changed the search state to paired elementary-tree and
  auxiliary-tree adjunction stacks. Recursive typed adjunction exposed
  terminals from opposite edges and enforced their character-yield equation
  before a pair could survive. The bounded run compared 21,158 structural
  derivation pairs (1,012,887 terminal comparisons), with zero exact joins;
  the held-out coordinated-predicate site repair also produced zero. It
  retained 40 base and 20 repair probes, rejecting 62 malformed derivations
  before rendering. The concrete successor is a new lexical realization at
  the recorded TAG adjunction frontier, not a larger bank, beam, or replay.

## Scalable-construction requirement (2026-09-15)

The active construction objective now has two explicit layers: (1) a
length-indexed exact core that can accept any requested letter count and
report unreachable lexical targets without fabricating text, and (2) a
boundary-crossing typed inventory that can make strict closures increasingly
readable.  `llm_palindrome/scalable.py` implements the memoized residual core;
`docs/SCALABLE-EXACT-LENGTH-CONSTRUCTOR-20260915.md` records its replay and
the explicit non-reader fallback.  Exact arbitrary-length output is a
construction capability, not evidence of English readability; the original
reader gate, provenance gate, and no-shortcut rules remain unchanged.

The arbitrary-length requirement is now part of the active goal, not a
replacement for the reader goal: for every requested `N`, the constructor must
either return an independently verified `N`-letter closure or a reproducible
strict-mode reachability report.  The explicit one-letter fallback is useful
only as a totality witness and is never eligible for promotion.  Reader-facing
work therefore measures progress on two separate axes—exact length coverage and
human-rated intact prose—and may claim success only when both gates are met.

- **Joint grammar/residual probe (2026-09-15).** A typed decoder selected POS
  roles while cancelling the opposite character residual, rather than
  generating prose first and filtering it afterward.  With Brown-derived word
  types (word types only, never intact corpus sentences), it found one 40-letter
  exact closure: `Some memo see some memos; some memo see some memos.`  The
  independent ASCII audit confirms exactness, while the shared admission gate
  rejects repeated content, repeated nontrivial units, and a proper
  self-palindromic span.  It is therefore not a candidate or reader material.
  The concrete successor is a memoized bilateral residual search with
  valency-compatible expansions and target length carried in state; arbitrary
  length remains separately covered by the total constructor and never by
  filler padding.

- **Content-disjoint repair (2026-09-15).** The same decoder was rerun with a
  live content-word ledger, so a closure could not reuse a noun or verb across
  either side.  It found no `>=39`-letter closure; its deepest frontier was the
  38-letter seed decomposition (`an aide rips nine memos` / `some men inspire
  diana`) plus a near-miss that left the residual `ight`.  This is a boundary
  result, not a reason to relax the gate: the next operator must vary typed
  lexical continuations at that residual while preserving disjoint content and
  independent exact-N auditing.

- **Typed semantic expansion family (2026-09-15).** Three independently
  authored event-clause families—basic transitive, adverbial, and adverbial
  plus object modifier—were searched with role selection and character
  cancellation in the same state.  All three reached only two matched outer
  characters and produced zero closures.  The failure is at the first outer
  boundary, not a readability measurement; the concrete successor is a
  pronoun-led family with terminal-`i` object alternatives.

- **Seed-boundary typed resegmentation (2026-09-15).** A fresh operator applied
  520 one-character mirrored insertions to the seed half, then independently
  resegmented both resulting 40-letter tapes into typed clause templates while
  excluding every seed content word.  It produced 2,080 typed-left failures,
  zero joint segmentations, and zero exact candidates.  The next operator is
  two-character insertion with variable-length clause templates and lexical
  valency carried in the memoized state; no seed wrapper or filler is allowed.

- **Two-character boundary insertion (2026-09-15).** The successor widened the
  same operator to all 13,520 two-character mirrored insertions (42-letter
  tapes) and seven variable-length typed clause templates.  It produced 94,640
  typed-left failures, zero joint segmentations, and zero exact candidates.
  This rules out the one-step insertion neighbourhood at the current lexical
  inventory; the next repair must change the lexical inventory and valency
  transitions, not merely increase insertion count.

- **Expanded bilateral typed inventory (2026-09-15).** Increasing the
  disjoint decoder to 3,000 Brown/word-frequency types and a 50,000-state
  budget per plan pair reached a 40-character residual frontier but still
  produced zero exact closures.  The strongest states are retained with their
  unresolved residuals (`ide` and `id`); their surfaces are not prose and were
  not promoted.  The next operator must make valency and discourse roles
  explicit while carrying those residuals, rather than simply widening the
  frequency pool again.

- **Global Brown POS-shape lattice (2026-09-15).** To reopen the grammar
  rather than add another narrow frame, the search mined 200 frequent Brown
  POS shapes of 8--15 words and kept syntax plus unique content live during
  center-out character matching.  Across 24 seeds and 700 parents it produced
  one 20-letter exact control and no closure at the `>=39` floor; no long
  surface was found or promoted.  The short control is independently exact but
  fails the length/short-word gates.  The next constructive change is to add
  explicit transitivity and discourse-event roles to this global lattice, not
  to widen the corpus pool blindly.

- **Typed lexical-graph residual search (2026-09-15).** The next operator
  carried number, tense, determiner, and transitivity features inside the
  character-cancellation state, with 12 complete clause shapes and 424,334
  independently generated left clauses. It made 5,092,008 residual attempts
  and found zero closures at the 39-letter floor. This is a concrete graph
  failure, not a readability result: the successor is to add attested
  subject--verb/object edges and inflectional variants to the same residual
  frontier, rather than widen isolated word lists.

- **Breadth-first typed semordnilap clauses (2026-09-15).** A separate
  residual-state search generated 151,484 independently authored clauses with
  explicit PERSON, THING, and transitive-verb roles, then parsed each reversed
  tape through four different clause templates. Semordnilap lexemes were
  allowed, but whole-word mirrors and seed reuse were rejected. It found zero
  closures at 39 letters; the durable successor is the attested lexical-graph
  operator, not another fixed template family.

- **Broad grammar residual probe (2026-09-15).** A compact 12-plan residual
  solver reopened clause order and reporting/discourse shapes with 350
  Brown-tagged lexical types per run (50,000 states per plan pair). It found
  zero exact closures at the 39-letter floor. The concrete successor is to
  keep lexical choices tied to attested valency edges while allowing
  inflectional and attachment states; simply adding more POS plans is now
  retired as a non-productive branch.

- **Brown-attested relation residual BFS (2026-09-15).** The search extracted
  3,533 compact SVO relations from 57,340 Brown sentences, expanded them to
  5,423 locally attested tense/number variants, and checked 6,061 left tapes
  against an independently indexed reversed residual. There were zero
  residual matches before any admission gate. The next constructive branch is
  two adjacent clauses with adjective/prepositional attachments, carrying
  dependency features through the residual state.

- **Two-clause Brown attachment residual BFS (2026-09-15).** The successor
  composed independently selected attested SVO relations into two clauses per
  side, adding bounded adjective and prepositional attachments. It indexed
  1,387,258 typed composite tapes and checked the same number of reversed
  residual states; all were misses, with zero exact closures or admitted
  candidates. The next branch is cross-clause dependency framing with relative
  clauses and controlled conjunctions, retaining the punctuation-boundary
  residual state.

- **Cross-clause dependency residual BFS (2026-09-15).** Relative-`who` and
  controlled `and/or/but` frames were composed from independently attested SVO
  relations, carrying relation identity, number, tense, dependency mode, and
  connector through the residual key. It indexed 20,448 frames and had 20,448
  residual misses, with zero pre-gate matches and zero exact closures. The
  next repair is agreement-aware shared-subject/object relative attachment.

- **Shared-coreference relative residual BFS (2026-09-15).** Agreement-aware
  relative frames shared either the subject or object across two independently
  attested relations, carrying co-reference mode, number, and tense as hard
  residual features. It produced 302 frames (298 indexed tapes), with all 302
  left states missing their reverse residual and zero exact closures. The next
  operator adds independently attested adjective/prepositional attachments
  inside the relative clause without relaxing agreement or tense.

- **Two-relative-chain residual BFS (2026-09-15).** The next dependency
  operator built two-edge chains from Brown-attested SVO relations, enforcing
  shared head-subject and relative-object agreement while omitting shared
  nouns from the surface. It produced 35 valid chains, all 35 reverse-residual
  misses, and zero exact closures. The next repair adds independently attested
  adjective/prepositional material on either relative edge while retaining both
  co-reference variables and hard agreement/tense constraints.

- **Two-relative-chain attachment residual BFS (2026-09-15).** Adjective–noun
  and preposition–noun attachments were added on either relative edge, with
  both co-reference variables, number, tense, relation identity, and
  attachment provenance retained in the residual state. It indexed 497
  frames and recorded 497 residual misses, zero matches, and zero exact
  closures. The next repair is one bounded lexical edit at a content boundary,
  not another grammar-only expansion.

- **Attached-chain bounded lexical repair (2026-09-15).** One tense-preserving
  verb or independently attested attachment lexeme was changed in each
  attached two-relative frame, producing 15,473 repair variants and 12,828
  indexed tapes. All 15,473 residual lookups missed; there were zero exact
  closures and no candidate promotion. The next repair jointly changes a
  subject/object pair only when the replacement relation is Brown-attested.

- **Attached-chain joint relation repair (2026-09-15).** A distinct repair
  replaced a complete SVO relation, requiring at least two changed
  subject/verb/object content slots and a Brown-attested replacement before
  residual matching. It generated 239,768 repaired states (497 tape keys), all
  residual misses, with zero exact closures and no promotion. The next search
  must change attachment topology or boundary segmentation rather than repeat
  relation repair.

- **Relative topology/boundary residual BFS (2026-09-15).** This branch
  explicitly excluded all 497 prior attachment tape keys and changed the
  construction topology: first-relative suffix PPs, second-relative prefix
  PPs, cross-boundary PPs, and attested determiner-present/absent variants.
  It explored 8,695 new frames, all residual misses, with zero reverse
  matches or exact closures. The next operator makes the PP noun co-referent
  with a chain variable and searches its boundary position jointly.

- **Coreferent variable-PP boundary residual BFS (2026-09-15).** This
  successor changed both PP length and placement: prep+noun,
  prep+determiner+noun, and prep+determiner+adjective+noun were inserted at
  every boundary, with the PP noun forced to co-refer to a chain variable.
  It explicitly excluded the 497 attachment keys and 8,695 topology keys,
  then searched 49,044 new frames (45,492 distinct tapes). Every residual
  lookup missed; there were zero reverse matches, exact closures, or admitted
  candidates. This closes the current Brown-attested attachment family; the
  next experiment must leave this family rather than replaying it with a
  larger pool or beam.

- **Independent semantic event-frame relexicalization (2026-09-15).** This
  genuinely new family authored temporal intransitive/copular event reports
  on the left and parsed their reversed tapes with independently authored
  event phrases on the right. It used no Brown corpus, SVO/relative relation,
  POS lattice, or semordnilap inventory, and fingerprinted 615,860 existing
  normalized strings before searching. Thirty event/template combinations
  were offered; all 30 failed at the reverse lexical prefix, with zero exact
  closures. The next step must change to a two-event discourse frame, not
  enlarge this finite inventory.

- **Fresh paired-clause ledger (2026-09-15).** A separate Luna authoring
  probe proposed four complete-looking clause pairs and fingerprinted 457
  repository tapes before auditing them. Two were exact but both failed the
  shared gate: one used reverse-word fragments and a hidden palindromic span,
  and the other did the same with an inflected fragment. No item was promoted
  or sent to readers; this is a provenance ledger, not readability evidence.

- **Character-ledger prompt bank (2026-09-15).** A distinct cross-product
  probe paired four fresh natural left clauses with four independently authored
  right-clause guesses while exposing the reversed tape constraint. It
  fingerprinted 616,270 repository tapes and retained all 16 proposals; none
  was exact or admitted. This is not a language-score rerun: its concrete
  successor is constrained right-side decoding that exposes only
  reverse-compatible prefixes.

- **Two-event discourse frame (2026-09-15).** The next construction changed
  the semantic state itself: eight fresh ordered frames paired an intransitive
  event with a result state, carrying strict event-before-result ordering into
  an independently lexicalized reverse discourse. It scanned 1,759 JSON
  artifacts and excluded 616,106 existing tape keys; all eight reverse parses
  failed, with zero exact closures. The recorded successor adds one
  independently lexicalized connective while preserving the temporal state;
  it must remain a new signature rather than an inventory/beam increase.

- **Connective-bearing event pair (2026-09-15).** This successor added an
  explicit independently lexicalized connective slot between the ordered
  event and result units. It used eight fresh frames, scanned 1,761 JSON
  artifacts, and excluded 616,313 existing tape keys; all eight reverse
  connective parses failed, with zero exact closures. The next operator is a
  constrained contrast/cause/consequence choice, still under the same strict
  temporal state and novelty audit.

- **Constrained reverse lexical decoding v2 (2026-09-15).** A separate route
  replaced guessed right clauses with memoized, frequency-ranked dictionary
  segmentations of each reversed tape, bounded to eight words and a 96-state
  frontier. Four fresh natural left clauses produced zero complete parses but
  four concrete partial probes (best prefixes: “so id ar”, “se”, empty, and
  “se ne”). The output contains exact audits and central-gate results for each
  probe; no item is reader-facing. The next method must add semantic clause
  structure over these prefixes rather than widen the lexical frontier.

- **Semantic connective-class event pair (2026-09-15).** This branch carried
  contrast, cause, or consequence as an explicit relation-class state through
  the ordered event/result residual, with an independently lexicalized right
  discourse. Six fresh class-conditioned frames were tested; all six reverse
  parses failed after excluding 616,608 repository tape keys. No exact closure
  or admitted candidate resulted. The next operator is a relation graph for
  cause/consequence direction and contrast polarity, not a larger inventory or
  replay of an earlier connective search.

- **Relation-graph event pair (2026-09-15).** This successor made the semantic
  edge explicit: directed `causes`/`leads_to` edges and symmetric opposed
  `contrasts_with` polarity were carried with strict event-before-result order
  through the reverse residual. Six fresh frames (two per edge class) were
  tested; all six reverse parses failed after excluding 616,820 existing tape
  keys. Every rejection includes the edge state, rendered probe, independent
  exact audit, central gate, and readability diagnostic. The next experiment
  must use a two-edge event-to-intermediate-to-result micrograph, not replay
  this single-edge state.

- **Two-edge event micrograph (2026-09-15).** The final rung of this semantic
  family inserted an intermediate state and carried two explicit edges across
  a three-rank temporal order. Four fresh micrographs were tested; all four
  reverse parses failed after excluding 616,955 existing tape keys. The run
  records rendered probes, topology/temporal residuals, exact audits, central
  admission, and readability diagnostics, then explicitly stops this
  event-family ladder. Any next construction must use a materially different
  semantic inventory.

- **Dialogue acknowledgment inventory (2026-09-15).** The next materially
  different family used hand-authored speech acts: 4,320 question/instruction
  left acts and 7,560 independently lexicalized answer/acknowledgment acts.
  All 4,320 residual lookups missed, with zero exact closures or admitted
  candidates. A repository fingerprint across 1,740 JSON files was recorded,
  and the retained frontier now includes rendered left-act probes, lengths,
  reverse targets, and independent exact audits. The next operator is
  elliptical answers and imperative acknowledgments; it is not a larger beam
  over this same act inventory.

- **Elliptical dialogue acknowledgment inventory (2026-09-15).** This
  successor changed the act grammar to paired `can we`/`what about` prompts
  and elliptical response or imperative-acknowledgment turns. It tested
  291,060 prompt turns against 1,806 response turns; every residual lookup
  missed, with zero exact closures or admitted candidates. The run preserved
  rendered residual probes and excluded the same repository palindrome
  fingerprint without enlarging the preceding inventory. Its next operator is
  a bounded shared discourse-topic slot, which must be a new signature.

- **Shared-topic elliptical dialogue pivot (2026-09-15).** The bounded
  successor paired prompt turns and elliptical/acknowledgment responses that
  shared one explicit topic variable, without enlarging the action or topic
  inventory.  It tested 5,130 prompt turns against 23,400 response turns;
  every residual lookup missed, with zero exact closures or admitted outputs.
  The artifact excludes its own output from the repository fingerprint and
  records 100 rendered probes with independent audits.  This closes the
  dialogue family; the next route must leave dialogue semantics entirely.

- **Multiword-unit transducer local repair (2026-09-15).** A new lexical-unit
  family composed independently authored idiomatic clause fragments and
  permitted one reversible affix or compound split/join while preserving
  syntax.  It evaluated 396 surfaces and 60 local repairs, with zero exact
  closures or mechanically admitted outputs; 40 rendered residual probes are
  retained.  This is a transformation-operator failure, not a reason to
  repeat the same inventory with a larger beam.

- **Bidirectional scene-slot graph (2026-09-15).** Six authored scene graphs
  linked `located_at`, `possesses`, and `describes` edges while an independent
  right-side slot graph consumed the reversed character residual.  All six
  reverse parses failed, with zero exact closures or admitted candidates.
  Each residual retains the semantic slot state, rendered probe, exact audit,
  and the one-terminal-replacement repair operator.

- **Morphological/derivational seam search (2026-09-15).** A separate
  hand-authored morphology grammar jointly selected inflectional and
  derivational forms and allowed reverse character seams to cross word and
  morpheme boundaries.  It explored 113,250 left surfaces, recorded 90,660
  reverse misses and 3,840 partial seam probes, and found zero exact closures
  or admitted outputs.  The deepest useful seam is `arena` → `an era`; the
  next repair changes one same-family inflectional/derivational form while
  preserving agreement and the full reverse reparse.

- **Constraint-programming semantic grammar (2026-09-15).** A finite-domain
  one-hot model selected a complete semantic rule, typed lexical slots, global
  character variables, and exact target length in one constraint system rather
  than decoding a reversed tape.  It explored 44,071 states and 212,947 slot
  branches across exact targets 40--80; 18 rendered partial probes reached
  their longest consistent prefixes, but there were zero exact closures or
  admitted outputs.  The next repair is one new lexeme at the first recorded
  character contradiction, not a larger solver budget or filler.

- **Independent clause-lattice joint DP (2026-09-15).** A separate
  length-indexed dynamic program paired 49 complete subject/tense/argument
  frames and solved left/right lexical choices jointly across word boundaries.
  It explored 805 states and retained 49 rendered residual probes; all pairs
  failed before exact closure, with zero admitted candidates.  The next
  operator replaces one lexicalization in the deepest same-frame slot while
  preserving the cross-boundary and no-repeat gates.

- **Character-clause finite-state transducer (2026-09-15).** Independent
  arithmetic/measurement clause banks were compiled into mirrored character
  tries whose product emits equal character pairs while retaining complete
  clause parses and inflection choices.  The bounded product visited four
  states and reached one complete parsed residual probe; it found zero exact
  closures or mechanically admitted outputs.  The next operator is a held-out
  comparative-quantity template with a frozen bank split, not an unbounded
  trie expansion.

- **Two-bank word-equation seam DP (2026-09-15).** A fresh pair of
  content-disjoint banks contained 10 complete, independently authored
  mini-clauses on each side.  A memoized prefix/suffix equation compared both
  clause tapes character-by-character, allowing the equality seam to cross
  ordinary word boundaries; 100 bank pairs (9 deliberately outer-compatible)
  yielded 100 rendered probes and zero exact closures or admitted outputs.  The
  concrete repair is a seam-preserving replacement of one complete lexical
  slot, not another residual decoder or a larger beam.

- **Seam-first complete-clause authoring (2026-09-15).** This run selected six
  outer-letter/terminal-width seam specifications before enumerating two
  independently authored complete SVO banks.  It enumerated 2,052 joint clause
  pairs, retained 13 boundary-crossing residual probes, rejected three pairs
  for repeated content units, and found zero exact closures or mechanically
  admitted outputs.  Its next repair preserves the seam specification while
  replacing one same-domain subject, verb, or object; it is a distinct
  authoring-order test, not a replay of the bank DP.

- **Semantic dependency outside-in solver (2026-09-15).** A fresh
  cause/preparation narrative used integer span boundaries, variable-length
  artifact noun phrases, dependency-state obligations, and outside-in arc
  consistency over paired character domains.  Targets 39--87 explored 12
  states and 195 paired span branches, leaving five partial probes and zero
  closures or admitted candidates.  The concrete repair adds one independently
  authored lexeme at the first contradicted mirrored span while preserving the
  dependency state; it does not reverse-segment a completed clause or widen a
  prior CSP/DP/FST.

- **Synchronous semantic-parse equations (2026-09-15).** Two independent
  semantic parses were expanded in lockstep against a shared character
  equation, with role and discourse state carried in the frontier rather than
  decoding a reversed tape.  The run evaluated 81 left parses × 81 right
  parses (6,561 lockstep states), retained 18 rendered probes and six
  parse-substitution repairs, and found zero exact closures or admitted
  outputs.  The next operator substitutes a complete parse at the first
  conflicting equation while preserving its semantic roles.

- **Internal lexical-center window repair (2026-09-15).** Sixteen complete
  authored clause pairs were each subjected to a bounded rewrite of one
  lexical center window while their exterior character equations remained
  fixed.  Fifty-six repairs produced 24 rendered probes, zero exact closures,
  and zero mechanically admitted outputs.  The concrete successor expands the
  center to a two-word constituent with held-out role agreement; it does not
  wrap or repeat an existing palindrome.

- **Discourse-plan coupled expansion (2026-09-15).** A typed narrative-plan
  graph selected two independently authored two-sentence narratives, enforcing
  semantic role agreement before online mirrored-character emission.  All 12
  plan pairs were rejected at an early character mismatch, with zero exact
  closures or admitted candidates.  The next repair is role-preserving branch
  substitution at the first mismatch, not post-hoc sentence reordering or a
  larger lexical beam.

- **Collocation-synchronous grammar (2026-09-15).** Five independently
  authored role-typed collocation frames on each side were expanded in
  lockstep while preserving complete-clause adjacency.  The 25 pair product
  produced zero exact closures or admitted outputs; every pair has a rendered
  residual probe and an independent two-pointer recheck.  The concrete repair
  replaces one collocation frame at its first residual offset and holds out
  the replacement frames for any later reader screen.

- **Human-compositional center-window search (2026-09-15).** Four intact
  two-sentence mini-scenes were authored first, then 24 content-word
  substitutions were tried from a finite natural-continuation table.  The
  deterministic character-window audits found zero exact closures or admitted
  outputs.  Its next operator expands the semantic window while preserving the
  sentence frames; no output is promoted from a proxy score.

- **Global typed semantic paraphrase rewrite (2026-09-15).** A whole-structure
  rewrite jointly varied agent, verb, and object slots in two complete clauses
  rather than mutating one seed position.  It evaluated 1,296 clause pairs and
  46,224 character-propagation states, with zero exact closures or admitted
  outputs.  The recorded repair substitutes all typed slots again at the first
  mirrored contradiction, preserving the no-catalogue and no-repeat gates.

- **Connected collocation-graph paths (2026-09-15).** A role-typed graph of
  16 hand-authored collocation edges was walked through 36 connected paths,
  with five same-role lexical repairs measured inside the graph.  Every path
  has a rendered residual and independent two-pointer audit; zero exact
  closures or admitted outputs were found.  The next operator is a same-role
  node substitution at the first failed edge, not a new post-hoc reorder.

- **Semantic sentence-pair alignment (2026-09-15).** Independently authored
  complementary complete sentences were paired across semantic paraphrase
  alternatives and aligned on the full normalized tape, allowing the mirror
  to cross both word and sentence boundaries.  The bounded 20-combination run
  found zero exact closures or admitted candidates and retained every rendered
  probe.  Its concrete repair adds tense/aspect and connective alternatives
  while preserving the complementary meaning frame.

- **Template-analogy semantic lexicalization (2026-09-15).** Four abstract
  palindrome role shapes were relexicalized independently with fresh semantic
  words and paired seam-width constraints.  The joint enumeration covered 499
  clause pairs and 30 rendered probes, with zero exact closures or admitted
  outputs.  The next repair changes one semantic role lexicalization at the
  recorded seam, while keeping the abstract shape and catalogue exclusion.

- **Neural dual-prefix beam, repaired (2026-09-15).** The first neural pass
  was invalidated because it reversed an unfinished right-reading prefix and
  rejected every state before scoring.  The repaired run emits the right edge
  as reversed characters and compares true shared-tape prefixes; its bounded
  typed beam made 25 expansions, all 25 failed at the first seam, and zero
  states reached GPT-2 scoring, exact closure, or admission.  The concrete
  next operator is a held-out same-role seam replacement; the invalid v1 is
  preserved only as implementation evidence and is not registered.

- **Seed-preserving mutation zero (2026-09-15).** A local constructive probe
  exhaustively changed one and two mirrored character pairs and inserted one
  mirrored pair into the 38-letter seed, then independently resegmented every
  lexical tape and applied the Brown sentence-shape gate.  It covered 475
  one-pair substitutions, 520 insertions, and 106,875 distinct two-pair tapes;
  2,169 had at least one lexical segmentation, but none had a complete
  sentence-shaped reparse or an admitted output.  This is a narrow failure of
  seed-local edits, not a readability result.  Its concrete successor is a
  typed word-boundary shift plus lexical replacement, jointly searched from
  the highest reverse-prefix frontiers rather than preserving the seed as an
  interior island.
- **Newly retired diagnostic frames.** The independently reparsed device
  causal control (109 letters) exhausted 74 states, failing at
  `devices:c` / `live:l`; the gardeners subject-action control (100 letters)
  exhausted 2,176 states, failing at `gardeners:a` / `the:e`; the clinical
  tinnitus control (115 letters) reached 14 character cancellations before
  `chronic:c` / `u`.  None is a candidate or reader-study material.
- **Provenance hardening.** Central admission now rejects the copied outer
  scaffolds `No it is ... position`, `See ... bees`, and `Go ... dog`, in
  addition to the existing catalogue-family, repeated-content, word-mirror,
  and proper multiword-palindrome exclusions.  Earlier controls using those
  scaffolds are retained only as rejected evidence.
- **Local-minimum reset.** These failures share the same defect: they author a
  conventional sentence from the left and defer incompatible character debt to
  a distant suffix.  Active experiments instead select left and right lexical
  constituents jointly: (1) a residual-driven bidirectional zipper grammar,
  and (2) a lexical-bridge search that permits character reversal to cross
  ordinary word boundaries before an independent semantic reparse.  Both begin
  with short complete sentences and may lengthen only after finding an
  admissible exact survivor.
- **Audited successor evidence.** A productive action/result grammar (224
  derivations) exhausted its declared finite inventory without an exact
  closure, reaching only `an` before incompatible next letters.  A 21-item
  question/reply product reached `arep`, but an independent audit found that
  its ``Yes'' witness tested plan membership rather than entailment; it is
  excluded from the semantic-construction frontier.  These are diagnostics,
  never prose outputs or a claim about the broader search space.  The active
  successors are now whole-sentence generators with all lexical and semantic
  roles represented in a single center-free character product.
- **Architecture correction.** Astra found that the former paired-clause
  tries force the palindrome center to the clause boundary and thereby miss
  every path whose midpoint lies inside a word or constituent.  The active
  kernel instead represents one complete sentence as a character DAG, matches
  its forward and reverse edges, and accepts both even and odd center meets.
  It is independently checked against exhaustive toy grammars and lexical-path
  replay.  It makes no semantic or readability claim on its own; active
  generators compile separately typed full-sentence plans into that kernel.
- **First valid whole-text zero.** A dependent-slot compiler produced 720
  euphony- and valency-valid SVO/imperative derivations at the 30--160 letter
  discovery band.  Every finite product exhausted with no closure; the
  deepest real path is one matched character.  This is a valid, narrow zero,
  not an output-quality finding.  Its successor uses a semantic endpoint
  algebra that joins typed initial and final phrase realizations before
  full-text construction, rather than hoping independent clause inventories
  share an outer character stream.
- **Endpoint-algebra zero and its repair.** The first endpoint-algebra
  product enumerated 1,320 typed intact-clause derivations, selected 120
  with at least two matched endpoint characters, and exhausted every selected
  product without a closure.  It is a finite grammar diagnostic, not a result
  about English or readability.  Its successor must carry endpoint residuals
  progressively and branch on all legal lexical realizations, rather than
  inspecting only a preferred surface realization.
- **Anti-shortcut topology guard.** When matched outer characters leave a
  proper interior span whose two ends are word boundaries, any successful
  continuation would make that interior a preassembled palindrome.  The
  active search therefore rejects a two-or-more-word interior at that state;
  a one-word center is allowed only for an ordinary repeatable function word.
  This is a construction prune, not a readability test, and will be exercised
  by path-local tests before it is used to describe a search result.
- **Excluded reverse-shell attempt.** A 16-clause reverse-boundary script
  constructed a source clause followed by a segmentation of its complete
  reversed tape.  It also hard-coded `today` and supplied no independent
  target grammar.  That is a reflected whole-clause shell, so its zero is
  excluded rather than counted as a failed generator.  The replacement is a
  boundary-disjoint character product in which both typed sides are selected
  jointly and neither copies the other side's whole tape.
- **Boundary-disjoint chart zero.** A new seven-domain, 11-layout character
  chart represents 325,320,804 finite lexical realizations without copying a
  complete source tape.  All 77 compiled grammars exhausted with no closure;
  the production inventory reached no more than four matched outer characters
  and never reached an island-prune state.  Its shifted-boundary and
  forbidden-island tests establish the construction guard, not improved
  language quality.  The immediate repair is an edge-compatible typed
  inventory: discover compatible subject/predicate/final-constituent channels
  from a broad fixed lexicon, require at least three real outer matches, then
  run the boundary-disjoint interior product and an independently declared
  reparse.
- **Broad semantic-edge zero.** The edge-compatible successor independently
  declared 112 plural human subjects, 812 typed noun forms, and 36 valency
  constrained predicates.  It discovered 1,171 channels and exhausted 18,880
  whole-surface products (1.82 billion lexical realizations represented by
  their character products).  Every selected path replayed at least three
  actual outer pairs; 345 reached five, but none reached a left word boundary
  or an exact closure.  Its separate validator rejects generator-only words.
  This is a deeper endpoint diagnostic, not an output or readability result.
  The immediate successor is a free-midpoint action--reason discourse grammar
  whose semantic reparser verifies an actual causal relation, rather than the
  previously excluded question/reply membership shortcut.
- **Causal-channel zero.** The first action--reason implementation gives the
  independent parser a qualitative state-transition test: a cooling action on
  a vessel corrects the hot state of its bound contents, while counterfactual
  heating-hot and broken-reference examples fail.  Its four discovered
  `doctors`/`hot cod` channels cover 28 products (1,166,116 lexical
  realizations represented), all of which exhaust after five matched letters.
  This validates causal checking without confusing it for a prose result.
  The successor changes causal order and coreference realization (condition
  first as well as reason last) before another endpoint search.
- **Cleaning-domain causal zero.** The active/passive, reason-last/condition-
  first cleaning grammar independently resolves `it`/`they`/`them` by type,
  number, syntactic role, and a unique antecedent, then checks that washing,
  cleaning, scrubbing, or wiping reduces the stated contamination of that
  artifact.  It exhausted 4,928 five-pair products after crossing a left word
  boundary; condition-first branches are explicitly ineligible at their
  outermost letters.  No closure was found.  The next construction therefore
  changes the representation itself: a variable-token character lattice
  carries unfinished lexical prefixes and one whole-sentence semantic parser,
  so final word boundaries may cross provisional constituent boundaries.
- **Variable-token lattice zero.** The art-conservation lattice carries both
  a semantic parser state and an unfinished lexical prefix on every character
  node, representing 250,033,280 finite lexical paths (18--124 letters) in
  2,592 character nodes.  Its independent art-conservation parser accepts
  only coherent repair statements and rejects counterfactual role/effect
  assignments.  The finite product exhausted 320 states: 13 reached five
  real outer pairs and eight reached six, with no exact closure.  A toy
  lattice verifies that its representation can move a final word boundary,
  but no production path exercised that behavior; this is therefore a narrow
  construction zero, not evidence that dynamic resegmentation improved the
  search.  The successor requires a production resegmentation witness at
  five or more matched pairs while intersecting independently compiled
  semantic grammars, followed by a separate whole-sentence reparse.
- **Boundary-sensitive semantic-intersection zero.** The successor keeps an
  acoustic measurement statement only when independent source and rendered
  target analyses agree on its semantic signature and disagree about an
  early word boundary.  Its finite target projection has 4,408,992 distinct
  rendered surfaces after the shift filter; derivation-pair and rendered-
  surface counts are recorded separately to prevent alternative
  segmentations from inflating coverage.  Eighteen palindrome-product states
  reached five true outer pairs.  In each audited production witness,
  source `time | keepers` crosses to rendered `timekeepers`, while the full
  111--116-letter statement passes a third, independent semantic parse.
  All such paths fail at pair six (`e` versus `l`), and the finite product has
  zero exact closures.  This is a narrow endpoint result, not an output or a
  readability finding.  The immediate repair requires coaccessible
  resegmentation at both edges of an argument-attachment grammar, with joint
  endpoint alternatives before another palindrome product.
- **Two-sided attachment screens are unqualified.** A credential/attendant
  attachment inventory records 1,440 target surfaces whose source and target
  analyses each have a real boundary disagreement, but none supplies the
  required six-pair, six-continuation production frontier; its palindrome
  product is never treated as a result.  A second finite screen independently
  parses 960 relative-clause/coordination attachment pairs with different
  attachment readings.  It finds 640 zero-pair, 288 one-pair, and 32 two-pair
  endpoints, so no surface qualifies for a full palindrome product.  These
  are failed qualification screens, not examples, candidate runs, or evidence
  about English.  The next repair is an audited local lexical-infill operator:
  it may propose only a missing attachment within a live finite character
  residual, must validate every proposal independently, and may not query a
  model until its mechanics and controls pass.
- **Residual-infill preflight zero.** The audited local-infill interface fixes
  a seven-pair art-repair residual and exhausts 192 independently parsed
  1--3-token possessor attachments before any authoring call is permitted.
  All 192 fail at pair eight (`w` against `n`, `r`, or `t`), with no exact
  closure; its two-query budget consequently remains at zero reserved and
  zero executed.  It is not a model result, a candidate, or a readability
  measurement.  The immediate repair changes the event's argument order so
  the possessive NP no longer occupies that fixed mismatch frontier, and it
  must demonstrate an eight-pair, multi-continuation residual before a new
  bounded infill interface exists.
- **Argument-order residual zero.** Moving the owner into the first object of
  a coordinated repair event produces 432 whole rendered statements for which
  an independent parser binds the later demonstrative to that same owned
  artwork.  Every statement passes the non-exact central gates and reaches
  nine outer pairs, but all fail at pair ten (`r`/`l`).  The eight-pair probe
  has 432 one-pair-compatible surfaces but only one next-letter class, and
  the finite grammar has no exact completion; no model transport is present
  and no query is made.  The next repair must jointly vary the clause onset
  and final material-property realization, require ten real pairs and six
  paired next-letter classes, and preserve independent ownership/event
  binding.
- **Joint onset/material screen is unqualified.** A typed repair grammar
  jointly varies onset event, substrate, finish, owner, and repair event across
  252 source--target pairs, retaining every rendered surface and two separate
  parses.  Three paths reach ten pairs and then fail `d`/`g` at pair eleven.
  Although 18 pairs have raw two-sided compound-boundary geometry, none has
  both boundary changes after ten real pairs; there are zero qualified
  channels, closures, full products, or model calls.  The next construction
  fronts a typed patient/locative repair clause and jointly generates its
  initial material noun phrase and final agent/tool attachment, with an
  eleven-pair live-boundary screen before expansion.
- **Patient-fronted screen is unqualified.** The passive and locative
  patient-first grammar preserves 210 rendered source/target surfaces and two
  independent parses, but reaches no more than four real pairs (deepest
  `c`/`e` at pair five).  It has no eleven-pair state, live two-sided boundary
  witness, qualified channel, closure, product, or model call.  Ninety
  attempted adhesive surfaces also fail the central lexicon on `epoxy`; that
  rejection is retained rather than relaxed.  The next construction is an
  agentless result-state/participial grammar that jointly selects a
  material-property onset and final measured-state complement, removing the
  tool-suffix bottleneck while retaining patient/event binding.
- **Excluded implementation records.** The fixed-`today` bridge never entered
  a target lexical state; the proposed online/FSM successors falsely counted
  word completions or node movement; the modal causal frames did not establish
  causal edges; and the endpoint confirmation used table membership rather
  than entailment.  Each has a run-local exclusion marker and no candidate.

## Current state (2026-09-12)

- The reflected-catalogue constructor is rejected as a shortcut. It is retained
  only as a failed-route artifact: it reaches length by repeating borrowed,
  individually palindromic utterances, not by generating a single readable,
  original whole. It is not served by the API, shown as product output, or used
  as a paper result.
- The existing v3 hierarchical composer produces exact, provenance-traceable
  output but its current 58--464-letter systems have weak local word order.
  A calibrated Brown-bigram audit gives order gains of -0.08/-0.03/+0.12 for
  short/medium/long systems versus +1.17/+1.08/+1.28 for matched prose.
- The first frozen human packet is not usable: its intended real-prose controls
  strip punctuation and often stop mid-sentence. Rater recruitment is paused.
- The initial study also confounds requested length with construction depth
  (approximately 1/7/15 mirror pairs), has only three system texts per band,
  gives all raters one fixed order, and lacks per-item completion and
  item-aware inference.
- Search/POS results are retained only as construction diagnostics. They do not
  meet the reader-quality acceptance gate.
- Three independent whole-sentence construction families have now been
  exhausted without a readable survivor: typed semantic SVO templates
  (`runs/semantic-slot-solver-2026-09-12/expanded-results.json`), lexicalized
  control/infinitive dependency trees
  (`runs/lexicalized-dependency-tree-2026-09-12/results-02.json`), and
  matrix-plus-relative-clause trees
  (`runs/relative-clause-sol-2026-09-12/result.json`).  They produced no
  candidate, not a negative-result paper claim.
- The possessive-name character-crossing relexicalizer is a rejected
  catalogue-family ablation. Although it produced two distinct 30-letter
  surfaces, both retain the famous ``Marge lets ... see ... telegram`` frame.
  Name substitution is not independent generation under the no-shortcut rule,
  so neither is a lead, a paper result, or a reader-study item.

## Evidence and artifacts

- `data/readable_palindrome_centres.json`, `llm_palindrome/reader_first.py`,
  and `runs/reader-first-showcase-2026-09-12/`: preserved rejected-route
  evidence. They must not be promoted as a baseline, a generated example, or
  a reader study item.
- `runs/readability-length-study-2026-09-12/`: superseded pilot packet. Do not
  collect ratings from it.
- `runs/readability-length-study-2026-09-12/programmatic-readability.json`:
  local-order, lexical, repetition, and segmentation diagnostics; not a human
  readability result.
- `runs/first-composition-rescue-2026-09-12/`: a frozen six-block
  counter-order screen.  It is not a reader study and must not be distributed
  as one.  The twelve source pairs have mean Brown local-order gain $-0.113$;
  the two orders are $-0.097$ and $-0.021$.  Under the preserved
  `gpt-oss:20b` development prompt, all six blocks were parsed as `neither`.
  These diagnostics do not establish human unreadability, but they reject
  reordering the current sentence-shaped bank as a promising rescue.
- `runs/cross-boundary-material-probe-2026-09-12/`: a second frozen
  development screen that removes the independent-sentence gate.  It samples
  six POS-shaped and six boundary-crossing-only blocks from the same 498-pair
  generated inventory, preserving both exact orders of each block.  The
  recorded `gpt-oss:20b` screen passed all six prose/shuffle controls and chose
  `neither` for every candidate block in both arms.  It is not human evidence,
  but it rejects the pair-boundary assumption as the main explanation for this
  bank's failure.  Its Brown diagnostics are descriptive only: crossing-only
  variants have positive local order gain (+0.14 to +0.21) yet no recoverable
  global reading under the calibrated development screen.
- `paper/POLICY-ROBUSTNESS-2026-09-12.md`: bounded traversal behavior, not an
  efficiency or language-quality result.

## Hypothesis frontier

1. **Material/constraint failure (active).** Both frozen screens reject the
   current 498-pair generated inventory before paragraph ordering matters.
   Relaxing POS-shaped pair boundaries can improve a local-order diagnostic,
   but it does not recover a subject, intent, or interpretable thought.  The
   next generator must create new bidirectionally meaningful material rather
   than select or reorder this bank.
2. **Character-crossing lexical repair (active).** The zero-yield dependency
   families fail predominantly at outer lexical boundaries.  The next route
   starts from a complete, independently authored semantic dependency tree,
   then replaces a mirrored *character span* with a jointly lexicalized
   modifier/relative-clause attachment.  It must preserve all unaffected
   letters and attach the repair as one grammatical sentence; it is not a
   reflected phrase catalogue or a word-order construction.
3. **Composition failure.** If a future pair inventory supplies interpretable
   material, whole-output planning is still needed to preserve a subject and
   discourse at more than one pair.
4. **Evaluation failure.** A revised
   study is still needed to measure any later improvement and to avoid selecting
   on an automatic proxy.

## Active decision

The bounded first-composition rescue is complete. Astra identified the local
minimum as treating POS-shaped halves as semantic units and then relying on
nesting to make prose; Sol confirmed that the appropriate first artifact was a
standalone experiment, not a v3 endpoint change.  The frozen material and the
recorded `neither` outcome rule out that specific reordering rescue for the
current sentence-shaped inventory.

The boundary probe is complete: the six POS-shaped and six boundary-crossing
blocks both received a calibrated `neither` decision, with exact word- and
letter-preservation enforced on every non-neither display. The v3 API is
retired. Do not run further selectors over this inventory or recruit a
reader study from it.

The next bounded artifact must test a genuinely new *bidirectional material
generator*: it must propose both lexicalizations together, verify their exact
letter reversal mechanically, freeze every proposal and rejection, and
preserve a distinct novelty check.  A local model may help propose or segment
the fixed character stream, but cannot act as the endpoint judge.  Only a
candidate that survives a separate, human-ready screening design can return
the project to a replacement reader study.

The attested-span intersection and constrained word-map authoring routes have
completed without reader-worthy material.  Do not widen either merely by
adding samples or a language score.  The active repair route is a new
character-crossing lexicalized attachment; its first run must record the
source dependency witness, both repaired contexts, the exact span equality,
and every rendered rejection.

### Active run

- **Next construction frontier:** the TAG and reverse-complement Euler runs
  are complete and their zeros are preserved as evidence, not as the thesis.
  The next run must change its construction dimension, preflight itself against
  the 53-entry novelty registry, and retain complete rendered probes with
  independent exact audits. It must not replay a prior bank, beam, chart, BPE,
  ILP, evolutionary, TAG, or Euler configuration. Any exact survivor remains
  gated on intact-prose versus shuffled-control blinded reading.

### Rejected catalogue-family ablation

The generator enumerated 100 typed derivations and found four exact closures.
An independent verifier now recomputes every hard gate without trusting the
generator's check dictionary. All four closures are exact, but all four are
promotion-ineligible because they instantiate the borrowed
`Marge lets ... see ... telegram` family. The existing 24-form package and its
programmatic audit are superseded development artifacts and must not be
distributed or cited as a reader study.

- Rejection run: `runs/possessive-name-relexicalizer-2026-09-12/rejected-catalogue-family-audit-04.json`
- Independent hard-gate audit:
  `runs/possessive-name-relexicalizer-2026-09-12/rejected-catalogue-family-independent-audit-04.json`

The background local-model centre-out attempt produced no result artifact: its
recorded process terminated before emitting a proposal. Preserve its empty
launcher record as infrastructure evidence only; do not treat it as a
generation result. No relexicalizer currently supplies admissible material.
The active route is an independently authored character-crossing construction
whose frame is checked against catalogue-family provenance before any reader
package is built.

In parallel, design but do not recruit a replacement reader study until it has
complete naturally punctuated prose controls, an externally separated key,
rater-specific randomized orders, per-item completion checks, and an
item-aware analysis plan.

## Model escalation record

- **Astra, 2026-09-12:** diagnosed material quality before paragraph ordering;
  proposed the controlled two-pair rescue as the smallest discriminating test.
- **Sol, 2026-09-12:** directed a pure standalone experiment, whole-text
  rendering, invariant tests, and a complementary-order baseline; rejected a
  speculative v3 API mode.
- **Astra, 2026-09-12 (second consultation):** diagnosed fixed SVO frames as
  an endpoint-compatibility local minimum and directed lexicalized dependency
  construction before any further beam expansion.
- **Sol, 2026-09-12 (second consultation):** implemented a character-crossing
  relative-clause test.  Its zero closure result rules out that frozen family,
  not dependency repair in general.
- **Evolutionary complete-prose genomes, 2026-09-15:** evolved pairs of
  independently rendered SVO-plus-PP sentences with typed constituent
  crossover, seam-biased terminal mutation, and a character-mismatch fitness.
  The 5,760-genome population produced a readable 73-letter control
  (`A poet keeps the red lantern near the river. A farmer keeps a winter
  garden under one cedar.`) but no exact closure; its best mismatch was 26
  character pairs.  The concrete repair is a held-out lexical-bank expansion
  that preserves population state and typed crossover, not another beam or
  pool increase.

- **Lexicalized dependency attribute chart, 2026-09-15:** built independent
  recursive head-dependent forests with number/tense/valency unification and
  joined 5,184 complete prose pairs from opposite character edges.  It found
  zero exact closures; representative rendered prose was
  `The makers repair the radios near a harbor; the guides sort the charts
  beside a station.`  The next operator is dependency-preserving rotation of
  one feature-compatible modifier or PP dependent, with a fresh lexical bank.

- **Recursive CFG chart intersection, 2026-09-15:** enumerated independently
  derived complete sentences from a recursive relative-clause grammar and
  intersected their normalized character language with its reversal.  The
  depth-stratified chart covered 644 unique surfaces (up to 53 letters), with
  414,736 bounded near-miss comparisons but no exact intersection.  This is a
  distinct construction route, not a larger beam over an earlier inventory.
  The concrete next repair is a held-out typed relative-clause frontier at the
  deepest surviving chart items; the zero is not treated as an impossibility
  result or reader evidence.

- **GPT-2 BPE dual continuation, 2026-09-15:** changed the search alphabet to
  actual local-cache byte-pair pieces.  Nine-slot and six-slot complete SVO
  clauses were sampled independently from disjoint agreement-filtered role
  banks; each right-side BPE piece was emitted in ordinary reading order and
  consumed the next character of the left clause's reversed residual.  The
  bounded lattice covered 900 left clauses and 1,800 base/held-out-repair
  targets, retaining 40 rendered near-miss probes and zero exact closures.
  The concrete next operator is seam-triggered held-out expansion of the
  right object inventory, followed by a fresh BPE-token residual run.  The
  probes are intact prose controls only; no programmatic readability claim is
  made.

- **Mined phrase-chunk clause composition, 2026-09-15:** mined bounded 2--4
  word phrase components from the count and WikiText snapshots, classified
  them as headed NPs and finite two-token VPs, and independently composed
  541,598 complete determiner-led NP--VP--NP clauses.  A length-stratified
  chart of 1,826 clauses produced no exact reverse-index join; 40 probes were
  rendered and independently audited.  The concrete next operator is
  held-out frequency-stratum phrase expansion at a surviving seam, preserving
  the strict transitive grammar and disjoint-content-word gate.

- **Variable-boundary character-tape ILP, 2026-09-15:** formulated one
  lexicalized feature-grammar path as a binary flow over word-start arcs.
  Each arc chooses both a word and its character offset, so token boundaries
  are variables rather than fixed slots; subject-number agreement and
  verb-sense/object compatibility are carried in the automaton, while
  mirrored character equalities are hard MILP constraints during assignment.
  The bounded 39-letter HiGHS run built 57,166 arcs and produced no exact
  closure within its time limit.  Its intact grammar probe was `a new child
  carries a new boat in a new garden now` (39 letters; independently audited
  with 18 mismatched pairs).  The concrete repair is a held-out
  seam-support lexical expansion of the same variable-boundary flow, not a
  slot cross-product or relaxed exactness test.

- **Lexicalized TAG yield equation, 2026-09-15:** represented complete prose
  as recursive elementary/auxiliary-tree derivations rather than fixed clause
  slots.  Typed adjunction stacks at NP/VP/S sites produced 291 independent
  base derivations and 21,158 paired structural states; the outside-in
  terminal-yield equation checked 1,012,887 character pairs and found zero
  exact joins.  A held-out coordinated-predicate auxiliary was then adjoined
  at the object/S boundary as a structural repair, covering 20,914 additional
  states with zero exact joins.  Forty base probes and twenty repair probes
  were rendered as intact prose; malformed article/agreement derivations were
  rejected and counted before the join.  The next repair is deeper typed
  adjunction at the best surviving tree site, not another lexical beam.

- **Reverse-complement Eulerian overlap graph, 2026-09-15:** mined 3-letter
  character-context transitions from the local prose snapshots, retained only
  transitions with an observed reverse-complement edge, and explored
  edge-disjoint Euler trails before attempting English segmentation.  The base
  graph had 4,452 eligible edges and produced 40 exact rendered probes, but no
  balanced trail survived the independent content-word and dependency proxy
  gates; a held-out singleton-context degree repair produced zero balanced
  trails.  The probes remain diagnostic and require human reading evidence.
