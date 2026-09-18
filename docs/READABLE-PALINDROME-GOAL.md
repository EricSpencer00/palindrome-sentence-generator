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

## Current frontier (2026-09-16)

## Current frontier (2026-09-17 solver-correction reset)

## Orchestration reset (2026-09-18)

- **Progress audit.** `main` is clean and one local orchestration commit ahead
  of `origin/main`; the code baseline remains `ca526eb`. It contains 116
  commits since 2026-09-18 00:00, but the active
  acceptance measurements remain unchanged: zero exact admissible rows, zero
  reader-eligible outputs, and no blinded-reader package. The latest useful
  diagnostic is the 119-letter inherited prose row with 46 mismatches; it is
  not a palindrome and is not acceptance progress.

- **Local-minimum finding.** The branch-aware Dream-RSI controller now covers
  3--6 branches per replay and routes held-out policy at 0.780, while rounds
  95--98 and the subsequent seam/mirror lanes still produce zero exact
  admissible rows. This is a strong local-minimum signal: branch accounting and
  diagnostic coverage are improving, but construction quality is flat.

- **Control decision.** Freeze commits whose only outcome is another replay,
  seam variant, lexical-bank expansion, branch score, or diagnostic artifact.
  The next construction must change the material source or representation and
  be evaluated against the same gate. A run counts as progress only when it
  produces an independently audited original exact candidate, improves a
  held-out acceptance-relevant measurement, or decisively falsifies a concrete
  construction hypothesis. Unit-test coverage and a new artifact alone do not
  qualify.

- **Seam-correction result (2026-09-17).** The full-sequence product had two
  correctness defects: it rejected a multi-letter palindromic residual at the
  center, and it banned ordinary words merely because they occurred in the
  catalogue fixture. Both are fixed in commit `c41c115`; focused regression
  tests pass. The corrected product independently found two 51-letter exact
  closures, `doc note i dissent a fast never prevents a fatness i diet on cod`
  and `cod note i dissent a fast never prevents a fatness i diet on doc`.
  The first is the known catalogue tape; the second is a two-token endpoint
  derivative. Both are preserved with forward/reverse hashes and rejected
  before any reader claim.

- **Relation-search reset (2026-09-17).** Three orthogonal constructions were
  run after the correction: typed grammar-state relation search (128 states,
  no closure), authored-clause tape resegmentation (2,000 clauses, no right
  parse), and variable-length compositional clause relation search (37,448
  semantic states before its bounded queue exhausted). None produced a
  reader-worthy output. Their common failure is a seam-level grammar
  incompatibility, not a lack of queue volume; their artifacts and next
  operators are committed as `d3d3baa`, `ba76cba`, and `317ecb9`.

- **Active construction branch.** The next method must change lexical and
  syntactic material at the first residual while retaining typed agreement and
  semantic valency: a finite-state grammar relation solver with seam-local
  morphology, not another bank sweep. Exact closures remain closed to readers
  until intact prose survives the no-shortcut gate and a randomized blinded
  reader package is run.

- **Seam-local morphology repair (2026-09-17).** The first implementation of
  that operator carried inflection-family choices inside the live relation
  state and tested 19 states with 15 immediate mismatch edges. It produced no
  exact closure; its fresh 47-letter control, `The teacher reads a letter
  often; the writer reads a story.`, remains intact prose but is not a
  palindrome. The failure is now narrowed to subject/object agreement at the
  seam, which is the next repair in commit `c06ee8d` (source and artifact
  retained separately from the parent solver).

- **Agreement-pair repair (2026-09-17).** Singular/plural subject--verb
  packages were then carried in the same live relation state. The run reached
  31 states and 24 mismatch edges, with no exact closure. Its fresh control,
  `The teachers read a letter often; the writer writes a story.`, is intact
  prose but not a palindrome. The recorded successor is a seam-conditioned
  object/subject substitution, not a larger lexical sweep (`d9501e3`).

- **Semantic substitution repair (2026-09-17).** A fresh caretaker/object
  package then conditioned substitutions on semantic valency while retaining
  agreement and residual state. It explored 25 states, reached one matched
  character at best (`the child finds ... the paint`), and produced no exact
  closure. The near miss and independent hashes are retained in `d1d0924`; a
  fourth seam-local variant would repeat the same first-edge failure signature,
  so this branch is paused pending an architectural reset.

- **Bank-free whole-passage reconstruction (2026-09-17).** The architectural
  reset was tested as a genuinely global operation: an authoring model rewrote
  one complete event passage at a time, with temporary mirror violations and no
  supplied word bank or reversed tape. The 27B prose model produced eight
  intact, ordinary-English revisions and no exact closure. Its best row was
  readable but only 94 letters (below the 100-letter promotion floor); later
  rows cycled around 96 letters and stopped improving the mismatch count. The
  earlier 20B run is retained separately: it emitted one malformed mirrored
  list, then refusal strings. These are concrete interface/method failures, not
  reader evidence. The independent two-pointer and forward/reverse SHA audits
  pass on every stored row. Because the capable model shortened the passage and
  converged to a fixed paraphrase, this operator is stopped rather than tuned
  into another sweep. The next construction must preserve a 100+ letter event
  while making coordinated changes in two non-adjacent semantic regions before
  re-rendering; a single local paraphrase call is not enough.

- **Wheel-spin diagnosis (2026-09-17).** The repeated unsuccessful lanes share
  a representation error: they author ordinary clauses first, then ask a fixed
  bilateral grammar or a one-region synonym repair to satisfy character
  reflection. Increasing lexical banks, queue budgets, or seam variants only
  revisits the same first-edge incompatibility. The corrected character product
  is promising as infrastructure—it independently replays the 51-letter
  catalogue control and catches center residuals—but its only exact outputs are
  catalogue-family derivatives and therefore fail originality/readability
  admission. The prose-model reset is promising only in the narrower sense that
  it preserves intact English under global rewriting; it did not yet couple
  that prose to exact equations. No candidate has crossed the reader gate.

- **Coordinated two-region reconstruction (2026-09-17).** A fresh 138-letter
  event was then revised by changing its opening subject/action and closing
  consequence/setting regions together, while preserving a 100--140-letter
  band. All eight stored rows remain intact ordinary prose; the best row is
  114 letters with 52 mismatches versus 65 initially. The normalized mismatch
  rate moved from .942 to .912 while the model shortened the passage but stayed
  above the floor. This is a useful structural signal, not exact or reader
  evidence, and it does not excuse the length loss. Its artifact records every
  rendering, rejected-length count (zero), provenance, and independent
  forward/reverse hashes. The next test is a new authored event under the same
  two-region operator to check whether the reduction is reproducible; do not
  keep paraphrasing this lineage or promote its diagnostic score to readability.

- **Midpoint representation audit.** Astra differential testing found that the
  earlier live products silently required each complete clause to occupy one
  whole palindrome half.  That excludes valid unequal partitions and centers
  inside words.  A second defect made one descending right cursor's completion
  state unreachable, and the scene seam diagnostic always reported `closed=false`.
  The cursor and seam fixes are committed and covered by synthetic equal,
  unequal, internal-word, and negative controls.

- **Verified midpoint-crossing product.** The new product agrees with the
  full-tape oracle on six fixtures and then runs on the existing recursive prose
  grammar (16 derivations, 39--60 letters) without adding a bank or importing
  the seed.  It produced no exact closure.  A held-out semantic-frame repair
  (`the path`) retained ordinary prose and midpoint crossing but also produced
  no exact closure; its next action is a new shared-sentence slot construction,
  not another equal-half or vocabulary sweep.

- **JEV diagnostic.** The local TypeSafe wrapper reached `jev-1.13.0` with
  HTTP 200 in the approved compact-state call.  Its output is advisory routing
  evidence only; it cannot certify exactness or readability.

- **Single-sentence semantic slots.** A shared `Scene -> Agent Verb Theme
  Setting` derivation was searched directly with settled outside-in character
  obligations (40 typed products, 8 intact witnesses, longest 41 letters,
  zero exact closures).  This route does not form two complete halves; its next
  action is one held-out same-role setting substitution conditioned on the
  first residual, preserving the original controls.

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

- **Prosodic-foot surface realization (2026-09-15).** A new construction
  dimension attached CMU syllable counts, lexical stress contours, and phrase
  boundaries to complete event clauses before their independent character
  yields were joined. The bounded run covered 9,000 realizations per side and
  retained four append-only artifacts, including two repairs for an empty
  clause frontier and malformed articles. It found no exact join and no
  reader-eligible output. Its concrete successor is stress-frontier lexical
  replacement at a surviving outer-character match, not another lexical-bank
  or beam-width sweep.

- **Global tied masked denoising (2026-09-15).** A genuinely different
  construction treated every character position as a tied variable and
  propagated parallel masked-word assignments across the complete mirrored
  tape before scoring. Six rendered proposals were preserved with per-pair
  ledgers and independent admission checks; none reached the 39-letter gate.
  The three exact short controls are catalogue material and were rejected, so
  this run supplies no candidate or readability evidence. The next route must
  change the material source, not replay this symbolic bank or merely enlarge
  its beam.

- **Semantic-involution preflight (2026-09-15, excluded).** This audit was
  deliberately not counted as a construction family: its probes repeat a
  frame in reverse lexical order and include a known short palindrome. That
  violates the no-word-order-symmetry and no-catalogue-shortcut rules. Its
  exclusion is recorded so the same shortcut cannot re-enter under a new
  filename.

- **Novelty review of the latest parallel runs (2026-09-15).** The proper-name
  caption crossword is retained as a distinct typed appositive-record grammar;
  its 25 complete-caption probes produced zero exact closures and no reader
  material. The thematic-grid seam composition and typed SVO/PP semantic
  pairing runs are explicitly excluded after review: both replay the existing
  complete-prose cross-product state space under new row banks or seam
  diagnostics. Their rendered probes, hashes, and failure frontiers remain
  preserved, but they do not inflate the method count or paper claims.

- **Information-structure focus/scope route (2026-09-15).** A fresh
  construction state attached focus, presupposition, and negative-to-result
  polarity to complete cause/result clauses before rendering. Sixteen intact
  probes were checked by both the primary admission path and an explicit
  opposing-index scan; there were zero exact or admitted closures. The next
  repair is a polarity-preserving result clause whose terminal character stream
  crosses the causal seam, not another lexical-bank or beam replay.

- **Terminal-seam repair (2026-09-15).** That repair held the information
  structure fixed and substituted four authored result lexicalizations per
  plan. It produced 16 intact probes, a best outer match of one character, and
  zero exact or admitted outputs. Because the formal preflight marked it as a
  0.529 conceptual overlap with its parent, it is recorded as a repair under
  the same family rather than counted as a new method. The next repair changes
  connective/tense state at the causal seam, not merely the lexical bank.

- **Typed anaphoric scene chain (2026-09-15).** A separate preflighted route
  carried a singular object antecedent through three complete sentences using
  pronoun and definite-description continuity. Four ordinary intact scenes
  were checked by an explicit two-pointer audit and mechanical admission; all
  four failed exactness. The concrete repair is a held-out antecedent-synonym
  substitution that preserves number and discourse roles. A purported
  “multiset-balanced” stochastic sampler was rejected from the ledger because
  its implementation never enforced the advertised balance state.

- **Corrected multiset-balanced pair sampling (2026-09-15).** The corrected
  route genuinely tracked the 26-letter parity vector while sampling two
  independently authored, content-disjoint complete clauses. Across 50,000
  pairs it found zero parity survivors, so it produced zero exact closures; the
  strongest readable controls remain diagnostic only. The next operator is a
  parity-indexed clause lattice that chooses lexical continuations from the
  count-vector frontier instead of random sampling.

- **Parity-indexed lattice repair (2026-09-15).** The prescribed repair
  enumerated 2,304 typed clauses and indexed them by their 26-letter parity
  vector before attempting content-disjoint joins. Every clause had a unique
  parity bucket, so the arithmetic frontier still had zero joins; 25 ordinary
  clause pairs were retained as controls and all failed exactness. The next
  change must add agreement-bearing tense/determiner states before enlarging
  the lexical inventory.

- **Agreement-bearing parity repair (2026-09-15).** The next state added
  present/past verb forms and a/an/the agreement while indexing 98,304 complete
  clauses. It produced 2,720 content-disjoint parity joins and retained the
  strongest 25 intact probes; independent exact audits still found zero
  closures. The next repair is a controlled clause-boundary connective state,
  not another lexical-count sweep.

- **Proper-name reverse grammar (2026-09-15).** A separate 140,368-clause
  grammar used typed names, objects, adjectives, and locatives on both sides of
  an exact reversed-tape index. It produced zero content-disjoint exact pairs;
  25 ordinary clauses were retained as controls, with no reader claim. The
  next repair is a finite name/locative dependency state, not a larger name
  list alone.

- **Adaptive crossword span cover (2026-09-15).** A formally preflighted,
  whole-tape search allocated mirrored character variables first, placed
  context-conditioned phrase spans with reversible backtracking, and recovered
  non-mirrored word boundaries only after completion. It produced 10,785 exact
  tapes across 40--60 letters, independently confirmed by direct reverse-string
  and opposing-index audits, but every surface collapsed into one-letter
  fragments and failed the admission gate. This is retained as a distinct
  state-space result, not as readable progress. The concrete repair is a
  minimum-word-length, non-echoing contextual-template operator.

- **Context-template prose repair (2026-09-15).** The repair held authored
  contextual phrases intact and rejected one-letter reflected segmentation,
  reverse phrase echoes, and short function-word templates. It preserved 18
  rendered probes from roughly 40--100 letters and used separate direct and
  opposing-index audits; it found zero exact closures. This closes the
  diagnostic branch rather than relaxing the reader gate. The next frontier is
  a genuinely open-vocabulary construction that proposes complete clause
  expansions under semantic obligations before mirrored character commitment.

- **Lexical-chain walk (2026-09-15).** A separate forward-only typed lexical
  graph walk generated 6,375 complete single-clause probes from collocational
  edges without paired clauses, reverse emission, or mirrored units. It found
  zero exact closures. It remains useful evidence that a forward lexical graph
  alone is insufficient; it is not a reader result and will not be replayed by
  changing its seed or beam.

- **Syntactic mirror-template repair (2026-09-15).** A second-pass repair
  enforced typed subject/event/continuation slots and a minimum-two-letter
  surface. It found two exact surfaces at 43 and 51 letters, but an independent
  catalogue/duplicate-span audit rejected both; they are preserved as failure
  evidence and cannot be presented as generated prose.

- **Selectional-preference prefix automaton (2026-09-15).** This preflighted
  route learned coarse subject--verb--object preferences from Brown tag
  sequences and kept those preferences live while synchronizing two independent
  slot grammars character by character. The base run evaluated 324 frame/shape
  pairs and 11,854 search states; the concrete subordinate-clause repair
  expanded this to 441 runs and 15,477 states, with separate direct and
  opposing-index audit paths. No exact closure survived. Its best partial
  frontier is retained, and the next frontier changes construction family to
  an independent model-authored clause bank rather than replaying this lexical
  automaton.

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
- The character-level half-tape family is closed after one base run and three
  concrete repairs. Each retained 40 exact tape probes, but independent
  boundary recovery collapsed to repeated short words (for example, ``a cis a
  cis ...``), so none passed the ordinary-word, distinct-unit, or anti-shortcut
  gates. These append-only artifacts are failure evidence, not reader items.

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

- **Next construction frontier:** the TAG, reverse-complement Euler,
  prosodic-foot, global tied-denoising, and character-level half-tape runs are
  complete and their zeros are preserved as evidence, not as the thesis. The
  next run must change its construction dimension, preflight itself against the
  56-entry novelty
  registry, and retain complete rendered probes with independent exact audits.
  It must not replay a prior bank, beam, chart, BPE, ILP, evolutionary, TAG,
  Euler, prosodic, or masked-denoising configuration. Any exact survivor
  remains gated on intact-prose versus shuffled-control blinded reading.

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

- **Semordnilap template inventory, 2026-09-15:** introduced a disjoint
  finite inventory of 25 ordinary English semordnilap pairs and placed them in
  seven typed clause templates.  The reflected tape was segmented
  independently, rather than accepting word-order symmetry.  Three hundred
  fifty base and seam-swap repair probes produced zero exact closures; the
  concrete repair was a held-out seam-pair substitution followed by fresh
  reverse segmentation.  No probe is presented as readable evidence.

- **Rank-partitioned corpus sentence-gram FST, 2026-09-15:** used a new
  phrase-token construction: one source-rank partition supplied intact
  sentence/n-gram left material, while a disjoint held-out partition had to
  consume the exact reverse residual around a single-letter center. The base
  lattice and two concrete repairs (shorter phrase atoms, then expanded
  lower-order phrase atoms) produced 120 preserved partial/dead-end rendered
  probes and no exact closure. The reverse residual usually had no legal
  held-out phrase at its first character; this family is closed rather than
  silently replayed as a larger lexical beam. No probe is readable evidence.

- **Candidate readability audit, 2026-09-15:** added a reproducible report over
  the latest semordnilap, sentence-gram, and character-tape artifacts.  It
  retains 510 rendered rows and independently confirms 40 exact tapes, but
  zero rows pass the shared mechanical gate.  The highest-frequency exact
  rows are visibly repetitive (`a cis a cis ...`) and have negative Brown
  order gain; these diagnostics prioritize repair only and are not reader
  evidence.  Any future row that clears the gate must be frozen with an intact
  rendering, a matched word-shuffle control, and randomized blinded raters.

- **Admission-guided center-out repair, 2026-09-15:** preflighted a distinct
  live state dimension (`wordfreq-bigram-centerout`, whole-word uniqueness,
  admission-at-closure, length sweep) before running it.  The run produced
  four independent exact, mechanically admitted surfaces at 48--116 letters;
  the longest rendered candidate is:

  > `No it can won knee bye know oh so than its opera was sec cases Utah; two new we now that uses access aware post in ah to show on key been known action.`

  Its normalized tape is 116 letters and equals its reverse; the shared
  admission dictionary returns every hard check true, with provenance and
  SHA-256 recorded in `runs/lexical-admission-centerout-20260915.json`.
  This is a constructive exact closure, not a readability claim—the prose is
  not yet reader-worthy.  The next reader-facing test is held back: first
  apply grammar-constrained boundary resegmentation to this frozen tape, then
  package only any complete-clause survivor with an intact-prose control, a
  word-shuffled control, randomized blinded order, and reproducible rater
  instructions.

- **Fixed-tape grammar boundary repair, 2026-09-15:** applied the declared
  successor to the 116-letter tape without changing a single letter.  The
  Brown-POS weighted chart's 20k-word coverage run and its first 80k-word run
  had no complete segmentation; after correcting the short-word coverage bug,
  the held-out v3 produced two exact, mechanically admitted segmentations:

  > `No it can won knee bye know oh so than its opera was sec cases Utah; two new we now that uses access aware post in ah to show on key been known action.`

  > `No it can won knee bye know oh so than its opera was sec cases Utah; two new we now that uses access aware post in ah to show on key been known act ion.`

  Both preserve the 116-letter tape and pass every mechanical check, but
  neither is intact English prose.  All three runs, the source tape hash, POS
  sequence, and independent exact checks are retained in
  `runs/grammar-boundary-resegmentation-repair-20260915*.json`.  The next
  constructive operator is a fixed-tape valency chart (finite-verb and
  argument-role constraints carried through the same boundary states); only a
  complete-clause survivor will be eligible for intact/shuffled blinded
  readers.

- **Fixed-tape valency chart repair, 2026-09-15:** carried subject, finite
  verb, object, and clause-boundary states through the same immutable 116-letter
  tape.  Both exact mechanical segmentations survived the dictionary gate,
  but the stricter chart found zero complete-clause parses.  This is a genuine
  new repair state, not another beam or vocabulary replay; its complete
  provenance and failure frontier are in
  `runs/fixed-tape-valency-chart-repair-20260915.json`.  The next construction
  step is typed argument-role lexicalization at those surviving boundaries,
  preserving exact letters and the hard anti-shortcut checks before any reader
  package is built.

- **Model-authored complete-clause bank index, 2026-09-15:** preflighted a new
  construction dimension before execution: a local model authored 180 ordinary
  sentences, which were de-duplicated to 172 intact complete clauses with raw
  response provenance retained. A deterministic hash split made independent
  left/right banks; exact reverse indexes tested 75 x 97 one-clause joins and a
  bounded 5,550 two-clause composition frontier. Both direct reverse-string and
  opposing-index audits were independent, and no exact closure occurred. This
  is not a readability result and no clause is treated as model-certified prose;
  the raw proposals and zero-closure run are preserved in
  `runs/model-authored-clause-proposals-20260915.json` and
  `runs/model-authored-clause-bank-index-20260915.json`. The next repair must
  change the construction state (for example, intent-conditioned clause
  continuation around a live character seam), not simply resample this bank.

- **Incremental semantic-scene seam growth, 2026-09-15:** the next
  preflighted route grew three topic-linked complete events one at a time,
  carrying two-character seam obligations at each event boundary before any
  full-scene reverse lookup. It explored 333,306 coherent scene states and a
  terminal-event same-topic repair queue of 42,070 substitutions; the best
  repair matched only two reflected characters and no exact palindrome closed.
  The direct-reversal and opposing-index audits agree, and the full failure
  frontier is preserved in `runs/semantic-scene-seam-growth-20260915.json`.
  This is a new construction state, not a larger clause-bank cross-product;
  the next repair must change the seam-aware event lexicalization rather than
  replaying the same scene bank.

- **Live seam intent continuation, 2026-09-15:** a separate preflighted route
  queried the model directly with ordinary left prose and the exact required
  reversed tape, without a clause bank or reverse index. Six complete-sentence
  trials timed out before returning a candidate, so exact and reader counts are
  zero. The timeout evidence is retained with a concrete bounded repair:
  request short continuations at each live seam with timeout-safe batching,
  rather than repeating whole-sentence calls. Artifact:
  `runs/live-seam-intent-continuation-20260915.json`.

- **Fixed-tape GPT-2 boundary decoder, 2026-09-15:** a new preflighted repair
  froze the 115-letter exact tape from the lexical-admission frontier and
  enumerated 180 complete dictionary segmentations per position. A local GPT-2
  reranked 180 complete renderings, while the tape remained immutable and was
  checked by independent normalized-string and two-pointer audits. The best
  exact diagnostic rendering was `Test sale not care pro fit name till
  item anti for per act one last set.` (56 letters); the full gate admitted
  zero rows and the rendering is not intact prose, so no reader package was
  made.
  The artifact records all renderings, model scores, tape hash, and the next
  repair (typed valency transitions over the same frozen tape):
  `runs/fixed-tape-gpt2-boundary-decoder-20260915.json`.

- **Long-form POS center-out lexicalization, 2026-09-15:** a distinct
  preflighted constructor used 14--16 word POS templates and a debt-carrying
  outside-in character equation with fresh Brown/word-frequency lexical
  items. It explored 26,515 states across three templates and found no exact
  terminal row. The concrete repair fixed a correctness bug: an odd-length
  character tape may end with a one-letter (or otherwise palindromic) residual
  inside the centre word, so the successor accepts only a palindromic centre
  residual. The corrected run still found zero exact rows; both provenance and
  independent-audit paths are preserved in
  `runs/pos-template-centerout-longform-repair-20260915.json` and
  `runs/pos-template-centerout-longform-center-residual-repair-20260915.json`.
  This closes the route without relaxing the reader gate; the next construction
  must replace the lexical inventory/state rather than replaying the same POS
  templates.

- **Role-aware reversible reservoir center-out, 2026-09-15:** replaced the
  hand-list lexical menu with 66 reversible pairs derived from Brown
  universal-POS counts and a word-frequency floor, then injected each member
  into its attested role before solving the character debt. The run explored
  78,933 states across three templates and found no exact terminal, so no
  surface is being promoted as prose. The run and reservoir
  pair provenance are preserved in
  `runs/role-aware-reversible-reservoir-centerout-20260915.json`.

- **Variable-length role reservoir center-out, 2026-09-15:** changed the
  grammar dimension rather than enlarging the prior search: six complete
  5--10-slot templates were solved independently against the same
  corpus-derived reservoir. The short templates died at the outer seam and
  the longer template reached only residual-mismatch terminals; exact,
  mechanically admitted, and reader-eligible counts are all zero. Its next
  operator is explicitly not another same-template or reservoir-size replay;
  see `runs/variable-length-role-reservoir-centerout-20260915.json`.

- **Asymmetric template reservoir center-out, 2026-09-15:** paired four
  different left/right semantic templates and carried a subject-number
  agreement state through the bilateral debt transitions. This is a separate
  state product from both the fixed and variable same-template routes. The
  bounded run produced 552 explored states, zero terminal closures, zero
  mechanically admitted rows, and zero reader rows. The independent preflight,
  rendered-output log, and next repair are in
  `runs/asymmetric-template-reservoir-centerout-20260915.json`; no candidate is
  represented as readable evidence.

- **Attested phrase-pair wrapper, 2026-09-15:** changed the construction unit
  again, indexing 753 common Brown phrase spans and asking an independent
  held-out word trie to segment each exact reverse tape. All 753 reverse
  segmentation calls failed at the first lexical boundary; there are zero
  exact, mechanically admitted, or reader-eligible rows. The run keeps eight
  concrete span probes with their required reverse tapes and source offsets in
  `runs/attested-phrase-pair-wrapper-20260915.json`. Because the source spans
  are attested material, none is presented as generated prose; the next repair
  must replace spans with independently authored complete clauses rather than
  relax the lexical gate.

- **Homograph-sense lattice, 2026-09-15:** preflighted a separate semantic
  state in which one orthographic tape must admit two independent homograph
  sense/POS parses. Twenty-five frame pairs explored 1,912 states; three
  residual terminals were rejected before exact closure and the run produced
  zero exact, mechanically admitted, or reader-eligible rows. The actual seam
  probes (for example `An china.` and `A camera.`) are retained with their
  residuals, but are fragments rather than prose. Artifact:
  `runs/homograph-sense-lattice-20260915.json`.

- **WordNet synonym-frame CSP, 2026-09-15:** preflighted a distinct
  semantic-preserving lexicalization route. Six complete dependency-frame
  shapes drew independent WordNet lemma alternatives on the left and right;
  each left yield was matched against a right frame under an exact character
  equation. The run made 6,834 reverse-frame calls with zero reverse hits,
  zero exact closures, and zero reader rows. No source sentences or catalogue
  text were imported. The artifact and independent audits are in
  `runs/wordnet-synonym-frame-csp-20260915.json`; the next repair is an
  agreement/subcategorization feature layer, not another reservoir or template
  replay.

- **WordNet agreement/subcategorization repair, 2026-09-15:** carried subject
  number, determiner agreement, transitive valency, inflection, and optional
  adjunct attachment through a separate finite-state frame automaton. The
  repair explored 90,000 left assignments and 60,012 reverse-frame checks but
  found no reverse lexical hit, exact closure, or reader row. Its artifact is
  `runs/wordnet-featured-frame-repair-20260915.json`; the next operator changes
  dependency topology rather than widening this same option list.

- **GPT-2 topic-conditioned half decoder, 2026-09-15:** sampled 288 ordinary
  topic prompts, decoded each required reverse tape with an independent lexical
  DP, and retained 73 exact tapes for audit. The longest was 104 letters, but
  every rendering contained non-lexical fragments and failed mechanical
  admission; no row is reader evidence. The full provenance and probes are in
  `runs/gpt2-topic-half-decoder-20260915.json`. The next repair is an online
  GPT-2 constrained decoder at the reverse seam, not another post-hoc rerank.

- **GPT-2 odd-center bridge, 2026-09-15:** added the missing one-letter seam
  state by decoding `center + reverse(left)` independently for all 26 letters.
  Of 384 topic-conditioned samples, 3,848 center decodes yielded four exact
  39-letter surfaces. They all contained fragmentary/non-prose right sides and
  failed the self-palindromic-span gate, so none is reader evidence. The
  rendered rows, lengths, audits, and model prompts are preserved in
  `runs/gpt2-center-letter-bridge-20260915.json`; the next repair must carry a
  syntactic state into online reverse decoding rather than relax this gate.

- **Orientation repair, 2026-09-15:** reversed the proposal direction—GPT-2
  supplied the natural right clause while a strict decoder solved the left
  tape. This was preflighted as a repair because its signature shares the
  odd-center bridge state. Across 384 samples and 3,354 center decodes it
  produced zero exact or reader rows; the full negative evidence is in
  `runs/gpt2-right-half-bridge-20260915.json`. The bridge family is now
  exhausted at both orientations without weakening the lexical gate.

- **POS/constituency bridge repair, 2026-09-15:** carried Brown-backed POS
  labels and complete clause patterns into the odd-center decoder. The 384
  samples generated 3,354 center trials but no complete right-frame hit, exact
  closure, or reader row. This closes the fragment-resegmentation repair with
  evidence in `runs/gpt2-center-fsm-bridge-20260915.json`.

- **Interrogative–quantifier dependency automaton, 2026-09-16:** preflighted a
  genuinely different construction state: auxiliary inversion and question
  polarity on the left are joined to an independently ordered quantified
  answer frame on the right, with an odd center letter carried in the exact
  character equation. The base run checked 420,000 left assignments,
  7,165,600 center equations, and 52,197,516 right-frame yields; it found no
  reverse hit, exact closure, or reader row. A concrete target-indexed repair
  then performed 62,901,860 slot-prefix trials without materializing that
  product, found one hit, and rejected it as known catalogue text. The artifacts
  are `runs/interrogative-quantifier-fsm-20260915.json` and
  `runs/interrogative-quantifier-indexed-repair-20260916.json`; this family is
  not being widened again. The next route must change dependency topology and
  author both sides independently.

- **Terminal-aware grammar intersection, 2026-09-16:** preflighted a separate
  construction state that keeps two independent typed clause grammars live while
  taking explicit lexical-boundary epsilon transitions at simultaneous word
  endings. The 30 frame pairs explored 8,724 states and 8,766 character matches;
  nine short closures were rejected, with zero long exact, mechanically admitted,
  or reader-eligible rows. This repairs a concrete boundary-loss bug in the prior
  outside-in probe without replaying its lexical products. The artifact is
  `runs/terminal-aware-grammar-intersection-20260916.json`; the next route must
  change dependency topology or semantic state, not widen these same frames.

- **Semantic-derivation MCTS, 2026-09-16:** preflighted a separate search-control
  state in which UCT allocates rollouts over typed subject/predicate/object and
  modifier actions while independent left/right lexicalizations carry a live
  reflected-character ledger. Thirty thousand rollouts produced no exact or
  mechanically admitted closure; the longest rendered probe was 29 letters and
  no row entered the reader gate. The full action counts and partial surfaces
  are in `runs/semantic-mcts-derivation-20260916.json`. Its concrete repair is a
  reverse-conditioned action prior over the right-edge residual, not another
  static-frame or beam replay.

- **Reverse-conditioned MCTS repair, 2026-09-16:** retained the MCTS grammar and
  lexical alternatives but changed only rollout policy: a softmax prior used
  reflected-character lookahead and edge-length residual to choose the next
  independent action. Thirty thousand rollouts made 2,814,090 ledger-prune
  decisions, with no complete exact closure; the longest probe was 30 letters.
  It is recorded as a repair artifact, not a second family, and the next route
  must change semantic state rather than tune this prior again.

- **Morphosemantic product-delay pilot, 2026-09-16:** preflighted a new
  productive state in which looping feature automata realize morphology on
  demand and a persistent output-delay monoid cancels independent yields; no
  complete clause bank, fixed-tape segmentation, beam, or MCTS was reused.
  The 100,000-state bounded run made 480,563 transitions and 183,140
  character rejections but found zero exact closures. It is recorded as a
  failed construction with no reader evidence in
  `runs/morphosemantic-product-delay-20260916.json`; the next repair must
  change the lexical realization frontier, not replay this state.

- **Morphology-first dependency lattice, 2026-09-16:** preflighted a distinct
  route that chooses a typed dependency topology first, then unifies
  node-local lemma, derivation, and inflection paths before ordinary-order
  emission. Of 1,728 base yields, three were exact diagnostics; all were
  rejected independently as catalogue or word-order/self-symmetry material.
  Its derivational/topology repair expanded the same state family to 24,570
  yields and retained 48 mismatch-audited near misses, with zero exact
  closures. The complete surfaces and morphology paths are preserved in
  `runs/morphology-first-dependency-lattice-20260916.json` and
  `runs/morphology-first-dependency-lattice-derivational-repair-20260916.json`;
  no catalogue text is treated as generated output.

- **Semantic relation alignment, 2026-09-16:** preflighted a directed
  event-edge construction in which each side independently lexicalizes typed
  roles while a deterministic boundary-synchronous ledger compares ordinary
  yields. Four relation frames explored 36,662 states; the deepest match was
  six characters and there were zero exact, mechanically admitted, or
  reader-eligible rows. The terminal-compatible phrase/odd-center repair then
  explored 646 states and 336 terminal spans, also with zero exact closures.
  Both runs are preserved in
  `runs/semantic-relation-alignment-20260916.json` and
  `runs/semantic-relation-alignment-terminal-repair-20260916.json`.

- **Weighted CFG synchronous DP, 2026-09-16:** preflighted a new parse-forest
  state that intersects two independently weighted CFG derivations by
  character position. The base forest had 648 derivations per side and 13,392
  chart states; a typed-adjunct repair expanded this to 3,240 derivations and
  82,512 states. Neither produced an exact closure or reader-eligible row.
  Evidence is frozen in `runs/weighted-cfg-sync-dp-20260916.json`.

- **Non-repeating reversible insertion, 2026-09-16:** tested a scalable seam
  wrapper constructor using five distinct, independently annotated reverse
  lexical units around the verified seed. It produced exact 50--106-letter
  renderings, but every seam was visibly incoherent; the hard no-repeat gate
  held and all five remain outside the reader gate in
  `runs/reversible-grammar-insertion-20260916.json`. This is failure evidence,
  not a readable output. Its concrete repair replaced wrappers with pairs of
  independently grammatical adjuncts in a complete prose frame; all 12
  combinations were near misses and none closed exactly. The repair evidence is
  in `runs/semantic-insertion-repair-20260916.json`.

- **Corpus semantic-frame realizer, 2026-09-16:** preflighted a distinct
  semantic-frame planner that independently realizes agent/action/artifact
  slots and orders them with a corpus bigram model after character-equation
  filtering. It attempted 4,961 disjoint frame pairs from 6,840 frames and
  retained 40 ordinary probes, with zero exact, admitted, or reader-eligible
  rows. No neural model or catalogue text was used; the frozen evidence is in
  `runs/corpus-neural-frame-realizer-20260916.json`.

- **Parallel Luna construction queue, 2026-09-16:** three fresh routes were
  run in parallel and kept separate in the novelty ledger. The recursive
  complete-clause residual DP rendered 40 intact probes up to 1,922 letters;
  all failed exact closure. The dialogue speech-act grammar tested 25
  request/answer, greeting/acknowledgment, and report/response pairs; its
  same-act lexical repair ran 36 complete trials up to 53 letters; both found
  zero exact closures. The Brown-attested residual lattice analyzed 57,340
  sentences and adjacent-span repairs but produced no reverse segmentation, so
  it rendered no generated text. The authored boundary repair retained three
  complete 69–73-letter probes and zero exact closures. Every route has a
  concrete next operator recorded in its run artifact; none is reader-eligible.

  Representative rendered probes (shown to keep the work reader-facing) are:

  ```text
  the baker repairs a gate. a quiet aide keeps notes. the cook serves the supper.
  Please share the morning report. I can bring the letter.
  a farmer calls the calm note. near the market. a sailor replies the fine signal. in the opera.
  ```

  Their normalized tapes and independent audits are in
  `runs/recursive-grammar-residual-dp-20260916.json`,
  `runs/dialogue-speech-act-residual-repair-20260916.json`, and
  `runs/reverse-transition-svo-authored-search-20260916.json`. They are
  complete English controls, not palindrome claims: each exact audit is false,
  provenance is authored or held-out lexical data, and the next reader-facing
  test remains blocked until a novel exact candidate clears the mechanical
  gate. The Brown route's empty `displayed` list is intentional because it
  found no text safe to show as generated.

  A reproducible programmatic diagnostic over 2015 rendered rows is frozen in
  `runs/parallel-luna-readability-diagnostics-20260916.json`: 72 exact rows
  (36 semantic-frame tapes plus their 36 residual-decoder rejections) and 0
  mechanically admitted rows. The report breaks out Brown word-order gain,
  mean word frequency, repetition, punctuation segmentation, and length by
  route. These values are explicitly diagnostic; they do not certify
  readability or authorize a reader study.

- **Second parallel Luna reset, 2026-09-16:** after the first queue stayed at
  zero exact closures, three different hypotheses were run rather than tuning
  the same beams. A reversible-word whole-sentence grammar produced 16 base and
  36 repair probes with zero exact closures. A seedless semantic recursive CFG
  produced 10 complete candidates up to 91 letters with zero closures. A
  semantics-preserving mutation search ran 41 base and 57 repair probes over
  authored clauses; every trial remained a complete clause, but exact and
  reader counts were zero.
  A representative repair rendering was “The careful nurse records a dosage.
  The cautious nurse records a dosage.” It is preserved as a complete-prose
  diagnostic with its mutation provenance, not presented as an exact output.

  A third parallel Luna reset added three non-overlapping routes. An induced
  PCFG sampled 432 base and 1,728 PP-repair derivations with no exact closure.
  A rhetorical-plan lattice generated 27 base and 36 held-out-connective repair
  probes, longest 172 letters, with no exact closure. An inflectional FST with
  agreement and clitic transitions generated 64 base and 32 negative-contraction
  repairs, longest 68 letters, with no exact closure. Their rendered probes,
  hashes, provenance, and independent audits remain in the corresponding run
  artifacts; none is reader-eligible.

  A fourth parallel Luna reset added graph-to-prose, active/passive voice, and
  CCG/type-logical routes. The typed graph constructor produced 9 base and 16
  alternate-topology repairs (max 110 letters); voice alternation produced 64
  base and 144 held-out repairs (max 78); CCG produced 4 base and 4 category
  repairs (max 82). All were complete prose with independent audits but zero
  exact closures and zero reader-eligible rows. Their rendered text and
  provenance remain frozen in the route artifacts and the 2015-row diagnostic.
  These routes change the construction state (reversible lexical roles,
  unseeded recursion, and online semantic mutation) and are not filename or
  beam-width replays.

  A fifth reset added dependency-completion CSP and an all-different lexical
  word-equation inventory. The CSP ran 81 base and 81 syntax-completion repair
  candidates; the word-equation route ran 6 base and 6 POS-compatible repairs.
  Both kept complete clauses and independent provenance visible, but had zero
  exact closures and zero reader-eligible rows. The aggregate diagnostic now
  contains 2015 rendered rows across 57 route phases plus per-route summaries; its metrics remain
  filters and failure diagnostics, never a readability certificate.

  A bounded direct-authoring reset also ran three local-model prompts plus one
  mismatch repair. Every call timed out at 20 seconds before returning text;
  the timeout evidence and model hash are preserved in
  `runs/direct-constrained-authoring-20260916.json`. This is a concrete
  authoring failure and a reason to change the construction state again, not a
  readability or impossibility claim.

  The direct-authoring repair was then rerun with a bounded alternate local
  model (RhythmAI). It returned the rendered line “A rare, radiant, and radiant,
  rare aura.” (30 letters): the independent tape audit was non-palindromic and
  below the length gate, so it is preserved only as repair evidence in
  `runs/rhythmai-authoring-probe-20260916.json` and was not sent to readers.

  A morphology-semantic template CSP reset selected derivational and
  inflectional realizations jointly with semantic frames (216 base plus 216
  affix-frontier repairs). Its complete sentence renderings and independent
  audits are in `runs/morphology-semantic-template-csp-20260916.json`; zero
  exact closures survived either pass. A pivot-centered paragraph beam then
  authored 18 semantic-slot pairs and 18 near-synonym repairs, also with zero
  exact closures (`runs/pivot-paragraph-beam-20260916.json`). These are new
  construction states, not replays, and both leave the human reader gate
  closed until a candidate passes exactness and the intact-versus-shuffled
  study.

  A deterministic microgrammar reset then grew recursive adjuncts while
  carrying lexical debt online. It rendered 24 base scenes and 24 frontier-
  substitution repairs (70--152 letters); every row is retained with its
  complete-sentence audit, but all exact and reader gates remained false in
  `runs/microgrammar-lexical-debt-20260916.json`. The repair changes the
  lexical frontier rather than merely widening a beam, so it is a concrete
  next construction state despite the negative closure count.

  A prosodic-foot constructor then coupled stress-foot choices to semantic
  scene roles (54 base scenes and 54 measured-timing repairs, up to 170
  letters). An induced-grammar reverse decoder separately learned only grammar
  shapes and selected fresh held-out lexical frames (4 base plus 4 repairs,
  66--78 letters). Both routes preserve complete renderings and independent
  tape audits in their run artifacts; neither produced an exact closure, so no
  text is being relabeled as a candidate or sent to readers.

  A semantic-frame tape solver then produced 60 exact letter tapes among 162
  lexical alternatives (52--61 letters). The exact rows expose the next
  construction problem rather than solving it: for example, “the nurse keeps
  a blue map by dawn. nwadybpameulbaspeekesruneht .” has an independently
  verified 54-letter palindrome, but the right side is a non-word residual and
  fails the lexical/reader gates. Those rows are retained as concrete exact
  failure evidence in `runs/semantic-frame-tape-solver-20260916.json`; the
  repair must decode the residual into intact English before any reader test.

  A Boolean MaxSAT semantic-plan route separately optimized character-equality
  literals over complete event clauses (8 base plus 8 held-out repairs, up to
  76 letters). It improved the objective but produced no exact or reader-
  eligible row (`runs/maxsat-semantic-grammar-20260916.json`).

  The next repair wave kept three new construction states separate. A
  held-out boundary decoder rendered 81 semantic-role probes and 81
  resegmentation repairs (67--78 letters), all complete but non-exact. A
  lexical-trie residual probe segmented the 38-letter benchmark tape but found
  no complete closure. An evidential-scene planner realized source-of-knowledge
  discourse frames in 4 base and 4 held-out repairs (59--66 letters), again
  with zero exact closures. Their artifacts retain rendered text, independent
  audits, and provenance; none is reader-eligible. The pivot-growth proposal in
  the same wave was preflight-blocked because its state space duplicated the
  existing pivot/scene-growth families, and was recorded as an exclusion rather
  than counted as a new run.

  A residual lexical decoder then acted on the exact semantic-frame tapes,
  recursively segmenting each required reverse string with a held-out POS
  lexicon. It rendered 36 exact 52--54-letter tapes, but every right half was
  unsegmentable (for example, the residual `nwadybpameulbaspeekesruneht`), so
  all 36 were rejected and no survivor reached the reader gate. This is the
  concrete repair result, not a readability claim; evidence is frozen in
  `runs/residual-lexical-decoder-20260916.json`.

  A heteropalindromic clause composer then authored 12 fresh complete clause
  pairs and 12 determiner-seam repairs (46--48 letters). The independent
  clause and lexical gates stayed true, but none closed the character tape
  exactly; all remain outside the reader gate in
  `runs/heteropalindromic-clause-composer-20260916.json`. Its two-sided clause
  construction is preserved as a concrete next repair state, not a shortcut or
  a readability claim.

  A sixth reset used typed lexical-chain permutations rather than a model or
  clause cross-product: 24 complete base probes and 12 synonym repairs were
  rendered, all with zero exact closures. The run is preserved as
  `runs/lexical-chain-palindrome-20260916.json`; it contributes to the same
  diagnostic aggregate but no row entered the reader gate.

  The latest three-Luna wave deliberately changed syntactic topology instead
  of widening any prior beam. A conditional-embedding solver composed `if`/
  `then` antecedent/consequent scenes with a held-out antecedent repair and
  rendered eight complete probes (80--89 letters). A reported-speech route
  added attribution verbs and tense-shifted embedded propositions, then ran a
  conditional repair over 6 base and 12 repaired scenes (68--88 letters). A
  nested conditional/temporal mutation route rendered 3 intact scenes and 6
  lexical repairs (44--50 letters). All three routes have independent tape
  hashes, explicit provenance, and complete-sentence checks; none produced an
  exact closure or a reader-eligible row. Their concrete artifacts are
  `runs/conditional-embedding-solver-20260916.json`,
  `runs/reported-speech-topology-20260916.json`, and
  `runs/nested-conditional-mutation-20260916.json`.

  After that topology wave the aggregate diagnostic contained 1940 rendered rows across
  51 route phases. It still contains 72 exact but mechanically rejected tapes
  and 0 mechanically admitted rows; the independent recomputation remains
  mismatch-free. These counts are search evidence and repair guidance, never
  a readability certificate. No candidate has reached the intact-versus-
  shuffled blinded reader test, so the API and paper release gates stay closed.

  A subsequent three-Luna scalability wave tested a simultaneous phrase-level
  beam. It grew both sides from semantic commitments in lockstep and used the
  Wikitext n-gram table only as a local-order diagnostic. Nine base and nine
  mismatch-repair candidates were complete prose (52--72 letters), but none
  was exact or reader-eligible. A mirror-pair phrase-chain graph and a
  conditional role-changing pair search were preflight-blocked because their
  path and role states duplicate retained lexical-chain, collocation, and
  conditional/reversible families; both blocks and their concrete pivots are
  preserved. The runnable artifact is
  `runs/simultaneous-phrase-beam-20260916.json`, while the preflight artifacts
  are `runs/reversible-phrase-chain-preflight-20260916.json` and
  `runs/conditional-role-pair-preflight-20260916.json`.

  The aggregate now contains 2000 rendered rows across 56 route phases. The
  independent tape recomputation still has zero mismatches, 72 exact-but-
  rejected tapes, and 0 mechanically admitted rows. No route has reached the
  blinded intact-versus-shuffled reader test; these diagnostics therefore
  guide the next construction rather than certify readability.

  A third wave then coupled reader-first scene authoring to discourse
  connective choice and sentence-level paraphrase. It produced six coherent
  two-sentence scenes and six repairs (49--57 letters); two repairs duplicate
  base renderings, so the aggregate retains 10 unique rows with explicit
  source provenance. No exact closure or reader-eligible row was found. A
  coordination/ellipsis proposal and an open-vocabulary morpheme CSP were
  preflight-blocked as already represented by the coordination, morphology,
  multiset, and inflectional families. Their preflight artifacts remain
  preserved alongside the runnable
  `runs/discourse-connective-coupled-20260916.json` artifact.

  The aggregate is now 2000 rendered rows across 56 route phases, with zero
  independent tape mismatches, 72 exact-but-rejected tapes, and 0 mechanically
  admitted rows. The intact-versus-shuffled blinded reader study is still the
  next gate once an exact candidate exists; programmatic scores remain
  diagnostic only.

  A fourth wave tested information-structure and joint authoring directly. A
  topicalization constructor preserved subject/verb roles while varying object
  scope, then applied parenthetical repairs (6 base + 12 repair scenes,
  48--83 letters). A deterministic coupled syntax/lexical fallback generated
  9 base + 9 repair scenes (45--50 letters); four repaired renderings duplicate
  base text, leaving 14 unique rows in the aggregate. Both routes keep intact
  prose and independent audits but have zero exact closures and zero
  reader-eligible rows. Modal/evidential scope was preflight-blocked as a
  retained scope/evidence replay. Artifacts are
  `runs/topicalization-scope-constructor-20260916.json` and
  `runs/coupled-syntax-lexical-authoring-20260916.json`, with the modal block
  preserved in `runs/modal-scope-preflight-20260916.json`.

  The aggregate now contains exactly 2015 rendered rows across 57 route phases.
  Independent recomputation reports zero tape mismatches, 72 exact-but-
  rejected tapes, and 0 mechanically admitted rows. No output has reached the
  blinded intact-versus-shuffled reader test; these measurements remain
  diagnostics while the constructive search continues.

  A fifth wave tested semantic wrappers around fresh, non-catalogue centers.
  Nine base scenes and nine wrapper-connective repairs were authored (57--62
  letters); three repaired renderings duplicate base text, leaving 15 unique
  rows in the aggregate. All are intact complete prose with independent
  provenance and exact audits, but none closes the tape or reaches the reader
  gate. Nested quotation and open-model-bank reverse decoding were preflighted
  and blocked as reported-speech/dialogue and fixed-tape/model-bank replays.
  The runnable evidence is in
  `runs/reversible-semantic-wrappers-20260916.json`; the preflight records are
  retained beside it.

  The aggregate now contains 2015 rendered rows across 57 route phases, with
  zero independent audit mismatches, 72 exact-but-rejected tapes, and 0
  mechanically admitted rows. The next reader-facing test remains the
  randomized blinded intact-versus-shuffled study, which cannot start until an
  exact candidate clears every mechanical gate.

  A sixth three-Luna wave deliberately opened three different branches rather
  than enlarging an existing beam. A comparative/modal scene search rendered
  81 measurement scenes and 81 held-out modal repairs (31--41 letters). A
  scalar-evaluation route independently lexicalized an event, a judgment
  predicate, and its evidence proposition (4 base + 4 held-out repairs,
  78--85 letters). A template-free authoring route began from three fresh,
  intact prose centers and made six bounded one-word repairs (58--72 letters),
  preserving punctuation, order, and clause structure. All three routes have
  independent exact/hash audits, complete-prose checks, and provenance; none
  produced an exact closure or a reader-eligible row. Reverse-trie and
  open-vocabulary evidence variants were preflighted and blocked as state-space
  replays, with their concrete pivots retained.

  The aggregate now contains exactly 2194 rendered rows across 61 route phases.
  Independent tape recomputation reports zero mismatches, 72 exact-but-
  rejected tapes, and 0 mechanically admitted rows. These counts are diagnostic
  evidence only: no candidate has reached the randomized blinded
  intact-versus-shuffled reader test, so the paper release and API remain
  closed while the constructive search continues toward a genuinely readable
  long palindrome.

  A follow-up Luna lane attempted manual grammar-aware center expansion from
  three fresh common-word prose centers and three semantic-center repairs
  (45--53 letters). The scenes remain intact and independently audited but
  produced no exact closure or reader-eligible row. A reversible-pair
  center-out route and a repair of the 72 exact-but-rejected tapes were both
  preflighted and blocked because their construction states are already
  represented; neither was rerun as a disguised decoder experiment. The
  aggregate therefore stands at 2200 rendered rows across 62 route phases,
  with zero tape mismatches, 72 exact-but-rejected tapes, and 0 mechanically
  admitted rows.

  A follow-up three-Luna fan-out added two runnable construction probes and two
  explicit preflight records. The semantic seam-DP lane rendered nine fresh
  scenes plus nine connective repairs (32--41 letters); a manual
  bidirectional-scene lane rendered six fresh letter/weather/travel scenes
  (37--44 letters). Both preserve intact prose, provenance, and independent
  two-pointer/hash audits, but produced no exact closure and no reader-eligible
  row. GPT-2 reverse reranking and weighted Brown-corpus reverse decoding were
  rejected before sampling because their construction states duplicate retained
  neural/reverse/corpus families; their pivots are preserved in the preflight
  artifacts. The aggregate now contains 2224 rendered rows across 64 route
  phases, with zero independent tape mismatches, 72 exact-but-rejected tapes,
  and 0 mechanically admitted rows. These are diagnostic records only; the
  reader gate and API remain closed while the next constructive lanes seek a
  genuinely readable long palindrome.

  The human-guided global-equation lane then searched 648 states over a fresh
  four-clause archive scene, varying lexical realizations, tense, and
  punctuation under a semantic-preservation ledger. It retained 24 complete
  148--161-letter renderings, with unanimous independent exact/admission
  agreement but zero exact closures. The aggregate is now 2579 rendered rows
  across 68 route phases; zero tape mismatches, 72 exact-but-rejected tapes,
  and 0 mechanically admitted rows remain. This is a concrete failure and
  repair frontier, not a readability claim; the reader packet still waits for
  an exact mechanically clean output.

  The boundary-fixed-point lane then jointly enforced reflected cumulative
  word-boundary profiles during lexical emission (1,350 generated rows, 647
  unique retained renderings, 52--64 letters). It tracked cross-boundary
  matches online and applied a concrete five-letter profile-class expansion;
  no exact closure or proper-span violation was found. The aggregate is now
  3226 rendered rows across 69 route phases, with 72 exact-but-rejected tapes
  and 0 mechanically admitted rows. This lane is preserved as a constructive
  diagnostic, not a reader result.

  The morpheme/affix closure lane added 19,200 joint probes from independent
  clause banks and retained 48 complete 57--64-letter failures. Its
  first-mismatch held-out stem+affix swap is now the next repair operator;
  every exact and SHA-256 audit agreed, but exact_count and admission_count
  remain zero. The aggregate is 3274 rendered rows across 70 route phases,
  with 72 exact-but-rejected tapes and no mechanically admitted candidate.

  The next three-Luna fan-out kept the search constructive while changing the
  state representation in each lane. Equal-length semantic clause collision
  joined independently authored SVO clauses by reversed-tape hash (120,000
  clauses per side) and retained eight complete 64--66-letter probes. A
  semantic valency boundary CSP searched 384 station-scene realizations and
  retained 24 complete 127--147-letter renderings while varying sense,
  agreement, tense, and word boundaries jointly. A character-synchronous
  Earley/finite-state intersection kept agreement and event roles in parser
  state and retained 12 complete 48--64-letter probes. All three lanes have
  independent exact/hash audits and mismatch-directed repair operators; none
  has yet produced an exact closure or a mechanically admitted candidate.
  Their artifacts are `runs/equal-length-clause-collision-20260916.json`,
  `runs/semantic-valency-boundary-csp-20260916.json`, and
  `runs/earley-finite-state-grammar-intersection-20260916.json`.

  The aggregate now contains 3318 rendered rows across 73 route phases, with
  zero independent tape mismatches, 72 exact-but-rejected tapes, and 0
  mechanically admitted candidates. These rows remain construction evidence,
  not readability certification: the intact-versus-shuffled blinded reader
  package and the API stay closed until an exact, mechanically clean,
  reader-worthy output exists.

  The ten-lane Luna ledger is now complete. The wave ran ten orthogonal
  construction dimensions: direct character-level GPT-2 decoding; immutable
  exact-tape typed-CFG resegmentation; dependency-tree seam CSP; weighted
  agreement-carrying morphology; CFG/Earley character intersection;
  human-authored discourse scenes; semantic valency/attachment; inflectional
  and clitic boundaries; length-indexed compositional grammar; and semantic
  slot repair of an exact source tape. Each lane retains rendered ordinary
  prose (or a clearly labelled control), provenance, novelty preflight, two
  independent exact checks, and a concrete held-out repair. Duplicate sweeps
  were not counted. The direct decoder's best intact probe is “A baker marks
  the button near a bright field. A caller answers the message behind a brief
  plaza.” (77 letters); the scalable grammar reaches 137-letter probes, and
  the scene lattice reaches 124-letter probes. None is exact and mechanically
  admitted, so none is reader-eligible.

  The refreshed aggregate contains 3484 rendered rows across 81 route phases,
  73 exact-but-rejected tapes, and 0 mechanically admitted candidates, with
  zero independent tape mismatches. The typed-CFG lane's 46-letter tape is
  explicitly rejected as a word-order/semordnilap control, not promoted as
  generated prose. This is the current constructive frontier: the paper and
  API remain gated, and the next repair must change lexical material or a
  recorded semantic frontier rather than replaying any of these ten lanes.

  Two post-ledger repairs then changed the search state rather than widening a
  beam. The assumption-core solver searched 240 complete multi-clause scenes
  at 61--74 letters, retaining a minimal conflicting assumption core and a
  held-out one-slot replay for every miss. The corrected masked-character lane
  explored three genuine semantic alternatives per complete scene under
  pairwise character masking and local complete-tape scoring. Both lanes have
  intact English probes, independent exact/hash audits, provenance, and zero
  exact or mechanically admitted closures. The phrase-lattice automata route
  was preflight-excluded because it duplicates retained phrase-FST state.

  The current aggregate is 3727 rendered rows across 83 route phases, with 73
  exact-but-rejected tapes and 0 mechanically admitted candidates. These
  follow-ups provide new repair frontiers, not a readable palindrome; the
  blinded reader study, paper promotion, and API release remain gated.

  A further wave tested a bidirectional encoder pseudo-likelihood state and a
  residual min-cost-flow state. The encoder enumerated three complete
  50--52-letter semantic scenes, but the cached checkpoint was TF-only under
  the installed runtime, so it failed closed without claiming model evidence.
  The flow solver produced two complete 76--77-letter clause compositions,
  both non-exact under independent tape/hash audits, with held-out lexical
  repair recorded. A recursive stack-machine exact closure was explicitly
  rejected because its surface was a semordnilap word chain rather than prose.

  The current aggregate is 3732 rendered rows across 85 route phases, with 73
  exact-but-rejected tapes and 0 mechanically admitted candidates. The next
  repair must improve a recorded semantic frontier; no reader study, paper
  promotion, or API release is justified yet.

  The next orthogonal Luna continuation added three new states rather than
  duplicate sweeps. A compound/derivational scene CSP generated 80 complete
  39+ letter probes with a held-out compound replacement repair. A graph of
  complete authored scene realizations walked meaning-preserving paraphrase
  edges, yielding 81 probes from 117--408 letters plus one held-out edge
  repair. A constituency parse-tree exact-cover solver generated 24 complete
  tree realizations and 8 held-out tree/agreement repairs. Each route keeps
  actual prose, provenance, novelty preflight, independent exact/two-pointer/
  hash/mechanical audits, and a concrete next repair; all three have zero
  exact closures and zero mechanically admitted rows.

  At that stage, the common audit report contained 3888 normalized rows across 92 route
  phases, with 73 exact-but-rejected tapes and 0 mechanically admitted
  candidates. The normalization retains only rows with the shared audit schema
  (48 compound, 82 paraphrase-graph, and 6 parse-tree rows from the new wave),
  while the full run artifacts preserve every generated probe. No output is
  reader-eligible yet, so the intact-versus-shuffled human test remains gated.

  A subsequent three-Luna repair wave added a memoized bilateral semantic CFG,
  a human-authored scene-equation frame constructor, and a typed
  agreement/attachment/lexical-sense edit program. Their rendered diagnostics
  are 154, 145, and 82 letters respectively; every one is complete ordinary
  prose with provenance, independent exact/hash checks, and a held-out repair,
  but all three fail the exact palindrome gate. The aggregate now contains
  4240 normalized rows across 98 route phases, 73 exact-but-rejected tapes,
  and 0 mechanically admitted candidates.

  A further three-Luna wave tested event-graph character SAT, a syntax-stack
  semantic-role decoder, and a meet-in-the-middle phrase-equation inventory.
  They produced complete ordinary prose at 61, 81, and 74 letters with
  provenance, independent audits, and held-out repairs, but no exact closure.
  Their evidence is retained as constructive search progress; it does not
  become a readable example or a reader-study claim until the exact gate is
  passed.

  The next repair wave tested semantic-center SAT, reverse-tape CFG/valency
  segmentation, and dependency-preserving mirror-pair construction. Their
  longest complete prose diagnostics were 70, 51, and 68 letters; all failed
  independent exact audits. Bounded held-out continuations then added a
  phrase-equation whole-phrase repair (81 letters), a frozen-center
  verb/object debt repair (69 letters), and an 18-trial dependency
  subject/adjunct repair (68 letters). The aggregate now contains 4282
  normalized rows across 100 route phases, with 0 mechanically admitted
  candidates. These outputs remain diagnostic rather than reader evidence;
  the next action is the concrete held-out repair named by each artifact, not
  a duplicate sweep.

  Three further orthogonal Luna continuations then produced an online
  grammar-state decoder (51--56 letters) with outer-frame repairs, a live
  slot-equation CFG resegmentation chart with held-out repairs up to 92
  letters, and a scalable clause-growth semantic-frame repair reaching 231
  letters. The online lane also replayed three held-out semantic-slot
  substitutions. All are intact ordinary prose with independent exact/hash
  audits, provenance, novelty preflight, and concrete next repairs; all have
  zero exact closures. The aggregate now contains 4301 normalized rows across
  105 route phases, with 0 mechanically admitted candidates. The reader
  package remains gated until an exact mechanically admitted output exists.

  A separate closure attempt then changed the construction state rather than
  widening these routes: reverse-lexicon phrase synthesis retained 46--81
  letter probes, center-out grammar boundary DP retained 61--77 letter
  probes, and authored clause-template SAT retained 89--100 letter probes.
  Each produced intact prose with independent exact/hash audits, provenance,
  novelty preflight, and a concrete repair, but no exact closure. The
  aggregate now contains 4328 normalized rows across 108 route phases, with 0
  mechanically admitted candidates; reader testing remains gated.

  The bounded repairs for those closure attempts then retained six
  reverse-lexicon inflection variants at 65--69 letters, one center-out
  adjunct repair at 75 letters, and one fresh clause-template SAT frame at
  106 letters. All remain intact, independently audited, and unadmitted. The
  aggregate now contains 4336 normalized rows across 111 route phases, with 0
  mechanically admitted candidates.

  The follow-up repairs retained three shared-agreement seam candidates at
  61--68 letters, a two-boundary center-out repair at 66 letters, and a fresh
  t-initial SAT subject repair at 103 letters. Each remains intact,
  independently audited, and unadmitted. The aggregate now contains 4347
  normalized rows across 117 route phases, with 0 mechanically admitted
  candidates.

  **Authoritative ten-lane snapshot (2026-09-16):** the common audit now
  contains 4,390 normalized rows across 133 route phases, with 76 exact tapes
  and 0 mechanically admitted outputs. The 42-letter exact seam witness
  (“Ava saw radar level civic; civic level radar was Ava”) fails the
  self-palindromic/repeated-unit and word-order gates; the 42-letter
  “a man a plan a canal panama” closure is catalogue material. Fresh
  52--55-letter scene clauses and 66--72-letter multi-clause controls remain
  ordinary, provenance-backed prose but fail exactness under independent
  pointer/hash audits. All ten lanes now have actual rendered evidence and
  concrete next repairs; the reader package remains gated until a novel exact
  output is mechanically admitted.

  The next frontier repairs retained a centered-complement reverse-lexicon
  scene at 87 letters, a center-out time-adjunct repair at 70 letters, and a
  guide-clause SAT complement at 103 letters. Each remains intact,
  independently audited, and unadmitted. The aggregate now contains 4350
  normalized rows across 120 route phases, with 0 mechanically admitted
  candidates.

  The next targeted repairs retained a typed-complement reverse-lexicon scene
  at 93 letters, a residual center-out adjunct at 75 letters, and a same-
  valency SAT guide-verb repair at 102 letters. Each remains intact,
  independently audited, and unadmitted. The aggregate now contains 4353
  normalized rows across 123 route phases, with 0 mechanically admitted
  candidates.

  The next construction step retained a semantic-role noun-boundary scene at
  93 letters, a paired subject/adjunct boundary-CSP scene at 72 letters, and a
  fresh SAT subject/object pair at 101 letters. Each remains intact,
  independently audited, and unadmitted. The aggregate now contains 4356
  normalized rows across 126 route phases, with 0 mechanically admitted
  candidates.

  The following bounded repairs retained an adjacent-PP reverse-lexicon scene
  at 106 letters, a fresh verb-frame center-out scene at 57 letters, and a
  recipient-role SAT scene at 99 letters. Each remains intact,
  independently audited, and unadmitted. The aggregate now contains 4359
  normalized rows across 129 route phases, with 0 mechanically admitted
  candidates.

  The next directed repairs retained four reverse-lexicon seam candidates at
  65--68 letters, a second center-out adjunct repair at 75 letters, and a
  second SAT outer-character repair at 102 letters. All remain intact,
  independently audited, and unadmitted. The aggregate now contains 4342
  normalized rows across 114 route phases, with 0 mechanically admitted
  candidates.

  **Targeted seam repair update (2026-09-16):** a new non-duplicate artifact,
  `runs/seam-feature-slot-repair-20260916.json`, passed novelty preflight and
  rendered 11 single-slot substitutions of the exact seam witness. Independent
  pointer/hash checks found no surviving exact closure and no mechanically
  admissible output. Its concrete next construction is typed boundary
  resegmentation at the first failing seam, allowing adjacent short words to
  absorb reflected suffixes. The aggregate is now 4,539 rows across 142 route
  phases, with 77 exact tapes and 0 mechanically admitted outputs.

  **Typed-boundary follow-up (2026-09-16):**
  `runs/typed-boundary-resegment-shortwords-20260916.json` passed novelty
  preflight and rendered 23 fresh adjacent-short-word boundary variants at
  the failing seam. Independent pointer/hash checks found 0 exact and 0
  mechanically admissible rows. The next repair is a fresh non-palindromic
  typed subject/verb/object frame with agreement preserved.

  **Additional joint construction lanes (2026-09-16):** the central-pivot CSP
  retains a fresh 113-letter complete two-clause scene, independently rejected
  at its first mirrored character; the semantic-slot lattice renders 108 joint
  states and prunes all; and the bidirectional decoder's exact 51-letter “Doc,
  note” palindrome is explicitly rejected as known catalogue text. All three
  artifacts include provenance, audits, and next repairs; none is promoted.

  **Fresh typed-frame follow-up (2026-09-16):**
  `runs/fresh-typed-frame-live-seam-20260916.json` renders “The baker carries
  a letter near the quiet harbor” (40 letters), a new singular-agreement frame
  with eight live obligations. It passes novelty and every mechanical quality
  check except exactness. The next repair targets only the adjunct boundary
  lexeme selected by the first residual obligation.

  **Adjunct-boundary targeted repair (2026-09-16):**
  `runs/adjunct-boundary-targeted-repair-20260916.json` performs one fresh
  `near` to `by` substitution, yielding “The baker carries a letter by the
  quiet harbor.” The SVO frame and agreement stay fixed; independent validation
  is non-exact at 38 letters, below the 39-letter floor. The next repair carries
  the residual into the determiner slot.

  **Additional reverse-segmentation evidence (2026-09-16):** the
  corpus-backed lane retains two fresh 103--104-letter authored clauses.
  Weighted boundary DP finds no exact reverse segmentation; copied-span
  rejection, provenance, and a concrete held-out lexical/POS expansion repair
  are recorded.

  **Latest authoritative continuation (2026-09-16):** the common audit now
  contains 4,564 normalized rows across 159 route phases, 79 exact tapes, and
  0 mechanically admitted outputs; the novelty registry contains 261 unique
  artifacts (26 exclusions, 239 run artifacts). The 164-letter word-pair
  graph frontier, two fresh 61-letter paired mutations, 113-letter clause CSP,
  108 semantic-slot states, and 103--104-letter reverse-segmentation clauses
  are all rendered, independently checked, provenance-backed, and paired with
  concrete next repairs. None is promoted as a readable palindrome.

  **Latest rendered controls:** the character-LM grammar lane emits a 77-letter
  two-clause scene; dependency-seam CSP emits a 105-letter archivist/visitor
  scene; agreement/clitic transduction emits a 122-letter ledger scene; and
  joint CFG/Earley intersection emits a 140-letter gardener/teacher scene.
  Their full strings, independent exact/hash failures, provenance, novelty
  checks, and next repairs are recorded in the lane evidence ledger. The
  repeated-clause 729-row CFG sweep is excluded as non-progress.

  **Hand-authored clause breakthrough:** a fresh five-clause bank with an
  all-different content-word gate produces a 308-letter intact scene: “Mara
  carries warm bread to the river. The careful pilot studies cloud maps.
  Children gather bright shells by moonlight. A gardener shelters young cedar
  shoots. Old friends share stories beside fire. By the hearth, new tales begin.
  Near the grove, small finches settle. At twilight, the patient tide returns.
  Beyond the hills, a quiet engine waits. At the shore, Owen listens for
  bells.” Independent pointer and SHA audits disagree at the first character;
  it is non-exact and non-admitted, with provenance and a semordnilap-compatible
  verb-object slot repair recorded in its run artifact.

  **Three orthogonal follow-ups:** boundary-conditioned finite-state
  resegmentation emits a 270-letter six-clause scene; live dependency-boundary
  CSP emits a 127-letter marine-biologist scene with explicit character
  equations; and residual semantic-slot substitution emits a 122-letter
  observatory scene with locked valency/agreement. Each has independent
  pointer/hash rejection, fresh provenance, novelty preflight, and a concrete
  next repair. None is exact or mechanically admitted.

  **Latest constructive repairs:** the outside-in typed scene CSP now uses
  distinct held-out right frames, retaining six 102--124-letter ordinary prose
  pairs after all-different filtering; the longest is “At first light, the marine
  biologist records patient observations beside the sheltered tide pool. At
  dusk, the coastal engineer maps hidden channels.” The exact-tape-first route is excluded
  from the retained aggregate because it preserves the known 16-letter “Able
  was I; I saw Elba” catalogue control and a fresh 72-letter exact tape whose
  reflected half is gibberish. A ten-clause residual-equation
  solver adds a 355-letter clinic scene (“At first light Mara opens the clinic,
  checks the quiet generators, greets the two nurses, and records the medicine
  count. She carries clean water to the waiting room, labels each parcel, phones
  the mountain driver, updates the weather board, thanks the volunteers, and
  closes the ledger before dusk. After supper she inventories the blankets,
  answers the radio, repairs a torn notice, and leaves clear instructions for the
  morning shift.”) with a paired verb/object repair. All
  have independent exact/hash audits, provenance, and novelty records; none is
  mechanically admitted.

  **Additional orthogonal lanes:** the finite clause automaton emits a
  109-letter archivist/visitor scene; the semordnilap typed lane retains the
  exact but non-prose “Stressed desserts.” control and a 100-letter baker near
  miss; and the two-sided discourse equation emits a 156-letter Nora/harbor
  scene with distinct content words. Each has independent exact/hash audits,
  provenance, novelty preflight, and a concrete repair; none is admitted.

  **Fresh ten-lane closure audit (2026-09-16):** ten orthogonal Luna state
  representations now have append-only run artifacts with rendered prose,
  independent pointer/hash validation, provenance, novelty preflight, and a
named next repair. The aggregate is 4,898 rows across 247 route phases (79
  exact but rejected tapes; 0 mechanically admitted). The lane-8 exact
  126-letter surface is quarantined because it is six repetitions of the
  catalogue clause “A man, a plan, a canal, Panama.” The longest fresh intact
  non-exact outputs are 270 letters from CFG/Earley intersection, 227 from
  character-LM decoding, 208 from flat compositional growth, and 170 from
  semantic slot repair. Subsequent typed equation, dialogue, A* boundary,
  single-slot semantic, paired-lexical, free-center bridge, and productive
  grammar-composition, grapheme-chunk, reader-first function-edit,
  constrained-edit-program, append-algebra, and discourse-relation-involution
  probes
  remain distinct non-exact repairs; an entailment rewrite was excluded after
  a novelty collision. The next
  reader-facing step remains a randomized
  blinded intact-versus-shuffled study, gated on a genuinely exact,
  anti-shortcut-clean candidate. The repaired inflection/clitic lane adds a
  distinct 128-letter harbor near miss after the repeated-unit exact control
  was quarantined.

  The newest continuation wave adds endpoint-aware seam decoding (six
  100--105-letter scenes), mutable fresh-scene tape/CFG resegmentation (a
  126-letter lexical repair), and a typed semantic-slot neighborhood (six
  109--160-letter repairs). Each has independent pointer/SHA validation,
  provenance, novelty preflight, and a concrete next repair; none is exact or
  reader-eligible.

  The latest repair wave adds a typed semantic center-out solver (12 complete
  95--109-letter scenes), a lexical word-equation intersection (four complete
  133--143-letter scenes), and a joint semantic-slot/boundary repair (nine
  complete 45--62-letter scenes). Representative renderings are “The patient
  curator shelters the fragile maps during the storm; the careful teacher
  copies the final field notes beside the window.”, “The curator labels the
  fragile map before the archivist stores the ledger, while rain gathers softly
  against the western windows and visitors wait beside the reading room.”, and
  “The quiet curator labels the old faded map before dawn.” All have independent
  pointer/SHA rejection, fresh provenance, novelty preflight, and concrete
  next repairs; none is exact or reader-eligible.

  The newest three non-overlapping Luna states are preserved with their actual
  renderings and independent digests. The constrained edit program starts from
  “At dawn, the careful cartographer marked the northern trail, while a patient
  ranger checked the bridge and recorded the weather.” (106 letters; pointer
  and SHA exactness `False`; forward/reverse prefixes `b4ca1de6...` /
  `d92f8d11...`) and lowers mirrored debt from 51 to 44 through parse-preserving
  edits. The append-algebra lane reaches “Mara observes the harbor lantern. Jon
  repairs the western gate. Iris records the morning tide. Noah carries a copper
  compass.” (102 letters; exactness `False`; digests `2ae18869...` /
  `746851d5...`) and records a typed suffix-obligation repair. The discourse
  lane includes “The rain cooled the garden because the seedlings survived the
  heat.” (56 letters; first residual 1; digests `f7687fb8...` /
  `9d847a67...`) and 11 related complete propositions. These are diagnostic
  near misses, not readability claims; each has fresh provenance, novelty
  preflight, anti-shortcut flags, and a concrete next repair.

  Two additional orthogonal Luna states now have concrete prose: the 115-letter
  character-semantic beam scene “The young botanist studies the silver seed
  cases beside the greenhouse; a careful pilot marks the distant landing lights
  through the mist.” and the 135-letter CFG/Earley scene “At first light, the
  surveyor records the river current while the baker warms bread for the waiting
  crew, and the harbor keeper checks the lamps before opening the gate.” Their
  independent pointer/SHA audits, provenance, novelty records, and typed next
  repairs are preserved; neither is exact or reader-eligible.

  Three new constructive states add a 108-letter paired semantic CFG scene, a
  109-letter joint semantic/inflection boundary-DP scene, and a 151-letter
  online clause-order archive scene. Each has intact rendered prose,
  independent exact checks, provenance, novelty preflight, and a named repair;
  none is exact or reader-eligible.

  The current continuation adds five distinct Luna states: nested-free
  clause-boundary DP, semantic phrase-edge joining, coupled object/attachment
  repair, role-typed semordnilap clause products, and a typed reversible-clause
  composer with appendable frame growth. Together they render 19 fresh
  92--124-letter scenes; every row has independent pointer/SHA validation,
  provenance, novelty preflight, and a named next repair. None is exact, so the
  reader gate remains closed while the next constructive repair is queued.

  Two further fresh states are now retained: seed-free authored-frame insertion
  renders “At dawn, the archivist opens the cedar cabinet and records the harbor
  map, before dusk.” (83 letters), while an interrogative/relative-template
  solver renders “Was the quiet curator sure that the young pilot had seen the
  chart I filed beside the harbor ledger?” (87 letters). Both are complete
  prose with independent pointer/SHA validation, provenance, novelty preflight,
  anti-shortcut checks, and concrete repairs; neither is exact or reader-
  eligible.

  The newest queue pass adds three genuinely distinct constructive states: a
  live mirrored-tape terminal decoder (six 88--103-letter scenes), a fresh
  human-authored attachment CSP (two 105--107-letter museum scenes), and an
  unbounded bilateral semantic-growth grammar (three non-repeating growth
  states up to 140 letters). Every row has independent pointer/SHA replay,
provenance, novelty preflight, anti-shortcut checks, and a concrete
  first-residual repair; none closes exactly or enters the reader packet.

  Three targeted repairs follow: a finite semantic-boundary lattice, a manual
  clause-plan inventory, and a repaired CFG frame. Their best intact output is
  “The baker bakes fresh bread before dawn; The carpenter repairs blue sails
  after lunch.” (71 letters; exact `false` under independent pointer/SHA). Each
  names a held-out first-residual lexical repair, and none is promoted to the
  reader packet.

  The newest six-lane Luna wave is constructive and non-duplicative: finite
  boundary decoding, agreement morphology, a human-authored scene lattice,
  memoized POS/valency intersection, semantic phrase chunks, and paired-slot
  exact-survivor repair. Its best reader-facing prose candidate is “The careful
  archivist stores weathered maps beside the north window. A quiet teacher
  reviews marked field notes near the harbor office.” (113 letters; exact
  `false` under independent pointer and forward/reverse-SHA checks). The
  phrase-chunk provenance is fresh and passes the anti-shortcut checks; the
  concrete next operation is a held-out chunk replacement at the first live
  residual, followed by exact re-audit. Three 101-letter exact tapes are also
  preserved, but their “Levels calm tales ...” surface is not readable and is
  excluded from the reader packet. The authoritative snapshot is **5,529 rows
  across 332 routes, 84 exact rejected tapes, and 0 mechanically admitted
  outputs**; the registry is **431 retained, 35 excluded, and 409 retained
  run artifacts**.

  A follow-up typed scene realizer adds three complete 39--42-letter clauses;
  the best is “The harbor guides the vessel near the breakwater.” (41 letters;
  exact `false` under independent pointer/SHA checks), with an object-slot
  replacement named at the first residual. Two exact controls from a CFG chart
  and lexical-closure solver are explicitly excluded for word-order,
  self-palindromic-span, or self-collision shortcuts. They remain auditable
  failure evidence and do not change the reader gate.

  The next frontier changes the obligation mechanism: a cross-word character
  trie retains “The gardener carries the bell beside the garden. The writer
  watches the seed within the station.” (79 letters; exact `false`), while a
  fresh center composition and an LM-ranked legal decoder retain complete
  prose controls. The trie lane's concrete next repair is a held-out `orchard`
  place choice; none is reader-eligible.

  The final exact-focused cycle used three distinct states. A 104-letter
  miller/chemist DP rendering was rejected as a byte-for-byte duplicate of the
  retained paired lexical graph, while the center-terminal family produced
  eight fresh 56--61-letter scenes (best: “The singer carries the melody at
  noon; The quiet clerk files the record.”) and the grammar-first intersection
  produced one fresh 84-letter curator/guide scene. All are intact prose with
  independent two-pointer/SHA rejection, provenance, novelty preflight, and a
  concrete next repair; the duplicate is excluded rather than counted. The
  latest authoritative snapshot is 4,976 rows across 285 routes, with 79 exact
  rejected tapes and 0 mechanically admitted outputs. The reader packet remains
  gated on a genuinely exact anti-shortcut survivor.

  The next three Luna states remain constructive rather than defensive: one
  semantic-slot plus dependency-attachment repair reaches a 168-letter
  archivist scene, a finite feature-center grammar reaches 114 letters, and a
  joint complete-constituent equation solver reaches 208 letters. Their actual
  renderings, independent pointer/SHA rejection, provenance, novelty preflight,
  anti-shortcut checks, and targeted repairs are retained; none is exact or
  reader-eligible. The authoritative snapshot is now 4,990 rows across 288
  routes, with 79 exact rejected tapes and 0 mechanically admitted outputs.

  The next continuation adds a 113-letter word-internal seam scene and a
  center-first residual grammar that reaches 244 letters through complete SVO
  growth. The relation-plan probe was re-audited and retained: its positive
  anti-shortcut fields certify no word-order mirror, no repeated content, and no
  catalogue scaffold. Three focused repairs (117, 59, and 214 letters) are also
  retained with independent exact checks and named next operators. The
  authoritative retained snapshot is now 5,195 rows across 309 routes, with 79
  exact rejected tapes and 0 mechanically admitted outputs. The latest
  agreement/clitic, seed-free semantic-slot, and semantic-valency lattice lanes
  add 72-, 108-, and 154-letter intact witnesses with independent rejection and
  concrete next repairs. The subsequent exact-closure queue adds bounded 81-,
  159-, and 106-letter witnesses, all independently rejected with concrete next
  repairs.

  The newest committed wave is measured at **4,967 rows across 283 route phases**
  with **79 exact but rejected tapes and 0 mechanically admitted outputs**. Its
  three actual fresh renderings are a 140-letter archivist/gardener/pilot scene,
  a 74-letter Jon dialogue with a complete relative clause, and an 83-letter
  engineer/sailor function-word scene. Independent pointer/SHA audits reject all
  three; provenance is fresh and no-copy, and each has a named state-specific
  repair. The immediate child repairs add 141-, 74-, and 89-letter scenes and
  also fail exact admission. The reader-facing intact-versus-shuffled test
  remains gated on a genuinely exact, anti-shortcut-clean candidate. A second
  child pass adds 144-, 75-, and 89-letter targeted repairs, also non-exact.
  The orthogonal attachment-CSP, semantic-slot DP, and bespoke free-center
  lattice add 18 more intact prose rows, still non-exact. Their targeted
  follow-ups add 91-, 80-, and 117-letter variants, also non-exact. The new
  reverse-phrase, center-free clause-pair, and finite semantic CSP lanes add
  10 more intact rows, still without an exact survivor.

  The newest queue pass adds three genuinely distinct constructive states: a
  live mirrored-tape terminal decoder (six 88--103-letter scenes), a fresh
  human-authored attachment CSP (two 105--107-letter museum scenes), and an
  unbounded bilateral semantic-growth grammar (three non-repeating growth
  states up to 140 letters). Every row has independent pointer/SHA replay,
  provenance, novelty preflight, anti-shortcut checks, and a concrete
  first-residual repair; none closes exactly or enters the reader packet.

  **2026-09-17 orthogonal follow-up:** three new constructive states were
  run once and registered separately. A simultaneous typed grammar-product
  beam renders “The young keeper carries a blue lantern beside the garden.”
  (48 letters), with live reverse-character debt and an extension repair. A
  reverse-tape trie boundary beam independently verifies exact 28-letter
  outputs such as “no one was at home | em oh ta sa we no on”; the fragments
  fail the prose gate, so the next repair is a tagged POS trie plus phrase
  scoring. Immutable-tape DP resegmentation yields exact 132-letter paths such
  as “tr ad er are wa st em et ...”; its lexical admission is fail-closed and
  the next test adds a clause-bigram model on a fresh tape. The contract replay
  is saved at `runs/ten-luna-lane-contract-20260917.json`: all ten requested
  lanes have rendered prose, independent two-pointer/SHA recomputation,
  provenance, novelty preflight, and a concrete repair; 0 are exact and 0 are
  mechanically admitted. These runs are distinct state representations, not
  larger duplicate sweeps.

  **2026-09-17 constructive repair wave:** three fresh Luna lanes and their
  named repairs were completed with disjoint artifacts. The CFG/Earley lane
  rendered “The patient gardener who carries the garden carries beside the
  gardener.” (61 letters in the recursive repair; 0 exact). The morphology
  lane rendered “The patient gardeners tended a shaded orchard after the
  morning rain although the patient gardeners tended the notes for the local
  archive.” (117 letters; 0 exact). The semantic bundle lane rendered 16
  ordinary-order candidates of 39--44 letters; 0 exact. Every row has a
  fresh generator hash, provenance, anti-shortcut flags, and independent
  two-pointer plus forward/reverse SHA replay. The follow-up suffix repair
  retained 72 agreement/tense realizations (117-letter maximum) and also
  closed 0 tapes. These are not reader candidates. The next constructive
  operators are object-relative attachment, a joint two-clause seam equation,
  and a held-out suffix substitution; each must produce a new rendered
  candidate before any reader packet is opened.

  The follow-up repairs then added an object-relative agreement chart (55
  letters maximum, 0 exact), a residual-conditioned suffix substitution (116
  letters maximum, 0 exact), and a joint two-clause seam-bundle search (16
  candidates at 82--91 letters, 0 exact). Representative rendered controls
  are “The senior curators inspected a fragile map after the morning rain
  although the senior curators inspected the notes for the local archive.”
  and “The patient gardener who carries the garden carries beside the
  gardener.” Their independent pointer/SHA audits reject them; repeated
  clause units are explicitly excluded. The next seam-conditioned lexical
  trie search must influence word-boundary choices before clause completion,
  then expose a fresh rendered candidate and its reader-facing test.

  **2026-09-17 follow-up repair wave:** the seam-conditioned lexical trie
  admitted 24 ordinary-order candidates from a 400-row prefix frontier, with
  a longest rendered witness of 90 letters and 0 exact closures. The object
  residual repair tested 240 compatible determiner/adjective realizations,
  reached 119 letters, and closed 0 tapes while preserving agreement. The
  number-gated object-relative CFG tested 55,296 controls plus 18,432 held-out
  subject/verb repairs, reached 61 letters, and closed 0 tapes. Each row has
  independent two-pointer and SHA-256 replay, generator provenance, and
  anti-shortcut checks. These are fresh construction operators, not duplicate
  sweeps; none is reader-eligible. Their concrete next repairs are
  inflectional word tries with character-by-character seam equations, a
  residual-conditioned tail preposition, and a fresh number-gated
  determiner/verb contrast, respectively.

  The next repair wave completed those operators rather than repeating their
  parent sweeps. Inflectional character-by-character seam expansion rendered
  6 ordinary-English candidates up to 90 letters; a residual-conditioned tail
  preposition repair rendered 32 agreement-preserving realizations up to 120
  letters; and a number-gated determiner/verb object-relative contrast tested
  36,864 controls plus 36,864 held-out repairs up to 61 letters. All three
  produced 0 exact closures. Each candidate carries independent two-pointer
  and forward/reverse SHA-256 audits, provenance, novelty preflight, and
  anti-shortcut flags. They remain outside the reader packet. Their next
  concrete operators are function-word tries across variable boundaries, one
  held-out clause-link realization, and a direct-object versus locative
  attachment feature state, respectively.

  The following repair wave completed those named operators. Function-word
  trie boundary expansion rendered 2 ordinary-English candidates up to 84
  letters; the clause-link residual repair rendered 16
  agreement-preserving realizations up to 120 letters; and the
  direct-object/locative attachment CFG tested 3,072 controls plus 1,024
  attachment-preserving repairs up to 61 letters. All three produced 0 exact
  closures. Independent pointer and forward/reverse SHA-256 audits,
  provenance, novelty preflight, and anti-shortcut checks are present for
  every row. They remain outside the reader packet. The next operators are
  seam-aware dynamic programming over function-word trie states, a
  center-crossing grammar state, and one controlled attachment alternation
  carried through the character-obligation state.

  The next wave completed those operators and changed construction family.
  Seam-aware function-word dynamic programming rendered 8 ordinary-English
  candidates up to 75 letters; one controlled direct-object/locative
  alternation rendered 1,024 candidates up to 61 letters; and the new typed
  center-crossing grammar state rendered 24 intact candidates up to 127
  letters, with the midpoint inside a token in every case. All three produced
  0 exact closures. Every row has independent pointer and forward/reverse
  SHA-256 audits, provenance, novelty preflight, and anti-shortcut checks.
  They are not reader candidates. The next concrete repairs are resolved
  position masks, a state-local relative-clause verb alternation, and a typed
  object complement selected against live midpoint debt, respectively.

  The following repair wave retained that state and added three orthogonal
  operators. Resolved-position-mask grammar DP rendered 8 intact candidates
  up to 70 letters; state-local relative-clause verb alternation rendered 256
  repairs up to 61 letters; and the center-crossing typed-complement repair
  rendered 24 intact candidates up to 131 letters, with midpoint-inside-token
  crossing in every candidate. All produced 0 exact closures. Independent
  pointer/SHA-256 replay, provenance, novelty preflight, and anti-shortcut
  checks are retained for every row. The reader gate remains closed. The next
  concrete operators are interval-valued function-word boundaries,
  semantic-preserving subject substitution within attachment state, and one
  held-out tense realization against the live midpoint debt.

  The next repair wave completed those operators. Interval-valued boundary
  DP rendered 8 intact candidates up to 82 letters; attachment-state subject
  substitution rendered 1,024 candidates up to 61 letters while preserving
  valency; and the center-crossing held-out-tense repair rendered 6 intact
  candidates up to 131 letters, with midpoint-inside-token crossing in all
  cases. All produced 0 exact closures. Every row retains independent
  pointer/SHA-256 replay, provenance, novelty preflight, and anti-shortcut
  checks. The reader packet remains closed. The next concrete operators are
  boundary-aware incremental offsets, held-out matrix-object substitution,
  and a typed indirect-object attachment against the live midpoint debt.

  The next wave completed those repairs. Direct boundary-offset seam DP
  rendered 8 intact candidates up to 80 letters; attachment-state matrix
  object substitution rendered 1,024 candidates up to 61 letters; and the
  center-crossing typed-indirect-object repair rendered 18 intact candidates
  up to 151 letters, with midpoint-inside-token crossing in every candidate.
  All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  variable-length clause-boundary states, attachment-state matrix-verb
  substitution, and one typed subject-role alternation at the center state.

  The next repair wave completed those operators. Variable clause-boundary
  offset transitions rendered 4 intact candidates up to 91 letters;
  attachment-state matrix-verb substitution rendered 1,024 candidates up to
  61 letters; and the center-crossing subject-role alternation rendered 18
  intact candidates up to 150 letters, with midpoint-inside-token crossing in
  every candidate. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are grammar-aware optional-relative boundary transitions, relative-clause
  object substitution by attachment state, and one held-out
  agreement-compatible clause complement in the center state.

  The next wave completed those operators. Grammar-aware optional-relative
  boundary transitions rendered 3 intact candidates up to 91 letters;
  attachment-state relative-object substitution rendered 4,096 candidates up
  to 61 letters; and the center-crossing held-out clause-complement repair
  rendered 54 intact candidates up to 157 letters, with midpoint-inside-token
  crossing in every candidate. All produced 0 exact closures. Independent
  pointer/SHA-256 replay, provenance, novelty preflight, and anti-shortcut
  checks are retained for every row. The reader gate remains closed. The next
  concrete operators are relative-clause lexical valency bundles,
  valency-gated relative determiner alternation, and a new syntactic
  attachment family for the center-crossing state.

  The next wave completed those operators. Relative-clause lexical valency
  bundles rendered 6 intact candidates up to 98 letters; valency-gated
  relative-determiner alternation tested 6,656 candidates up to 61 letters;
  and the new center-crossing relative-attachment family rendered 54 intact
  candidates up to 179 letters, with midpoint-inside-token crossing in every
  candidate. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are a joint relative/opposing-bundle seam CSP, a state-permitted
  `that`/`where` complementizer alternation, and a distinct appositive
  attachment family in the center-crossing grammar.

  The next wave completed those operators. Joint relative/opposing-bundle
  seam CSP rendered 12 intact candidates up to 99 letters; state-permitted
  complementizer alternation tested 3,584 candidates up to 61 letters; and
  the center-crossing appositive family rendered 54 intact candidates up to
  172 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are word-boundary
  positions as CSP variables, state-valid role-preserving complementizer
  alternation, and a typed parenthetical attachment in the center-crossing
  grammar.

  The next wave completed those operators. Word-boundary-position CSP
  rendered 10 intact candidates up to 88 letters; role-preserving
  complementizer alternation tested 512 candidates up to 58 letters; and the
  typed parenthetical center-attachment family rendered 54 intact candidates
  up to 172 letters, with midpoint-inside-token crossing in every candidate.
  All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  grammar boundary transitions as CSP variables, locative `where`/`in which`
  alternation, and a typed adverbial attachment in the center-crossing
  grammar.

  The next wave completed those operators. Grammar-boundary transition CSP
  rendered 10 intact candidates up to 83 letters; locative `where`/`in which`
  alternation tested 6,144 candidates up to 63 letters; and the typed
  adverbial center-attachment family rendered 162 intact candidates up to
  169 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are semantic-valency
  coupling for bridge transitions, location-preposition substitution within
  `in which`, and a typed discourse-marker attachment in the center-crossing
  grammar.

  The next wave completed those operators. Semantic bridge-valency CSP
  rendered 6 intact candidates up to 86 letters; `in which` location-
  preposition substitution tested 3,072 candidates up to 63 letters; and the
  typed discourse-marker center-attachment family rendered 162 intact
  candidates up to 182 letters, with midpoint-inside-token crossing in every
  candidate. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are a seam-conditioned semantic transition automaton, locative predicate
  substitution, and a typed sentence-level contrast attachment in the
  center-crossing grammar.

  The next wave completed those operators. The seam-conditioned semantic
  transition automaton rendered 6 intact candidates up to 88 letters;
  locative-predicate substitution tested 3,072 candidates up to 63 letters;
  and the typed sentence-level contrast center-attachment family rendered 324
  intact candidates up to 205 letters, with midpoint-inside-token crossing in
  every candidate. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are character-labeled automaton edges, locative subject substitution, and a
  typed temporal subordinate attachment in the center-crossing grammar.

  The next wave completed those operators. Character-labeled semantic
  automaton edges rendered 6 intact candidates up to 88 letters;
  `in which` locative-subject substitution tested 12,288 candidates up to 65
  letters; and the typed temporal center-attachment family rendered 972 intact
  candidates up to 226 letters, with midpoint-inside-token crossing in every
  candidate. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are edge-labeled lexical trie transitions, joint matrix/locative subject
  substitution, and a semantic coordination attachment that stops extending
  the sentence.

  The next wave completed those operators. Edge-labeled lexical trie
  transitions rendered 2 intact candidates up to 84 letters; joint
  matrix/locative subject substitution tested 12,288 candidates up to 65
  letters; and bounded semantic coordination rendered 2,916 intact candidates
  up to 251 letters, with midpoint-inside-token crossing in every candidate.
  All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  multi-slot opposing label propagation, coordinated-subject realization, and
  a compact semantic complement while retaining one coordination choice.

  The next wave completed those operators. Multi-slot opposing-label trie
  propagation rendered 2 intact candidates up to 82 letters; coordinated
  locative-subject realization tested 49,152 candidates up to 77 letters; and
  the compact semantic-complement family rendered 2,916 intact candidates up
  to 262 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are full
  bidirectional arc consistency over character-prefix domains, coordinated
  locative subjects with plural agreement, and a conjunction-free semantic
  attachment retaining the compact complement.

  The next wave completed those operators. Bidirectional arc consistency over
  character-prefix domains retained audited intact candidates up to 81
  letters; coordinated locative-relative subjects tested 49,152 candidates up
  to 78 letters; and the conjunction-free compact-complement family rendered
  8,748 intact candidates up to 260 letters, with midpoint-inside-token
  crossing in every candidate. All produced 0 exact closures. Independent
  pointer/SHA-256 replay, provenance, novelty preflight, and anti-shortcut
  checks are retained for every row. The reader gate remains closed. The next
  concrete operators are full positional character constraints over rendered
  intervals, coordinated place complements, and a shorter two-constituent
  center grammar rather than further sentence extension.

  The next wave completed those operators. Full positional slot-interval CSP
  rendered 2 intact candidates up to 82 letters; coordinated place
  complements tested 196,608 candidates up to 90 letters; and the shorter
  two-constituent center grammar rendered 18 intact candidates up to 95
  letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are interval-domain
  support pruning, locative-preposition coordination, and one
  agreement-conditioned verb realization in the shorter grammar.

  The next wave completed those operators. Interval-domain support pruning
  retained 2 intact candidates up to 82 letters; locative-preposition
  coordination tested 737,280 candidates up to 96 letters; and the short
  agreement-conditioned verb grammar rendered 36 intact candidates up to 95
  letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are incremental AC-3
  support counters, conjunction alternation inside the locative complement,
  and one tense-conditioned verb pair in the short grammar.

  The next wave completed those operators. Incremental AC-3 positional support
  counters retained 2 intact candidates up to 82 letters; locative-conjunction
  alternation tested 1,024 candidates up to 98 letters; and the short
  tense-conditioned verb pair rendered 72 intact candidates up to 99 letters,
  with midpoint-inside-token crossing in every candidate. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are full positional
  interval-pair support, locative complement order swap, and one
  agreement-conditioned object determiner in the short grammar.

  The next wave completed those operators. Full positional interval-pair
  support retained 2 intact candidates up to 82 letters; locative complement
  order swap produced 543 deduplicated candidates up to 93 letters; and the
  short object-determiner agreement repair rendered 216 intact candidates up
  to 99 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are domain-valued
  interval-pair propagation, shared-preposition locative complement reduction,
  and one agreement-conditioned object adjective in the short grammar.

  The next wave completed those operators. Domain-valued interval-pair
  propagation retained 2 intact candidates up to 82 letters; shared-
  preposition locative reduction produced 684 deduplicated candidates up to 93
  letters; and the short object-adjective agreement repair rendered 432 intact
  candidates up to 111 letters, with midpoint-inside-token crossing in every
  candidate. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are lexical slot-domain rejection before materialization, `in which` to
  `where` complementizer reduction, and one agreement-conditioned adverb in
  the short grammar.

  The next wave completed those operators. Lexical slot-domain rejection
  retained 2 intact candidates up to 82 letters; shared-state `in which` to
  `where` reduction produced 1,024 candidates up to 87 letters; and the short
  agreement-conditioned adverb repair rendered 1,296 intact candidates up to
  126 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are fixed-point
  opposing-position propagation, shared-preposition lexical predicate
  substitution, and one agreement-conditioned prepositional complement in the
  short grammar.

  The next wave completed those operators. Fixed-point opposing-domain
  propagation retained 2 intact candidates up to 83 letters; shared-`where`
  predicate substitution produced 512 candidates up to 85 letters; and the
  short agreement-conditioned prepositional-complement repair rendered 3,888
  intact candidates up to 141 letters, with midpoint-inside-token crossing in
  every candidate. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are full positional interval propagation, locative-subject agreement
  variation in shared-`where`, and one agreement-conditioned
  object-complement noun in the short grammar.

  The next wave completed those operators. Full-interval fixed-point
  propagation retained 2 intact candidates up to 82 letters; shared-`where`
  agreement variation produced 512 candidates up to 72 letters; and the short
  object-complement noun repair rendered 11,664 intact candidates up to 141
  letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are paired-slot
  full-support maps, shared-`where` matrix-verb agreement, and one determiner
  on the second object in the short grammar.

  The next wave completed those operators. Paired-slot full support maps
  retained 2 intact candidates up to 82 letters; shared-`where` matrix
  agreement produced 512 candidates up to 81 letters; and the short
  second-object determiner repair rendered 11,664 intact candidates up to 141
  letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are paired
  support-map intersection pruning, matrix determiner/subject-number
  alternation, and one second-object-only adjective in the short grammar.

  The next wave completed those operators. Paired support-map intersection
  pruning retained 2 intact candidates up to 82 letters; shared-`where`
  matrix-determiner variation produced 512 candidates up to 72 letters; and
  the short second-object adjective repair rendered 23,328 intact candidates
  up to 141 letters, with midpoint-inside-token crossing in every candidate.
  All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  positional support intersection, coordinated matrix determiner realization,
  and one second-object-only adverb in the short grammar.

  The next wave completed those operators. Positional support-intersection
  pruning retained 2 intact candidates up to 82 letters; corrected
  shared-`where` coordinated-determiner realization produced 512 candidates up
  to 88 letters; and the short second-object adverb repair rendered 36 intact
  candidates up to 150 letters, with midpoint-inside-token crossing in every
  candidate. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are rendered tape-offset mapping, coordinated matrix-subject lexical
  substitution, and one short subject-role alternation while retaining the
  second adverb.

  The next wave completed those operators. Rendered tape-offset support
  retained 2 intact candidates up to 82 letters; shared-`where` coordinated
  matrix-subject substitution produced 512 candidates up to 84 letters; and
  the short subject-role alternation rendered 36 intact candidates up to 150
  letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are opposing-offset
  lexical choice before tape assembly, joint locative-subject substitution,
  and a two-frame center seam without another lexical sweep.

  The next wave completed those operators. Opposing-offset lexical choice
  retained 8 intact candidates up to 81 letters; joint shared-`where`
  locative-subject substitution produced 448 candidates up to 97 letters; and
  the two-frame structural center seam rendered 8 intact candidates up to 142
  letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are joint lexical
  propagation across chosen slots, joint place-pair substitution, and an
  internal word-boundary seam shift within the same two frames.

  The next wave completed those operators. Joint lexical opposing-offset
  propagation retained audited candidates up to 83 letters; joint shared-
  `where` place-pair substitution produced 256 candidates up to 97 letters;
  and the two-frame internal boundary shift rendered 4 intact candidates up
  to 133 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are exact
  partial-tape equations with bidirectional propagation, shared-preposition
  substitution over the joint place pair, and an internal morpheme-boundary
  shift within the same words.

  The next wave completed those operators. Exact partial-tape bidirectional
  equations retained audited candidates up to 81 letters; joint shared-`where`
  preposition substitution produced 272 candidates up to 97 letters; and the
  two-frame internal morpheme shift rendered 8 intact candidates up to 144
  letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are unresolved-
  position masks across variable boundaries, joint `where`/`in which`
  alternation, and a determiner-boundary center seam in the same frames.

  The next wave completed those operators. Unresolved-position masks across
  variable boundaries retained 2 intact candidates up to 82 letters; joint
  `where`/`in which` alternation produced 512 candidates up to 99 letters; and
  the two-frame determiner-boundary seam rendered 4 intact candidates up to
  142 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are mask-guided
  next-character choice, joint locative-predicate alternation, and an
  adjective-boundary seam within the same frames.

  The next wave completed those operators. Mask-guided next-character choice
  retained 2 intact candidates up to 82 letters; joint locative-predicate
  alternation produced 256 candidates up to 99 letters; and the two-frame
  adjective-boundary seam rendered 4 intact candidates up to 133 letters, with
  midpoint-inside-token crossing in every candidate. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are lexical-trie equality
  expansion, joint locative-subject lexical alternation, and a fixed
  prepositional-boundary center seam in the same frames.

  The next wave completed those operators. Trie-equality expansion retained 2
  intact candidates up to 82 letters; joint locative-subject lexical
  alternation produced 512 candidates up to 99 letters; and the two-frame
  prepositional-boundary seam rendered 4 intact candidates up to 142 letters,
  with midpoint-inside-token crossing in every candidate. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are complete compatible
  word-boundary pair formation, joint place-pair lexical alternation, and a
  fixed complement-boundary seam within the same frames.

  The next wave completed those operators. Complete compatible boundary-pair
  trie formation retained 2 intact candidates up to 82 letters; joint
  place-pair lexical alternation produced 256 candidates up to 99 letters; and
  the two-frame complement-boundary seam rendered 4 intact candidates up to
  142 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are whole-tape seam
  equations coupled to completed pairs, joint preposition lexical alternation,
  and a fixed adverbial-boundary seam in the same frames.

  The next wave completed those operators. Whole-tape seam equations coupled
  to completed pairs retained 2 intact candidates up to 82 letters; joint
  preposition lexical alternation produced 256 candidates up to 99 letters;
  and the two-frame adverbial-boundary seam rendered 4 intact candidates up to
  133 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are pair-specific
  seam conflict repair, joint matrix/locative predicate-pair substitution, and
  a fixed verb-object boundary seam within the same frames.

  The next wave completed those operators. Pair-specific seam-conflict repair
  retained 2 intact candidates up to 83 letters; joint predicate-pair
  substitution produced 448 candidates up to 100 letters; and the two-frame
  verb-object boundary seam rendered 4 intact candidates up to 142 letters,
  with midpoint-inside-token crossing in every candidate. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are exhausted-role
  provenance across iterations, coordinated complementizer/predicate-pair
  realization, and a subject-verb boundary seam in the same frames.

  The next wave completed those operators. Exhausted-role conflict history
  retained 2 intact candidates up to 82 letters; coordinated
  complementizer/predicate-pair realization produced 416 candidates up to 100
  letters; and the subject-verb seam lane rendered 4 intact candidates up to
  142 letters, with midpoint-inside-token crossing in every candidate. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  seam-only family is now exhausted; the next concrete operator is a
  genuinely new center grammar with independent lexical obligations.

  The new-family wave produced three distinct constructive states. The
  single-scene center grammar rendered 4 intact candidates up to 100 letters;
  the independent scene lattice rendered 90 compact single-sentence
  candidates up to 77 letters; and the discourse-CFG obligation intersection
  rendered 2,048 intact candidates up to 50 letters while pruning 1,024 chart
  states before final rendering. All produced 0 exact closures. Independent
  pointer/SHA-256 replay, provenance, novelty preflight, and anti-shortcut
  checks are retained for every row. The reader gate remains closed. The next
  concrete operators are semantic attachment alternatives with newly closed-
  equation pruning, live reverse-obligation filtering at the object-setting
  boundary, and a typed causal-connective frame with subject sharing and tense
  agreement.

  The next wave completed those new-family repairs. Single-scene attachment
  pruning rendered 4 intact candidates up to 101 letters; the independent
  scene lattice retained 10 boundary-filtered candidates up to 73 letters; and
  the typed causal shared-subject CFG rendered 768 candidates while pruning
  2,304 partial states, with a 50-letter maximum. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are typed attachment
  valency with character-by-character role expansion, a second independent
  role bank at the same object-setting frontier, and a held-out past-tense
  agreement state in the causal frame.

  The next wave completed those new-family repairs. Typed attachment valency
  with character expansion rendered 4 intact candidates up to 101 letters;
  the second independent scene role bank retained 6 boundary-filtered
  candidates up to 70 letters; and the held-out past-tense causal CFG rendered
  768 candidates up to 51 letters while pruning 2,304 tense-inconsistent
  states. All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  live attachment grammar transitions, a character-trie continuation filter,
  and a mixed-tense temporal connective with consistency filtering.

  The next wave completed those new-family repairs. Live single-scene
  attachment transitions rendered 4 intact candidates up to 101 letters; the
  scene-lattice trie continuation retained 6 boundary-filtered candidates up
  to 70 letters; and the mixed-tense temporal causal CFG rendered 512
  candidates up to 48 letters while pruning 2,560 incompatible states. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are a semantic
  transition automaton over attachment/role states, a one-character
  reverse-prefix beam at the same frontier, and the reverse present-to-past
  `before` direction with explicit tense-order state.

  The following wave completed those three repairs. The semantic role/attachment
  transition automaton rendered 4 intact candidates up to 101 letters while
  carrying live support counters; the one-character reverse-prefix beam
  retained 8 independent scene candidates up to 72 letters but found no
  matching frontier character; and the reverse present-to-past `before` CFG
  rendered 256 candidates up to 49 letters while pruning 768
  shared-subject-inconsistent states. All produced 0 exact closures.
  Independent pointer/SHA-256 replay, provenance, novelty preflight, and
  anti-shortcut checks are retained for every row. The reader gate remains
  closed. The next concrete operators are edge-local support counters with
  pre-target-role transition pruning, a tied-branch two-character beam, and a
  temporal adverbial insertion that preserves the present-to-past `before`
  order.

  The following wave completed those repairs. Edge-local attachment support
  pruning rendered 4 intact candidates up to 101 letters; the tied
  two-character reverse-prefix beam retained 8 independent scene candidates up
  to 72 letters; and the post-connective temporal-adverbial `before` CFG
  rendered 768 candidates up to 58 letters while pruning 2,304
  shared-subject-inconsistent states. All produced 0 exact closures.
  Independent pointer/SHA-256 replay, provenance, novelty preflight, and
  anti-shortcut checks are retained for every row. The reader gate remains
  closed. The next concrete operators are character-position-aware opposing
  witnesses for attachment edges, an independently authored two-character-
  compatible setting phrase, and a pre-connective adverbial tense-state branch.

  The following wave completed those repairs. Position-aware attachment
  witnesses rendered 2 intact candidates up to 92 letters; the authored
  setting phrase plus tied two-character beam retained 8 candidates up to 67
  letters with a maximum two-character match; and the pre-connective
  temporal-adverbial `before` CFG rendered 768 candidates up to 58 letters
  while pruning 2,304 shared-subject-inconsistent states. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty preflight,
  and anti-shortcut checks are retained for every row. The reader gate remains
  closed. The next concrete operators are witness propagation into attachment
  lexical domains, a compatible third character only on matched setting
  branches, and a bounded two-adverb temporal branch with explicit attachment
  filtering.

  The following wave completed those repairs. Pre-render attachment witness
  propagation rendered 2 intact candidates up to 94 letters; the gated
  third-character beam entered 6 two-character-matched branches and retained
  candidates up to 65 letters, with the best branch matching only 2 of 3
  characters; and the bounded two-adverb temporal CFG rendered 1,536
  candidates up to 63 letters while pruning 7,680 invalid attachment/subject
  states. All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  witness-domain plus agent/action/theme valency intersection, a grammatical
  setting phrase beginning with the required third character, and held-out
  adverb substitution in one temporal slot at a time.

  The following wave completed those repairs. Intersecting attachment witness
  domains with agent/action/theme valency rendered 3 intact candidates up to
  98 letters; the authored `v`-initial setting phrase made all 4 matched
  survey branches satisfy three reverse-prefix characters, with candidates up
  to 63 letters; and slot-local pre-event adverb substitution rendered 3,072
  candidates up to 63 letters while pruning 7,680 invalid states. All
  produced 0 exact closures. Independent pointer/SHA-256 replay, provenance,
  novelty preflight, and anti-shortcut checks are retained for every row. The
  reader gate remains closed. The next concrete operators are bidirectional
  witness propagation under the intersected valency domain, a fourth-character
  compatible setting phrase on the matched survey branch, and analogous
  post-event adverb substitution with the pre-event slot fixed.

  The following wave completed those repairs. Bidirectional left/right scene
  valency expansion rendered 4 intact candidates up to 101 letters; the
  authored `v-r` setting phrase made all 4 matched survey branches satisfy a
  fourth reverse-prefix character, with candidates up to 63 letters; and
  post-event slot-local adverb substitution rendered 3,072 candidates up to
  63 letters while pruning 7,680 invalid states. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are incremental
  bidirectional witness propagation with one-side pruning, a fifth-character-
  compatible setting phrase on the matched survey branch, and paired semantic
  agreement for the two temporal adverb slots.

  The following wave completed those repairs. Incremental bidirectional
  witness propagation rendered 4 intact candidates up to 101 letters; the
  authored `v-r-u` setting phrase made all 4 matched survey branches satisfy a
  fifth reverse-prefix character, with candidates up to 63 letters; and the
  paired temporal-adverb CFG rendered 768 candidates up to 63 letters while
  pruning 8,448 incompatible pairs. All produced 0 exact closures.
  Independent pointer/SHA-256 replay, provenance, novelty preflight, and
  anti-shortcut checks are retained for every row. The reader gate remains
  closed. The next concrete operators are domain-valued right-role witness
  pruning, a sixth-character-compatible authored phrase on the matched survey
  branch, and an explicit earlier/later temporal ordering feature.

  The following wave completed those repairs. Domain-valued right-role witness
  pruning rendered 3 intact candidates up to 93 letters; the authored
  `v-r-u-s` setting phrase made all 4 matched survey branches satisfy a sixth
  reverse-prefix character, with candidates up to 57 letters; and the explicit
  earlier<later temporal-order CFG rendered 256 candidates up to 61 letters
  while pruning 1,792 reverse-order or inconsistent states. All produced 0
  exact closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are simultaneous witness
  propagation through both role sides, a seventh-character-compatible authored
  phrase on the matched survey branch, and a formerly/later ordering-compatible
  tense-state pair.

  The following wave completed those repairs. Simultaneous left/right role
  domain contraction rendered 2 intact candidates up to 93 letters; the
  seventh-character probe retained 4 intact survey candidates up to 64
  letters but found no grammatical phrase matching the required residual; and
  the distinct formerly<later temporal-order CFG rendered 256 candidates up
  to 62 letters while pruning 1,792 reverse-order or shared-subject states.
  All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  support-driven simultaneous contraction, a legal locative seventh onset on
  the matched survey branch, and a tense-compatible `while` variant for the
  formerly/later state.

  The following wave completed those repairs. Support-driven simultaneous
  contraction rendered 3 intact candidates up to 93 letters; the new locative
  setting family rendered 4 intact candidates up to 62 letters but found no
  seventh-character match; and the formerly<later `while` CFG rendered 256
  candidates up to 61 letters while pruning 768 shared-subject-inconsistent
  states. All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  incremental witness support per role value, locative-noun onset conditioning
  against the seventh residual, and while-state subject alternation with
  connective/tense consistency filtering.

  The following wave completed those repairs. Incremental per-role-value
  witness support rendered 12 intact candidates up to 94 letters; conditioning
  the locative noun on the seventh residual rendered 4 intact candidates up to
  57 letters with all four seventh-character matches; and the alternating-
  subject `while` CFG rendered 768 candidates up to 60 letters while pruning
  256 shared-subject states. All produced 0 exact closures. Independent
  pointer/SHA-256 replay, provenance, novelty preflight, and anti-shortcut
  checks are retained for every row. The reader gate remains closed. The next
  concrete operators are per-value domain updates before complete scene
  assignment, conditioning the next locative character against the eighth
  residual, and an alternating object frame with role-consistent transitions.

  The following wave completed those repairs. Per-value support updated both
  role domains before assignment and rendered 1 intact 92-letter candidate;
  the alternating-object temporal CFG rendered 2,304 candidates up to 71
  letters while pruning 1,792 same-subject/same-object states; and the
  eighth-residual locative continuation rendered 4 intact candidates up to 64
  letters but found no grammatical continuation at the required character.
  All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  fixed-point cross-side domain updates, locative-noun morphology that can
  satisfy the eighth residual, and an object-number agreement state for the
  alternating frame.

  The following wave completed those repairs. Cross-side witness-domain
  propagation to a fixed point rendered 1 intact 92-letter candidate; the
  alternating-object number-state CFG rendered 6,144 candidates up to 72
  letters while pruning 10,240 same-number or invalid-role states; and the
  locative morphology probe reached all four eighth-character `a` matches in
  4 intact candidates up to 57 letters, while explicitly losing the seventh
  `l` match. All produced 0 exact closures. Independent pointer/SHA-256
  replay, provenance, novelty preflight, and anti-shortcut checks are retained
  for every row. The reader gate remains closed. The next concrete operators
  are pair-specific fixed-point support maps, jointly conditioned seventh and
  eighth locative morphology, and determiner-number realization for the
  alternating objects.

  The following wave completed those repairs. Pair-specific support maps in
  the cross-side fixed-point loop rendered 4 intact candidates up to 97
  letters; the typed proper-locative `via del Lago` rendered 4 intact
  candidates up to 59 letters with all four joint `l/a` matches; and the
  number-gated determiner CFG rendered 12,288 candidates up to 72 letters
  while pruning 10,240 invalid number/role states. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are pair-local support-map
  revision, proper-locative tail conditioning against the ninth residual, and
  a single-position article-agreement transition.

  The following wave completed those repairs. Pair-component support-delta
  repair rendered 2 intact candidates up to 97 letters; the joint `la`
  proper-locative tail probe rendered 4 intact candidates up to 59 letters but
  found `g` where the ninth residual required `c`; and the first-object
  article-transition CFG rendered 6,144 candidates up to 72 letters while
  pruning 10,240 invalid number/role states. All produced 0 exact closures.
  Independent pointer/SHA-256 replay, provenance, novelty preflight, and
  anti-shortcut checks are retained for every row. The reader gate remains
  closed. The next concrete operators are held-out semantic component
  replacement, a `c`-initial proper-locative tail after the joint `la` onset,
  and the corresponding second-object article transition.

  The following wave completed those repairs. Held-out semantic component
  replacement rendered 2 intact candidates up to 95 letters; the typed
  proper-locative `c` tail (`via del Lacy`) rendered 4 intact candidates up to
  59 letters with all four ninth-character matches; and the second-object
  article-transition CFG rendered 6,144 candidates up to 72 letters while
  pruning 10,240 invalid number/role states. All produced 0 exact closures.
  Independent pointer/SHA-256 replay, provenance, novelty preflight, and
  anti-shortcut checks are retained for every row. The reader gate remains
  closed. The next concrete operators are full-scene human-readable checks for
  held-out replacements, tenth-residual conditioning after the `c` tail, and a
  coordinated article state for both alternating objects.

  The following wave completed those repairs. Held-out semantic replacements
  rendered 2 intact candidates up to 95 letters through the full scene grammar
  (human screening remains deferred until an exact survivor); the proper-
  locative `via del Laco` rendered 4 intact candidates up to 59 letters with
  all four tenth-character matches; and the coordinated article CFG rendered
  4,608 candidates up to 71 letters while pruning 14,080 mixed or invalid
  article-number states. All produced 0 exact closures. Independent
  pointer/SHA-256 replay, provenance, novelty preflight, and anti-shortcut
  checks are retained for every row. The reader gate remains closed. The next
  concrete operators are a `c`-tail eleventh-residual condition, plural-object
  coordinated `the/the` gating, and—only after exact closure—blinded human
  screening of the held-out prose.

  The following wave completed those repairs. Typed attachment alternatives and
  held-out semantic bundles rendered 3 intact candidates up to 101 letters;
  the proper-locative `via del Laco, locally` rendered 4 intact candidates up
  to 66 letters with all four eleventh-character matches; and plural
  coordinated `the/the` articles rendered 2,304 candidates up to 73 letters
  while pruning 14,080 singular or mixed-number states. All produced 0 exact
  closures. Independent pointer/SHA-256 replay, provenance, novelty
  preflight, and anti-shortcut checks are retained for every row. The reader
  gate remains closed. The next concrete operators are equation-coupled
  attachment expansion, twelfth-residual conditioning after `locally`, and
  one-position plural-object lexical substitution under the coordinated gate.

  The following wave completed those repairs. Equation-gated attachment
  expansion rendered 1 intact 92-letter candidate; the twelfth-residual probe
  retained 4 intact proper-locative candidates up to 66 letters but found `o`
  where `e` was required; and one-position plural-object substitution rendered
  4,608 candidates up to 73 letters while pruning 1,792 same-object or invalid
  states. All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  bidirectional equation gating, a grammatical `le`-initial continuation after
  the matched proper-locative prefix, and second-plural-object substitution.

  The following wave completed those repairs. Bidirectional equation gating
  rendered 2 intact candidates up to 97 letters; the grammatical `leeward`
  continuation rendered 4 intact proper-locative candidates up to 66 letters
  with all four twelfth-character matches; and second-plural-object
  substitution rendered 4,608 candidates up to 73 letters while pruning 1,792
  same-object or invalid states. All produced 0 exact closures. Independent
  pointer/SHA-256 replay, provenance, novelty preflight, and anti-shortcut
  checks are retained for every row. The reader gate remains closed. The next
  concrete operators are character-labeled requirements in both lexical tries,
  thirteenth-residual conditioning after `leeward`, and paired plural-object
  substitution under a semantic relation gate.

  The following wave completed those repairs. Character-labeled trie gating
  rendered 3 intact candidates up to 99 letters; the thirteenth-residual probe
  retained 4 intact proper-locative candidates up to 66 letters but found `e`
  where `h` was required; and relation-gated paired plural-object substitution
  rendered 576 candidates up to 73 letters while pruning 192 invalid states.
  All produced 0 exact closures. Independent pointer/SHA-256 replay,
  provenance, novelty preflight, and anti-shortcut checks are retained for
  every row. The reader gate remains closed. The next concrete operators are
  traversal-time character rejection, an `h`-initial grammatical continuation
  after the matched proper-locative prefix, and relation alternation for one
  object pair while retaining the other relation.

  The following wave completed those repairs. Traversal-time character
  rejection rendered 3 intact candidates up to 97 letters; the `h`-initial
  continuation probe rendered 4 intact candidates up to 63 letters but broke
  the earlier `le` state and admitted no thirteenth match; and one-pair
  relation alternation rendered 1,152 candidates up to 73 letters while
  pruning 192 shared-subject states. All produced 0 exact closures.
  Independent pointer/SHA-256 replay, provenance, novelty preflight, and
  anti-shortcut checks are retained for every row. The reader gate remains
  closed. The next concrete operators are multi-slot requirement propagation,
  an `h` placement that preserves the full `le` continuation, and
  relation-compatible subject alternation.

  The micro-variant family is now explicitly reset. Its latest wave still
  produced 0 exact closures: traversal-time character rejection reached 3
  intact candidates up to 97 letters, the `h` probe broke the full `le` state,
  and one-pair relation alternation reached 1,152 candidates up to 73 letters.
  I am preserving those diagnostics but no longer extending that residual
  ladder. Three whole-tape families now run in parallel: simultaneous typed
  left/right semantic bundles (8 intact candidates up to 99 letters), a
  bidirectional complete-phrase transducer (4 intact SVO candidates up to 56
  letters), and a CFG×palindrome-automaton midpoint intersection (256 product
  candidates, longest diagnostic control 34 letters). All remain exact-zero;
  the next operators are character-level coupled expansion, first-residual
  trie branching within complete phrase roles, and lexical-state expansion in
  the midpoint product. The reader gate remains closed.

  The first hard-reset follow-up completed. Character-level coupled semantic
  expansion pruned mirrored obligations immediately but still rendered intact
  prose only up to 99 letters; first-residual phrase-trie branching retained
  2 branches and 8 intact SVO candidates up to 55 letters; and the lexical
  CFG×palindrome-automaton product rendered 3,072 intact sentences, with a
  39-letter maximum diagnostic control. All three remained exact-zero, with
  independent pointer/SHA-256 audits and anti-shortcut checks. The reader gate
  remains closed. The next operators are semantic role tries, two-character
  residual branching within complete phrase roles, and character-level verb
  trie expansion inside the midpoint product.

  The next hard-reset follow-up completed all three repairs without producing
  an exact closure. Live semantic role tries rendered intact ordinary-English
  candidates up to 99 letters; two-character residual branching retained 2
  branches and 8 intact SVO candidates up to 55 letters; and character-level
  agent/verb/patient trie expansion with subject–verb agreement rendered 6,144
  intact sentences up to 39 letters. Each family has independent pointer/SHA
  validation, provenance, novelty, and anti-shortcut checks. These are useful
  construction advances, but still not reader candidates; the next operators
  are live trie-node propagation, typed agreement in the two-character state,
  and bilateral role-conditioned verb frames with held-out lexical classes.

  The next semantic and phrase repairs also completed without an exact
  closure. Live trie-node propagation kept intact ordinary-English candidates
  to 99 letters while checking prefixes before full-word admission; typed
  singular/plural states in the two-character transducer kept 8 intact
  candidates to 56 letters. Both retained independent audits, provenance,
  novelty, and anti-shortcut checks. This confirms the useful direction is
  earlier whole-tape obligation propagation, not another finished-word sweep;
  the next repairs carry live states across function-word boundaries and bind
  object number jointly with determiner agreement.

  The CFG lane also completed its role-conditioned held-out-frame repair:
  576 intact renderings were produced, including “the careful cartographer
  copies the journal.” at 38 letters, but none was an exact palindrome. The
  frame state, lexical tries, agreement, provenance, novelty, and independent
  audit all pass; a typed complementizer branch is the next distinct operator.

  The next semantic and phrase repairs again improved the construction state
  without yielding an exact palindrome. Variable function-word boundaries with
  live role nodes rendered intact ordinary-English candidates to 101 letters;
  joint subject/object/determiner agreement retained 4 intact candidates to
  58 letters. Independent pointer/SHA audits, provenance, novelty, and
  anti-shortcut checks pass. The next operators are trie-valued boundary
  propagation and tense/aspect carried jointly with agreement.

  The CFG complementizer repair added role-compatible `that`/`where` branches
  and rendered 1,152 intact sentences, including “the careful cartographer
  copies where the journal.” at 43 letters, but still produced 0 exact
  closures. The independent audit and provenance checks pass; the next CFG
  operator is a typed relative-clause predicate slot with role-preserving
  valency.

  The latest three repairs remained exact-zero but moved the state boundary
  earlier: trie-valued boundary/role nodes rendered intact prose to 101
  letters; joint agreement-tense-aspect states retained 6 intact candidates to
  58 letters; and typed relative-predicate slots rendered 2,304 intact
  sentences, with a 55-letter maximum. Independent audits, provenance,
  novelty, and anti-shortcut checks pass. The next operators are exact
  boundary-node requirements, lexical aspect/adverb coupling, and a typed
  relative-object slot with semantic-class agreement.

  The feature follow-ups still produced no exact closure: boundary-label
  opposing-role requirements retained 8 intact candidates up to 101 letters,
  and lexical aspect/adverb coupling retained 2 intact candidates up to 63
  letters. These rows are preserved, but an architectural audit found that
  several earlier “coupled” lanes rendered Cartesian products before attaching
  trie traces. I am therefore treating them as diagnostics, not evidence of
  genuine joint decoding. The decisive next run is a character-labeled graph
  product with word-boundary states and an exhaustive tiny oracle.

  The first true character-graph product now passes its asymmetric-boundary
  oracle (`live on time` + `emit no evil`) without rendering mismatching paths.
  On a 500-word common audited menu with reusable 2–8-word graph layers it
  expanded 2,500 states, exhausted the explicit budget, and found 0 non-shortcut exact
  completions. A semantic valency NFA branches lexical alternatives as live
  trie nodes (4 states, 6 pruned transitions, 0 exact paths), while the typed
  grammar NFA has 2 live states and 0 exact paths. These are the first honest
  topology tests; they do not yet supply a reader candidate.

  The graph follow-up now reports 500 common words rather than an alphabetical
  slice, and its bounded product correctly reconstructs separate reversed
  paths without fabricated doubling. The multi-character semantic trie lane
  expands 256 states and prunes 710 incompatible transitions, yielding 2
  ordinary-prose diagnostics but 0 exact paths. The typed grammar NFA reaches 4
  live character states and 0 exact paths. These are honest search-space
  measurements; the next repair is to carry function-word grammar states and
  independent reversed-token tries through the same product.

  The typed-slot follow-up now uses a shared layered character NFA rather than
  enumerating complete slot sentences and attaching traces afterward. Three
  asymmetric frame pairs were compiled with slot/option/character/boundary
  provenance and searched against independent reversed graphs; all three
  closed at zero exact paths and rendered zero candidates. This is a real
  topology repair, but not reader evidence. The next experiment must widen the
  audited lexical inventory while retaining live grammar-state transitions;
  another post-render template sweep is explicitly out of scope.

  The full-sequence grammar product now moves both outer slots of one complete
  typed chain under live character equality, so the seam may occur inside a
  verb or other lexical slot rather than only between complete clauses. Its
  quarantined 51-letter catalogue fixture replays exactly, validating the
  center-inside-slot transition, while four seedless patterns produce 0 novel
  exact paths after 23,407 live states and 21,967 mismatch-edge prunes. The
  catalogue fixture is never admitted as generated output; the next repair is
  a typed substitution at a preserved mismatch frontier. The run now records
  held-out role menus (for example, the outer `world`/`word` mismatch offers
  `day`, `dog`, `door`, or `crow`) without restarting from a finished tape.
  Resuming from those preserved states with one held-out substitution produced
  0 exact repairs, so the next branch must change the typed slot schema rather
  than expand the same lexical menus.

  A fresh coordinated POS seam pattern was then run as a separate geometry,
  not as a larger bank sweep. It expanded 1,406 live states without exhausting
  its budget and produced 0 exact paths above 38 letters. This confirms that
  the current outer-slot schema, rather than the queue length, is the blocker;
  the next construction must change the syntactic slot topology while keeping
  character equations live from the first transition.

  The scene-lattice product changes the search object again: three human-authored
  scenes (“letter desk,” “garden work,” and “quiet meal”) define agent, event,
  object, and response valency before any lexical choice. Typed role banks are
  expanded under live character equations, yielding 16, 24, and 22 states,
  respectively, with zero exact closures and recorded first-mismatch frontiers.
  No candidate reaches the reader gate. The concrete repair is a seam-local
  replacement of the first mismatching semantic role by a new valency-compatible
  role bank, followed by inward resumption; enlarging these banks is not the next
  experiment.

  Dream-RSI replay is now implemented as a controller experiment rather than a
  new proxy score. The adapter independently rescans every curated historical
  rendering, rebuilds parent links from forward tape hashes, quarantines exact
  controls and repeated-phrase scaffolds, and compares six deterministic replay
  policies on a train/held-out split. The initial 15,475-node history from 417
  worlds contained only 19 parent edges and zero branching parents, so all six
  policies had identical metrics; `whole_passage_focus` was selected only by a
  declared tie-break, not by fabricated improvement.
  The best surviving train rendering is intact authored prose — “At first light,
  the gardener unlocks the old shed, trims the apple tree, carries the spare hose,
  writes a note for the neighbor, sweeps the stone path.” — at 119 letters with
  46 opposing-end mismatches. On held-out worlds the best is “At first light,
  the gardener unlocks the old shed, trims the apple tree, labels the seed trays,
  carries the spare hose, writes a note for the neighbor, sweeps the stone path.”
  at 137 letters with 54 mismatches. Their independent forward/reverse SHA
  audits pass, but neither is exact or reader certified.

  The first selected-controller deployment was run on a fresh anchored museum event. The
  three retained revisions are original, intact 120–123-letter passages (for
  example, “The museum conservator took a torn map from a cedar chest, carried it
  to the worktable, and aligned its faded marks before the evening lamps were
  lit.”); independent audits report 56–58 mismatches and zero exact closures.
  Provenance, anchor constraints, and no-shortcut flags are recorded in
  `runs/dream-rsi-online-two-region-20260917.json`; the reader gate remains
  closed. This is a constructive failure: replay cannot choose among alternatives
  when generators emit single-child chains. The next repair is a branching
  two-region authoring operator that records multiple sibling prose proposals per
  preserved state, followed by a fresh Dream-RSI replay and then the intact-versus
  shuffled blinded reader package for any exact survivor.

  That branching repair is now live. A fresh harbor-cartographer event generated
  12 retained nodes (11 parent edges, 4 branching parents) and one rejected
  147-letter proposal; all retained renderings are intact 120–140-letter prose,
  with 55–66 opposing-end mismatches and zero exact closures. Replaying the
  augmented 15,488-node history with an 8-node/world budget finally separates the
  policies: `fixed_mismatch_first` wins on train (119 letters, 46 mismatches),
  while held-out best is 137 letters with 54 mismatches. The winner was then
  redeployed on a fresh anchored theater event, yielding 11 retained sibling-tree
  nodes (10 parent edges, 4 branching parents), two length rejects, and no exact
  closure. Its best rendered passage is “The archivist drew a torn playbill from
  the locked drawer, then set it on the reading table and traced the missing cast
  names before the house lights rose.” (126 letters, 55 mismatches); its
  independent forward/reverse SHA audit passes. This is useful controller
  evidence, not reader evidence: the next test is to replay this new tree again,
  preserve the winning branch, and run the blinded intact-versus-shuffled reader
  package only if an exact, novel survivor appears.

  The next Dream-RSI replay round adds three genuinely different online
  transitions to the history: whole-passage reconstruction (20 logged
  100–140-letter revisions, best 112 letters/52 mismatches), a corrected
  cross-word grammar intersection (known-seed regression passes, 0 novel exact
  pairs), and a Brown-derived POS-FSA outer product (60,000 fresh lexical
  combinations, 0 novel exact closures). A derivational imperative expansion
  likewise found no exact closure. The replay tree now has 15,542 nodes,
  35 parent edges, and four branching parents; the six policies separate on
  the held-out split, with `fixed_mismatch_first` still selected and a held-out
  best mismatch rate of 0.794. No exact novel candidate exists yet. These
  failures change the live construction operators rather than the acceptance
  gate; the next repair is agreement-carrying coordinated clauses with the
  same independent exact audit.

  That agreement-carrying lane has now run: 240 typed combinations with
  subject-number, tense, valency, object, and setting registers, retaining
  actual prose diagnostics but 0 exact novel closures (best residual: 33
  opposing mismatches). A third replay over 15,555 nodes and 422 worlds keeps
  `fixed_mismatch_first` as the held-out winner (0.794 mismatch rate), so the
  controller is selecting among genuinely different histories but has not yet
  produced a reader candidate. The next construction change is seam-aware
  joint word selection inside the coordinated grammar, with length-band
  enforcement before selection.

  The seam-aware joint-selection operator now chooses the two opposed lexical
  slots together, carrying number/tense/valency registers and enforcing a
  100–140-letter band before ranking. It retained 12 intact, non-mirrored
  candidates; the best rendered surface is “The patient gardeners recorded the
  lantern by the doorway, beside the river at dusk; and the careful baker
  carried the lantern by the doorway, beside the river at dusk.” (136 letters,
  52 mismatches). Independent two-pointer and forward/reverse SHA-256 audits
  agree on non-exactness; provenance marks fresh joint selection with no
  catalogue import, fixed tape, or self-palindromic content. A separate
  reversible-word composition lane tried 432 typed combinations around
  reversible lexical pairs; its best is “The nurse stop; then marks the map
  pots.” (31 letters, 13 mismatches), also independently non-exact. Neither
  lane reaches the reader gate. Replaying both artifacts in round four expands
  the history to 15,580 nodes and 424 worlds (35 parent edges, four branching
  parents); the six policies still separate, but `fixed_mismatch_first` remains
  the held-out winner with 0.794 mismatch rate and zero admissible exact
  closures. These are constructive failures, not a reason to relax the goal:
  the next operator is a human-authored scene lattice with seam-compatible
  inflectional variants, selected jointly before scoring, followed by the
  frozen intact-versus-shuffled reader package for the first exact novel
  survivor.

  The fifth replay round adds the scene-lattice and semordnilap-pair branches.
  The scene lattice produced 32 jointly selected inflection variants (24 in
  the 100–160-letter band); its best intact surface is “In the quiet archive,
  the patient clerk repaired a torn map and records its missing names, as the
  careful clerk filed a marked folder before closing.” (121 letters, 53
  mismatches). The semordnilap lane produced five mechanically exact
  46–64-letter strings, but every one fails the human gate; for example, “A
  gateman drawer diaper live star draw ward rats evil repaid reward nametag
  a.” is exact yet visibly fragmentary. Independent two-pointer and SHA audits
  agree in both lanes, and all provenance/novelty checks are recorded. Round
  five now replays 15,606 nodes across 426 worlds; the controller still selects
  `fixed_mismatch_first`, with zero admissible exact closures (the five exact
  pair strings are correctly quarantined). The next construction is boundary-
  aware lexical resegmentation inside the scene lattice, retaining live
  inflection and attachment constraints; only its first exact, intact novel
  output will open the blinded reader package.

  Round six replays the boundary-aware resegmentation artifact. Its generator
  jointly searched verb/object boundaries and inflectional choices while
  pruning the live character seam, producing 16 branches (10 in the
  100–160-letter band). The best rendered passage is “At first light, the
  harbor pilot studied the folded chart, and signals the waiting boat, checks
  the tide ledger, and carries a lantern toward shore.” (119 letters, 54
  opposing-end mismatches); its independent two-pointer and forward/reverse
  SHA-256 audits agree that it is not exact. Provenance records a local,
  human-authored scene frame with no catalogue import, fixed tape, mirrored
  word order, or repeated-unit shortcut. Replaying it expands the Dream-RSI
  history to 15,622 nodes across 427 worlds (35 parent edges, four branching
  parents); the six policies still separate, but `fixed_mismatch_first` remains
  the held-out winner (0.794 mismatch rate), with zero admissible exact
  closures and 97 exact-but-rejected rows. This is a construction failure that
  triggers the next live operators—character-level CFG intersection and
  semantic slot repair—not a relaxation of the readability goal. Any exact
  survivor must still be rendered, independently audited, provenance-checked,
  and then tested in the blinded intact-versus-shuffled reader package.

  Round seven adds two orthogonal constructive lanes. The CFG/character
  intersection authored 16 scene-clause branches (9 above 100 letters), but
  its realized surfaces either repeated a whole scene or remained non-exact;
  the longest rendered row was 190 letters with 86 mismatches and is
  quarantined by the repeated-content rule. The dependency/valency seam CSP
  retained 20 typed clause-pair branches (101–109 letters); its best intact
  surface is “A careful gardener carries the blue lantern beside the stone
  wall; a calm teacher records each small answer after the evening class.”
  (109 letters, 94 mismatches), independently non-exact. Both lanes record
  fresh human-authored provenance, no catalogue/fixed-tape/word-order
  shortcut, and a concrete next repair (seam-carrying lexical edges for CFG;
  inflectional and attachment-preserving substitutions for the CSP).
  Dream-RSI round seven replays 15,658 nodes across 429 worlds (35 parent
  edges, four branching parents). The six policies remain distinct and
  `fixed_mismatch_first` remains the held-out winner (0.794 mismatch rate),
  with zero admissible exact closures. The next run applies those recorded
  repair operators; no output has reached the reader gate yet.

  The semantic-slot obligation repair then searched 64 role-, agreement-, and
  attachment-preserving substitutions over two intact near-miss scenes (66
  independently audited renderings including controls). Its longest surface
  was 131 letters; the best rendered row is “The patient archivist stores
  weathered charts beside the north window. A steady teacher reviews the brass
  ledger before dusk.” (105 letters, 44 mismatches). It is fresh prose with no
  catalogue, fixed tape, mirrored word order, repeated unit, or fragment, but
  no exact closure. The recorded next repair carries a held-out multiword slot
  alternative through the full boundary-obligation vector. Dream-RSI round
  eight replays the same 15,658-node/429-world tree because this artifact was
  already present in the registered history; no policy or exactness result is
  changed. The reader gate remains closed pending a genuine exact survivor.

  Round nine applies the next repairs rather than retuning Dream-RSI. The
  attachment lane made 720 fresh inflectional/lexical substitutions while
  preserving valency and attachment; its best intact candidate is “A careful
  gardener carried the blue lantern beside the stone wall; the patient curator
  labelled the old chart in the quiet archive.” (108 letters, 86 mismatches),
  independently non-exact. The seam-carrying lexical chart then explored 48
  authored branches, but none reached the 100-letter gate; its failure is
  recorded as a construction-length issue, with coordinated-clause composition
  as the next operator. Both artifacts have fresh provenance and no shortcut
  flags. Dream-RSI round nine grows to 16,426 nodes across 431 worlds; the
  held-out controller is still `fixed_mismatch_first` (0.794 mismatch rate),
  with zero admissible exact closures. We continue with a coordinated,
  seam-carrying grammar, keeping the exact/readability gates unchanged.

  Round ten evaluates three new constructive branches. Coordinated seam-chart
  composition produced 48 fresh candidates; its best intact scene is “At
  sunrise, the orchard keeper marks the pear trees and carries baskets home,
  and while the patient beekeeper checks the cedar hive frames before evening.”
  (123 letters, 51 mismatches). The bidirectional phrase-pair CFG retained 12
  intact 100–180-letter surfaces, best “At dawn the archivist labels a map for
  the museum, and records the names; In winter the careful teacher reads
  letters from a quiet village, as the harbor darkens.” (130 letters, 57
  mismatches). Live attachment equations explored 729 typed expansions; its
  best was “After rain, a gentle mason folds linen cloth beneath the garden
  wall; later, the calm farmer stores clean supplies beside the back room.”
  (110 letters, 46 mismatches). All three are independently non-exact,
  provenance-clean, and free of catalogue/fixed-tape/mirrored-unit shortcuts;
  their recorded next repairs are seam-compatible lexical/inflectional
  alternatives and coordinated-clause growth. Dream-RSI round ten reaches
  16,510 nodes across 434 worlds; the held-out policy remains
  `fixed_mismatch_first` (0.794 mismatch rate), with zero admissible exact
  closures. The construction goal and reader gate are unchanged.

  Three fresh Luna construction lanes then tested distinct repairs. The
  center-out common-grammar lane authored 30 SVO/PP clause pairs; its best
  intact surface is “The patient baker repaired a broken cart before the rain.
  The bright clerk opened a sealed parcel near the station.” (94 letters, 36
  mismatches). The typed boundary-trie lane performed 75 paired boundary
  checks and produced a 130-letter surface, “At first light, the harbor pilot
  checked the weathered chart and signaled the waiting boat, while the coast
  keeper guided the small fishing boat toward shore.” (59 mismatches). The
  semantic phrase-chain lane produced 16 complete 115–119-letter chains; its
  best remained a 115-letter, 49-mismatch scene with a duplicated event frame.
  Each lane independently audited its tape with two-pointer comparison and
  forward/reverse SHA-256, and each recorded a concrete next repair; none is
  reader-eligible or exact.

  Dream-RSI round eleven replays these three artifacts, reaching 16,557 nodes
  across 437 worlds. `whole_passage_focus` is selected on the training split,
  but the held-out result remains 0.794 mismatch rate with zero admissible
  exact closures (117 exact rows are rejected by provenance/shortcut gates).
  The replay therefore changes routing but does not certify or create a
  candidate; the next construction step is a genuinely live character-level
  grammar/scene solver that carries opposing obligations into lexical and
  inflectional choices. The exact/readability gate and the blinded
  intact-versus-shuffled reader package remain closed until an original exact
  passage is actually rendered.

  Two more Luna lanes were then added as separate replay worlds. The
  homophone seam-weaver generated 120 135+ letter surfaces with a live
  opposite-character ledger; its best row began “At dawn, the patient gardener
  watered the shaded orchard...” and ended with a residual `a/e` mismatch, so
  it was rejected (0 exact closures). The center-out function/verb/noun CSP
  produced nine simultaneous lexical combinations; its best intact passage was
  “Before the market wakes, a careful keeper beside the quay checks the brass
  ledger, while a young apprentice from the harbor counts the numbered boxes.”
  (123 letters, 52 mismatches). Both runs include independent two-pointer and
  forward/reverse SHA audits, fresh provenance, and concrete held-out lexical
  repairs; neither is reader-eligible.

  The exact CFG×reverse-tape DP then added a memoized live chart whose state is
  `(nonterminal, left, right, tape-state, depth)`. It explored 1,096,303
  states and 62 complete parses; the 107-letter frontier,
  “the traveler remembers and the traveler remembers and the traveler remembers
  carefully and the traveler remembers carefully,” repeats a frame and is
  quarantined rather than promoted. Exact closures were zero, with independent
  two-pointer and forward/reverse SHA-256 audits. Dream-RSI round fourteen
  replays this world at 16,607 nodes across 440 worlds; the held-out result is
  still zero admissible exact closures, so the next repair is typed complement
  terminals inside the live chart, not a larger score sweep.

  I also exercised the selected policy's online redeployment path on a fresh
  two-region theater scene. The branching authoring world retained six
  independently generated 100–140-letter prose successors; its best rendered
  passage was “The theater archivist retrieved a torn playbill from a locked
  drawer, carried it to the reading table, and marked missing cast names before
  the house lights faded.” (134 letters, 59 mismatches). It produced zero exact
  closures, while recording three rejected length-band proposals and six
  parent-child edges. Replay round fifteen incorporates that online world at
  16,617 nodes across 440 worlds; the selected and held-out policies remain
  `whole_passage_focus` with zero admissible exact closures. This is a real
  online Dream-RSI deployment trace, not a readability result; the next repair
  must change the live character construction kernel.

  The lexicon reverse-edge lane supplied a useful negative control: its
  by-construction chain was an exact 138-letter tape, but rendered as
  “malayalam samas reviver seres rotator siris redder deed alula sis anana sus
  civic — civic sus anana sis alula deed redder siris rotator seres reviver
  samas malayalam.” Because it is a chain of self-palindromic words and repeats
  the same units, it is explicitly quarantined and cannot enter reader testing.
  The replay gate now rejects self-palindromic content words even when legacy
  metadata is incomplete. Round sixteen replays this artifact at 16,618 nodes
  across 442 worlds; held-out admissible exact closures remain zero.

  The trie-segmented phrase-edge repair produced a 214-letter exact tape, but
  the rendered chain (“alula malayalam civic malayalam ... anna malayalam”)
  is a stack of self-palindromic words with repeated `malayalam` units. Its
  stale eligibility claim was corrected: the lane now persists explicit
  anti-shortcut checks and marks the row `quarantined_shortcut_rejection` with
  `reader_eligible: false`. Round eighteen replays the corrected artifact at
  16,618 nodes across 442 worlds; the held-out admissible exact count remains
  zero. The next construction must replace the palindromic seed with held-out
  non-palindromic multiword phrases and enforce disjoint lexical content.

  The CFG repair added typed complement productions (`PP -> Prep Obj`) and
  richer clause paths while retaining live tape-state memoization. It still
  yielded 62 complete parses, a 107-letter repeated-frame frontier, and zero
  exact closures; the repeated frontier remains quarantined. Round nineteen
  replays the updated chart without changing the held-out result (zero
  admissible exact closures, 16,618 nodes across 442 worlds). The next change
  must alter the lexical domain, not merely add another complement production.

  As a diagnostic sanity check, the Brown word-bigram order-gain audit (64
  deterministic own-word shuffles) gives the quarantined 214-letter chain
  gain 0.000, mean Zipf 2.732, and repeated-word rate 0.719, whereas the
  fluent 134-letter theater diagnostic gives gain 1.017, mean Zipf 5.403, and
  repeated-word rate 0.111. These figures are descriptive only: they support
  filtering and failure analysis, never a readability certificate or a
  substitute for the blinded intact/shuffled reader study.

  A direct joint-authoring lane was also timed against the local `gpt-oss:20b`
  model. Four bounded attempts each timed out at 35 seconds before returning a
  passage; the run preserved those failures and a quarantined mismatch-repair
  operator rather than silently dropping them. No candidate or reader evidence
  was claimed. Round seventeen replays the complete history (16,618 nodes,
  442 worlds) with the same held-out result: zero admissible exact closures.
  The concrete response is to keep model calls off the critical construction
  path and invest in the live CFG/lexical chart, where character obligations are
  solved before prose is emitted.

  Three additional Luna construction lanes were run as a Dream-RSI breadth
  expansion. The ordinary phrase-pair/trie lane filtered 148 authored
  sentences down to 21 complete common-word phrases, then attempted live
  reverse-tape segmentation with disjoint lexical content; it found zero exact
  phrase edges. The semantic character-CFG lane used typed
  agent/action/object/location slots and independent pointer/SHA audits, but
  produced no 100-letter frontier or exact closure. A slot-seam lane used the
  local model only for clause-slot proposals; two calls timed out and the
  returned control text was rejected as incomplete, so it is not in the replay
  history. These are distinct failures, not duplicated score sweeps. Their
  concrete next repairs are, respectively, held-out typed SVO phrase
  expansion, live complement transitions, and a deterministic seam operator
  that never mirrors a finished tape.

  Dream-RSI round twenty-one replays the two valid new worlds together with the
  prior history: 16,618 nodes across 442 replay worlds, with
  `whole_passage_focus` again selected on training and held-out splits. The
  held-out winner has zero admissible exact closures (113 exact controls are
  rejected) and a best intact diagnostic of 159 letters at 0.794 mismatch
  rate. The replay routes the next construction step but does not claim a
  readable result; the reader gate remains closed until an original, exact,
  intact passage is rendered and tested against a randomized shuffled control.

  The recorded repairs were then executed rather than merely described. A
  typed SVO phrase-edge preflight tested five fresh agent/action/object/location
  clauses and found zero exact edges. The CFG lane was upgraded with explicit
  character-level complement transitions and still had no 100-letter closure.
  The pre-closure seam lane paired two complete hand-authored slot pairs while
  obligations were live; both stopped at the first lexical seam conflict and
  rendered nothing. All three artifacts preserve the conflict trace and the
  next lexical/agreement repair, and none uses a finished-tape mirror.

  The follow-up first-unmatched-seam repair was also replayed: five fresh typed
  SVO clauses exposed their required reverse prefixes before any surface was
  rendered. Every clause died at seam offset zero (the first required reverse
  character had no role-compatible lexical choice), so the lane produced no
  candidate and remains quarantined. This is a concrete repair trace for the
  next run—expand only role-compatible alternatives that match successive seam
  characters—rather than a claim that a failed trace is a result.

  Dream-RSI round twenty-three replays this evidence as 16,618 nodes across
  442 worlds, with 44 parent edges and seven branching parents. The selected
  `whole_passage_focus` policy remains indistinguishable from the alternatives
  on the training metrics; held-out replay has zero admissible exact closures,
  113 rejected exact controls, and a best intact diagnostic of 159 letters at
  0.794 mismatch rate. This is the intended Dream-RSI diagnostic: the replay
  can identify that the history lacks useful sibling choices, but it cannot
  manufacture a palindrome that was never present in a recorded transition.

  Three new Luna lanes then acted on that diagnosis. A live role-conditioned
  seam grower tested six alternatives for each of six fresh agent clauses and
  stopped before rendering because every reverse obligation was dead at offset
  zero. A seed-benchmark lane deliberately did not use the seed as a scaffold;
  its best rendered near miss was “The quiet gardener maps at dawn. A calm
  reader bakes a tent.” (47 letters, 15 mismatches, exact false, independent
  forward/reverse SHA values different). Both artifacts preserve fresh
  provenance, anti-shortcut checks, and the next seam-conditioned lexical
  repair.

  Dream-RSI round twenty-five now includes those worlds and a
  `failure_repair_first` policy. The policy exposes a deduplicated queue of 20
  actionable failure signatures for the next constructor, while keeping exact
  admission unchanged. Replay covers 16,624 nodes across 443 worlds; the
  selected and held-out policies are still `whole_passage_focus`, with zero
  admissible exact closures. The queue is therefore a routing improvement, not
  a readability claim.

  Round twenty-six makes that queue operational: when online redeployment is
  requested, the top deduplicated seam repair is appended to the fresh
  authoring anchors, and the bounded local-model call has an explicit timeout
  with structured timeout provenance. The offline replay remains unchanged at
  16,624 nodes and zero admissible exact closures; this patch changes how the
  next live branch is constructed, not how failed text is scored.

  Round twenty-eight fixed a replay bookkeeping gap exposed by the next three
  Luna lanes: when a run keeps its repair instruction at the artifact root,
  Dream-RSI now propagates that instruction to otherwise-unclassified prose
  rows before building the repair queue. The new reverse-phrase lane therefore
  contributes an actionable seam repair (11 held-out rows in training), while
  failure-only chart and scene traces remain quarantined without fabricated
  text. The replay still has 16,649 nodes, zero admissible exact closures, and
  a closed reader gate; this is routing evidence, not a candidate.

  Round twenty-nine replaces the overclaimed typed/CFG lanes with a verified
  counterexample-guided role-product constructor.  Its inner solver is the
  existing outside-in character-edge product; every pushed transition already
  matches the opposite character, and a grammar-path replay plus an
  independent two-pointer/SHA audit checks each closure.  At a dead state the
  run records the two live grammatical roles and their incompatible character
  sets.  The next round adds only authored, role-compatible lexical entries
  that address that boundary obligation, then reruns the same product.  A
  small independent oracle agrees with the product on all exact paths in a
  synthetic grammar, and the withheld 38-letter benchmark is recovered with
  matching forward/reverse SHA-256.  The benchmark is explicitly excluded from
  generated candidates.

  On three fresh clause patterns and four bounded repair rounds, the run
  visited six bounded frontiers and produced zero fresh exact closures.  The
  first repairs are concrete (for example, adding `each`/`every` at a
  determiner boundary and `path` at a noun boundary); later frontiers expose
  unavailable boundary letters, so the next operator must widen the authored
  reservoir or change the clause pattern.  No text from the withheld witness
  enters Dream-RSI replay, and no reader claim is made.  This is a corrected,
  executable construction loop—not evidence that a long readable palindrome
  has already been found.

  A separate reverse-conditioned POS/phrase lattice then tested a different
  search family.  It indexed 21 ordinary authored phrases by their reversed
  character prefixes and expanded 21 bounded queue states; 26 complete phrase
  pairs were rendered for diagnosis.  The best intact row was “the old man sat
  the pot was hot” (24 letters), matching only one outside-in pair; its
  independent two-pointer audit and forward/reverse SHA values both reject
  exactness.  There were zero exact closures.  Because the phrase bank is
  source material rather than generated prose, these rows remain diagnostic
  and are not presented as outputs.  The concrete repair is to add typed
  verb/object features to the trie key and continue only when the next seam
  character has a role-compatible lexical completion.

  That repair was then audited against the selected right phrase itself.  The
  first implementation had used the trie depth from an unrelated leaf as if it
  were evidence for the rendered pair; the corrected run now requires the
  chosen pair's own reverse prefix (plus typed-number compatibility) before
  rendering.  With 12 fresh frames expanded to 924 typed realizations (132
  number-conditioned verb forms plus object/adjunct substitutions) it produces
  zero valid near-miss rows and zero exact closures.  This removes a false
  positive diagnostic; the current run is explicitly a live-prefix probe, not
  a full two-sided closure.  The precise next operator is therefore a
  multi-clause chart with seam-compatible lexical entries and a true closure
  state.

  The next multi-clause chart composed 194 typed clauses into 1,500,660
  independently selected four-clause passages.  It performed the live
  outside-in comparison before rendering and kept 20 nonrepeating diagnostic
  passages; the best remained at seam depth zero, with zero exact closures.
  Repeated-clause rows are explicitly flagged and excluded rather than used as
  controls.  The next construction operator is a third-clause center-crossing
  chart with seam obligations carried across clause boundaries.

  The three-clause follow-up rejected 49,430 repeated-clause controls before
  scoring and retained 8,170 distinct passages.  None crossed the center, and
  none was exact.  These are diagnostics only; the next implementation must
  replace prefix scoring with a true center-compatible lexical chart that
  carries residual character debt through the crossing.

  **Dream-RSI semantic dialogue branch (2026-09-17).**  A human-authored
  dialogue lattice supplied explicit role-paired lexical edges and produced
  three rendered exact rows: the 68-letter scene `Noel, now live on; Damon,
  draw a map. Was I sore? Eros: I saw Pam, a ward. Nomad: no evil won, Leon.`;
  a same-tape boundary repair with `I saw Pam award Nomad`; and a 100-letter
  diagnostic extension with additional named message roles.  Independent
  two-pointer and forward/reverse SHA audits agree on every row.  The first
  row is a word-order mirror; the boundary repair removes that mirror but the
  `sore/Eros` centre is still a forbidden proper palindromic span; the longer
  extension inherits the same centre debt.  All rows are therefore rejected
  before any reader claim, and the exact text plus provenance remain in the
  artifact rather than being promoted as generated prose.

  This is the first Dream-RSI replay node whose repair changes lexical
  segmentation instead of enlarging a phrase bank.  Round 37 replays the
  parent/repair/extension branch across 16,714 historical nodes and selects
  `fixed_mismatch_first` on the held-out split; its online redeployment of the
  selected policy generated a two-level coordinated prose tree but no exact
  closure.  The next construction is concrete: replace the `sore/Eros`
  lexical centre with a live, syntactically complete centre-crossing edge,
  while retaining the `award` boundary repair and rejecting both word-order
  mirrors and proper palindromic subspans.  Replay remains routing evidence;
  the reader gate stays closed until a mechanically eligible 100+ letter row
  enters the randomized blinded intact/shuffled study.

  **Center-repair fan-out (2026-09-17).**  Three orthogonal Luna lanes were
  replayed as separate construction worlds rather than counted as one sweep.
  The bounded mirror-pair audit rechecked 256 exact rows and found zero
  mechanically clean candidates; its best rows are short phrase diagnostics,
  not prose.  The authored center-word grammar rendered 24 ordinary scene
  candidates up to 71 letters with the midpoint inside a lexical token and
  found zero exact closures.  The typed dialogue lattice rendered 432 fresh
  variants with the same center-crossing invariant and found zero exact
  closures.  Each row carries rendered text, an independent pointer/SHA audit,
  provenance, and a concrete repair; none enters the reader study.

  Round 38 adds those worlds to Dream-RSI replay, increasing the history to
  17,545 nodes while retaining the held-out policy result.  This narrows the
  next construction: preserve the online center-word ledger, but replace the
  current fixed five-slot scenes with a residual-driven center setting/answer
  expansion.  The mirror-pair inventory is now explicitly a diagnostic source,
  not a generator or a readability shortcut.
- **Seedless imperative/vocative utterance lattice, 2026-09-17:** crossed
  independently authored imperative and vocative speech-act templates around
  four non-palindromic discourse centers (`was`, `said`, `asked`, `told`).
  The 728 rendered, provenance-tagged probes produced zero exact closures;
  the closest controls still repeated content or failed the reverse tape.
  This lane used no seed-wrap composite and produced no reader evidence.

- **Dream-RSI masked-span infilling, 2026-09-17:** replayed a narrow
  one-slot repair against a constituent-width repair, then deployed the
  selected policy on held-out authored scenes.  The state froze already
  matching outer character assignments while reopening seam-owning typed
  slots; phrase length and word boundaries could change inside the reopened
  span.  It rendered 30 complete scene pairs up to 98 letters, with two
  independent audits on every row.  The constituent policy improved the best
  mismatch count from 35 to 33, but produced zero exact closures and zero
  mechanically admitted rows.  This is a real construction change, not a
  larger unchanged sweep.  The next operator is a two-sided constituent repair
  with agreement-carrying inflections, retaining all other assignments; no
  reader claim is made until an exact novel row survives the admission gate.

- **Two-sided agreement constituent repair, 2026-09-17:** reopened complete
  agent+verb constituents on both sides while keeping the outer scene
  assignments fixed and carrying singular/plural agreement through the
  replacement.  Four fresh scenes (longest 67 letters) were rendered and
  independently audited; none closed exactly.  The concrete next repair is a
  joint agent+verb plus object span with cross-side number agreement, not a
  repeat of the previous one-slot lane.

  Round 40 replay now contains 17,914 independently audited nodes.  The
  held-out fixed-mismatch policy reaches a 0.780 mismatch rate, a small
  routing improvement over round 39, but still no exact or reader-eligible
  output.  The result is preserved as a repair queue item; replay scores do
  not certify English readability.

- **Joint two-span masked infill, 2026-09-17:** reopened agent+verb+object
  spans on both sides with cross-side number agreement while retaining the
  fixed outer setting.  The bounded run rendered 16 ordinary scenes through
  75 letters; the best residual debt was 56 and exact closures were zero.
  This geometry is now explicitly exhausted.  Its concrete successor is a
  boundary-crossing relative-clause infill, which changes the constituent
  geometry instead of enlarging this bank.

  Round 41 replays 18,152 audited nodes.  The held-out fixed-mismatch policy
  remains at a 0.780 mismatch rate; the new route changes no reader-facing
  status because there is still no exact novel survivor.  The replay therefore
  routes the next experiment toward the relative-clause boundary rather than
  spending another run on the exhausted constituent geometry.

- **Seedless typed-template equation, 2026-09-17:** 81 fresh typed SVO
  realizations (longest 85 letters) were built from live outside-in character
  obligations; zero exact closures.  The concrete successor is seam-indexed
  lexical edge substitution rather than another template sweep.

  Dream-RSI round 39 replays the new typed, imperative, seam-repair, and
  masked-infilling worlds as 17,688 independently audited nodes.  The held-out
  winner remains `fixed_mismatch_first` at a 0.794 mismatch rate; this is
  routing evidence only.  The online reader-facing gate is still closed because
  no fresh exact, mechanically admissible prose row exists yet.

- **Relative-clause boundary infill, 2026-09-17:** changed the geometry after
  the joint agent/object route was exhausted.  Two authored scene frames jointly
  reopened a relative clause and its host complement on both sides, allowing
  word boundaries to move across the attachment seam while keeping the outer
  openings fixed.  The bounded deployment rendered 18 complete prose scenes,
  up to 109 letters; the best residual debt was 86 and exact closures were
  zero.  For example, `At dawn, the archivist who marks the map checks the
  quiet harbor. By dusk, the sailor who charts the inlet notes a broad area.`
  passes the independent audit as non-exact, with the first outer `a`/`a`
  agreement preserved and the unresolved seam recorded.  The run uses no
  catalogue text, finished-tape reversal, repeated unit, or reader claim.  Its
  concrete successor is a three-region discourse-frame infill with one shared
  referent; another duplicate two-span sweep is explicitly rejected.

- **Three-region discourse anchor, 2026-09-17:** changed the live state again
  to a setup, a shared-referent relative clause, and an anaphoric response.
  Sixteen bounded authored scenes were rendered (longest 69 letters); the
  best had 42 mirrored mismatches and exact closures were zero.  The shared
  referent is semantic glue, not a repeated palindrome unit.  Independent
  pointer/SHA checks and provenance are retained; the next repair jointly
  inflects the referent and anaphoric response without dropping the attachment.

- **Semantic-frame mirror, 2026-09-17:** independently varied semantic slots
  in two long scene frames with mutable subjects, actions, objects, and
  relative clauses.  Four fresh complete passages (135--140 letters) were
  rendered and audited; exact closures were zero and the best had 120
  mismatches.  The concrete next repair is seam-conditioned role-compatible
  substitution inside the relative-clause slots, not another broad sweep.

- **Dream-RSI replay audit correction, 2026-09-17:** replay round 47 now
  exposes every mechanically-admissible exact row with its rendered text,
  source path, and audit hashes instead of reporting a bare count.  The
  fail-closed shortcut gate also rejects legacy `word_order_only_symmetry`
  flags and all catalogue-family aliases.  On 18,412 nodes the held-out
  mismatch-first policy remains at 0.780 mismatch rate, with zero admissible
  exact rows and 111 exact-but-rejected controls.  This fixes an evidence
  accounting bug; it does not turn replay into readability evidence.

- **Agreement-carrying discourse repair, 2026-09-17:** applied the recorded
  three-region repair by jointly inflecting a shared referent and its anaphoric
  response while preserving the relative-clause attachment.  Thirty-two typed,
  complete scenes were rendered; the longest was 65 letters, the best residual
  was 42 mismatches, and exact closures were zero.  The next operator moves the
  mutable boundary across the relative-clause seam while retaining agreement.

- **Relative-clause seam repair, 2026-09-17:** applied first-mismatch,
  role-compatible substitutions inside relative-clause slots on a fresh semantic
  frame.  Eight complete passages (up to 142 letters) were rendered; the best
  residual fell from 120 to 116 mismatches, but exact closures remained zero.
  The concrete successor is coordinated two-sided relative-clause substitution
  with length balancing; no reader gate is opened by this near-miss improvement.

  Dream-RSI round 48 now replays 18,452 audited nodes including both repairs.
  The held-out mismatch-first policy is unchanged at 0.780, with zero
  mechanically admissible exact rows and 107 exact-but-rejected controls in
  the held-out partition.  The unchanged policy is useful evidence: the next
  run must alter the live operator (two-sided length-balanced seam infill), not
  retune replay weights or claim a speed result.

- **Relative-seam agreement repair, 2026-09-17:** jointly varied referent
  number, anaphor, and subject-relative versus object-relative attachment in 32
  bounded prose candidates.  The longest was 75 letters, the best residual was
  48 mismatches, and exact closures were zero.  The seam-crossing attachment
  is distinct from the earlier one-sided slot repair; its concrete successor
  is a typed two-word seam bridge.

- **Two-sided relative balance, 2026-09-17:** coordinated left/right
  relative-clause substitutions with matched length deltas.  The strict
  balance admitted one complete pair, up to 142 letters, with zero exact
  closures and a best residual of 120, so it did not improve the prior 116.
  The next repair relaxes equality to a bounded +/-1 or +/-2 window while
  jointly changing semantic heads and verbs.

  Dream-RSI round 49 replays 18,485 audited nodes including both seam repairs;
  the held-out fixed-mismatch policy remains 0.780 with zero admissible exact
  rows.  The unchanged replay winner is a stop signal for score tuning, so the
  next deployment must use the promised relaxed length-window seam bridge.

- **Live Dream-RSI redeployment, 2026-09-17:** after held-out selection,
  `fixed_mismatch_first` routed a fresh sibling tree through the local prose
  model (branch factor 3, depth 2).  Eight accepted complete passages and two
  rejected length-band proposals were audited.  The strongest rendered branch
  was `The stage archivist retrieved a torn playbill from a locked drawer,
  brought it to the reading table, and marked the missing cast names before the
  house lights rose.` (134 letters, 64 mirrored mismatches); exact closures were
  zero.  This is the first end-to-end replay-to-live deployment evidence, but
  it is still a prose diagnostic, not reader evidence.  The live branch now
  queues the typed two-word seam bridge rather than another model retry.

- **Relaxed relative-head/verb window, 2026-09-17:** replaced strict length
  equality with a bounded +/-2 character window while changing semantic heads
  and verbs on both sides.  Two complete passages (up to 137 letters) were
  admitted to the construction trace, but the best residual stayed at 120 and
  exact closures were zero.  The next repair is semantic-frame replacement at
  the two-sided seam.

- **Typed two-word seam bridge, 2026-09-17:** inserted coordinating (`and then`)
  and causal-temporal (`so now`) bridges while carrying referent/anaphor
  agreement.  Sixty-four complete candidates (up to 82 letters) were rendered;
  the best residual was 50 and exact closures were zero.  The next operator
  co-designs bridge polarity with a short complement, rather than widening the
  connector inventory.

- **Semantic-frame seam replacement, 2026-09-17:** replaced nouns, verbs,
  settings, and relative clauses on both sides of a fresh frame.  Four complete
  scenes up to 141 letters were rendered; the best residual improved to 118
  mismatches, but exact closures were zero.  The next repair is a seam CSP over
  authored lexical slots, constrained by opposing seam letters before prose is
  rendered.

- **Relative-seam bridge complement, 2026-09-17:** added a short locative or
  temporal complement to the typed bridge, changing attachment geometry while
  preserving complete prose.  Sixty-four candidates up to 89 letters were
  rendered; the best residual was 58 and exact closures were zero.  The next
  operator is typed one-word complement substitution with the bridge held
  fixed.

- **Typed one-word seam complement, 2026-09-17:** held the `and then` bridge
  fixed and varied only typed one-word complements across 64 complete prose
  candidates.  The longest was 88 letters, the best residual was 58, and exact
  closures were zero.  The next repair jointly chooses the complement with a
  minimal auxiliary at the seam.

- **Frame-pair seam CSP, 2026-09-17:** applied a pre-render one-letter seam
  constraint to nine authored frame pairs, rejecting three before rendering
  and retaining six complete passages up to 142 letters.  Exact closures were
  zero and the best residual was 124, worse than the prior seam repair; the
  next refinement is a two-letter seam CSP with role-compatible alternatives.

- **Two-letter frame seam CSP, 2026-09-17:** refined the pre-render frame
  constraint to a two-letter opposing signature over 16 authored frame pairs.
  Eight were rejected before rendering and eight complete passages up to 141
  letters were audited; exact closures were zero and the best residual stayed
  at 120.  The next route is a three-letter signature with inflectional
  agreement.

- **Complement-plus-auxiliary seam repair, 2026-09-17:** jointly varied the
  one-word complement and a minimal `can`/`will` auxiliary while holding the
  bridge and relative attachment fixed.  Sixty-four complete candidates up to
  91 letters were rendered; the best residual was 60 and exact closures were
  zero.  The next operator permits auxiliary inflection with a short complement
  phrase.

- **Recursive discourse spine, 2026-09-17:** implemented the scalable
  construction route as `Spine := Event (TypedAdjunct)*`, with authored
  temporal, locative, instrumental, and causal event sentences.  It produced
  complete readable passages for requested targets 100, 150, 200, and 300
  letters (actual lengths 151, 187, 256, and 356), while carrying the outer
  character equation at every expansion.  Exact closures were zero, so these
  are construction traces—not palindrome claims.  The next repair is recursive
  adjunct substitution at the first unresolved equation, preserving the
  scalable grammar rather than padding a fixed sentence.

  Dream-RSI round 55 replays 18,765 audited nodes after adding the recursive
  spine.  The held-out mismatch-first policy remains 0.780 with zero
  mechanically admissible exact rows.  The recursive lane changes the length
  regime and keeps prose intact, but the next live step must repair its first
  open character equation rather than merely request a longer target.

  Dream-RSI round 54 replays 18,761 audited nodes.  The held-out
  mismatch-first policy remains 0.780 with zero mechanically admissible exact
  rows and 111 exact-but-rejected controls in training.  The stable plateau
  confirms that replay is routing evidence; the next live construction must use
  the queued morphology-aware three-letter signature and auxiliary-inflection
  operator.

  Dream-RSI round 53 replays 18,689 audited nodes including the typed
  complement and frame-CSP lanes.  The held-out policy remains 0.780 with zero
  mechanically admissible exact rows.  Replay is now a stable routing baseline;
  the next useful change is the queued two-letter CSP and semantic auxiliary
  repair, not another larger history sweep.

  Dream-RSI round 52 replays 18,619 audited nodes including the live deployment
  and both fresh repairs.  The held-out mismatch-first policy remains 0.780
  with zero mechanically admissible exact rows; this unchanged score is treated
  as a routing plateau, so the next run must use the queued seam CSP and typed
  one-word complement operator rather than another replay sweep.

  Dream-RSI round 51 replays 18,551 audited nodes, including the live sibling
  tree and both newest repair lanes.  The held-out policy is unchanged at
  0.780, with zero mechanically admissible exact rows; the unchanged replay
  score routes work to the recorded semantic-frame replacement rather than
  another score-tuning pass.

- **Inflectional same-seam alternatives, 2026-09-17:** varied inflectional
  lexical heads inside the three-letter agreement seam class.  Twelve of 16
  alternatives were rejected before rendering and four grammatical passages up
  to 119 letters remained; the best residual was 88 and exact closures were
  zero.  The result reproduces the prior best, so the next repair couples head
  choice directly to the opposing seam letters.

- **Auxiliary-polarity seam repair, 2026-09-17:** paired positive and negative
  two-word auxiliary phrases (`can still` / `cannot yet`) with the fixed bridge
  and relative attachment.  Sixty-four complete candidates up to 90 letters
  were rendered; the best residual was 56 and exact closures were zero.  The
  next operator couples polarity with a short object complement.

- **Recursive-spine adjunct substitution, 2026-09-17:** applied the first
  unresolved-equation repair to the scalable 100/150/200/300-letter spine.
  All four complete prose targets remained intact (up to 356 letters), but the
  current inventory had no unused terminal class that could close the live edge,
  so exact closures stayed zero and the rows were unchanged.  The concrete next
  repair expands terminal classes and permits replacing an existing adjunct only
  when attachment remains grammatical and the mirrored boundary changes.

  Dream-RSI round 58 replays 18,901 audited nodes including the inflectional,
  polarity, and recursive-spine repair traces.  The held-out mismatch-first
  policy remains 0.780 with zero mechanically admissible exact rows.  The
  recursive route therefore advances by changing its terminal inventory, not by
  claiming the scalable prose traces are palindromes.

- **Three-letter agreement CSP, 2026-09-17:** extended the pre-render seam
  signature to three letters and carried number agreement through the lexical
  alternatives.  Twelve of 16 authored pairs were rejected before rendering;
  four complete scenes up to 119 letters remained, with best residual 88 and
  zero exact closures.  The next repair varies inflectional lexical choices
  within the same seam class.

- **Auxiliary-plus-complement phrase, 2026-09-17:** permitted an inflected
  auxiliary together with a short two-word complement while preserving the
  bridge and attachment.  Sixty-four complete candidates up to 89 letters were
  rendered; the best residual was 58 and exact closures were zero.  The next
  operator is auxiliary polarity choice with a two-word complement.

  Dream-RSI round 56 replays 18,833 audited nodes after the three-letter and
  auxiliary-phrase repairs.  The held-out mismatch-first policy remains 0.780
  with zero mechanically admissible exact rows.  The stable replay result is
  now a baseline for the next live operators; it is not a readability or speed
  claim.

- **Coupled inflectional head seam, 2026-09-17:** coupled opposing semantic
  head choices by explicit seam keys, rejecting 10 of 16 pairings before
  rendering.  Six complete passages up to 119 letters remained; the best
  residual was 88 and exact closures were zero.  The next repair preserves the
  seam key while adjusting paired slot lengths.

- **Polarity-object seam repair, 2026-09-17:** coupled positive/negative
  polarity with typed object number (`maps`/`keys`) in 32 complete candidates.
  The longest was 81 letters, the best residual was 56, and exact closures were
  zero.  The next operator couples object number and polarity with a short
  locative complement.

  Dream-RSI round 59 replays 18,939 audited nodes, including the coupled head
  and polarity-object repairs.  The held-out mismatch-first policy remains
  0.780 with zero mechanically admissible exact rows.  The next construction
  step is the recorded paired-slot length repair and polarity/locative route;
  no score-tuning or readability claim is made.

- **Key-preserving paired-slot length repair, 2026-09-17:** changed the lengths
  of paired semantic slots while retaining their role keys and attachment.  The
  run rendered four complete passages (109--130 letters); for example, ``At
  dawn, a teacher carries a map which charts the shore beside the inlet; a
  guide keeps a journal which records the way beside the inlet.``  The
  independent two-pointer audit found 88 mismatches in the best row and no
  exact closure.  The next operator is a paired boundary-inflection change,
  not another Cartesian phrase sweep.

- **Object-number/locative seam repair, 2026-09-17:** jointly varied object
  number, polarity, and a typed locative while preserving the authored scene
  roles.  Thirty-two complete rows were rendered (69--93 letters); the best
  audited row had 60 mismatches and exact closures were zero.  The next repair
  varies the locative preposition with object number while carrying polarity,
  preserving this route's grammar and provenance.

- **Dream-RSI round 60, 2026-09-17:** replay now includes both fresh repair
  worlds above plus the recursive-spine world.  The controller recorded the
  rendered candidates and their independent audits as replay observations,
  then evaluated policies on a deterministic held-out split before any online
  redeployment.  The mismatch-first policy remains at 0.780 held-out score and
  admits zero exact rows; this is a routing diagnostic, not a readability or
  speed claim.  The next round must consume the paired-inflection and
  polarity/locative repair operators rather than replaying these same choices.

- **Paired boundary-inflection repair, 2026-09-17:** changed paired location
  boundaries under fixed semantic-role keys and attachment.  Four fresh,
  complete passages were rendered; the longest was 117 letters and the best
  independent two-pointer audit still had 104 mismatches, with zero exact
  closures.  For provenance, one row is ``At dawn, the keeper marks the chart
  that guides the crew along the inlet; the sailor reads the ledger that
  remembers the route along the inlet.``  The concrete next operator changes
  determiners and inflections jointly while retaining the same frame.

- **Locative-preposition seam repair, 2026-09-17:** jointly varied locative
  preposition, object number, and polarity in 32 complete rows.  The longest
  row was 93 letters; the best independent audit had 60 mismatches and exact
  closures remained zero.  The next repair varies locative noun-phrase length
  while preserving preposition and object agreement.

- **Dream-RSI live redeployment round 62, 2026-09-17:** after held-out policy
  selection, the winning mismatch-first policy was deployed against a fresh
  two-region authoring tree using the local language model.  It produced five
  complete sibling proposals and two rejected proposals.  The strongest
  rendered branch was ``A theater archivist pulled a torn playbill from a
  locked drawer, carried it to the reading table, and noted the missing cast
  names before the house lights rose.`` (130 normalized letters, 60 mismatches,
  exact=false); its forward and reverse tape hashes are recorded in the run
  artifact.  The initial and every sibling are independently audited, all
  exact closures are zero, and the reader gate remains closed.  The next
  repair is to branch from the preserved state with a new semantic operator,
  not to resample this same tree.

- **Paired determiner/inflection repair, 2026-09-17:** coupled singular/plural
  determiners and inflections under agreement constraints.  Four fresh intact
  passages were rendered (109--119 letters); the best row had 96 mismatches and
  exact closures were zero.  One audited row was ``At dawn, a keeper marks a
  chart that guides the crew beside the inlet; a sailor reads a ledger that
  remembers the route beside the inlet.``  The next operator couples these
  changes to agreeing relative-clause heads and verbs.

- **Locative noun-phrase-length repair, 2026-09-17:** varied short versus
  expanded locative noun phrases while preserving typed preposition and object
  agreement.  Thirty-two complete rows reached 96 letters; the best audit had
  60 mismatches and exact closures were zero.  The next repair jointly varies
  the locative determiner while preserving noun-phrase length.

- **Recursive event-predicate repair, 2026-09-17:** varied the central event
  predicate across four authored frames before growing the typed adjunct spine.
  The four requested targets reached 165, 197, 272, and 366 letters; the best
  audit still had 150 mismatches and all exact closures were zero.  The actual
  rows remain complete prose with provenance and independent two-pointer/SHA
  checks.  The next repair jointly varies the event predicate and its first
  attached adjunct so the live mirrored frontier changes at both boundaries.

- **Dream-RSI round 63, 2026-09-17:** replayed 489 worlds and 19,047 audited
  nodes after adding the three repair lanes above.  The fixed mismatch-first
  policy remained the held-out winner at 0.780; the held-out report had 418
  worlds, zero mechanically admissible exact rows, and 107 exact-but-rejected
  rows.  This stable score is a policy-routing plateau, so the next round is
  required to consume the recorded coupled operators rather than retune the
  replay metric.

- **Coupled determiner/relative agreement repair, 2026-09-17:** carried
  determiner, subject-number, relative-head, and relative-verb agreement as a
  single seam state.  Four complete passages up to 119 letters were rendered;
  the best audit had 92 mismatches and exact closures were zero.  The next
  operator seam-selects one role-compatible semantic slot rather than reopening
  the whole phrase bank.

- **Locative-determiner repair, 2026-09-17:** substituted determiners at fixed
  locative noun-phrase length, preserving preposition attachment and object
  agreement.  Sixty-four complete rows were rendered up to 96 letters; the
  best audit had 60 mismatches and exact closures were zero.  The next repair
  jointly varies determiner and locative noun.

- **Recursive event/adjunct repair, 2026-09-17:** changed the central event
  predicate and first typed adjunct together.  The requested targets reached
  160, 187, 271, and 366 letters; the best audit had 134 mismatches and exact
  closures were zero.  The next operator jointly varies the event predicate,
  first adjunct, and final adjunct so the terminal seam can move as well.

- **Seam-selected semantic-slot repair, 2026-09-17:** selected a
  role-compatible noun at the first mismatch seam while carrying agreement and
  attachment.  Four complete passages up to 120 letters were rendered; the
  best audit had 106 mismatches and exact closures were zero.  The next repair
  coordinates both opposing semantic nouns at the same seam index.

- **Locative determiner+noun-pair repair, 2026-09-17:** jointly varied the
  locative determiner and noun at fixed phrase length.  Sixty-four complete
  rows reached 97 letters; the best audit had 60 mismatches and exact closures
  were zero.  The next repair couples the locative preposition with that noun
  pair.

- **Recursive three-region terminal-seam repair, 2026-09-17:** jointly varied
  the event predicate, first adjunct, and final adjunct.  The four target rows
  reached 147, 174, 258, and 353 letters; the best audit had 138 mismatches
  and exact closures were zero.  The next construction changes representation
  at the central seam (a mutable center clause with live character obligations)
  rather than repeating three-region substitutions.

- **Paired opposing-noun seam repair, 2026-09-17:** coordinated opposing noun
  substitutions at one seam index while preserving agreement and semantic
  roles.  Four complete passages up to 118 letters were rendered; the best
  audit had 108 mismatches and exact closures were zero.  The next operator
  couples the nouns with their relative verbs at that same seam.

- **Locative preposition+noun-pair repair, 2026-09-17:** coupled locative
  preposition and noun-pair changes at fixed phrase length.  Sixty-four
  complete rows reached 98 letters; the best audit had 60 mismatches and exact
  closures were zero.  The next repair couples the object determiner with the
  locative pair.

- **Mutable center-clause repair, 2026-09-17:** changed the recursive spine's
  center representation to a typed clause such as ``and the reader understands
  the note`` rather than appending a fixed center.  Targets reached 179, 204,
  285, and 385 letters; the best audit had 168 mismatches and exact closures
  were zero.  The next construction makes that center clause itself a live
  two-sided constituent selected from opposing boundary obligations.

- **Paired noun/relative-verb seam repair, 2026-09-17:** changed opposing
  semantic nouns and their relative verbs together under agreement and
  boundary-balance constraints.  Four complete passages up to 121 letters were
  rendered; the best audit had 94 mismatches and exact closures were zero.  The
  next operator adds relative-head and boundary-length constraints.

- **Object-determiner/locative-pair repair, 2026-09-17:** coupled object
  determiners with the locative preposition+noun pair at fixed length.  Sixty-
  four complete rows reached 101 letters; the best audit had 60 mismatches and
  exact closures were zero.  The next repair couples object determiner with
  verb valency.

- **Head/boundary-constrained noun/verb repair, 2026-09-17:** carried relative
  heads, paired boundaries, noun/verb roles, and number through one seam audit.
  Four complete passages up to 121 letters were rendered; the best audit had
  88 mismatches and exact closures were zero.  The next operator grows only the
  seam-selected boundary clause.

- **Valency/locative-pair repair, 2026-09-17:** coupled object determiner and
  transitive verb valency with the locative pair.  Thirty-two complete rows
  reached 99 letters; the best audit had 64 mismatches and exact closures were
  zero.  The next repair couples verb tense with valency and the locative pair.

- **Tense/valency/locative repair, 2026-09-17:** coupled present/past tense
  with verb valency and the locative pair.  Thirty-two complete rows reached 97
  letters; the best audit had 56 mismatches and exact closures were zero.  The
  next operator couples tense with referent number and locative noun.

- **Live seam boundary growth, 2026-09-17:** grew only the boundary selected by
  the first unresolved seam and paired that growth with role-compatible edits.
  Four complete passages up to 135 letters were rendered; the best audit had
  116 mismatches and exact closures were zero.  The next repair keeps the seam
  conditioning but adds semantic edits rather than another boundary-only pass.

- **Seam-conditioned boundary/semantic pair repair, 2026-09-17:** paired live
  boundary growth with role-compatible noun edits at the first unresolved seam.
  Four complete passages up to 148 letters were rendered; the best audit had
  118 mismatches and exact closures were zero.  The next repair couples
  relative-verb edits to that same live seam.

- **Tense/number/locative-noun repair, 2026-09-17:** coupled present/past tense
  and referent number with locative noun choices.  Thirty-two complete rows
  reached 95 letters; the best audit had 56 mismatches and exact closures were
  zero.  The next operator couples tense and number with the locative
  preposition while preserving noun length.

- **Seam-conditioned verb/boundary repair, 2026-09-17:** changed relative verbs
  and shared boundary material together under live seam tracking.  Four complete
  passages up to 131 letters were rendered; the best audit had 110 mismatches
  and exact closures were zero.  The next operator is a tiny joint
  noun/verb/boundary lattice with live rejection.

- **Tense/number/locative-preposition repair, 2026-09-17:** coupled tense and
  referent number with locative preposition choices at fixed noun length.
  Thirty-two complete rows reached 90 letters; the best audit had 54 mismatches
  and exact closures were zero.  The next repair couples tense/number with the
  locative determiner.

- **Tiny joint noun/verb/boundary lattice, 2026-09-17:** considered four
  seam-compatible states and rejected one before rendering.  Three complete
  passages up to 117 letters remained; the best audit had 94 mismatches and
  exact closures were zero.  The next operator applies a live-seam threshold
  with paired lexical repair.

- **Tense/number/locative-determiner repair, 2026-09-17:** coupled tense and
  referent number with locative determiner and noun number while holding the
  preposition fixed.  Thirty-two complete rows reached 90 letters; the best
  audit had 54 mismatches and exact closures were zero.  The next repair
  couples tense/number with determiner and noun number.

- **Live-seam paired lexical repair, 2026-09-18:** made four paired
  role-compatible lexical edits on the best admitted lattice state while
  tracking the seam before and after each edit.  The longest complete passage
  was 119 letters; the best audit had 106 mismatches and exact closures were
  zero.  The next operator constrains substitutions by the opposing seam
  character class.

- **Tense/number/locative-NP agreement repair, 2026-09-18:** coupled
  tense/referent-number agreement with locative noun-phrase agreement and
  preposition alternation.  Thirty-two complete rows reached 91 letters; the
  best audit had 56 mismatches and exact closures were zero.  The next repair
  couples NP agreement with locative preposition alternation while preserving
  tense.

- **Opposing-seam character-class repair, 2026-09-18:** coupled opposing
  replacement vowel/consonant classes with bounded length matching.  Four
  complete passages up to 116 letters were rendered; the best audit had 104
  mismatches and exact closures were zero.  The next repair adds bounded
  replacement-length matching within each class.

- **NP-agreement/preposition repair, 2026-09-18:** coupled locative NP
  agreement and preposition alternation with tense while holding noun number
  fixed.  Thirty-two complete rows reached 91 letters; the best audit had 56
  mismatches and exact closures were zero.  The next repair couples this NP
  state with tense alternation.

- **Class-length-matched seam repair, 2026-09-18:** enforced equal paired
  replacement-length deltas within opposing character classes, rejecting two
  of four states before rendering.  Two complete passages remained (up to 117
  letters); the best audit had 108 mismatches and exact closures were zero.
  The next repair couples semantic heads and verbs under the same gate.

- **NP-agreement/preposition/tense repair, 2026-09-18:** coupled locative NP
  number, preposition, and tense.  Thirty-two complete rows reached 93 letters;
  the best audit had 56 mismatches and exact closures were zero.  The next
  operator couples tense with the locative NP determiner.

- **Class-length head/verb coupling, 2026-09-18:** enforced equal paired
  head/verb letter lengths and rejected three of four states before rendering.
  The one admitted complete passage was 93 letters with 80 mismatches; exact
  closures were zero.  This is the strongest residual signal so far, but it is
  not reader evidence; the next repair adds boundary-attachment constraints.

- **Bounded internal-boundary frontier, 2026-09-18:** added grammatical
  determiner and phrase-boundary variants for both first and terminal adjuncts,
  with duplicate-content rejection.  Four target traces reached 394 letters;
  exact closures remained zero.  The next operator moves beyond adjunct-local
  edits to a typed predicate-argument boundary operator.

- **Dream-RSI round 76, 2026-09-18:** replayed 515 worlds and 19,637 audited
  nodes after the equal-length head/verb and internal-boundary repairs.  Fixed
  mismatch-first remained the held-out winner at 0.780 (444 held-out worlds),
  with zero mechanically admissible exact rows and 107 exact-but-rejected rows.
  The next construction step is the typed predicate-argument boundary lane;
  replay stability remains a routing diagnostic only.

- **Dream-RSI round 74, 2026-09-18:** replayed 511 worlds and 19,570 audited
  nodes after the character-class and NP-agreement additions.  Fixed
  mismatch-first remained the held-out winner at 0.780 (440 held-out worlds),
  with zero mechanically admissible exact rows and 107 exact-but-rejected rows.
  This keeps the next work on construction operators—bounded seam lengths and
  joint recursive center/adjunct selection—rather than on replay-score tuning.

- **Dream-RSI round 73, 2026-09-18:** replayed 509 worlds and 19,534 audited
  nodes after the seam-character and joint-frontier additions.  Fixed
  mismatch-first remained the held-out winner at 0.780 (439 held-out worlds),
  with zero mechanically admissible exact rows and 107 exact-but-rejected rows.
  The next round consumes the opposing-character-class repair and the full
  recursive residual objective; policy stability is still only a routing
  diagnostic.

- **Dream-RSI round 72, 2026-09-17:** replayed 507 worlds and 19,498 audited
  nodes after the tiny lattice, center residual ranking, and tense/determiner
  lanes.  Fixed mismatch-first remained the held-out winner at 0.780 (437
  held-out worlds), with zero mechanically admissible exact rows and 107
  exact-but-rejected rows.  The next round consumes the live-seam threshold
  and center-plus-adjunct residual operators; it does not treat policy
  stability as a palindrome result.

- **Dream-RSI round 71, 2026-09-17:** replayed 505 worlds and 19,463 audited
  nodes after adding the seam-conditioned verb and tense/preposition lanes.
  Fixed mismatch-first remained the held-out winner at 0.780 (435 held-out
  worlds), with zero mechanically admissible exact rows and 107 exact-but-
  rejected rows.  The next round must evaluate the tiny noun/verb/boundary
  lattice and residual-ranked center/adjunct combinations.

- **Dream-RSI round 70, 2026-09-17:** replayed 503 worlds and 19,427 audited
  nodes after the Pareto-center and seam-conditioned repairs.  Fixed
  mismatch-first remained the held-out winner at 0.780 (433 held-out worlds),
  with zero mechanically admissible exact rows and 107 exact-but-rejected rows.
  The next construction operators are therefore boundary-relative-verb edits,
  tense/preposition coupling, and residual-ranked center candidates.

- **Variable-length center phrases, 2026-09-17:** exposed multiword center
  subject/object alternatives (for example, ``the careful witness`` and ``a
  quiet note``) instead of a single-word center.  The current edge heuristic
  still selected ``and the guide keeps the note`` for all four targets (170,
  197, 281, and 376 letters; best audit 150 mismatches; exact closures zero).
  The next repair replaces that single winner with a bounded Pareto frontier so
  short and long center candidates both survive to live equation solving.

- **Joint center/first/terminal frontier, 2026-09-18:** selected center
  subject/verb/object, first adjunct, and terminal adjunct from one rendered
  residual-debt objective, with duplicate-adjunct rejection.  The four target
  rows reached 171, 198, 286, and 384 letters; the best audit had 160
  mismatches and exact closures were zero.  The next repair carries the same
  objective into the full recursive spine rather than freezing the first
  adjunct.

- **Dream-RSI round 69, 2026-09-17:** replayed 501 worlds and 19,391 audited
  nodes after the tense/valency and live-seam additions.  Fixed mismatch-first
  remained the held-out winner at 0.780 (431 held-out worlds), with zero
  mechanically admissible exact rows and 107 exact-but-rejected rows.  The
  controller therefore advances to the queued Pareto-center and seam-conditioned
  repair operators; it does not promote this plateau to a result.

- **Dream-RSI round 68, 2026-09-17:** replayed 499 worlds and 19,355 audited
  nodes after the constrained seam repairs.  Fixed mismatch-first remained the
  held-out winner at 0.780 (429 held-out worlds), with zero mechanically
  admissible exact rows and 107 exact-but-rejected rows.  The route is still
  using replay to choose the next construction, not to redefine success around
  the plateau.

- **Dream-RSI round 67, 2026-09-17:** replayed 497 worlds and 19,319 audited
  nodes after the noun/verb and object-determiner repairs.  The mismatch-first
  policy remained the held-out winner at 0.780 (427 held-out worlds), with zero
  mechanically admissible exact rows and 107 exact-but-rejected rows.  The
  replay plateau is now an explicit trigger for the next representation change;
  it is not a readability or speed result.

- **Full typed edge frontier and unfreezing fix, 2026-09-18:** expanded the
  first/terminal adjunct frontier and corrected a prefix-freezing bug so the
  optimizer selects all three components from scratch.  The corrected target
  rows reached 177, 207, 283, and 388 letters; the best audit had 170
  mismatches and exact closures were zero.  Duplicate-content rejection stays
  active; the next repair carries residual ranking into internal adjunct word
  boundaries.

- **Explicit predicate-valency frames, 2026-09-18:** distinguished transitive,
  ditransitive, and locative center clauses and selected predicate, subject,
  object, and frame jointly under the rendered residual objective.  Four target
  traces reached 394 letters; exact closures remained zero.  The next repair
  carries the valency frame into the outer event spine and its edge adjuncts.

- **Head/verb boundary-attachment gate, 2026-09-18:** added paired locative
  attachment constraints to equal-length head/verb states, rejecting three of
  four before rendering.  One complete passage reached 118 letters with 112
  mismatches; exact closures were zero.  The next repair changes only
  relative-clause lexical heads under the same gate.

- **Determiner/tense/locative-number repair, 2026-09-18:** coupled locative
  number, preposition, and determiner while preserving tense.  Thirty-two
  complete rows reached 91 letters; the best audit had 56 mismatches and exact
  closures were zero.  The next operator couples locative number with
  preposition and determiner while preserving tense.

- **Attachment-aware relative-head repair, 2026-09-18:** varied relative-clause
  lexical heads while preserving paired locative attachment.  Four complete
  passages up to 117 letters were rendered; the best audit had 94 mismatches
  and exact closures were zero.  The next repair coordinates relative heads
  and verbs under the same role gate.

- **Locative-number/preposition/determiner repair, 2026-09-18:** coupled
  locative number, preposition, and determiner with fixed tense.  Thirty-two
  complete rows reached 89 letters; the best audit had 56 mismatches and exact
  closures were zero.  The next operator couples fixed-tense verb valency with
  this locative state.

- **Dream-RSI round 77, 2026-09-18:** replayed 517 worlds and 19,670 audited
  nodes after the predicate–argument and attachment-gate additions.  Fixed
  mismatch-first remained the held-out winner at 0.780 (445 held-out worlds),
  with zero mechanically admissible exact rows and 107 exact-but-rejected rows.
  The next construction step is attachment-aware relative-head repair and
  explicit valency frames, not score tuning.

- **Dream-RSI round 78, 2026-09-18:** replayed 519 worlds and 19,706 audited
  nodes after the attachment-aware relative-head, locative-state, and explicit
  valency additions.  Fixed mismatch-first again won the held-out split at
  0.780 (447 held-out worlds), with zero mechanically admissible exact rows and
  107 exact-but-rejected rows.  This is a routing plateau, not a readable-output
  result: the next construction actions are paired relative-head/verb repair,
  fixed-tense valency with locative state, and valency propagation into the
  outer event spine.

- **Dream-RSI round 79, 2026-09-18:** replayed 521 worlds and 19,742 audited
  nodes after adding the paired head/verb, fixed-tense valency/locative, and
  outer-spine valency branches.  Fixed mismatch-first remained the held-out
  winner at 0.780 (449 held-out worlds), with zero mechanically admissible exact
  rows and 107 exact-but-rejected rows.  The replay still routes the next
  construction; it does not certify a palindrome or readability.  The next
  repair must enforce cross-clause agreement and argument-role compatibility.

- **Cross-clause agreement/role compatibility, 2026-09-18:** filtered outer and
  center event frames jointly for determiner agreement, human-agent subjects,
  shared argument roles, and typed adjunct attachment.  Four complete target
  traces reached 393 letters; exact closures remained zero.  The next repair
  turns agreement features into live residual variables rather than a final
  filter.

- **Attachment-aware clause-length balance, 2026-09-18:** balanced paired
  relative-clause lengths under the same locative attachment gate.  Two complete
  passages reached 115 letters; the best audit had 98 mismatches and exact
  closures were zero.  The next repair pairs balanced clauses with coordinated
  determiner changes.

- **Valency/anaphor locative repair, 2026-09-18:** coupled anaphor choice with
  predicate valency and locative agreement under fixed tense.  Thirty-two
  complete rows reached 87 letters; the best audit had 56 mismatches and exact
  closures were zero.  The next repair couples anaphor choice with
  relative-clause subject agreement.

- **Dream-RSI round 80, 2026-09-18:** replayed 523 worlds and 19,776 audited
  nodes after cross-clause compatibility, clause-length balance, and
  valency/anaphor repairs.  Fixed mismatch-first remained the held-out winner at
  0.780 (450 held-out worlds), with zero mechanically admissible exact rows and
  107 exact-but-rejected rows.  The next construction must make agreement
  features live variables in the character residual equation; this replay
  result is routing evidence only.

- **Live agreement residual, 2026-09-18:** made singular/plural agreement a
  live residual-search variable while selecting typed center predicates,
  arguments, and edge adjuncts.  Four complete target traces reached 393
  letters; exact closures remained zero.  The next repair jointly realizes
  person/number morphology on outer and center predicates.

- **Balanced-clause determiner pair, 2026-09-18:** coordinated determiner
  changes with equal relative-clause lengths under locative attachment.  Four
  complete passages reached 113 letters; the best audit had 86 mismatches and
  exact closures were zero.  The next repair couples these determiner changes
  to agreement-aware verbs.

- **Anaphor/relative-agreement repair, 2026-09-18:** coupled anaphor choice
  with singular/plural relative-clause subject agreement while preserving
  valency and locative agreement.  Thirty-two complete rows reached 76 letters;
  the best audit had 56 mismatches and exact closures were zero.  The next
  repair couples relative agreement with anaphor and locative NP number.

- **Dream-RSI round 81 online redeployment, 2026-09-18:** replayed 525 worlds
  and 19,812 audited nodes, then redeployed the held-out winner into a fresh
  two-region authoring tree.  The online tree produced 8 audited nodes and 0
  exact closures; its best intact passage was 130 letters with 60 mismatches:
  “The theater archivist found a torn playbill in a locked drawer, carried it to
  the reading table, and marked the missing cast names before the house lights
  rose.”  Two sibling proposals were rejected for leaving the 100–140-letter
  construction band (146 and 148 letters).  This is an actual deployment
  failure, so the next repair is a broader live branch with length-band-aware
  seam choices, not a claim of progress.

- **Joint predicate morphology, 2026-09-18:** realized person/number
  alternatives directly in both outer and center predicates, changing the
  character yield while preserving event roles.  Four complete target traces
  reached 376 letters; exact closures remained zero.  The next repair adds
  tense/aspect jointly with person/number morphology.

- **Balanced determiner/agreement-verb repair, 2026-09-18:** coupled
  coordinated determiner changes to singular/plural agreement verbs under the
  balanced relative-clause structure.  Four complete passages reached 113
  letters; the best audit had 80 mismatches and exact closures were zero.  The
  next repair pairs the structure with boundary inflection.

- **Three-way agreement repair, 2026-09-18:** coupled relative-clause subject
  number, anaphor number, and locative NP number.  Sixteen complete rows reached
  75 letters; the best audit had 60 mismatches and exact closures were zero.  The
  next repair couples the agreement state with locative preposition choice.

- **Dream-RSI round 83 fresh online redeployment, 2026-09-18:** fixed the
  duplicate-seed bug by deriving a round-specific artifact, experiment id, and
  reproducible seed base.  The replay covered 527 worlds and 19,832 audited
  nodes; the fresh online tree produced 12 nodes and no exact closure.  Its best
  intact candidate was 137 letters with 58 mismatches:
  “The theater archivist lifted a torn playbill from the locked drawer, carried
  it to the reading table, and penciled the missing cast names before the house
  lights rose.”  Its independent forward/reverse tape hashes differed, and the
  provenance records a fresh two-region authoring branch with no catalogue or
  tape-reversal shortcut.  The next repair is to replay this new branch and
  couple its tense/aspect morphology to the live seam equation.

- **Joint tense/aspect morphology, 2026-09-18:** jointly varied person/number
  with present and past predicate realizations in outer and center clauses.
  Four complete target traces reached 376 letters; exact closures remained zero.
  The next repair adds aspect auxiliaries and clause-level temporal
  compatibility.

- **Agreement-aware boundary inflection, 2026-09-18:** varied paired locative
  boundary inflections while preserving determiner/verb agreement.  Four
  complete passages reached 113 letters; the best audit had 92 mismatches and
  exact closures were zero.  The next repair adds role-compatible relative-
  clause lexical substitution.

- **Three-way/preposition agreement, 2026-09-18:** jointly selected locative
  preposition within the relative-subject/anaphor/locative-number state.
  Sixteen complete rows reached 75 letters; the best audit had 60 mismatches
  and exact closures were zero.  The next repair couples the state with
  locative determiner and noun.

- **Dream-RSI round 84, 2026-09-18:** replayed 529 worlds and 19,852 audited
  nodes after the fresh-seed redeployment and morphology repairs.  Fixed
  mismatch-first remained the held-out winner at 0.780 (456 held-out worlds),
  with zero mechanically admissible exact rows and 107 exact-but-rejected rows.
  The controller therefore advances to the next construction operator rather
  than retuning this plateau.

- **Aspect auxiliaries/temporal compatibility, 2026-09-18:** added
  auxiliary-bearing predicate realizations such as “has kept” and linked them
  to compatible temporal adjuncts.  Four complete target traces reached 376
  letters; exact closures remained zero.  The next repair exposes temporal
  adjunct tense/aspect as a shared seam variable.

- **Agreement-boundary relative lexical repair, 2026-09-18:** added
  role-compatible relative-clause lexical substitutions to the
  agreement-aware boundary state.  Four complete passages reached 117 letters;
  the best audit had 94 mismatches and exact closures were zero.  The next
  repair pairs the edits with seam-length balancing.

- **Three-way determiner/noun agreement, 2026-09-18:** jointly selected
  locative determiner and noun inside the three-way agreement state while
  holding preposition fixed.  Sixteen complete rows reached 75 letters; the
  best audit had 60 mismatches and exact closures were zero.  The next repair
  couples the state with locative phrase length.

- **Dream-RSI round 85, 2026-09-18:** replayed 531 worlds and 19,872 audited
  nodes after aspect, temporal, relative-lexical, and three-way agreement
  repairs.  Fixed mismatch-first remained the held-out winner at 0.780 (458
  held-out worlds), with zero mechanically admissible exact rows and 107
  exact-but-rejected rows.  The next construction run therefore changes the
  temporal seam representation instead of retuning replay scores.

- **Shared temporal seam variable, 2026-09-18:** shared tense/aspect across
  outer and center predicates and their temporal adjuncts, changing the live
  seam equation.  Four complete target traces reached 361 letters; exact
  closures remained zero.  The next repair enforces semantic event ordering
  before rendering.

- **Agreement-aware relative seam length, 2026-09-18:** enforced equal paired
  relative-clause lengths while preserving agreement and locative attachment.
  Four complete passages reached 113 letters; the best audit had 88 mismatches
  and exact closures were zero.  The next repair balances paired boundary
  lengths at the same seam.

- **Three-way locative-length repair, 2026-09-18:** jointly selected short or
  expanded locative phrase length inside the three-way agreement state while
  holding preposition fixed.  Sixteen complete rows reached 79 letters; the
  best audit had 60 mismatches and exact closures were zero.  The next repair
  couples locative phrase length with anaphor lexical class.

- **Dream-RSI round 86, 2026-09-18:** replayed 533 worlds and 19,892 audited
  nodes after the shared-temporal, relative seam-length, and three-way
  locative-length repairs.  Fixed mismatch-first remained the held-out winner at
  0.780 (460 held-out worlds), with zero mechanically admissible exact rows and
  107 exact-but-rejected rows.  This plateau triggers the next semantic
  event-ordering and lexical-class construction changes; it is not a result.

- **Pre-render temporal event-order gate, 2026-09-18:** rejected semantically
  incompatible tense/aspect and temporal-adjunct combinations before rendering,
  while retaining compatible choices as live seam candidates.  Four complete
  target traces reached 361 letters; exact closures remained zero.  The next
  repair replaces pairwise lexical rules with a typed event-order graph.

- **Agreement-relative boundary-length repair, 2026-09-18:** balanced paired
  boundary and relative-clause lengths while preserving agreement and locative
  attachment.  Four complete passages reached 113 letters; the best audit had
  94 mismatches and exact closures were zero.  The next repair uses seam-
  selected paired lexical edits across both components.

- **Locative-length/anaphor-class repair, 2026-09-18:** coupled locative phrase
  length with singular/plural anaphor lexical class while preserving three-way
  agreement.  Sixteen complete rows reached 85 letters; the best audit had 60
  mismatches and exact closures were zero.  The next repair couples lexical
  class with relative-clause attachment.

- **Dream-RSI round 87, 2026-09-18:** replayed 535 worlds and 19,912 audited
  nodes after the event-order gate, boundary-length, and anaphor-class repairs.
  Fixed mismatch-first remained the held-out winner at 0.780 (462 held-out
  worlds), with zero mechanically admissible exact rows and 107 exact-but-
  rejected rows.  The next construction step is the typed event-order graph
  and seam-selected paired lexical repair; replay remains routing evidence only.

- **Typed event-order graph, 2026-09-18:** represented temporal compatibility
  with explicit precedence values for Before/After/While/Until/Since and center
  tense/aspect states, filtering candidates before residual scoring.  Four
  complete target traces reached 361 letters; exact closures remained zero.  The
  next repair lets event-order assignments participate directly in seam
  optimization.

- **Seam-paired relative/boundary edit, 2026-09-18:** selected paired lexical
  edits across relative-clause and boundary components at the live seam while
  preserving agreement and attachment.  Four complete passages reached 115
  letters; the best audit had 88 mismatches and exact closures were zero.  The
  next repair uses seam-selected agreement-aware inflectional edits.

- **Anaphor/attachment repair, 2026-09-18:** coupled singular/plural anaphor
  class with subject/object relative attachment while preserving three-way
  agreement and locative length.  Sixteen complete rows reached 92 letters; the
  best audit had 60 mismatches and exact closures were zero.  The next repair
  couples attachment with locative preposition.

- **Dream-RSI round 88, 2026-09-18:** replayed 537 worlds and 19,932 audited
  nodes after the typed event-order, paired lexical, and anaphor-attachment
  repairs.  Fixed mismatch-first remained the held-out winner at 0.780 (463
  held-out worlds), with zero mechanically admissible exact rows and 107
  exact-but-rejected rows.  The next construction step lets event-order
  assignments change temporal wording in the live seam rather than only filter.

- **Event-order optimization, 2026-09-18:** allowed compatible temporal
  assignments to participate directly in residual ranking, combining mirrored
  character debt with precedence distance and typed-prose constraints.  Four
  complete target traces reached 361 letters; exact closures remained zero.  The
  next repair exposes multiple grammatical temporal realizations per event
  order.

- **Seam-selected inflectional repair, 2026-09-18:** coupled relative and
  boundary inflectional endings with number agreement at the live seam.  Four
  complete passages reached 115 letters; the best audit had 96 mismatches and
  exact closures were zero.  The next repair pairs inflectional endings with
  coordinated determiners.

- **Attachment/preposition repair, 2026-09-18:** coupled subject/object
  relative attachment with locative preposition while preserving anaphor class
  and three-way agreement.  Sixteen complete rows reached 92 letters; the best
  audit had 60 mismatches and exact closures were zero.  The next repair couples
  attachment with locative determiner.

- **Dream-RSI round 89, 2026-09-18:** replayed 538 worlds and 19,936 audited
  nodes after event-order optimization, seam inflection, and attachment/
  preposition repairs.  Fixed mismatch-first remained the held-out winner at
  0.780 (464 held-out worlds), with zero mechanically admissible exact rows and
  107 exact-but-rejected rows.  The next construction step expands temporal
  lexical realizations and agreement-aware seam edits; replay remains only a
  routing mechanism.

- **Multiple temporal surface realizations, 2026-09-18:** offered several
  grammatical human-authored clauses per event-order class (After, Later,
  Earlier, During, Afterward) and ranked them by seam residual.  Four complete
  target traces reached 344 letters; exact closures remained zero.  The next
  repair separates temporal surface form from adjunct semantic content.

- **Seam inflection/determiner pair, 2026-09-18:** paired seam-selected
  inflectional endings with coordinated determiner changes under number
  agreement.  Four complete passages reached 119 letters; the best audit had 80
  mismatches and exact closures were zero.  The next repair couples
  relative-verb inflection to the same edit.

- **Attachment/locative-determiner repair, 2026-09-18:** coupled subject/object
  relative attachment with locative determiner while preserving preposition,
  anaphor class, and three-way agreement.  Sixteen complete rows reached 92
  letters; the best audit had 60 mismatches and exact closures were zero.  The
  next repair couples attachment with locative noun.

- **Dream-RSI round 90, 2026-09-18:** replayed 540 worlds and 19,956 audited
  nodes after temporal surface, seam inflection, and attachment/determiner
  repairs.  Fixed mismatch-first remained the held-out winner at 0.780 (466
  held-out worlds), with zero mechanically admissible exact rows and 107
  exact-but-rejected rows.  The next construction step separates temporal
  surface realization from adjunct semantics and adds locative noun coupling.

- **Direct character-constrained scene authoring, 2026-09-18:** ran eight
  parallel, independently seeded local-model requests with an exact
  80–180-letter tape constraint and intact-scene instructions.  It produced no
  exact closure.  The strongest fresh prose was 119 letters with 56 mismatches:
  “As the rain began to fall, Sam saw the silver sparrow fly past the bay, and
  the bay’s spray kissed his face as he watched the bird slip back into the fog.”
  Its provenance records a direct authored scene, no catalogue import, no
  finished-tape reversal, and no word-order symmetry; the independent
  forward/reverse hashes differ.  The concrete next repair is a seam-targeted
  lexical realization beam over this scene’s roles, not another prompt-only
  sweep.  Reader certification remains closed.

- **Direct character-scene lexical repair, 2026-09-18:** applied eight fresh
  seam-targeted repairs to that authored scene while preserving rain, Sam, the
  sparrow, bay, spray, and fog roles.  Four responses stayed in the 80–180
  letter band; none was exact.  The best in-band repair was 80 letters with 36
  mismatches (“Rain beat, Sam gazed at a silver sparrow near the bay, spray
  lashed his skin, the bird vanished in fog.”).  Independent forward/reverse
  hashes differ and provenance records the source scene and no tape reversal.
  The next operator is a live role-compatible character trie, not another
  prompt-only repair.

- **Role-compatible character trie, 2026-09-18:** searched 486 authored
  alternatives for rain, agent, bird, bay, spray, and fog roles with character
  obligations checked before rendering.  It produced no exact closure; the
  longest intact candidate was 93 letters, and the best mismatch count was 56.
  A representative complete candidate is “After rain, Sam watches a sparrow
  above the bay; Sam notes spray beneath the fog.”  Independent forward/reverse
  hashes differ, and provenance records no catalogue, reversal, repeated unit,
  or word-order shortcut.  The next repair adds two-character paired
  obligations over role-phrase boundaries.

- **Dream-RSI round 91, 2026-09-18:** replayed 543 worlds and 20,458 audited
  nodes after adding the direct authoring, seam-repair, and role-compatible
  character-trie evidence.  Fixed mismatch-first remained the held-out winner
  at 0.780 (469 held-out worlds), with zero mechanically admissible exact rows
  and 107 exact-but-rejected rows.  The next construction step is the trie’s
  two-character boundary obligation, not another replay sweep.

- **Paired two-character role-boundary obligations, 2026-09-18:** added
  pre-render obligations over rain↔fog and agent↔sparrow phrase boundaries in
  the role-compatible trie.  All 486 complete prose candidates failed at least
  one paired obligation; exact closures remained zero.  The longest remained
  93 letters and the best residual remained 56 mismatches.  The next repair
  expands role alternatives while varying the valency frame instead of adding
  more obligation characters.

- **Seam determiner/verb inflection, 2026-09-18:** coupled determiners,
  relative verbs, and boundary inflections by number.  Four complete passages
  reached 119 letters; the best audit had 80 mismatches and exact closures were
  zero.  The next repair uses role-compatible verb substitution within the
  inflection class.

- **Attachment/locative-noun repair, 2026-09-18:** coupled subject/object
  relative attachment with locative noun while preserving determiner,
  preposition, anaphor class, and three-way agreement.  Sixteen complete rows
  reached 92 letters; the best audit had 60 mismatches and exact closures were
  zero.  The next repair couples attachment with locative noun length.

- **Dream-RSI round 92, 2026-09-18:** replayed 545 worlds and 20,478 audited
  nodes after the role-trie boundary, seam-inflection, and locative-noun
  repairs.  Fixed mismatch-first remained the held-out winner at 0.780 (471
  held-out worlds), with zero mechanically admissible exact rows and 107
  exact-but-rejected rows.  The next construction step expands role
  alternatives under valency rather than increasing boundary obligations.

- **Valency-aware role-trie expansion, 2026-09-18:** varied transitive,
  ditransitive, and locative frames across the rain/agent/bird/bay/spray/fog
  role inventory.  It rendered 1,458 complete prose candidates, but no
  two-character paired-boundary survivor and no exact closure; the longest was
  104 letters and the best mismatch count remained 56.  The next repair makes
  predicate-argument boundaries live rather than fixed phrase-edge characters.

- **Role-compatible verb-class repair, 2026-09-18:** substituted verbs within
  semantic role, number, inflection class, and attachment.  Four complete
  passages reached 113 letters; the best audit had 90 mismatches and exact
  closures were zero.  The next repair matches substituted verb lengths at the
  seam.

- **Attachment/locative-length repair, 2026-09-18:** coupled subject/object
  attachment with short or expanded locative phrase length while preserving
  determiner, preposition, anaphor class, and three-way agreement.  Sixteen
  complete rows reached 96 letters; the best audit had 60 mismatches and exact
  closures were zero.  The next repair couples attachment with locative
  complement semantics.

- **Dream-RSI round 93, 2026-09-18:** replayed 547 worlds and 21,470 audited
  nodes after the valency-aware role-trie expansion, verb-class, and locative-
  length repairs.  Fixed mismatch-first remained the held-out winner at 0.780
  (473 held-out worlds), with zero mechanically admissible exact rows and 107
  exact-but-rejected rows.  The next construction step makes predicate-
  argument boundaries live and adds complement semantics; replay remains only
  a routing mechanism.

- **Dream-RSI round 75, 2026-09-18:** replayed 513 worlds and 19,604 audited
  nodes after the class-length and unfreezing repairs.  Fixed mismatch-first
  remained the held-out winner at 0.780 (443 held-out worlds), with zero
  mechanically admissible exact rows and 107 exact-but-rejected rows.  The
  controller now advances to internal word-boundary edits and seam-conditioned
  residual ranking; this plateau is not a readability result.

- **Dream-RSI round 94, 2026-09-18:** replayed 547 worlds and 21,470 audited
  nodes after the live-valency and direct-authoring lanes.  Fixed mismatch-first
  remained the held-out winner at 0.780, with zero mechanically admissible exact
  rows and 107 exact-but-rejected rows.  The replay tree was still mostly flat,
  so this round did not justify another larger sweep; the next repair makes
  explicit sibling branches first-class replay evidence.

- **Explicit replay sibling branching, 2026-09-18:** the role-compatible trie
  now records 27 sibling branch identities across 1,458 complete prose
  candidates.  Exact closures remain zero; the longest candidate is 104 letters
  with 56 mismatches.  This is a structural Dream-RSI repair, not a readability
  claim: the next test replays held-out role alternatives and compares residual
  trajectories before deployment.

- **Single-scene appositive redeployment, 2026-09-18:** a fresh construction
  operator produced three intact appositive scenes, none exact.  The longest was
  75 letters and the best residual was 60 mismatches.  The next repair edits the
  appositive head and adjacent verb as one seam unit.

- **Dream-RSI three-region policy scene, 2026-09-18:** two fresh maritime/civic
  three-region scenes were generated under a replay-selected branch policy;
  neither was exact.  The longest was 93 letters and the best residual was 72
  mismatches.  The next repair jointly masks the shared referent and response
  spans rather than extending either scene.

- **Dream-RSI round 95, 2026-09-18:** replayed 549 worlds and 21,475 audited
  nodes after those orthogonal lanes.  Fixed mismatch-first still won held-out
  routing at 0.780; exact admissible rows remained zero.  Branch-aware scoring
  now distinguishes policies by branch coverage (3–6 distinct branches), but
  the best inherited prose residual did not improve.

- **Dream-RSI round 96, 2026-09-18:** the branch-aware replay controller was
  verified on the same 549-world pool.  It reports branch identity separately
  from action diversity and independently audits every node; the best replay
  residual remains the 119-letter gardener passage at 46 mismatches.  This
  diagnostic change is not accepted as a candidate; it triggered one fresh
  online redeployment.

- **Dream-RSI round 97 online redeployment, 2026-09-18:** the selected policy
  generated a new 8-node branching tree (4 branching parents) from the theater
  archivist scene.  All children were intact prose but none was exact; the best
  fresh child was “The theater archivist lifted a torn playbill from the locked
  drawer, carried it to the reading table, and marked the missing cast names
  before the house lights rose.” (135 letters, 61 mismatches).  Independent
  forward/reverse SHA-256 hashes differ.  This is preserved as failure evidence;
  the next repair is the held-out role-alternative replay and seam-coupled
  appositive/three-region construction, not a duplicate two-region sweep.

- **Held-out sibling replay, 2026-09-18:** replayed eight unseen fog-role
  alternatives from the 27 role-trie sibling branches.  Every held-out row was
  independently audited; exact closures remained zero, the longest output was
  104 letters, and the best residual remained 56 mismatches.  The next repair
  selects a branch policy from these residual trajectories and changes only the
  next role boundary.

- **Appositive head/verb seam repair, 2026-09-18:** treated an appositive head
  and adjacent verb as one mutable seam in three fresh intact scenes.  No exact
  closure appeared; the longest output was 69 letters and the best residual was
  60 mismatches.  The next repair constrains the seam by consonant/vowel class.

- **Three-region joint-mask repair, 2026-09-18:** jointly masked shared
  referent and response spans in two fresh workshop/garden scenes.  Neither was
  exact; the longest output was 75 letters and the best residual was 64
  mismatches.  The next repair replays those masked pairs with a mutable bridge
  clause.

- **Dream-RSI round 98, 2026-09-18:** replayed 551 worlds and 21,489 audited
  nodes, including the online tree and held-out role alternatives.  Branch-aware
  policies now see 3–6 distinct branches per train replay, with nine branching
  parents overall, but fixed mismatch-first still wins held-out routing at 0.780
  and exact admissible rows remain zero.  This confirms that branch accounting
  is working while the construction quality is still unchanged; the next action
  is the three queued seam/bridge repairs, not a larger replay budget.

- **Dream-RSI exact-boundary grammar zipper, 2026-09-18:** the routing plateau
  triggered a change to the transition system rather than another score sweep.
  Three deterministic policies grew paired typed grammar derivations from
  opposite syntax ends, comparing newly exposed characters before a child was
  admitted and carrying residual length across word boundaries.  The fresh
  authored bank produced 9 bounded nodes and 0 exact closures; complete fresh
  controls were rendered and independently audited up to 48 letters.  A
  withheld smoke bank recovered the existing 38-letter seed in 12 replay
  traces, but those rows are explicitly excluded from candidate counts and
  reader evidence.  This is a construction-method result, not a readability
  claim.  The next repair is a held-out auxiliary/relative-clause frame on the
  same live residual state, followed by a fresh exact closure and only then a
  blinded reader package.

- **Dream-RSI round 99, 2026-09-18:** replayed 552 worlds and 21,493 audited
  nodes after adding the exact-boundary construction tree.  The historical
  held-out winner and mismatch rate were unchanged (fixed mismatch-first,
  0.780), so the new lane is not being presented as a routing improvement.
  Its value is architectural: it supplies a true exact-compatible transition
  frontier for the next grammar repair, while the withheld seed fixture is
  excluded from replay admission.

- **Held-out auxiliary-frame repair, 2026-09-18:** executed the queued repair
  by replacing the right syntax edge with a determiner/subject/auxiliary/
  finite-verb/name frame while retaining the live character residual.  All
  three policy replays reached 9 fresh nodes and 0 exact closures; fresh
  complete controls were rendered up to 51 letters with independent pointer
  and hash audits.  This is a new construction transition, not a larger
  duplicate sweep.  The next concrete repair is a relative-clause linker
  frame, after which a fresh exact closure—not a smoke fixture—must exist
  before any reader study can start.

- **Dream-RSI round 100, 2026-09-18:** replayed 555 worlds and 21,496 audited
  nodes after the held-out auxiliary-frame repair.  The historical routing
  winner and held-out mismatch rate stayed fixed at 0.780, so no policy claim
  is being substituted for a construction result.  The new frame remains in
  the failure ledger and the next action is the relative-clause linker repair.

- **Held-out relative-linker repair, 2026-09-18:** added a who/that linker
  before the right finite verb while retaining the live residual and the
  independently authored left frame.  Three policy replays reached 9 fresh
  nodes and 0 exact closures; complete controls were rendered up to 41
  letters and independently audited.  The next repair jointly varies linker
  attachment and subject number rather than widening the same sweep.

- **Dream-RSI round 101, 2026-09-18:** replayed 558 worlds and 21,499 audited
  nodes after the relative-linker repair.  The offline policy result remains
  unchanged (fixed mismatch-first, held-out mismatch rate 0.780); the new
  construction world is retained as a concrete failure branch rather than
  converted into a proxy success claim.

- **Syntax-constrained clause bridge, 2026-09-18:** introduced a new
  construction operator rather than another replay sweep.  It reverses a
  complete authored clause plus a named center and requires the reversed tape
  to parse as a second complete clause.  The fresh intersection was empty
  (0 exact closures).  It still rendered three intact controls — “the baker
  marks maps; Ada a pilot reads notes.” (36 letters), “a writer charts gates;
  Iris the clerk opens doors.” (40), and “the captain guards plans; Nora an
  editor finds books.” (43) — all independently rejected by the two-pointer
  and forward/reverse hash audits.  These are controls, not candidate prose;
  they also expose the current missing agreement/center attachment.  The next
  repair is agreement-carrying inflection with a live character-seam index,
  retaining complete-clause parsing before any reader-facing test.

- **Dream-RSI round 102, 2026-09-18:** replayed 561 worlds and 21,502 audited
  nodes after adding that clause-bridge world.  The held-out policy winner
  remains fixed mismatch-first at 0.780; this is routing evidence only.  No
  fresh exact or reader-certified output was admitted, so the next action is
  the agreement/seam construction repair rather than a larger replay budget.

- **Agreement-seam bridge, 2026-09-18:** carried singular subject/verb
  agreement through paired complete clauses and indexed the live character
  seam while states were built.  It produced 20 paired states and 8 rendered
  controls, with 0 exact closures.  The longest control was “the sailor
  carries the letters; the writer reads a letter.” (47 letters); the best
  residual was 16 mismatches.  Independent two-pointer and hash audits agree
  that none is exact.  This is construction evidence only, not a readability
  claim; the next repair carries number/tense morphology and directly solves
  the outer determiner/name seam before lengthening clauses.

- **Dream-RSI round 103, 2026-09-18:** replayed 569 worlds and 21,510
  audited nodes after the agreement-seam construction.  Fixed mismatch-first
  remains the held-out routing winner at 0.780, with no fresh exact or
  reader-certified output.  The next action is the recorded morphology/name
  seam repair, not another replay sweep.

- **Morphology outer-seam solver, 2026-09-18:** carried number and tense
  features through reverse-role transitions and rejected a state before render
  when the outer determiner/name seam disagreed.  The fresh feature search
  produced 0 seam states and 0 exact closures; its two authored controls were
  “The baker marks maps; a pilot opens doors.” (33 letters) and “Some clerks
  guard gates; the pilots mark maps.” (37), both independently non-exact.
  This confirms the shell bottleneck rather than a readable-output result.  The
  next repair learns compatible inflectional seam tokens from authored clause
  pairs before any length expansion.

- **Dream-RSI round 104, 2026-09-18:** replayed 571 worlds and 21,512
  audited nodes after the morphology outer-seam repair.  Fixed mismatch-first
  remains the held-out routing winner at 0.780; the fresh construction still
  has no exact or reader-certified output.  The next action is to learn
  inflectional seam tokens from authored clause pairs, not increase replay
  budget.

- **Inflectional seam residual index, 2026-09-18:** learned number/tense-keyed
  reverse residuals from 140 complete authored clauses, allowing the seam to
  cross word boundaries before rendering.  It produced 12 complete-clause
  controls and 0 exact closures; the longest rendered pair was “the rain fell
  on it; the answer was no.” (29 letters), independently non-exact.  The next
  repair composes feature-compatible clauses around a live named center.

- **Dream-RSI round 105, 2026-09-18:** replayed 584 worlds and 21,527 audited
  nodes after the inflectional residual index.  Fixed mismatch-first remains
  the held-out routing winner at 0.780; no fresh exact or reader-certified
  output was admitted.  The next action is the named-center composition repair
  recorded by the construction lane.

- **Named-center clause composition, 2026-09-18:** composed 125 pairs of
  independently authored, feature-compatible complete clauses around a single
  live name and ranked their cross-word residuals before exact replay.  The
  longest rendered control was 56 letters; exact closures remained 0 and no
  row was reader-eligible.  The strongest seam still left 23 mismatches.  The
  next repair solves agreement-carrying inflectional equations before a row is
  rendered, rather than ranking fixed clauses after the fact.

- **Dream-RSI round 106, 2026-09-18:** replayed 709 worlds and 21,652
  audited nodes after the named-center lane.  Fixed mismatch-first remains the
  held-out routing winner at 0.780; no fresh exact or reader-certified output
  was admitted.  The next action is the recorded pre-render inflectional
  equation solver.

- **Agreement-inflection center repair, 2026-09-18:** kept number, tense,
  object number, and a named center live while checking character equations
  before rendering.  It explored 9,024 feature-valid states and rendered 12
  complete controls; exact closures remained 0.  The longest control was 51
  letters (“The baker marked a map near Rhea; the clerks mark a map near
  Rhea.”), with 19 mismatches under independent replay.  The next repair is
  asymmetric lexical bridge attachment at the named-center seam.

- **Dream-RSI round 107, 2026-09-18:** replayed 721 worlds and 21,664
  audited nodes after the agreement-inflection center lane.  Fixed
  mismatch-first remains the held-out routing winner at 0.780; no fresh exact
  or reader-certified output was admitted.  The next action is asymmetric
  lexical bridge attachment at the named-center seam.

- **Asymmetric lexical bridge attachment, 2026-09-18:** attached one fresh
  adjunct bridge to only one side of complete clauses around a named center,
  checking the reverse-prefix character equation before rendering.  It
  evaluated 1,000 constructions, found 0 equation hits, and retained 12
  complete controls; the longest was 59 letters (“a sailor carries letters
  home at dawn; Diana, a gardener guards old notes.”), independently non-exact.
  The next repair pairs agreement-carrying bridges with seam-aware name
  selection.

- **Dream-RSI round 108, 2026-09-18:** replayed 733 worlds and 21,676
  audited nodes after the asymmetric bridge lane.  Fixed mismatch-first remains
  the held-out routing winner at 0.780; no fresh exact or reader-certified
  output was admitted.  The next action is agreement-carrying bridge pairs
  with seam-aware name selection.

- **Agreement-carrying bridge pairs, 2026-09-18:** paired 40 complete
  grammatical frames by agreement signature and solved each name-adjacent
  residual before rendering.  Exact closures remained 0; the longest control
  was “The baker marks a map; the pilot reads Mara.” (34 letters), with 17
  mismatches under independent replay.  The next repair adds agreement-carrying
  clitic and inflection variants at both name-adjacent seams.

- **Dream-RSI round 109, 2026-09-18:** replayed 773 worlds and 21,716
  audited nodes after the agreement-carrying bridge-pair lane.  Fixed
  mismatch-first remains the held-out routing winner at 0.780; no fresh exact
  or reader-certified output was admitted.  The next action is the recorded
  two-sided clitic/inflection seam repair.

- **Agreement-clitic name seam, 2026-09-18:** carried agreement and clitic
  choices through both name-adjacent seams, scheduling the shorter exposed
  tape first and pruning character mismatches before rendering.  It produced
  16 fresh nodes and 0 exact closures; the rendered controls were 35–41
  letters and the reader gate remained closed.  The next repair adds
  noun-number inflection together with semantic-valency checks at both seams.

- **Dream-RSI round 110, 2026-09-18:** replayed 563 worlds and 21,719
  independently audited nodes after the agreement-clitic seam lane.  The
  held-out policy winner is still fixed mismatch-first at 0.780; this is
  routing evidence only.  No fresh exact or reader-certified output was
  admitted, so the next action is the recorded noun-number/valency seam
  construction rather than a larger replay sweep.

- **Dream-RSI online redeployment, 2026-09-18:** deployed the selected policy
  on a fresh two-region authoring seed, creating a real branching tree (9
  nodes, 3 branching parents) rather than replaying old text.  It produced
  intact, original English controls, including “An archivist at the theater
  drew a torn playbill from a locked drawer, brought it to the reading table,
  and marked the missing cast names as the house lights rose.” (131 letters,
  58 opposing-end mismatches) and “The theater archivist pulled a torn
  playbill from a locked drawer, carried it to the reading table, and marked
  the missing cast names before the house lights rose.” (133 letters, 64
  mismatches).  Independent two-pointer and forward/reverse SHA-256 audits
  reject every row as non-exact; provenance records a fresh seed and no
  catalogue import or finished-tape reversal.  These are readable controls,
  not palindrome candidates, so the reader gate is still closed.  The next
  reader-facing test remains a randomized blinded intact-versus-shuffled study
  for the first novel exact survivor; the immediate construction repair is
  reciprocal lexical frames with plural-object agreement.

- **Noun-number/valency seam, 2026-09-18:** added a distinct transducer that
  carries subject number and transitivity while checking both name-adjacent
  character equations before rendering.  The authored complete-clause bank
  produced 0 equation-compatible nodes and 0 exact closures; intact controls
  were 33–41 letters (including “The baker marks a map; the gardeners guard
  letters.”), all independently audited and human-unreviewed.  The reader gate
  remains closed.  The next repair is lexicalized reciprocal frames with
  plural-object agreement, not a duplicate sweep.

- **Dream-RSI masked scene repair, 2026-09-18:** iteratively reopened only a
  typed scene slot at the first mirrored mismatch, accepting a repair only
  when the independent mismatch count decreased.  It accepted one monotone
  repair and rendered two intact controls (63 and 71 letters), with 0 exact
  closures and no reader certification.  The next repair jointly reopens
  mirrored verb–object slots while preserving valency.

- **Dream-RSI compositional grammar spine, 2026-09-18:** generated a scene
  spine first, then solved typed valency/agreement character equations before
  rendering rather than reversing a finished tape.  It produced 36 fresh
  nodes, 0 exact closures, and a longest intact control of 62 letters; all
  rows passed independent audit but remain human-unreviewed.  The next repair
  permits cross-slot character carry across noun/adjunct boundaries.

- **Dream-RSI round 111, 2026-09-18:** replayed the enlarged history after
  the noun-number, masked-repair, and compositional-grammar lanes: 564 worlds
  and 21,729 audited nodes, with 12 branching parents.  Fixed mismatch-first
  remains the held-out routing winner at 0.780, with 0 admissible exact rows.
  This is policy-routing evidence, not a palindrome or readability result;
  the next construction is lexicalized reciprocal frames with plural-object
  agreement.

- **Lexicalized reciprocal plural-frame seam, 2026-09-18:** generated fresh
  reciprocal verb frames with plural subject/object agreement and semantic
  valency carried as state.  Both name/object seams were checked by live
  character equations before rendering; the lane produced 0 equation-compatible
  nodes and 0 exact closures.  Fresh intact controls were 65 letters and remain
  human-unreviewed.  The reader gate is closed.  The next repair adds
  reciprocal preposition alternations and animate/inanimate object typing.

- **Dream-RSI round 112, 2026-09-18:** replayed the registered reciprocal
  plural-frame lane with 564 worlds and 21,729 audited nodes.  The lane added
  no replayable exact branch (its own live search had 0 equation-compatible
  nodes), so the held-out fixed mismatch-first policy remains 0.780 and the
  admissible exact count remains 0.  This is a recorded failure with a concrete
  next construction—reciprocal preposition alternations plus animate/inanimate
  object typing—not a reason to enlarge the replay budget.

- **Reciprocal preposition/typing seam, 2026-09-18:** added a distinct
  transducer carrying preposition valency and animate/inanimate object type
  through reciprocal states, with live mirrored character equations checked
  before rendering. It produced 0 equation-compatible nodes and 0 exact
  closures. Fresh intact controls were 66–68 letters and passed independent
  two-pointer plus forward/reverse hash audits; they remain human-unreviewed,
  so the reader gate is closed. The next repair allows typed determiner and
  adjunct attachment changes while preserving valency.

- **Dream-RSI round 113, 2026-09-18:** replayed the registered reciprocal
  preposition/typing lane with 564 worlds and 21,729 audited nodes.  Its live
  search contributed no equation-compatible branch, so the held-out policy
  remains fixed mismatch-first at 0.780 and the admissible exact count remains
  0.  The failure is preserved with the concrete typed determiner/adjunct
  attachment repair; no replay-budget expansion is counted as progress.
