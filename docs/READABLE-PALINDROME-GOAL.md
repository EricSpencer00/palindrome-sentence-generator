# Readable palindrome finder: goal record

## Objective and acceptance gate

Build a reproducible exact-English-palindrome finder and an ACL/NAACL paper
whose central claim is supported by evidence. The objective is a long readable
palindrome, and longer is better after an output is exact, independently
reproducible, and judged by
independent readers as grammatical, with a recoverable subject or intent and
coherent meaning. The paper must distinguish those reader outcomes from
mechanical exactness and automatic diagnostics.

## Current frontier (2026-09-13)

- **Acceptance gate remains unchanged.** A result must be an original,
  exact letter-level palindrome, rendered as intact English prose, mechanically
  verified independently, and later supported by blinded human reading.  No
automatic score or search statistic can promote an item.

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

- **Centre-out incomplete-phrase authoring** —
  `runs/centreout-incomplete-phrase-2026-09-12/`.  A frozen local-model prompt
  requests twelve whole English utterances composed outward from unfinished
  middle phrases.  The shared verifier enforces exact normalized symmetry,
  30--100 letters, ASCII-letter input, lexical form, unique non-self-palindromic
  words, no whole-word reversal or repeated multiword unit, and local and
  catalogue-family exclusion. Promotion requires at least one rendered
  survivor; it will then receive a human-ready blinded screen, never an
  automatic readability label.

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
