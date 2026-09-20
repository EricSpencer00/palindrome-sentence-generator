# v4 construction and evaluation ledger

This note is the current evidence-led core for the paper. The working claim
is constructive: choose grammatical lexical paths while satisfying character
seams during search. Exactness is independently checked; automatic language
scores and AI feedback only diagnose historical lanes. The active search is
exact-by-construction, with grammar boundaries and mirrored character orbits
chosen together. No output below is human certified yet.

The readability target is broad English: an intact, grammatical, scene-bearing
line that a blinded reader can understand. “Shakespearean,” where it appears
in legacy run names or diagnostic fields, is not a diction or imitation
requirement. Literary vocabulary is neither a shortcut nor a certificate;
reader ratings decide whether an exact output is readable.

## Anchor and frontier

The strongest reader-plausible exact output remains:

> **An aide rips nine memos; some men inspire Diana.**

It has 38 ASCII letters, normalized tape
`anaideripsninememossomemeninspirediana`, and forward/reverse SHA-256
`ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6`.
The independent outside-in pointer audit and the fail-closed mechanical gate
both pass. It has not yet been rated by blinded humans.

## Cross-role clause CSP

As a new constructive lane, we paired complete contemporary clauses while
deliberately crossing grammatical endpoint roles: names, determiners, nouns,
quantities, prepositions, and places can occupy different positions on the two
sides. A shared character obligation is consumed online as each side expands;
the search never materializes a finished clause to reverse or repair. The
25 typed shape pairs produced 1,768 live states and pruned 17,885 conflicting
obligations. Six intact prose controls and independent two-pointer/SHA audits
were retained. It produced no fresh exact candidate above 38 letters, so the
reader gate remains closed; this is a construction result, not evidence that
the controls are palindromic.

Here, “Shakespearean” in earlier experiment labels is only shorthand for a
broad English-readability target, not a literal style requirement. The active
construction lanes therefore use ordinary contemporary English; literary-style
lanes are retained only as clearly disclosed historical diagnostics.

The longest mechanically admitted exact control is not readable:

> **To new one post is an evening. Is sign in even as its open owe. Not.**

It has 50 letters and SHA-256
`855d51fa2b5cb8b4f63b9e043494f066702f8abbb329671ee5c68b82ca7788e3` in both
directions. A post-hoc `gpt-oss:20b` AI-feedback pass scored intact English,
scene coherence, and Shakespearean cadence 0/3, 0/3, 0/3. This is diagnostic,
not a readability certificate and not a search reward.

The 66-letter exact row is rejected mechanically for a hidden proper
multiword palindrome span:

> **Erased on forever event is an evening. Is sign in even as it never ever. Of nodes are.**

## Constructive methods

| Method | Search-space change | Rendered evidence | Exact / admitted |
|---|---|---|---:|
| Fixed half-tape grammar CSP | Character aliases are assigned before word boundaries; agreement and valency are live state | 38-letter anchor | 2 / 2 at 38; none above |
| Phrase-valued half-tape CSP (z2) | Phrase edges keep determiner/noun boundaries live while assigning mirrored character variables | **An aide rips nine memos; some men inspire Diana.** (38 letters; 24 target runs; 48,999 nodes) | 1 / 1 at 38; no >38 closure |
| Character-trie grammar decoder | POS/inflection terminals choose individual word boundaries while each emitted character assigns its mirrored variable | 69 target runs; 290,359 nodes; anchor recovered at 38 letters | 1 / 1 at 38; no >38 closure |
| Character-trie relative decoder | Adds typed `who`/`that`, relative subject, finite verb, and object transitions to the live character trie | 84 target runs (39--80 letters); 306,725 nodes; no exact closure | 0 / 0 |
| Authored scene lattice | Human-authored semantic edge pairs are joined only when their live character equations close | 45 exact diagnostics (42 letters); 10 collide with prior run tapes | 35 novel / 0 admitted; reader-rejected |
| Compositional shell seam DP | Joins independently authored complete SVO shells with live outside-in seam obligations; no finished-tape reversal | 21,660 compositions; 164 intact retained controls; longest 75 letters; best seam 4 chars | 0 / 0 |
| Seam-conditioned shell substitution | Role-preserving lexical and inflectional substitutions inside one complete SVO shell | 11,340 grammatical variants; 420 retained controls; longest 27 letters; best seam 4 chars | 0 / 0 |
| Fresh grammatical clause-pair lattice | Newly authored reversible clause/response pairs, checked as live equations before rendering | 49 exact diagnostics; **“Stressed, Deliver; reviled, desserts.”** (30 letters) | 49 / 0; reader-rejected |
| Anchor-preserving overhang search | Center-out overhang expansion around the 38-letter anchor with debt tracking | **132-letter exact frontier**; anchor embedded as a proper palindromic span | 0 admitted; withdrawn shortcut |
| Bidirectional typed half-tape CSP | Expands the more constrained unfinished grammar edge from either end while propagating character, agreement, and valency state | 70 cells; 2,421,192 edge attempts; anchor regression recovered; no >38 closure | 0 / 0 above 38 |
| Agreement-carrying adjunct center CSP | Solves subject/adjunct character equations before selecting the finite verb and object, with number carried into temporal/locative slots | 16,800 semantic shells; 15,120 outer-pruned; 24 intact controls up to 47 letters; **“The quiet poet guards an open journal while he waits.”** | 0 / 0 |
| Finite SVO character-orbit search | Intersects two complete finite SVO tries while assigning mirrored character orbits and refusing partial-fragment closure | 8 orbit states; 7 matched transitions; 3 intact controls up to 48 letters; no exact closure | 0 / 0 |
| Two-sided semantic orbit product | Builds independent complete SVO/adjunct story paths on both sides, then assigns each mirrored character orbit while grammar boundaries and semantic roles remain live | 12,000 left paths × 10,200 right paths; 34 product states; 20 matched and 140 rejected orbit transitions; 4 intact 41-letter controls; no exact closure | 0 / 0 |
| Two-sided semantic orbit product — held-out setting frame | Adds one initial setting-preposition/object frame at the prior OBJECT/SUBJECT dead frontier, then reruns the same center-out product with determiner and SVO roles fixed before expansion | 2,000 paths per side; 37 product states; 20 matched and 201 rejected orbit transitions; 4 intact 59-letter controls; no exact closure | 0 / 0 |
| Lexical boundary-state product | Selects inflection/clitic boundary states and mirrored orbit locks before rendering two independent scene clauses | 4 bounded assignments; intact controls to 78 letters; no exact closure | 0 / 0 |
| CFG/character-orbit chart intersection | Intersects a finite semantic CFG with live terminal character orbits and requires agent/action/theme/setting closure | 2 complete English clauses; no exact closure or admission | 0 / 0 |
| Semantic slot/orbit product | Chooses valency, attachment, agreement, and center-out character equations jointly over authored vivid contemporary-English scene frames | 12 agreement-valid scene pairs; intact controls to 75 letters; no exact closure | 0 / 0 |
| Large-lexicon CFG/trie orbit search | Expands a 70-word finite SVO+PP grammar into a character trie while selecting two ordinary-order clauses and their word boundaries before orbit checking | 2,781 trie nodes; 200 bounded states; complete controls to 63 letters; no exact closure | 0 / 0 |
| Morphology-first orbit grammar | Selects agreement, tense, article-boundary, and cadence states before lexical emission across two complete clauses | 16 agreement-valid variants; longest 66 letters; no exact closure | 0 / 0 |
| Semantic-role character FSM | Carries semantic role, valency, agreement, and word-boundary state on every center-out character transition with a complete-clause terminal gate | 4,608 bounded states; two complete prose controls to 36 letters; no exact closure | 0 / 0 |
| Live paired-slot clause DFS | Selects both ordinary-order clauses from grammatical slots while carrying the unmatched character stream into the next slot; no rendered-string repair | 26 live states in the long SVO+double-modifier form; a short-form calibration independently re-finds the 38-letter anchor; no >38 closure | 1 / 1 at 38; 0 / 0 above |
| Synchronous typed grammar product | Intersects complete transitive, copular, and locative clauses in a forward/reverse character trie before either side is rendered | 2,116 authored utterances; 12 live trie states; the near-`an arena` orbit reaches `aneranar` before an `r`/`a` conflict; no exact closure | 0 / 0 |
| Semantic dialogue-relation product | Chooses requester, agent, object, answer, and closing roles from authored alternatives while checking the full two-sided character obligation and request/answer/confirmation parse | 243 role states; two intact diagnostic frames to 78 letters; 0 constructive closures | 0 / 0 |
| Dialogue recipient-bridge extension | Adds a distinct recipient-obligation bridge relation while retaining the complete request/answer/confirmation parse and live two-sided character debt | 729 role states; diagnostic frames only to 78 letters; 0 constructive closures | 0 / 0 |
| Live phrase-boundary FSM | Advances authored subject, predicate, adjunct, and one held-out complement with one-sided character debt across phrase edges; right constituents are opened backward and the combined tape is audited at closure | 159 live transitions; 0 complete clause closures; no exact candidate over 38 letters | 0 / 0 |
| Luna bilateral character-LM/grammar beam | Selects two ordinary typed SVO clauses while a character 3-gram prior ranks only transitions that already satisfy the live mirrored orbit | 36,134 live character expansions; 1,200 rendered diagnostic controls to 69 letters; 0 exact closures | 0 / 0 |
| Luna recursive CFG scene lattice | Adds independent weather, agent, and purpose beats through a compositional scene grammar, with terminal choices intersected before rendering | 8 fresh scene paths at 68–74 letters; 0 exact closures | 0 / 0 |
| Luna dependency/valency CSP | Fixes fresh ditransitive donor/recipient/theme dependency frames and audits every reflected terminal obligation | 4 fresh frames at 17–19 letters; 0 exact closures | 0 / 0 |
| Hand-authored POS bilateral CFG orbit | Selects determiner/noun/verb and modified-SVO slots independently on both sides, matching each live character before advancing a word boundary | 2 typed templates; 88 live states; 0 complete closures | 0 / 0 |
| Brown PCFG bilateral orbit | Composes new POS-template clauses from Brown frequency domains while matching both ordinary clauses character-by-character | 961 independent template pairs; 179,205 live states; 0 exact closures | 0 / 0 |
| Luna relative-clause CFG orbit | Intersects independent relative-clause/coordination parses from a held-out lexical bank while consuming opposite-end characters live | 200,000 constructive states; 0 closures; no prose candidate reached the render gate | 0 / 0 |
| Brown character decoder center-out | Emits the left half with a Brown-derived character/word-boundary decoder and mirrors each character immediately, rejecting repeated units before rendering | 20 exact 40-letter closures with independent pointer/SHA audits; all failed the human readability gate because the right half was not English-segmentable | 20 / 0 admitted |
| Right-boundary WFSA decoder | Constrains mirrored right-side word boundaries and POS transitions during character decoding, while retaining immediate exact mirroring and fresh-word rejection | 0 segmented exact closures at 40+ letters; no candidate reached the reader gate | 0 / 0 |
| Agreement/valency WFSA decoder | Adds subject-number, transitivity, object-role, and clause-finality states to mirrored lexical decoding before a character is accepted | 0 exact closures at 40+ letters; no candidate reached the reader gate | 0 / 0 |
| Variable-boundary lattice decoder | Allows 1–3-word phrase chunks and live boundary movement on both mirrored sides while preserving immediate character equality | 0 exact closures at 40+ letters; no candidate reached the reader gate | 0 / 0 |
| Paired clause center-out lattice | Selects fresh SVO, copular, locative, imperative, relative, and appositive clauses jointly with typed semantic roles before character emission | 0 exact closures; intact controls reached 38 letters but never crossed the live equation gate | 0 / 0 |
| Indexed phrase-boundary center-out | Pre-indexes fresh NP/VP phrase pairs by exposed boundary characters and length difference, then consumes full debt before opening the next grammatical seam | 12 index keys, 30 pair options; 0 states reached the frontier; no candidate rendered | 0 / 0 |
| Boundary-indexed typed clause growth | Places each fresh subject/verb/number/object/name/adjunct at its real tape offset and rejects conflicts before opening the next slot | 66,967 live nodes and 23,668 terminal leaves across 40–70 letters; no exact closure | 0 / 0 |
| Asymmetric boundary-indexed growth | Tests a six-slot left clause against a four-slot response while carrying every outer character obligation before lexical placement | 150,719 live nodes and 1,868 terminal leaves; the withheld fresh inventory produced no exact closure | 0 / 0 |
| Character-level grammar beam | Keeps word-boundary and role state live in a bounded two-sided character beam with fixed lexical proposal scores | 3,010 bounded expansions over 38–45-letter targets; no exact closure | 0 / 0 |
| Dictionary reverse-segmentation DP | 51,129 authored SVO seeds are matched against a 52,927-headword POS-neutral reverse segmentation | No exact closure above 38; no candidate promoted | 0 / 0 |
| Indexed half-tape path CSP | Inverted character-slot seam index over typed grammar paths | Anchor recovered from 1,205 nodes | 2 / 2; longest 38 |
| Connector character product | Typed clauses and connectors cross an outside-in character product | 100 frontier witnesses; longest 65 | 0 / 0 |
| Residual-seam scene lattice | Complete grammatical clauses indexed by remaining length, required character, agreement, and valency | 246 grammatical renderings; longest 66 | 0 / 0 |
| Semantic shell growth | Held-out event pairs require live edge overlap at each incremental shell step | 7,272 renderings; longest 183 | 0 / 0 |
| Held-out terminal repair | Same-valency terminals selected by residual character and typed state | 252 grammatical renderings; longest 67 | 0 / 0 |
| Relative indexed boundary CSP | Agreement-carrying relative complement with indexed word-boundary offsets | 12,588 search nodes; longest exact 0 | 0 / 0 |
| Explicit relative-slot CSP (z2) | `who`/`that`, finite relative verb, and object are separate live edges rather than one opaque clause | 194,216 nodes across 39–70 letters; no exact closure | 0 / 0 |
| Shared relative complement | Shared participant plus finite `that`/`and` markers across the relative seam | 4,042 indexed nodes; no exact closure | 0 / 0 |
| Residual-prefix attachment lattice | Two-character seam-prefix state with attachment, agreement, and valency | 258 candidates; longest 77 | 0 / 0 |
| One-time typed lexical bank | Local model authors lexical alternatives once; deterministic CSP searches them without per-candidate feedback | 703,172 nodes; 390 grammatical frontier controls | 0 / 0 |
| Three-beat alias grammar | Three finite SVO beats with live boundary aliases and conjunction state | 1,344 renderings; longest frontier 81 | 0 / 0 |
| Scope-conditioned event frames | Independently authored event frames enforce subject-number agreement, transitivity, semantic scope, and adjunct attachment before surface realization | 22 complete forward-English candidates; **“The patient nurses carry a quiet message through town, while a young pilot checks the clear signal before rain.”** (91 letters) | 0 / 0 |
| Four-slot carried-character grammar | Agent, verb, object, and attachment slots propagate mirrored character obligations after each slot and prune before complete rendering | 6,144 states; all pruned before rendering; no candidate promoted | 0 / 0 |
| Relation/connector event frames | Carries contrast, cause, or sequence relation choice alongside agreement-valid event frames and scores outer agreement before rendering | 24 complete prose candidates; longest 73 letters; best pre-render outer agreement 5 | 0 / 0 |
| Typed semordnilap phrase graph | Uses mirror-pair strings only as a vocabulary index, then enforces phrase role and article agreement before shell rendering | 4,656 indexed pairs; all rejected before rendering; malformed fragments quarantined | 0 / 0 |
| Dependency attachment reset | Pairs active, passive, and locative dependency realizations and measures attachment seams after independent clause selection | 36 complete natural-English candidates; longest 77 letters; diagnostic only | 0 / 0 |
| Held-out endpoint/function frames | Enforces bilateral first/last character equality on new event frames before selecting a held-out connector | 27 complete natural-English candidates; longest 79 letters | 0 / 0 |
| Held-out subject/object inner-class frames | Carries vowel/consonant class obligations across independently authored subject and object boundaries before rendering | 9 frame pairs; 8 rejected before rendering, 1 complete 72-letter prose control; no exact closure | 0 / 0 |
| Semantic relation-frame orbit | Pairs independently authored agent/relation/object/setting frames under a live outer lexical orbit before rendering | 9 frame pairings; 8 pruned before rendering, 1 complete 64-letter prose control; no exact closure | 0 / 0 |
| Semantic relation realization lattice | Gates active, passive, and locative realizations by attachment signature before rendering | 36 relation-pair states; 24 rejected before rendering, 12 complete controls to 59 letters; no exact closure | 0 / 0 |
| Live residual relation slots | Carries width-one and width-two residual equations through relation/object/setting slots before rendering | 9 states; 0 width-one survivors, 0 rendered candidates | 0 / 0 |
| Cross-word boundary grammar DP (diagnostic) | Records per-word cross-boundary traces over fresh frame pairs, with full audits but no enforced global residual | 63 transition states; 9 complete prose controls to 77 letters; quarantined diagnostic | 0 / 0 |
| Relation/setting debt trace (diagnostic) | Records residual traces across two slots while retaining an explicitly arbitrary filter | 6 states; 4 complete prose controls to 90 letters; diagnostic only | 0 / 0 |
| Unequal center-crossing grammar | Carries a two-character buffer across an unequal center before rendering | 16 typed pairings; 0 buffer survivors and 0 rendered candidates | 0 / 0 |
| Wider unequal-center buffer | Propagates an exact unmatched buffer through unequal subject/verb slots | 243 typed states; all pruned before rendering | 0 / 0 |
| Two-sided unmatched-buffer DP | Stores actual unmatched buffers on both word streams in reverse-facing order and prunes on mismatch | 9 orientation-correct transitions; 9 pruned, 0 surviving states, 0 rendered candidates | 0 / 0 |
| Variable-buffer adjunct-trie DP | Adds variable-length word-trie transitions and independently authored adjunct slots to the live buffer | 8 slots; 4 transitions, 4 pruned, 0 rendered candidates | 0 / 0 |
| Center-out scene buffer | Grows independently authored event scenes from the center with variable buffers and attachment state | 3 scene transitions; all pruned before rendering | 0 / 0 |
| Reverse-trie typed grammar | Traverses a reverse-facing right-clause trie with deque residuals, typed agreement, center transitions, and unequal word boundaries | 12,001 nodes; 0 rendered candidates; diagnostic row quarantined | 0 / 0 |
| Prosodic-skeleton diagnostic | Ranks fresh clauses by aggregate word-length shapes without a live character equation | 16 fresh controls to 79 letters; post-render diagnostic only | 0 / 0 |
| Endpoint-seeded inward scene | Seeds compatible scene endpoints, then expands full interior grammar and unequal buffers | 2 endpoint seeds; both pruned before interior expansion | 0 / 0 |
| Endpoint-seed interior equations | Gates surface grammar, then solves live interior cross-word equations | 9 endpoint seeds; 0 interior closures; malformed article forms excluded | 0 / 0 |

The manual clause-seam check is retained as a separate construction
discriminator: 16 independently authored clause pairs were rendered above 38
letters, and all failed at the outer seam (the first pair compared `a` with
`n`). It never edited a completed sentence. The next useful step is therefore
to change the endpoint lexical inventory, not to repair those 16 renderings.

A prior 256-clause product is explicitly withdrawn from the results: its first
implementation compared only first/last word characters rather than consuming
the full mirrored streams. The regression test now rejects that shallow gate,
so it cannot inflate the construction ledger.

Every lane records a literal two-pointer audit, forward/reverse SHA-256, source
provenance, novelty preflight, and a concrete next construction discriminator.
None reverses a finished sentence, imports catalogue text, or scores each
search state with an LM/RLAIF reward.

The constructive result is therefore specific, not a proxy claim: the z2
half-tape representation independently recovers the 38-letter readable anchor,
but the bounded 38--45-letter pilot does not yet improve its length. A failed
construction is not repaired after rendering; the next construction exposes
relative-clause internals as separate character-constrained edges. Its
194,216-node run also closes at zero, so it remains an auditable discriminator,
not a promoted example.

An orthogonal dictionary-DP check also returned zero above 38. Its failure is
not folded into a general sparsity claim: it chooses a complete seed before
reverse segmentation, so the concrete construction change is a POS/inflection-aware
character trie that inserts boundaries during half-tape search.

The character-trie decoder now implements that construction change and independently
recovers the anchor across individual word boundaries. Its 69-run pilot still
has no >38 closure, so the representation is a verified construction step,
not a claim that the length target has been met.

The next character-trie construction made the relative clause internal rather than
opaque: marker, relative subject, finite verb, and object each participate in
the character seam and agreement state. Across 84 deterministic targets from
39 through 80 letters it visited 306,725 nodes and returned zero exact rows.
This is a concrete negative result with a live construction change, not a
readability or sparsity claim. The next repair is a shared-participant/anaphor
state plus one adjunct edge, gated on first finding a valid relative closure.

An authored scene lattice reproduced 45 exact 42-letter diagnostics, including
“Was Noel an item stressed? Desserts met in a leon saw.” The independent
two-pointer and forward/reverse SHA checks pass; 10 tapes collide with prior
run artifacts and the remaining 35 fail the mechanical construction gate. The
reverse-compatible witness phrases do not form a coherent scene, so none is
reader-worthy and the 42-letter example is withdrawn as a duplicate. The next
repair must author fresh grammatical clause pairs rather than assembling known
reverse-compatible witness fragments.

As a separate constructive baseline, compositional shell seam DP joined
independently authored complete SVO clauses rather than reverse-compatible
fragments. It visited 21,660 unique compositions, retained 164 intact prose
controls up to 75 letters, and matched at most four outside-in characters; no
exact closure occurred. The longest controls are useful reader materials, but
they are not palindromes. The next repair is seam-conditioned inflectional
substitution inside one complete shell while keeping the semantic roles fixed.

The seam-conditioned substitution repair evaluated 11,340 role-preserving
variants and retained 420 controls, but still closed at zero exact rows. Its
best intact controls are short (“Some scribe inspires some memos.”), so it is
a diagnostic repair rather than a length advance. A separate fresh clause-pair
lattice closed 49 exact rows, the longest 30 letters, including “Stressed,
Deliver; reviled, desserts.” All were mechanically rejected as incomplete or
fragmentary clause pairs. The next repair is to require finite subject/verb
clauses before any reverse-compatible pair can enter the lattice.

The earlier center-out overhang search reached a 132-letter exact tape, but it
did so by embedding the complete 38-letter anchor as a proper self-palindromic
span and repeating a content word. The independent audit is retained as a
frontier diagnostic; the shared admission gate withdraws it, and no reader
study is run on it. The next constructive search must forbid anchor embedding
before expansion rather than treating this overhang as a result.

The bidirectional half-tape pilot then changed the expansion order itself: it
selected the more constrained unfinished edge from either end and propagated
the mirrored character orbit, agreement, and object type before choosing the
opposite edge. The remote replay completed all 70 cells from 39 through 52
letters (2,421,192 edge attempts and 10,460 search nodes) without exhausting
its budgets, recovered the 38-letter anchor in regression mode, and produced
zero exact closures above 38. This excludes that bounded grammar, not the
overall readable-palindrome goal; the next repair must change sentence
structure or lexical boundary possibilities rather than spend more time in
the exhausted cells.

The agreement-carrying adjunct center CSP changed the seam state again by
solving a subject/temporal-or-locative equation before choosing the verb and
object. Its remote run evaluated 16,800 semantic shells, pruned 15,120 at
that outer equation, and rendered 24 intact controls up to 47 letters. For
example: “The quiet poet guards an open journal while he waits.” No exact row
closed, so these are reader-facing controls for the next study, not candidate
palindromes. The next repair carries one held-out inner verb/object edge into
the residual character state rather than widening every slot.

The finite-SVO character-orbit lane is a separate complete-constituent
baseline: two hand-authored finite SVO tries are intersected while mirrored
character orbits are assigned, and only complete SVO terminals may close. The
remote run expanded 8 orbit states, matched 7 transitions, and retained three
intact controls up to 48 letters, including “A patient keeper guards charts;
The baker records a sonnet.” It returned zero exact closures. The next repair
adds one held-out subject/object noun bundle at the first live orbit frontier,
without relaxing complete-clause or independent-audit gates.

The next lane removes the remaining repair framing from the active method. Two
independent semantic path banks are authored first, with complete subject,
finite-verb, object, and optional adjunct roles. A two-sided trie product then
assigns a shared character to each mirrored orbit while both ordinary-order
paths are still unfinished; no rendered sentence is edited afterward, and the
right path is never rendered in reverse order. In the remote 39–60-letter
run, 12,000 left paths and 10,200 right paths produced 34 product states, 20
matched transitions, and 140 rejected transitions. The product reached depth
three but no simultaneous complete-path closure. It retained four intact
41-letter controls, including “A baker carries a map; an artist answers some
bells.” The failure is therefore a construction-boundary result, not a reason
to enlarge a repair queue: the next discriminator adds one held-out complete
story frame at the first dead frontier and reruns the same orbit product.

The held-out setting-frame discriminator adds a grammatical initial
preposition/object pair (bare time expressions such as “at noon” and
determined locatives such as “over the river”) before the same finite SVO
grammar. On the bounded remote replay, 2,000 paths per side produced 37
product states, 20 matched transitions, and 201 rejected transitions. No
39–60-letter exact closure appeared; four intact 59-letter controls remain
available, and their first failures are retained at the
setting-preposition/object boundary with independent pointer, SHA, and
mechanical audits. This frame is a falsifiable extension of the grammar, not
a repair to a rendered sentence.

Three orthogonal Luna probes then tested whether the remaining failure was
caused by boundary choice, grammar representation, or semantic-slot selection.
The lexical boundary-state product selected inflection/clitic boundaries and
mirrored locks before rendering four independently authored clause pairs. It
retained a grammatical 78-letter control (“the quiet ranger marks the trail at
dawn; then the patient baker packs the loaves for the market.”) but closed at
zero. The CFG/character chart intersection rendered two complete clauses—
“the artist admires canvas near the river bridge” and “a gardener waters
garden beside the school bridge”—and likewise found no exact closure. Its
static lexical order was not a candidate reward.

The semantic-slot product added a separate valency/attachment lattice with
agreement-valid morphology and rejected repeated frame pairs. It evaluated 12
complete scene pairs, including “the herald carries the letter through the
hall; the actors keep the oath near the grove.” Every row has independent
pointer and forward/reverse hash audits; none is exact. These are concrete
construction outcomes, not a readability claim or a reason to launch a repair
queue. The next discriminator holds out attachment prepositions and measures
whether any valency frame gains a live closure frontier.

The larger-lexicon CFG/trie search widened the finite language rather than
adding a reward model: 70 authored/common-English words became 2,781 trie
nodes and 200 bounded two-sided states. It retained complete ordinary-order
controls to 63 letters and closed at zero. The morphology-first lane then
selected agreement, tense, article-boundary, and cadence states before
emission. After excluding malformed agreement variants, 16 valid rows reached
66 letters with zero exact closures. Both lanes leave a concrete next state
change—held-out transitive verbs for the trie, and a held-out boundary state
for morphology—rather than another repair pass.

The semantic-role character FSM carried the obligation at finer resolution:
every emitted character retained its agent/predicate/patient or scene-modifier
role, valency, agreement, and word-boundary state. It visited 4,608 bounded
states and retained two complete prose controls (“Ranger maps harbor near
bridge quietly.” and “Scribe marks signal carefully under tower.”), but no
terminal closure or near-miss survived. Its next test adds a plural
agent/object pair to the same FSM; it does not edit either control.

The paired-slot clause DFS makes the construction invariant explicit. It opens
the left clause from the beginning and the right clause from its final word;
when one selected word is longer, the unmatched character stream is carried
into the next grammatical slot on the other side. The long form (a finite SVO
clause with two modifiers) visited 26 live states and closed
at zero above 38 letters. Its short-form calibration independently recovered
the existing 38-letter anchor, with identical forward/reverse SHA-256 and a
separate two-pointer audit. This is the active algorithmic direction: an
off-tape English draft is never repaired into a palindrome, and the 38-letter
calibration is not presented as progress beyond the current frontier.

The boundary-indexed typed-growth lane is the next construction discriminator,
not a repair queue. It places each word at its actual mirrored tape offset and
rejects the first incompatible character before the following subject, verb,
number, object, name, or adjunct slot is opened. Its bounded fresh inventory
visited 66,967 states and 23,668 terminal leaves from 40 through 70 letters;
no exact closure survived. The residual is therefore a boundary-inventory
signal for the next search, while the paired-slot DFS remains the active
generation method.

The synchronous typed grammar product tests the same invariant with a different
representation: complete transitive, copular, and locative clauses are placed
in forward and reverse tries, and only shared character prefixes are expanded.
The 2,116-utterance inventory reached twelve live states. The endpoint grammar
extended the `ane` branch through the fresh `an era ... arena` shell to
`anerae`, then through the parsed noun-phrase orbit `an era narrates an arena`
to `anerana`, and the fresh “near an arena” realization extends that branch to
`aneranar` before an `r`/`a` conflict; the branches expose concrete `n`/`h`,
`r`/`a`, and `e`/`g` conflicts, while
the older `th` branch still dies at `e`/`g`. This is a small, honest frontier
diagnostic: it demonstrates that endpoint design can deepen the live orbit
before rendering, and gives the next construction a concrete continuation
target rather than a repair queue. The endpoint family is now exhausted at the
`aneranar` `r`/`a` conflict; forcing another continuation would be repair-like,
so the next branch is an orthogonal phrase-boundary finite-state grammar.

The corrected phrase-boundary FSM then consumed one-sided debt across constituent
edges rather than discarding states when one unit was longer. Its remote replay
visited 159 live transitions but produced no complete clause closure; the reader
gate remains closed. The earlier draft that audited the two clauses separately
is withdrawn and covered by a regression test, so neither version is presented
as a readable candidate.

The semantic dialogue-relation product is a separate grammar family. It tests
243 combinations of requester, agent, object, answer, and closing roles while
preserving a complete request/answer/confirmation parse. It produced two
useful intact controls—“Mira asks the baker for warm bread, and the baker
answers with a clear yes; Mira thanks the baker.”—at 77 letters and a 78-letter
variant, but no constructive closure. Those controls are not palindrome
outputs and do not enter the reader gate.

Adding the recipient-obligation bridge made a genuinely new construction rather
than a residual edit: the same live character equations were carried through
729 role states before rendering. It still produced zero constructive closures;
the 78-letter frames remain diagnostic controls only. Its distinct experiment
identity passed novelty preflight, so the dialogue family is now closed instead
of being expanded with post-hoc repairs.

Three Luna lanes then changed the search representation rather than patching a
failed tape. The bilateral character-LM beam expanded 36,134 live character
states and retained ordinary controls such as “A careful pilot maps rivers
after rain; The quiet teacher reads letters after rain.” (68 letters), but no
exact closure. The recursive scene lattice produced eight fresh 68–74-letter
weather/agent/purpose paths, and the dependency lane tested four fresh
ditransitive frames; both closed at zero. A Brown-derived PCFG widened the
typed vocabulary to 961 independent template pairs and 179,205 live states,
also with zero closure. These outputs are diagnostic controls, not palindrome
candidates; each has an independent two-pointer and forward/reverse hash
record. The next construction is a held-out relative-clause grammar, selected
on the live orbit before any prose is rendered.

The relative-clause follow-up supplied that held-out grammar directly: it
tested 200,000 relative-clause/coordination state pairs and still found zero
constructive closures. Because the outer character obligation failed before a
complete parse could be rendered, it produced no reader-facing text; the
failure is retained as a grammar-family discriminator, not converted into a
repair pass.

The synchronous lexical center-out lane then tested the alternative suggested
by the seam failures: begin at a grammatical center and grow outward while
carrying the entire unmatched character debt across phrase boundaries. A
fresh NP/VP phrase grammar was expanded to depth 5 with beam 200 on
`hst-bench`; the corrected implementation rejected repeated phrases and
content words before insertion and propagated both grammar states. It reached
zero frontier states, zero exact closures, and therefore emitted no prose for
the reader gate. This is a clean construction failure, not a near-palindrome
repair result; the next discriminator indexes phrase pairs by exposed
boundary while retaining the complete debt.

The indexed follow-up changed the frontier before any tape was built. It
bucketed the same fresh phrase grammar by exposed first/last characters and
length difference, leaving 12 index keys and 30 pair options, but no
compatible initial state. The result is useful precisely because it is not a
repair: the incompatible grammar family is retired at the index boundary and
the next construction must add NP/NP and VP/VP seam families rather than
editing a rendered string.

The Brown character-decoder lane supplied the first nonzero exact frontier
above the 38-letter anchor: 20 fresh 40-letter tapes closed under immediate
character mirroring, with matching two-pointer and SHA-256 audits. The actual
rendered controls make the remaining problem explicit—for example,
“the about nevertheless sselehtreventuobaeht” is exact but its right half is
not English. These are not readable outputs and are not promoted. The next
decoder version therefore carries a right-side lexical/POS boundary automaton
inside generation; raw reversal or post-hoc resegmentation is disallowed.

The right-boundary WFSA was then put inside that decoder. It required the
mirrored side to remain lexically and POS-segmentable before a character was
accepted, rather than splitting the finished tape afterward. The held-out
40+-letter run produced zero segmented exact closures, so it supplies no
reader-facing output; the next state change is agreement and valency on the
right WFSA, not a repair pass.

Adding agreement and valency states to that WFSA was a separate held-out
construction. The fresh singular/plural, transitive/intransitive, object-role,
and clause-finality frames produced zero exact closures at 40+ letters on
`hst-bench`; no text was sent to readers. Its next state is tense/aspect and
semantic-role compatibility, again selected before emission rather than used
to patch a near miss.

The variable-boundary lattice next allowed one-to-three-word phrase chunks and
live boundary movement on both sides. It still produced zero 40+-letter
lexical closures, so the failure is not being hidden behind a fixed token
boundary. A separate paired-clause lattice then widened the authored scene
bank across SVO, copular, locative, imperative, relative, and appositive
frames; its best intact controls were ordinary English, but none survived the
first exact equation. Both lanes are retained as construction evidence, not
as repair attempts.

The connector-clause debt follow-up corrected the clause renderer and reran on
`hst-bench`. It selected the connector and both ordinary clause texts jointly,
checked the full cross-clause character debt before rendering, and recorded
eight valid intact-English controls. It produced zero exact closures. The
controls are useful because they are real prose; the earlier type-label
rendering is withdrawn and covered by a regression test.

The strategy then changed representation rather than repairing a draft. A
bidirectional semordnilap grammar intersects fresh lexical token pairs while
filling typed clause templates from the first character. The first bank
produced 20 exact closures at 26--36 letters. Adding auxiliary, agreement,
and clause-finality states produced 20 exact closures up to 48 letters,
including “was deliver desserts drawer; reward stressed reviled saw.” Every
one failed the intact-prose gate: these are exact lexical demonstrations, not
readable outputs. They are retained as independently audited method evidence;
the proper-name/scene extension then generated the following 56-letter exact
diagnostic (shown with editorial punctuation only): “No evil, Noel, deliver
desserts raw; war—stressed, reviled—Leon, live on.” Its normalized tape has
zero pointer mismatches and identical forward/reverse SHA-256, but the entire
construction is an aligned reversed-token chain. We therefore withdraw it
from reader comparisons under the no-shortcuts gate; it is not evidence of
readability.

The first cross-word-seam follow-up required a boundary shift on both sides
and retained complete authored scene clauses. Its 512 remote controls reached
77 letters but had zero exact closures. A phrase-segmentation prototype then
closed three 62--64-letter tapes, but its greedy fallback emitted isolated
letters on the opposing side; those rows are rejected as gibberish. The strict
held-out lexical/scene parser removes that fallback and returns zero admissible
closures. This is the current construction boundary: every future exact row
must be fully lexical and grammatical on both sides before it can enter the
reader queue.

The morphology/adjunct expansion made that parser more permissive without
relaxing the gate: 4,320 bounded trials added finite-verb variants, plural
objects, and adjunct vocabulary, yet still produced zero admissible or exact
closures. We therefore retire that bank rather than widening it indefinitely;
the next method changes the boundary representation itself.

The first whole-sentence grammar intersection implements that change directly.
One compact CFG is expanded into complete role-labelled sentences, and a
center-out character gate rejects each derivation at its first mirrored
mismatch, before a finished tape can enter a repair queue. An expanded
held-out lexicon yielded 10,800 distinct-slot derivations; all died at that
first gate (zero complete states, zero exact closures). This is a useful
negative construction result—not a readable candidate—but it verifies the
intended search geometry: the next expansion is a lexical trie and richer
grammar, not a repair operator or a pre-paired reverse clause.

The subsequent slot-pair construction makes the boundary state explicit. For
three complete determiner--subject--verb templates (a modifier, an adjunct,
and a prepositional object), it chooses the first and last words independently,
compares the exposed prefix and reversed suffix immediately, and retains
unequal word lengths in buffers so a match may cross a word boundary. With up
to 24 Brown-PCFG entries per role plus singular-subject, third-person
transitive-verb, and determiner-object filters, first-character boundary
indexing left 61 live states across the three templates. Replacing the coarse
role buckets with NLTK Brown universal DET/NOUN/VERB counts did not produce a
complete sentence: all states were pruned before closure. A frame extraction
over the same corpus found 864 `DET NOUN VERB DET NOUN` sentences and 1,991
`DET NOUN VERB ADP` sentences; the next pass used their observed lexical
combinations rather than only counting frames. Keeping subject and object
banks separate produced 230 live states, 220 early prunes, and zero exact
closures. A Penn-tag preflight found 2,231 NN→VBZ and 4,377 NN→VBD
transitions but no NNS→VBP transitions in the bounded corpus. Those counts
are availability evidence only. The corrected implementation now carries
1,275 subject and 609 verb word-to-Penn-feature mappings through recursion and
checks them before deeper expansion. No agreement-conflict frontier was
reached (`agreement_pruned=0`), so this is an implementation result, not a
readability gain; the next state expands feature-conditioned banks and inner
states.

The feature-conditioned run then used separate Brown Penn-tag banks for
singular+VBZ, plural+VBP, and past+VBD frames. A valid follow-up extracted
preposition--noun adjuncts only from the same `DET NOUN VERB ADP NOUN` frame,
kept subject, verb, and object banks separate (64/64/64, 58/53/58, and
64/64/64), and searched a distinct prepositional-object template. It visited
82 bounded states and pruned all 82 before an exact closure. No candidate
reached the reader gate. A second-complement follow-up admitted only one
past-feature context (`of aluminum`) from the bounded Brown frame inventory;
adding that slot left the same 82 states all pruned and produced zero exact
closures. This grammar family therefore needs a larger frame-attested
second-complement bank or retirement, rather than an off-tape repair pass.

An earlier adjunct follow-up is explicitly rejected in the audit: it copied one global
preposition--noun inventory into every feature bank rather than extracting
adjuncts from matching tagged clause frames, and it did not add a distinct
prepositional-object derivation. It contributes no candidate or method claim;
the next implementation must carry frame identity into the adjunct slot.

These historical repair and frontier rows now motivate a strategy reset rather than more
residual patching. The active construction policy is exact-by-construction:
grammar boundaries and mirrored character orbits must be selected together,
so an off-tape prose draft is never promoted into a repair queue. The next
construction is a whole-sentence grammar intersection: compile one fresh
sentence grammar into lexical character tries, then search the forward and
reverse derivation frontiers together while carrying POS, semantic-role, and
word-boundary state. The complete sentence—not two pre-paired clauses—is the
object being generated. The paired-slot clause DFS, two-sided semantic orbit
product, its semantic-slot extension, and the semantic-role character FSM are
retained as prior exact-by-construction baselines; earlier repair runs and the
semordnilap rows remain auditable diagnostics, not the paper's proposed route
to readable prose. Exactness is necessary, while human readability remains an
independent gate.

### Synchronous whole-frame intersection

To test whether a richer frame could avoid the short five-slot frontier, we
implemented a seven-role `DET NOUN VERB DET NOUN ADP NOUN` grammar in
`experiments/synchronous_cfg_intersection_20260919.py`. Both outer roles are
selected together from frequency-ranked Brown frame partitions; newly exposed
characters are compared immediately while recursion carries cross-word
buffers inward, consuming matched prefixes and retaining unequal-length
residuals. This is one grammatical derivation, not a finished tape followed by
repair or reversal. Pairing 101 complete Brown-attested frame paths produced
8,613 synchronous states, all pruned by live character obligations. The
independent SHA-256 audit
therefore reports zero exact closures and no rendered candidate; complete
provenance is in `runs/synchronous-cfg-intersection-20260919.json`. This is a
construction boundary, not a readability claim or a reason to promote repair;
the next lane must change the grammar or lexical character constraints.

### Live-buffer invariant correction

The preceding slot implementation exposed a concrete construction bug: its
first-character index inspected the newest right word even when an older right
word still held unmatched character debt. That incorrectly rejected the seed
at the `Diana`--`inspire` seam. The new live-buffer constructor removes that
index whenever the buffers differ, carries the complete unmatched prefix and
suffix, and independently re-audits every closure. It regenerates the exact
38-letter control and a 37-letter lexical variant from the typed two-clause
grammar, but finds no closure above 38 letters. This is a valid invariant and
regression correction, not a new reader result; the rendered control remains
the only candidate eligible for future blinded comparison.

Two follow-ups kept that invariant but changed the lexical geometry. A
two-clause product consumed unmatched characters across 65,536 complete typed
clause pairs; every pair was pruned and no closure above the 38-letter control
appeared. A separate reverse-index prototype queried 287 Brown-typed
transitions and 721 reverse-index keys across 120,407 states, but its
implementation audit found that the residual consumer was not actually called;
its zero-closure result is therefore not a valid frontier measurement. These
are distinct construction attempts, not repairs: the next lane must invoke
residual consumption over multiword typed phrase transitions so local English
joins are selected jointly with the character orbit.

The corrected phrase follow-up queried 1,600 two-word Brown-derived units and
called residual consumption before each exact audit. It evaluated 120,000
phrase-pair states, pruned 119,997, and found no closure above 38 letters.
Because the units were recombined rather than replayed as complete sentences,
this is a valid lexical-geometry result but not reader evidence; the next
construction must place these phrase transitions inside a complete typed clause
grammar.

The six-slot phrase-clause construction then made that change explicitly. Its
outer-to-inner roles were `NP, VP, PP, PP, VP, NP`, with 180 transparent
two-word units per role. A reverse trie was queried from the *current residual*
obligation: an unmatched left buffer constrains the right phrase's final
character, while an unmatched right buffer constrains the next left phrase's
first character. When neither buffer exists, the full role bank is opened and
the overlap is checked immediately. This avoids the earlier seam-index error
and never edits a completed sentence. The run pruned 32,220 branches, reached
zero complete states, and produced zero exact closures above 38 letters. Its
independent artifact is `runs/six-slot-phrase-clause-20260919.json`; the lane
is therefore a valid stopped construction family, not a candidate or a
readability result. The next lane changes the grammar's phrase-length and
semantic-frame structure rather than adding a repair pass.

The corrected bilateral word-trie decoder makes the same principle explicit at
individual word boundaries. It chooses complete lexical words from typed role
tries on both sides, uses a boundary index only when no residual exists, and
otherwise lets outstanding character debt cross any number of word boundaries.
The expanded ordinary bank visited 7,933 live states and pruned 8,022
transitions. It independently regenerated the exact 38-letter anchor, with
identical forward/reverse SHA-256, but produced no longer exact row. This is a
constructive regression—the seam is now represented correctly—yet it is not a
length or reader win; the next construction must widen the grammar and vary
phrase lengths while retaining this word-level invariant.

An authored Shakespearean scene lattice tested the semantic side of that
construction. Six human-written court frames carried explicit subject,
transitive, adjunct, relative, and object valencies. After correcting its
residual consumer, all 16 outer frame pairs were pruned at the first character
equation, with zero orbit steps and zero exact closures. The pre-correction
run is marked invalid because it retained matched characters; only the corrected
artifact is evidence. No candidate reached the reader gate.

The variable-length phrase grammar is the first constructive lane to return a
second intact prose rendering without any repair step. It jointly grows paths
of lengths two through five (`NP VP`, optional adjuncts, and a second event),
and consumes the character debt after every phrase addition. In a 150,000-state
bounded run it evaluated 5,940,178 rejected transitions and independently
audited two exact 38-letter rows:

> **Some men inspire Diana; an aide rips nine memos.**

> **An aide rips nine memos; some men inspire Diana.**

Both are generated from the transparent authored/Brown-bigram bank, have
distinct normalized tapes, and pass the independent exact/SHA audit. Neither
is longer than the existing anchor, and no row above 38 letters closed; this is
therefore a real readability-preserving construction win and a clear next
experiment (wider variable phrase banks), not evidence that repair can scale.

The next variable-grammar construction changed the semantic shape rather than
just increasing the bank: both sides had two finite events, with optional
relative-clause or prepositional expansions from a held-out authored bank.
Residual character buffers were consumed after every phrase edge. The run
visited 180,000 states and rejected 1,243,165 transitions immediately; it
closed zero exact rows above 38 letters. This is a valid stopped grammar family
because its complete-path invariant is explicit. The next construction must
change how semantic roles are paired across the center, not repeat this
two-event state budget.

An independent bilateral word-trie implementation then tested the same
residual invariant with a different scheduler: variable complete clause paths
(`NP V OBJ`, optional `PP`/`REL`) were selected independently, and lexical
units were opened from both outside edges. The held-out role bank produced 105
live states, 910 transitions, and 841 immediate rejections. It regenerated the
two 38-letter exact renderings above with identical pointer and SHA audits,
but no longer row. This cross-implementation agreement is a useful correctness
win; the next construction changes the semantic center pairing rather than
adding another duplicate word sweep.

The cross-paired seam constructor made that center change explicit. Its left
clause used `subject–event–object–setting`, while the held-out right clause
used `setting–object–event–subject`; each frame carried an explicit valency.
Matched characters were consumed after every frame pair. All 12 outer frame
pairings were rejected at the first seam equation, with zero orbit steps and
zero exact closures. This is a valid semantic-pairing negative, not a failed
repair queue; the next construction adds tense/agreement variants inside this
cross-role order.

The morphology-aware follow-up carried subject number, finite-verb agreement,
and present/past tense as live feature state in the same cross-role grammar,
using held-out right-clause realizations. Feature checks ran before the seam
comparison, and residual characters were still consumed immediately. All nine
reachable states were pruned at the first character seam, with zero feature
conflicts, orbit steps, or exact closures. This rules out morphology as the
missing ingredient for this role order; the next construction adds a central
conjunction with independently inflected subordinate clauses.

The central-conjunction construction then generated a complete main clause,
`while`/`though`/`when`, and a complete subordinate clause with held-out
lexicalizations. Subject number and event agreement were carried before each
seam comparison, and residual characters were consumed immediately. Its nine
reachable states were all pruned at the first character seam; there were zero
feature conflicts, orbit steps, or exact closures. The next construction puts a
relative clause inside the subordinate subject to change the boundary geometry.

The embedded-relative follow-up inserted a relative event inside the
subordinate subject while retaining the central conjunction, held-out
subordinate vocabulary, and agreement state. Residual characters were still
consumed after each frame pair. It reached nine states, pruned all nine at the
outer seam, and produced zero exact closures. The construction is therefore
stopped with a concrete next step—another held-out relative attachment site
conditioned on valency—rather than a post-hoc repair.

The dual-relative follow-up added two distinct attachment sites: a
subject-modifying relative in the main clause and an object-modifying relative
in the subordinate clause. Valency labels, agreement/tense state, held-out
subordinate lexicalizations, and live residual consumption all remained
active. The nine reachable states were again pruned at the first character
seam, with zero exact closures. The next construction moves the relative
attachment onto the bridge conjunction and adds explicit complement selection.

The bridge-complement construction selected a finite subordinate complement
before expanding its attached relative clause. Agreement state and held-out
subordinate lexicalizations were carried through the same live residual seam;
all nine reachable states were pruned at the first character comparison, with
zero exact closures. The next construction conditions bridge complement choice
on tense and mood before seam expansion.

The mood/tense bridge lane carried the bridge's mood and tense as equations
into subordinate event selection before seam expansion, while retaining the
attached relative clause and agreement state. It visited nine states and
pruned all nine at the first character seam, with zero feature conflicts,
orbit steps, or exact closures. This closes the current conjunction/attachment
family for now; the next productive step is a broader character-conditioned
lexical grammar rather than another local attachment variant.

The broader character-conditioned lexical-trie grammar then changed the search
space rather than the attachment template. It indexed each role in forward and
reverse tries, selected individual words under the live character obligation,
and supported complete SVO, PP, and transitive relative paths (`REL V DET N`).
The corrected run visited 180,000 bounded states, produced no incomplete
relative path, and closed zero exact rows above 38 letters. The first draft's
bare `REL V` path is explicitly superseded and contributes no evidence. This
is the current stopped lexical baseline; the next construction must add a
different semantic center or a larger complete-role grammar, not repair a
rendered tape.

### Center-seeded semantic expansion

The center-seeded lane tested the opposite construction order: choose a center
first, reject non-palindromic grammatical centers by the exact central invariant,
then grow complete semantic role pairs outward while checking the exposed
characters before each transition. Four lexical centers (`level`, `radar`,
`refer`, and `noon`) survived the center gate; 100 outer transitions were tested
and all were rejected at the first character obligation. It produced no complete
candidate and no reader-facing text. This is a valid negative construction, but
it also identifies why a simple word-level center is too rigid: ordinary English
phrase pairs need residual debt to cross word boundaries, not whole-word outer
matches. The next lane therefore changes the chart representation to combine
complete constituents with live character residuals; it does not repair a
rendered draft. The independent artifact is
`runs/center-seeded-semantic-expansion-20260920.json`.

### Bottom-up synchronous CFG chart

The next representation builds grammatical constituents before pairing them:
87 noun phrases, 1,105 verb phrases, 200 prepositional phrases, and 96 complete
relative clauses. It then intersects complete clause derivations bottom-up,
consuming exposed character debt at each paired constituent boundary. The
120,000-state bounded run pruned every combination and closed no exact row above
38 letters. Three ordinary complete-clause controls are retained with their
independent pointer/SHA audits. This is a direct construction result, not a
repair queue; the next construction changes the grammar's semantic nonterminals
and keeps the same pre-render residual invariant. Artifact:
`runs/bottom-up-cfg-character-intersection-20260920.json`.

### Recipient and adjunct semantic chart

The chart was then widened by changing the grammar, not by mutating failed
strings. New recipient, ditransitive, and adjunct-attachment nonterminals yielded
12 noun phrases, 13 verbs, 6 recipients, 6 adjuncts, 546 ditransitives, and 546
adjunct verb phrases. A bounded run tested 6,400 live constituent combinations;
all were rejected by the character equation before closure. It retained three
complete ordinary controls, including a 39-letter ditransitive clause, with
independent pointer/SHA audits. No exact candidate was admitted. The semantic
recipient family is therefore stopped; the next construction adds feature
carrying to a different chart state rather than repairing any rendered tape.
Artifact: `runs/bottom-up-recipient-adjunct-chart-20260920.json`.

### Recursive scene chart

An independent recursive scene grammar tested imperative, copular/locative, and
finite-conjunction alternatives with explicit agreement and valency. It produced
3,570 complete grammar paths and paired them with variable word boundaries in a
300,000-state character-synchronous chart. Every state was pruned before exact
closure; no candidate was promoted. Complete scene controls and diagnostic
witnesses retain their two-pointer/SHA audits, but they are not readability or
palindrome evidence. The next construction adds typed dialogue complements as a
new grammar family, not a repair operator. Artifact:
`runs/earley-scene-chart-orbit-20260920.json`.

### Tense/agreement recipient chart

Feature conditioning was tested as a separate construction: present and past
realizations, singular/plural subject agreement, and recipient valency were
carried into the chart before clause pairing. The resulting inventory contained
9 noun phrases, 14 verb states, 6 recipient states, 504 ditransitives, and 3,612
agreement-valid clauses. A bounded run tested 1,972 live paired combinations;
all were pruned and no exact row above 38 letters closed. Three complete prose
controls were independently audited. This rules out the missing-feature
hypothesis for this chart family; the next construction changes aspectual
grammar rather than repairing a candidate. Artifact:
`runs/typed-tense-agreement-recipient-chart-20260920.json`.

### Typed dialogue quotation chart

The recursive-scene branch then added a typed dialogue grammar: a speaker,
speech predicate, `that` complement, and a complete quoted clause whose subject
agreement and valency were carried into the chart. Conjunction alternatives and
variable phrase boundaries were retained. The run produced 41,616 complete
dialogue paths and tested 300,000 seam states; every state was pruned and no
exact closure appeared. Generated dialogue controls (including 39-letter
controls) remain ordinary prose diagnostics, not palindrome evidence. The next
construction adds a typed question complement; no rendered string is repaired.
Artifact: `runs/dialogue-quote-chart-orbit-20260920.json`.

### Boundary-conditioned clause growth

The boundary-conditioned lane changed the search order again: it first indexed
complete clause frames by compatible exposed characters (including proper-name
objects), then grew both clauses inward while carrying residual debt across word
boundaries. It found 69 compatible seeds and tested 180,000 states with
1,438,093 immediate residual rejections. No exact candidate above 38 letters
closed. Its retained control is two complete clauses with explicit punctuation;
the article-agreement gate was corrected before the final run. The next
construction adds a typed relative frame to the same boundary index; no near
miss is repaired. Artifact:
`runs/boundary-conditioned-clause-growth-20260920.json`.

The indexed-relative follow-up added a complete `DET SUBJ V REL V DET OBJ`
frame and a final-object boundary index. It expanded 128 compatible seeds across
180,000 states and rejected 1,438,669 residual transitions, with no exact row
above 38 letters. The same punctuated controls remain the only reader-facing
text. Relative agreement is the next construction; failed strings are not
repaired. The run is retained in the same boundary-construction artifact.

### Recursive compositional clause series

To address length directly, a recursive grammar `C → SUBJ VP | SUBJ VP CONJ C`
was searched at depths one through three. Every predicate was a complete
transitive clause, and boundary-compatible lexicalization plus residual
consumption happened during construction. The run found 90 compatible seeds and
tested 180,000 states with 1,128,405 residual rejections; no exact candidate
above 38 letters closed. Controls are rendered as two punctuated complete clause
series and independently audited. This is a scalable construction boundary,
not a reason to repair near misses. Artifact:
`runs/recursive-clause-series-grammar-20260920.json`.

### Aspectual auxiliary chart

The recipient branch next carried progressive and perfect auxiliaries (`is/are`,
`was/were`, `has/have`, `had`) together with subject agreement and recipient
valency. It produced 336 aspectual VPs, 2,016 recipient VPs, and 8,064
agreement-valid clauses. A bounded run tested 3,328 live paired combinations;
all were pruned before closure and no exact row above 38 letters appeared. Three
ordinary aspectual controls were independently audited. The next construction
would add clitic/negation states only if they form a genuinely new grammatical
geometry; no candidate repair is used. Artifact:
`runs/aspect-auxiliary-agreement-recipient-chart-20260920.json`.

### Typed question-complement chart

The dialogue branch then replaced `that` complements with typed `whether`
questions. Subject number selected `do/does` and the bare question verb before
the seam chart advanced; conjunction alternatives and complete objects were
retained. The run produced 23,436 complete question-dialogue paths and tested
300,000 states, all pruned with zero exact closures. Generated controls reached
45 letters but are ordinary prose diagnostics only. The next construction is a
typed wh-question complement, not a repair pass. Artifact:
`runs/question-quote-chart-orbit-20260920.json`.

The wh-extraction follow-up added typed object extraction sites (`which letter`,
`what sign`) to complete question clauses while preserving subject/auxiliary
agreement and speech valency. It retained the 23,436-path chart and 300,000
state bound; every state was pruned and zero exact closures appeared. Controls
such as “the bard asks which letter does the king read” are complete questions,
not palindrome candidates. The next construction is pied-piping; no question
string is repaired. Artifact:
`runs/wh-question-chart-orbit-20260920.json`.

### Lexical grammar intersection with collocation ordering

The next lane returned to the successful boundary geometry but changed the
search object: complete SVO and SVO+PP clause paths were generated from
independent phrase banks, seeded by compatible outer character classes, and
expanded with unequal phrase lengths under live residual debt. A collocation
prior ranked only already-compatible states, with boundary-class coverage
reserved so scoring could not erase exact possibilities. The run tested 688
compatible seeds and states, rejected all 688, and closed no exact row above 38
letters. Twelve complete generated controls were independently audited. This
is a direct lexical intersection result, not a repair queue. Artifact:
`runs/lexical-grammar-lm-intersection-20260920.json`.

### Unequal phrase-count scheduler

The lexical intersection then removed its equal-path-count restriction. Left and
right clause paths could be SVO, SVO+PP, SVO+relative, or SVO+PP+relative, and
an independent scheduler advanced whichever side still had residual character
debt. It tested 420 outer-character seeds; every branch failed its first live
constituent obligation, so there were zero closures and no exact row above 38
letters. Complete controls reached 48 letters and were independently audited.
This is the direct scheduler result, not a repaired tape. Artifact:
`runs/independent-unequal-clause-scheduler-20260920.json`.

The recipient-bearing follow-up added typed ditransitive paths, recipient/theme
attachments, adjuncts, and relatives while retaining unequal phrase counts and
independent-side scheduling. It tested 2,160 outer-character seeds; every seed
failed the first live obligation, with zero closures and no exact row above 38
letters. Recipient and adjunct controls were independently audited. The next
construction carries recipient agreement states; no rendered candidate is
repaired. Artifact:
`runs/recipient-unequal-attachment-scheduler-20260920.json`.

The agreement-conditioned follow-up carried singular/plural subject and verb
realizations plus typed recipient number through the same unequal scheduler. It
tested 906 feature- and boundary-compatible seeds; all failed at the first live
obligation, with zero closures and no exact row above 38 letters. Two complete
agreement controls were independently audited. The next construction adds
recipient case alternants, not a repair pass. Artifact:
`runs/recipient-agreement-unequal-scheduler-20260920.json`.

The case-alternant follow-up made predicate-conditioned `to` and `for` frames
explicit while retaining number agreement and unequal SVO/ditransitive/PP/
relative paths. It tested 331 compatible feature/class seeds; every seed failed
the first live obligation, with zero closures and no exact row above 38 letters.
Complete `to` and `for` controls were independently audited. The next
construction adds definiteness and pronoun-case variants; no rendered string is
repaired. Artifact:
`runs/recipient-case-alternant-scheduler-20260920.json`.

The definiteness/pronoun follow-up added definite and indefinite recipient NPs
plus `him/her/them` pronoun states to the same unequal scheduler. It tested 360
compatible feature/class seeds; all failed the first live obligation, with zero
closures and no exact row above 38 letters. Definite-recipient and
pronoun-recipient controls were independently audited. The next construction
adds animacy/semantic selection states; no candidate is repaired. Artifact:
`runs/recipient-definiteness-pronoun-scheduler-20260920.json`.

### Negation/clitic polarity chart

The aspectual recipient grammar then added polarity as a first-class state:
positive auxiliaries and finite `auxiliary + not` clauses shared agreement,
tense, aspect, and recipient valency constraints. The inventory contained 672
polarity-conditioned VPs, 4,032 recipient VPs, and 16,128 clauses. A bounded
run tested 3,328 live pairs and pruned every one; no exact row above 38 letters
closed. A complete 40-letter negative recipient control was independently
audited, but it is not an exact candidate. The next construction is interrogative
inversion, not repair. Artifact:
`runs/negation-clitic-agreement-recipient-chart-20260920.json`.

### Typed relative recursive series

The recursive series was independently extended with a typed relative
nonterminal (`REL` plus a complete transitive predicate). Three recursive paths
produced 54 compatible boundary seeds; 39,992 states and 202,214 residual
rejections yielded zero exact candidates above 38 letters. Complete punctuated
controls include 53- and 56-letter relative-clause series, but remain ordinary
prose diagnostics. The next construction adds relative-object valency variants;
no near miss is repaired. Artifact:
`runs/recursive-series-typed-relative-20260920.json`.

### Semantic multiword phrase-trie orbit

To move away from single-word boundary loops, this lane indexed complete
authored phrase units (noun phrases, verb phrases, adjunct PPs, objects, and
proper names) by their first and last letters. Two ordinary-order semantic
templates were assembled from opposite ends while a live residual tape was
consumed across phrase and word boundaries. The run tested 112 endpoint seeds
and 3,552 incompatible phrase transitions; it closed no exact path and
produced no candidate above the 38-letter frontier. Three intact controls were
rendered at 38, 38, and 42 letters and independently checked with a two-pointer
comparison and forward/reverse SHA-256. This is a construction failure, not a
repaired-tape result: phrases remain in ordinary order, proper names are
selected as grammatical endpoints, and no mirrored token or catalogue text is
admitted. The next construction adds complete relative phrase units with
name-compatible endpoints, preserving the live cross-boundary equations.
Artifact: `runs/semantic-phrase-trie-orbit-20260920.json`.

The relative-endpoint extension added complete relative phrase units such as
“that name Diana” and “who sees Nora”, selected through a proper-name endpoint
trie. It tested 252 compatible seeds and 6,120 live residual transitions;
again, no exact path above 38 letters closed. Complete controls reached 36,
37, and 42 letters and passed the same independent pointer/SHA audit. This is
still direct construction: the relative units remain ordinary-order prose and
are never used to patch a rendered near miss. The next lane therefore changes
the search geometry to a full grammar/character product over complete paths,
rather than adding another local feature axis. Artifact:
`runs/relative-name-endpoint-phrase-trie-20260920.json`.

### Complete-grammar product automaton

The next reset built complete ordinary-order grammar paths before any
character intersection: 2,100 SVO, SVO+PP, SVO+relative, and SVO+PP+relative
paths were materialized, then paired in a weighted product automaton. The
weight ranked grammatical transitions only; it did not score or repair a
rendered candidate. The product traversed 14,400 live states, all rejected at
the first incompatible character, with zero exact closures above 38 letters.
Two complete prose controls (38 and 40 letters) passed independent
two-pointer/SHA audits. Because both sides were complete grammar paths before
the product, this result is distinct from endpoint-seeded schedulers and does
not grow a damaged near miss. The next construction moves the intersection to
a typed center seam with complement frames. Artifact:
`runs/grammar-product-automaton-intersection-20260920.json`.

### Typed center-seam complement product

This lane changed the join geometry again: 127 complete matrix/complement
frames were derived first, then paired at a typed center seam with live
two-pointer equations. It tested 16,129 seam products and pruned 16,127
incompatible joins. Crucially, it independently recovered the existing
38-letter construction in both orientations: “An aide rips nine memos; some
men inspire Diana.” and its reverse clause order. The recovered rows have
matching forward/reverse SHA-256 and exact pointer audits; they are retained as
a baseline recovery, not a new length claim. No new exact candidate above 38
letters closed. The next seam family is an interrogative complement grammar,
not a repair pass. Artifact:
`runs/typed-center-seam-complement-product-20260920.json`.

### Fresh complete-frame length extension

To pursue length without repairing the 38-letter row, the constructor generated
160 fresh ordinary-order frames with three, four, or five semantic
constituents (SVO, SVO+adjunct, SVO+finite-complement, and combined forms)
before any seam equations were evaluated. It tested 26,244 center-seam
products and pruned 26,242 incompatible joins. No fresh exact candidate above
38 letters closed. The same run recovered the known 38-letter geometry only in
two explicitly marked baseline-control rows; it did not wrap or edit those
rows. Fresh complete prose controls reached 38, 50, and 72 letters and passed
independent pointer/SHA audits. The next construction changes seam scheduling
to allow unequal constituent lengths. Artifact:
`runs/complete-frame-length-extension-20260920.json`.

### Unequal complete-frame seam scheduler

The zip-only seam was then replaced with independent constituent advancement:
one complete semantic frame may consume a residual across several phrase
boundaries before the other frame advances. The run tested 150 fresh
three-to-five-constituent frames and 23,128 live states, with 23,102 residual
prunes and 24 surviving transitions. It closed only ten explicitly isolated
baseline rows at 38 letters; no fresh exact candidate above 38 closed. The
fresh prose controls remain 38, 50, and 72 letters, all independently audited.
This is a direct unequal-length construction, not a repair or wrapper around a
finished palindrome. The next lane carries the scheduler into typed complement
frames. Artifact:
`runs/unequal-complete-frame-seam-scheduler-20260920.json`.

The same independent advancement was then carried into typed finite-complement
frames. It tested 160 complete matrix/complement or matrix/complement+adjunct
frames and 26,268 live states, with 26,242 residual prunes and 24 surviving
transitions. No fresh exact row above 38 letters closed; the known row was
recovered only as a baseline control. Complete complement controls reached 40,
50, and 55 letters and passed independent audits. The next construction adds
typed question/answer complement frames, not a repair pass. Artifact:
`runs/unequal-typed-complement-frame-scheduler-20260920.json`.

### Typed question/answer complement scheduler

The complement grammar was broadened to complete dialogue frames: 80 typed
`asks/whether` questions and 80 typed `answers/that` responses were materialized
before unequal seam scheduling. The product tested 12,800 states; every state
failed live character compatibility before closure. Complete dialogue controls
reached 39, 42, and 43 letters and passed independent pointer/SHA checks. The
known 38-letter palindrome is retained separately as a baseline control and is
not counted as generated by this lane. The next construction adds typed
wh-question/answer valency, not a repair pass. Artifact:
`runs/typed-question-answer-complement-scheduler-20260920.json`.

The wh follow-up added complete `who`, `what`, and `which book` question
constituents with matching transitive answer valency. It tested 160 complete
wh-dialogue frames and 12,800 live states; every state failed compatibility
before closure, with no fresh exact row above 38 letters. Complete wh-dialogue
controls reached 38, 40, and 43 letters and passed independent audits. The
next construction changes argument structure to wh-recipient questions with
typed ditransitive answers. Artifact:
`runs/typed-wh-question-answer-scheduler-20260920.json`.

The final local dialogue variant separated wh-recipient and wh-theme roles,
with typed ditransitive answer valency. It tested 160 complete frames and
12,800 live states; no fresh exact candidate above 38 letters closed. Complete
controls reached 37, 40, and 43 letters and passed independent audits. This
exhausts the local feature-axis family for this run: the next construction is a
lexical reverse-segmentation intersection over complete sentence paths, rather
than another recipient/case toggle or repair pass. Artifact:
`runs/typed-wh-recipient-ditransitive-scheduler-20260920.json`.

### Lexical reverse-segmentation grammar

This reset enumerated 3,403 complete ordinary-order sentences, then parsed
each sentence's reversed character obligation through role-specific tries with
variable phrase boundaries. It used 4,231 parser states and found three
complete reverse parses, recovering the known 38-letter palindrome in both
orientations (plus one duplicate control). No fresh exact parse above 38
letters appeared. The reverse side is a complete grammatical parse, not an
edited output: no finished tape is reversed for presentation, no near miss is
repaired, and no catalogue text is replayed. The next construction permits
independent right-side role permutations while retaining the complete parse
gate. Artifact:
`runs/lexical-reverse-segmentation-grammar-20260920.json`.

The reverse parser was then widened to independent right-side role paths:
SVO, SVO+PP, SVO+relative, and ditransitive frames. It tested 4,159 complete
left sentences and 19,081 trie/parser states. The known 38-letter geometry was
recovered and both baseline orientations were explicitly excluded from the
fresh result set; no fresh exact parse above 38 letters remained. This keeps
the reverse side grammatical while varying its role sequence, with no
post-hoc repair or presentation reversal. The next construction adds
attachment-conditioned role transitions. Artifact:
`runs/lexical-reverse-independent-role-parser-20260920.json`.

Attachment-conditioned parsing added typed object PPs, object relatives,
subject PPs, and recipient attachments to the reverse grammar. It tested 1,401
complete left sentences and 7,717 trie/parser states; attachment validity was
checked before any right parse was emitted. Only the excluded baseline reverse
parse survived, with no fresh exact candidate above 38 letters. No tape was
repaired or reversed for presentation. The next construction adds semantic
selection constraints for recipient and relative referents rather than another
surface attachment variant. Artifact:
`runs/attachment-conditioned-reverse-parser-20260920.json`.

Semantic selection was then made explicit during reverse parsing: recipients
had to be animate, `who` relatives required animate antecedents, and object
relatives required an object attachment. The parser tested 1,401 complete
forward sentences and 7,017 states; one baseline reverse parse was recovered
and excluded, with no fresh exact candidate above 38 letters. The next reset
must change the authored semantic grammar or vocabulary rather than add
another local constraint. Artifact:
`runs/semantic-selection-reverse-parser-20260920.json`.

To change the lexical geometry rather than add another constraint, a fresh
Shakespearean scene lattice was authored with bard, king, queen, knight, crown,
rose, moon, lute, throne, and court vocabulary. It produced 1,681 complete
scene paths and tested 5,043 reverse-trie states; no reverse parse closed, so no
fresh exact candidate above 38 letters appeared. The old seed remained an
excluded control. No catalogue text, repair, wrapping, or mirrored unit was
used. The next construction adds authored complement cadence and pronoun
attachment paths. Artifact:
`runs/shakespearean-scene-reverse-lattice-20260920.json`.

The Shakespearean lattice was expanded with authored matrix cadence
(`says/hears/vows/declares`), complete `that` complements carrying he/she/they
attachments, and PP/relative extensions. It produced 1,001 complete scene
paths and 3,003 reverse-trie states; no reverse parse closed. Complete
authored controls reached 50 and 51 letters and passed independent audits. The
next construction adds authored dialogue-response complements before the
search space is reassessed. Artifact:
`runs/shakespeare-complement-pronoun-reverse-lattice-20260920.json`.

The final scene-vocabulary lane added authored dialogue-response predicates
(`asks`, `replies`, `answers`, `speaks-to`) with pronoun-bearing complements and
PP/relative attachments. It produced 801 complete dialogue paths and 2,403
reverse-trie states; no fresh exact parse above 38 letters appeared. Complete
dialogue controls reached 53 and 63 letters and passed independent pointer/SHA
audits. The next step is reader packaging for the strongest intact controls,
not another repair or surface feature. Artifact:
`runs/shakespeare-dialogue-response-reverse-lattice-20260920.json`.

The broad-vocabulary lane used Brown-derived word forms and frequency scores
only, while composing new semantic SVO, adjunct, recipient, and relative
frames. It built 45,000 complete left frames and tested 180,000 reverse-parser
states; no complete reverse parse or exact candidate closed. Controls reached
31, 40, and 42 letters. Collocation scores ranked complete parses only and did
not certify readability; no Brown sentence text, catalogue palindrome,
finished-tape reversal, or repair was admitted. The next construction changes
grammar geometry to coordination rather than enlarging lexical scoring.
Artifact: `runs/brown-authored-semantic-reverse-decoder-20260920.json`.

The coordination lane composed two complete semantic clauses with a Brown-bank
connector before reverse parsing. It tested 12,000 complete left frames and
24,000 coordination attempts, with no complete reverse parse or exact
candidate. Intact controls reached 41 and 49 letters. The frozen lexical bank
contained only `but` among the requested conjunctions; this is disclosed rather
than fabricated. No catalogue sentence, repair, wrapping, or mirrored unit was
used. The next construction changes grammar geometry to subordination and
embedded clauses. Artifact:
`runs/brown-authored-coordination-reverse-decoder-20260920.json`.

The subordination lane composed complete matrix plus embedded `that`-complement
and `who`-relative frames from Brown-derived word forms, then parsed the
reverse obligation through a second grammar. It tested 9,000 forward frames
and 9,000 reverse states, with no complete reverse parse or exact candidate.
Controls reached 33 and 43 letters. `that` and `who` were admitted only when
present in the frozen lexical bank; unavailable `whether` was not fabricated.
No catalogue text, repair, wrapping, or mirrored unit was used. The next
construction changes to discourse-frame grammar. Artifact:
`runs/brown-authored-subordination-reverse-decoder-20260920.json`.

The discourse-frame lane added source-admitted temporal markers (`then`, `now`)
to complete semantic clauses before reverse parsing. It tested 10,000 complete
frames and 10,000 reverse states, with no complete reverse parse or exact
candidate. Intact controls reached 45 and 53 letters. `however` and `therefore`
were unavailable in the frozen lexical bank and were not fabricated. No
catalogue text, repair, wrapping, or mirrored unit was used. The next
construction changes to an event-chain grammar. Artifact:
`runs/brown-authored-discourse-reverse-decoder-20260920.json`.

The event-chain lane composed complete temporal sequences with source-admitted
`first`, `then`, `after`, and `before` markers before reverse parsing. It
tested 10,000 complete event frames and 10,000 reverse states, with no
complete reverse parse or exact candidate. Intact controls reached 49 and 55
letters. No catalogue text, repair, wrapping, or mirrored unit was used. The
next construction changes to causal-frame grammar. Artifact:
`runs/brown-authored-event-chain-reverse-decoder-20260920.json`.

The causal lane composed complete cause/result frames with source-admitted
because/so-style markers before reverse parsing. It tested 10,000 complete
frames and 10,000 reverse states, with no complete reverse parse or exact
candidate. Intact controls reached 42 and 54 letters. No catalogue text,
repair, wrapping, or mirrored unit was used. The next constructor is a joint
bidirectional beam decoder that grows both complete grammar sides under exact
character constraints instead of enumerating fixed frame products. Artifact:
`runs/brown-authored-causal-reverse-decoder-20260920.json`.

The first joint bidirectional decoder expanded both grammar sides
synchronously, matching character obligations while word boundaries remained
live. It tested two complete SVO/SVO+place shapes across 16 states; all died at
the initial boundary, with zero closures. Collocation scores ranked only
already-compatible states, and complete 38- and 46-letter prose controls were
audited independently. This failure identifies a missing state dimension, not
a case for a wider beam: the next topology allows free cross-word boundary
offsets through a lexical boundary transducer. Artifact:
`runs/brown-bidirectional-beam-decoder-20260920.json`.

The free-offset boundary transducer allowed independent word entry and exit on
the two sides while retaining complete role-specific grammar paths. It still
explored only 16 states and closed no exact candidate; the same 38- and
46-letter prose controls were independently audited. This isolates the next
missing geometry: a seam-indexed lexical generator must choose compatible
character spans before selecting complete grammar roles, rather than widening
the same beam. Artifact:
`runs/brown-boundary-transducer-free-offsets-20260920.json`.

The seam-indexed lexical generator moved compatibility earlier: Brown role words
were bucketed by exposed character and span length before complete grammar
finish. It explored three residual-compatible seam states and closed no exact
candidate; complete 38- and 46-letter prose controls were independently
audited. This is distinct from the live beam/transducer, but the frontier is
still too sparse. The next construction widens semantic frame topology rather
than refining the seam index. Artifact:
`runs/brown-seam-indexed-lexical-generator-20260920.json`.

The authored phrase-pair graph moved construction to the seam itself: 12 left
scene/dialogue chunks and 11 right chunks yielded eight exposed-span-compatible
pairs, followed by three complete frame-topology checks. No fresh exact row
closed. The 38-letter seed was independently audited and retained only as an
excluded baseline control; new chunks were not wrappers around it. This closes
the current constructive sweep and supplies evidence for a method-level
reassessment rather than another ad hoc feature lane. Artifact:
`runs/authored-phrase-pair-graph-scene-20260920.json`.

The phrase-pair graph was expanded to 20 human-authored chunks on each side.
It yielded 35 non-palindromic exposed-span-compatible pairs and 1,920 complete
four-to-six-chunk scene/dialogue composition states with valency and agreement
checks. No fresh exact candidate above 38 letters closed. Complete controls
reached 37 and 60 letters; the known 38-letter row remained an excluded
baseline. This is the current constructive frontier: no wrapping, repair,
finished-tape reversal, catalogue text, or mirrored unit was used. Artifact:
`runs/authored-phrase-pair-graph-scene-expanded-20260920.json`.

The seed-conditioned paired grammar reused only the successful 38-letter
slot geometry; all phrase banks and both complete frames were newly authored.
It tested 4,608 fresh left frames and 13,824 reverse-segmentation DP states in
the 40–100-letter target, with zero complete reverse parses. The seed was a
calibration-only control, never generated or wrapped. Complete controls reached
40 and 44 letters and passed independent audits. Artifact:
`runs/seed-conditioned-paired-grammar-dp-20260920.json`.

The packed CFG/Earley lane memoized recursive `S`, `NP`, `VP`, relative, and
conjunction productions, then intersected independent lexical expansions with
live residual character debt. It tested four packed chart paths, 64 compatible
boundary seeds, 180,000 lexical-intersection states, and 1.44 million
residual rejections. No exact candidate above 38 letters closed. The complete
CFG control was independently audited; no repair, finished-tape reversal,
mirrored token, or catalogue text was used. The next construction changes the
semantic representation to a dependency-tree seam CSP. Artifact:
`runs/packed-cfg-earley-intersection-20260920.json`.

The dependency-tree seam CSP generated three typed dependency trees and paired
their exposed character spans before expanding complete role-bearing clauses.
It tested 9 tree pairs, 18 compatible boundary seeds, 84,852 live states, and
307,864 seam rejections, with no exact candidate above 38 letters. Independently
audited complete controls reached 39, 41, 49, 51, 60, 65, and 67 letters.
The character equations were enforced before rendering; no repair,
finished-tape reversal, mirrored token, catalogue text, or fragment output was
admitted. This is a topology-level failure, so the next construction changes
the semantic representation again: a role-permuted scene-graph transducer
with cross-boundary spans, rather than another agreement or morphology toggle.
Artifact: `runs/dependency-tree-seam-csp-20260920.json`.

The role-permutation chart then generated complete scene graphs independently
on each side and permuted agent/action/object/recipient/adjunct order before
carrying character spans across role boundaries. It evaluated 600 left graphs
against 600 right graphs (360,000 chart pairings), but found zero
span-compatible pairs and therefore zero fresh exact candidates above 38
letters. Complete controls at 44 and 50 letters passed the prose-structure
check and were independently audited with the pointer and forward/reverse
SHA-256 tests. This is a genuine topology miss rather than a repairable near
miss; no post-hoc edit, finished-tape reversal, mirrored token, catalogue
text, or fragment was admitted. Artifact:
`runs/dependency-role-permutation-chart-20260920.json`.

A second role-permuted scene-graph transducer used typed agent, patient,
recipient, time, and place roles with independently ordered complete graphs.
It found 48 compatible boundary seeds and explored 2,718 live states with
12,465 seam rejections, but no exact candidate above 38 letters closed.
Rendered complete controls included “The baker sends the child; the baker
sends the poet” and “The farmer sends the child; the baker sends the poet”;
both received independent pointer and SHA audits, but are controls rather
than palindrome outputs. The lane changes the role topology without any
repair, tape reversal, mirrored unit, catalogue text, or fragment emission.
Artifact: `runs/role-permuted-scene-graph-20260920.json`.

The masked-scene infill CSP represented every target character as a paired
variable while independently choosing complete typed scene frames. It tested
144 left frames against 144 right frames (20,736 length-band grid states), with
12,867 paired-slot conflicts and 7,869 left-fill conflicts; no fresh exact
candidate closed. Complete 40- and 44-letter prose controls were independently
audited. Because conflicts occurred during construction, this lane used no
post-hoc repair, finished-tape reversal, mirrored units, catalogue text, or
fragments. Artifact: `runs/masked-scene-infill-csp-20260920.json`.

The parallel blank-verse scene lattice jointly selected speaker, adjective,
verb, noun, and utterance roles for two authored Shakespearean-style lines.
The outer seam rejected all 9 states before any inner lexical advance, so it
produced no exact candidate; complete controls included “my lord sees the
crimson moon,” “dear friend marks the silent bell,” and “the queen names our
hidden vow.” This is a representation bottleneck, not a near-miss repair
case, and no reversal, mirrored token, catalogue text, or fragment output was
used. Artifact: `runs/blank-verse-parallel-scene-20260920.json`.

The complement/coordination hypergraph changed clause topology before lexical
selection, jointly considering single events, `that` complements, coordinated
events, and alternatives. It tested 4 hypergraphs across 16 topology pairs,
64 compatible seeds, and 29,712 states with 109,390 seam rejections. No exact
candidate above 38 letters closed. Complete controls reached 43, 44, 59, and
60 letters, including a `that`-complement sentence, and were independently
audited. No repair, finished-tape reversal, mirrored unit, catalogue text, or
fragment was admitted. Artifact:
`runs/complement-coordination-hypergraph-20260920.json`.

The event-indexed compositional hypergraph then represented two-event
discourses with cause, contrast, and temporal relation edges before solving the
paired character tape. Six authored event nodes per side yielded 2,700 graph
pair states, but zero span-compatible graph pairs and zero fresh exact
candidates above 38 letters. Complete 57- and 63-letter discourse controls
were independently audited. This is a relation-topology frontier result, not
a repair opportunity; no reversal, mirrored token, catalogue text, or
fragment output was used. Artifact:
`runs/event-indexed-compositional-hypergraph-20260920.json`.

The CCG/supertagged seam constructor changed the grammar representation to
forward/backward application and composition over independently generated
lexical categories. It tested 3 complete templates, 12 compatible seeds,
86,004 states, and 292,832 seam rejections, with no exact candidate above 38
letters. Complete CCG controls reached 35, 44, 60, and 69 letters and were
independently audited; repeated controls are retained as controls, not claimed
as outputs. No repair, finished-tape reversal, mirrored unit, catalogue text,
or fragment was admitted. Artifact:
`runs/ccg-supertagged-seam-20260920.json`.

The broad name-bank generator expanded the lexical inventory while retaining
complete ordinary-order SVO, ditransitive, and relative frames with semantic
valency checks. It tested 5,936 complete frames and 1,416,056 paired lexical
states, but found zero span-compatible pairs and zero fresh exact candidates
above 38 letters. Complete controls included “Alice guards the crown; Diana
praises the red letter” (43 letters) and “The young bard writes a quiet song;
Marie seeks the moon” (45 letters), each independently audited. No catalogue
sentence text, repair, finished-tape reversal, mirrored token, or fragment was
used. Artifact: `runs/broad-name-bank-complete-sentence-generator-20260920.json`.

The broad-lexicon envelope transducer removed the whole-frame alignment
restriction: a character-level WFSA carried independent word-boundary offsets
while complete SVO, ditransitive, and PP semantic paths stayed open. It
explored 255 envelope states across the expanded common-word/name bank, with
zero fresh exact candidates above 38 letters. The same 43- and 45-letter
complete prose controls were independently audited. No post-hoc repair,
finished-tape reversal, mirrored token, catalogue text, or fragment was
admitted. Artifact:
`runs/broad-lexicon-envelope-transducer-20260920.json`.

The scaled endpoint-indexed envelope replaced the small bank with the Brown
POS inventory plus 45 ordinary authored names: 102 agent forms, 60 verbs, 60
objects, 50 places, and 30 prepositions. Its endpoint index contained 285
keys and 30,600 indexed envelope starts, yet no fresh exact candidate above 38
letters closed. Complete controls at 34 and 43 letters were independently
audited, with the Brown bank hash recorded for reproducibility. No repair,
finished-tape reversal, mirrored unit, catalogue text, or fragment was used.
Artifact: `runs/scaled-endpoint-indexed-envelope-20260920.json`.

An implementation audit found that the preceding envelope compared raw phrase
strings, allowing spaces to enter the character equation. The corrected
normalized-envelope transducer stores rendered words separately from each
phrase's letter tape, excludes spaces and punctuation from matching, and
advances roles only after normalized tapes are exhausted. It explored 50
corrected states and still produced zero fresh exact candidates above 38
letters. Complete 38- and 40-letter controls passed independent pointer/SHA
audits. No repair, finished-tape reversal, mirrored unit, catalogue text, or
fragment was admitted. Artifact:
`runs/corrected-normalized-envelope-transducer-20260920.json`.

The normalized multiword-slot envelope extended that corrected state model to
explicit multiword agent, object, PP, and relative slots. It explored 27 live
states with complete normalized tapes and separate rendered boundaries, and
closed zero fresh exact candidates above 38 letters. Complete controls reached
40 and 48 letters and passed independent pointer/SHA audits. No repair,
finished-tape reversal, mirrored unit, catalogue text, or fragment was
admitted. Artifact:
`runs/normalized-multiword-slot-envelope-20260920.json`.

The discourse-plan delayed-realization lane changed the search object rather
than the seam: it first chose one coherent authored Shakespearean scene plan,
then jointly realized two utterances with shared speaker, events, referents,
tense, and rhetorical relation while character positions and word boundaries
remained latent. Eight plans with 3--5 lexical realizations per role yielded
3,848 attribute/pushdown states and 11,544 residual rejections. It produced 24
complete prose controls, including a 61-letter scene realization, but no exact
candidate above 38 letters. All controls and the exact gate use independent
pointer/SHA audits; no repair, tape reversal, mirrored unit, catalogue text,
or fragment was admitted. Artifact:
`runs/discourse-plan-delayed-realization-20260920.json`.

The typed-valency follow-up made transitive, intransitive, and ditransitive
predicate features, subject agreement, and referent binding explicit before
surface emission. It retained the shared discourse plan and delayed paired
character generation, testing 165 states and 165 residual rejections with 24
complete controls (up to 60 letters) and zero exact candidates above 38
letters. The lane is retained as a grammar result; its authored vocabulary was
still literary, so it is not treated as a readability claim. No repair,
reversal, mirrored unit, catalogue text, or fragment was used. Artifact:
`runs/discourse-plan-typed-valency-20260920.json`.

Because “Shakespearean” is a broad readability target rather than a literal
style requirement, the same typed discourse representation was rerun with
plain contemporary English plans and lexical choices. It produced 24 complete
natural-English controls (maximum 53 letters), explored 41 states, and had 41
residual rejections with zero exact candidates above 38 letters. Controls
included “I notice the storm and track my friend,” “We notice that storm, but
track our friend,” and “The teacher notices the rain, so tracks the neighbor.”
This is the same joint delayed-realization geometry with the style bias
removed; no repair, reversal, mirrored unit, catalogue text, or fragment was
used. Artifact:
`runs/discourse-plan-typed-valency-plain-20260920.json`.

The plain tense/aspect extension initially exposed malformed past stems and
auxiliary agreement in its controls; that raw artifact is not counted as
readability evidence. After a focused morphology correction (`noticed`,
`tracked`, `waited`, `sent`; `I am`, `we are`, `the teacher is`, `I/we were`,
`the teacher was`) and strict surface validation, the rerun produced 32
natural contemporary-English controls (maximum 52 letters), 164 states, and
164 residual rejections, with zero exact candidates above 38 letters. The
corrected run is the only tense/aspect result retained for evaluation. Artifact:
`runs/discourse-plan-tense-aspect-20260920.json`.

The multi-plan plain-English expansion kept the corrected morphology and added
eight ordinary discourse plans (weather, school, travel, meeting, letter,
garden, work, and home). Each plan was realized in present/past simple and
present/past progressive states with typed valency, agreement, and referent
binding. The run produced 32 complete contemporary-English controls, including
natural 59-letter realizations such as “the manager is reviewing the report, so
is sending the client the report,” while the work plan was constrained to a
grammatical transitive “share the results” frame. It explored 128 paired
character states, rejected 128 residual states, and closed zero exact
candidates above 38 letters. Every control and candidate gate was checked with
the independent two-pointer audit and forward/reverse SHA-256 tapes. No repair,
reversal, mirrored unit, catalogue text, or fragment was admitted. Artifact:
`runs/discourse-plan-multi-plain-20260920.json`.

The open-constituent-stack lane then changed the delayed realization state
itself. A shared contemporary-English discourse plan carried referent IDs,
event and tense attributes, and an attachment choice (PP, adverb, or adjective
phrase). Two sides popped the same typed constituent stack while emitting
surface letters into the live residual equation; neither side was first
materialized as a complete sentence. Three plans visited 12 paired states and
pruned 12 outer character seams before any inner chart advance. The lane still
closed zero exact candidates, but it rendered and independently audited 20
complete prose controls (maximum 37 letters), including “The keeper guarded a
book under the high roof.” The extended novelty signature is distinct from the
earlier delayed-realization lanes because it exposes stack ownership and
attachment selection before output. No repair, reversal, mirrored unit,
catalogue text, or fragment was admitted. Artifact:
`runs/discourse-stack-synchronous-20260920.json`.

The anaphoric two-clause lane kept stack ownership but bound the second clause
to the first clause's referent before any character equation was tested. Eight
ordinary subjects, eight verb/object pairs, and four discourse linkers yielded
256 complete contemporary-English frames and 24 intact controls (maximum 54
letters), such as “The analyst reviews the report and they review the report.”
Plural pronoun realization uses the correct uninflected verb (`they review`,
not `they reviews`). No frame was residual-compatible at the tested boundary,
so the exact count remained zero. The independent pointer/SHA audit and
novelty preflight passed; no repair, reversal, mirrored unit, catalogue text,
or fragment was admitted. Artifact:
`runs/anaphoric-two-clause-discourse-20260920.json`.

The reference-form constructor made definite-NP versus pronoun choice an
explicit discourse attribute before paired character emission. Five ordinary
scenes (classroom, meeting, garden, travel, and home) supplied 25 complete
controls, maximum 52 letters, with agreement-correct forms such as “The
gardener waters the plant and it needs the soil” and “The parent calls the child
and they answer the call.” The initial subject seam admitted zero live states,
so there were zero exact candidates above 38 letters. Independent pointer/SHA
audits passed, and no repair, reversal, mirrored unit, catalogue text, or
fragment was admitted. Artifact:
`runs/discourse-reference-form-20260920.json`.

The semordnilap-aware full-clause constructor tested a different lexical
geometry: reverse-indexed word spans were consumed across word boundaries, but
both sides had to close as complete ordinary clauses. The fresh bank used two
frames (SVO and SVO+PP), rejected repeated or self-palindromic units at exact
admission, and kept complete-clause controls separate from that shortcut gate.
The corrected run rendered 24 contemporary controls (maximum 36 letters),
indexed 22 reverse lexical keys, and reached zero live exact states at the
initial subject seam. It therefore produced zero exact candidates above 38
letters. Independent pointer/SHA audits passed; no catalogue text, repair,
finished-tape reversal, mirrored unit, or fragment was admitted. Artifact:
`runs/semordnilap-full-clause-20260920.json`.

The asymmetric-role lane changed the reversal topology rather than the
lexicon: a complete left SVO clause was paired against a complete right OVS
clause so the left subject aligned with the right object and vice versa. Ten
subjects, ten objects, and ten verbs yielded one million live role states, but
zero span-compatible renderings survived the first character equation. Twenty
authored clause-pair controls remained intact contemporary prose (maximum 52
letters), including “Alice reviews the report; the analyst checks the file.”
Independent pointer/SHA audits passed; no repair, reversal, mirrored unit,
catalogue text, or fragment was admitted. Artifact:
`runs/asymmetric-role-semordnilap-clause-20260920.json`.

The boundary-seeded follow-up changed search order: it first indexed partial
subject/object endpoint spans under the reversed alignment of two ordinary SVO
clauses, then opened independently chosen verbs while carrying the residual
characters across phrase boundaries. Twenty-five endpoint seeds produced 2,500
live verb states, all rejected at the next residual equation; no
rendered-compatible candidate or exact closure above 38 letters appeared. A
separate authored control set contained 20 complete contemporary-English
sentences (maximum 39 letters), including “Alice reviews the report near the
station.” Independent pointer/SHA audits passed; no repair, reversal, mirrored
unit, catalogue text, or fragment was admitted. Artifact:
`runs/boundary-seeded-asymmetric-clause-lattice-20260920.json`.

The free-center discourse lane changed the object again: one complete discourse
template was selected around a free semantic pivot (verb, object, or connector),
then independently chosen phrase chunks grew outward on both sides under a
center-adjacent character equation. Five pivot positions produced 320 states,
all pruned at the first boundary comparison, with zero exact candidates. The
separate authored set contained 20 complete contemporary-English controls,
including “Mara reads the letter and the sailor marks the seal.” Every control
and candidate gate received the independent pointer/SHA audit. No repair,
finished-tape reversal, mirrored unit, catalogue text, or fragment was admitted;
human reading remains pending until an exact row exists. Artifact:
`runs/free-center-discourse-growth-20260920.json`.

The authored cross-word seam lane independently lexicalized both complete
clauses from a small hand-authored inventory while exposing phrase boundaries
to the live character equation. It explored 120,000 bounded seam states across
eight agents, eight verbs, eight objects, and five attachment phrases, but no
seam-compatible closure survived. Twenty authored controls were complete
contemporary English (maximum 42 letters), including “The engineer checks the
schedule beside the river.” Independent pointer/SHA audits passed; no repeated
unit, repair, reversal, catalogue text, or fragment was admitted. Artifact:
`runs/authored-crossword-seam-phrase-design-20260920.json`.

The cross-word equation scene lane made the lexical equation itself the
construction object. It selected a complete contemporary two-clause scene
with typed agent, event, patient, and connective roles, then consumed lexical
characters across independently chosen phrase boundaries before either clause
was complete. The 25 live states all failed at the first outer role equation,
so there were no equation completions and no exact candidate above 38 letters.
The separate control set contains 20 intact contemporary-English clauses,
with a maximum of 42 letters; examples include “Mara reads the letter and the
queen opens the gate.” Every state and control has an independent two-pointer
and forward/reverse SHA audit. The lane uses no repair, finished-tape reversal,
mirrored unit, catalogue text, or fragment. Its next constructive expansion is
two-word subject chunks with agreement carried in the live equation rather than
post-hoc repair. Artifact:
`runs/cross-word-equation-scene-20260920.json`.

The agreement-carrying subject-chunk lane widened each clause's subject to a
two-word-capable number-bearing inventory. Singular/plural features were bound
to the corresponding verb before character emission, while independent phrase
boundaries continued to consume the live reverse-compatible equation. It
visited 40 states, all pruned at the first equation, with zero equation
completions and zero exact candidates above 38 letters. Twenty intact
contemporary-English controls were rendered and independently audited (maximum
42 letters); the set includes “A sailor opens the book while some guards mark
the seal.” No repair, reversal, mirrored unit, catalogue text, or fragment was
admitted. The next constructive expansion is a two-word object chunk carrying
animacy and number while retaining subject agreement. Artifact:
`runs/agreement-subject-chunk-equations-20260920.json`.

The typed object-chunk lane added number- and animacy-bearing object chunks to
both clauses while retaining subject agreement before verb emission. A small
verb/object compatibility table rejects ill-typed readings (for example,
greeting an animate object versus opening an inanimate one) at closure rather
than relying on the controls to hide them. The lane visited 64 states, all
pruned at the first live equation, and produced zero equation completions and
zero exact candidates above 38 letters. Its 20 controls are intact
contemporary English (maximum 43 letters), including “Some sailors greet the
poets and the queens mark a book.” Independent pointer/SHA audits passed; no
repair, reversal, mirrored unit, catalogue text, or fragment was admitted. The
next construction is typed recipient/dative structure with object number still
carried live. Artifact:
`runs/typed-object-chunk-equations-20260920.json`.

The typed recipient/dative lane changed the clause frame to a complete
ditransitive scene: animate recipients and inanimate themes were separate
semantic roles, theme number remained live, and subject number still governed
verb agreement before output. It visited 36 states, all pruned at the first
equation, with zero equation completions and zero exact candidates above 38
letters. Twenty intact contemporary-English ditransitive controls were
rendered and independently audited (maximum 65 letters), including “The poet
gives the child the letter and the queen sends a friend a book.” No repair,
reversal, mirrored unit, catalogue text, or fragment was admitted. The next
constructive expansion is an explicit prepositional benefactive recipient
frame. Artifact:
`runs/typed-recipient-dative-equations-20260920.json`.

The benefactive-preposition lane replaced the bare ditransitive recipient with
an explicit `for`/`to` phrase while retaining typed animate recipients,
inanimate theme number, and subject agreement. The corrected implementation
tracks both clause sides independently, uses reverse arrival order only for
the live character equation, and audits the complete grammatical surface in
surface order. It visited 36 states, pruned 36 seams, and produced zero
equation completions and zero exact candidates above 38 letters. Twenty intact
contemporary-English benefactive controls were independently audited (maximum
68 letters), including “A sailor brings the seal for the poet while some guards
offer books to the children.” The earlier run was invalid because its closure
accumulator was never populated; it is superseded and not counted. No repair,
reversal, mirrored unit, catalogue text, or fragment was admitted. The next
construction alternates recipient prepositions under an explicit discourse
relation. Artifact:
`runs/benefactive-preposition-equations-20260920.json`.

The relation-conditioned recipient lane made the preposition a live discourse
state: `benefit` selects `for`, while `transfer` selects `to`. It retains
animate-recipient and inanimate-theme typing, theme number, subject agreement,
and complete grammatical surface reconstruction. A verb/preposition gate
keeps destination-only readings such as *show ... for* out of the closure
grammar. The run explored two relation states and 72 live states, pruned all
72 at the equation boundary, and produced zero equation completions and zero
exact candidates above 38 letters. Twenty intact contemporary-English
controls were independently audited (maximum 71 letters), including “The poet
gives the letter for the child and the queen sends a book for a friend.” No
repair, reversal, mirrored unit, catalogue text, or fragment was admitted. The
next construction conditions the relation on a contrastive discourse
connective. Artifact:
`runs/relation-conditioned-preposition-equations-20260920.json`.

The contrastive-connective lane coupled the relation state to the connective:
benefit frames use “although” with `for`, while transfer frames use “while”
with `to`. It retains typed recipient/theme roles, number and agreement, and
the verb/preposition compatibility gate, with complete surface rendering
separate from reverse equation arrival. Two relation states yielded 72 live
states; all 72 were pruned at the equation boundary, with zero equation
completions and zero exact candidates above 38 letters. Twenty intact
contemporary-English controls were independently audited (maximum 73 letters),
including “A sailor brings the seal for the poet although some guards offer
books for the children.” No repair, reversal, mirrored unit, catalogue text, or
fragment was admitted. The next construction introduces a contrastive
subordinate clause with explicit subject shift while retaining the recipient
state. Artifact:
`runs/contrastive-relation-equations-20260920.json`.

The contrastive subject-shift lane changed the second clause's agent identity
while retaining the benefit/transfer relation and its although/for versus
while/to coupling. It kept typed recipient/theme roles, subject agreement, and
the verb/preposition compatibility gate, with complete surface rendering
separate from reverse equation arrival. Two relation states yielded 60 live
states; all 60 were pruned at the equation boundary, with zero equation
completions and zero exact candidates above 38 letters. Twenty intact
contemporary-English controls were independently audited (maximum 73 letters),
including “Mara gives the letter for the child although Noah sends a book for a
friend.” No repair, reversal, mirrored unit, catalogue text, or fragment was
admitted. The next construction realizes the shifted subject anaphorically
while retaining explicit referent identity. Artifact:
`runs/contrastive-subject-shift-equations-20260920.json`.

The anaphoric-subject lane replaced the shifted second-clause name with a
pronoun (`she`, `he`, or `they`) bound to an explicit antecedent ID before
equation closure. It retained relation-conditioned although/for versus
while/to, typed recipient/theme roles, subject agreement, and the
verb/preposition compatibility gate. Two relation states yielded 36 live
states; all 36 were pruned at the equation boundary, with zero equation
completions and zero exact candidates above 38 letters. Twenty intact
contemporary-English controls were independently audited (maximum 65 letters),
including “Mara gives the letter for the child although she sends a book for a
friend.” No repair, reversal, mirrored unit, catalogue text, or fragment was
admitted. The next construction allows anaphoric recipient pronouns with an
explicit antecedent type. Artifact:
`runs/anaphoric-subject-relation-equations-20260920.json`.

The anaphoric-recipient lane added typed recipient antecedents and realized
them as `him`, `her`, or singular/plural `them`, while retaining the
anaphoric subject, relation-conditioned connective/preposition, and semantic
verb gate. Recipient identity and number must match the antecedent exactly
before closure. Two relation states yielded 24 live states; all 24 were
pruned at the equation boundary, with zero equation completions and zero exact
candidates above 38 letters. Twenty intact contemporary-English controls were
independently audited (maximum 60 letters), including “Mara gives the letter
for the child although she sends a book for him.” No repair, reversal,
mirrored unit, catalogue text, or fragment was admitted. The next construction
jointly realizes anaphoric subject and recipient with plural agreement and
explicit coreference constraints. Artifact:
`runs/anaphoric-recipient-relation-equations-20260920.json`.

The joint-plural anaphora lane bound both clauses to plural antecedents: a
plural subject realized as `they`, a plural recipient as `them`, and both
verbs were checked for plural agreement before output. Relation-conditioned
although/for versus while/to and the semantic verb gate remained live. Two
relation states yielded two live states; both were pruned at the equation
boundary, with zero equation completions and zero exact candidates above 38
letters. Twenty intact contemporary-English controls were independently
audited (maximum 71 letters), including “The guards give the letters for the
children although they send books for them.” No repair, reversal, mirrored
unit, catalogue text, or fragment was admitted. The next construction branches
mixed singular/plural antecedents with explicit agreement. Artifact:
`runs/joint-plural-anaphora-equations-20260920.json`.

The mixed-agreement lane branched singular and plural antecedents while
retaining joint subject/recipient anaphora, relation-conditioned
although/for versus while/to, explicit agreement, and semantic
verb/preposition gates. It visited 18 live states, pruned all 18 at the
equation boundary, and produced zero equation completions and zero exact
candidates above 38 letters. Twenty intact contemporary-English controls were
independently audited (maximum 63 letters), including “Noah brings the letter
to a friend while he offers books to her.” No repair, reversal, mirrored unit,
catalogue text, or fragment was admitted. The next construction adds
cross-clause number-mismatch alternatives with explicit non-coreferential
subjects. Artifact:
`runs/mixed-agreement-anaphora-equations-20260920.json`.

The non-coreferential number-mismatch lane removed the pronoun link between
clause subjects and independently branched singular/plural agreement, while
retaining relation-conditioned prepositions/connectives, typed recipient/theme
roles, and semantic verb gates. It visited 48 live states, pruned all 48 at
the equation boundary, and produced zero equation completions and zero exact
candidates above 38 letters. Twenty intact contemporary-English controls were
independently audited (maximum 73 letters), including “The poet shows a book to
the child while the queens give letters to the children.” No repair, reversal,
mirrored unit, catalogue text, or fragment was admitted. This closes the
current discourse-feature family; the next run pivots to a materially
different character/lexical search. Artifact:
`runs/noncoreferential-number-mismatch-equations-20260920.json`.

The endpoint-indexed lexical-envelope pivot changed the search representation
rather than adding another discourse feature. It draws a larger Brown-derived
inventory plus 20 authored names, indexes lexical endpoints, and advances two
independent complete grammar templates across live word boundaries. Explicit
transitive, ditransitive, and location-preposition compatibility gates run at
closure; they are diagnostic grammar filters, not readability certification.
Across SVO, SVO+PP, and ditransitive template pairs it visited 1,659 live
lexical-envelope states and 177 endpoint keys, with zero fresh exact candidates
above 38 letters. Its separate control set contains 20 unique intact
contemporary-English sentences, independently audited (maximum 47 letters),
including “Alice reads the report near the station.” No repair, reversal,
mirrored unit, catalogue text, or fragment was admitted. The next constructive
expansion is phrase-level relative clauses in this lexical-envelope search.
Artifact: `runs/large-phrase-endpoint-reverse-envelope-20260920.json`.

The relative-clause expansion kept the endpoint envelope but added subject-gap
(`subject who/that verb object`) and object-gap (`object who/that subject verb`)
templates. Relative-role and transitivity gates run before exact admission,
while word boundaries remain live. The combined run visited 2,912 states
(1,659 base and 1,253 relative), indexed 179 endpoint keys, and produced zero
fresh exact candidates above 38 letters. It rendered 40 unique intact controls,
including 20 complete relative-clause controls such as “The teacher who read
the essay praised the student.” All controls received independent pointer/SHA
audits; no repair, reversal, mirrored unit, catalogue text, or fragment was
admitted. Artifact:
`runs/large-phrase-endpoint-reverse-envelope-20260920.json`.

The optional-slot live-buffer lane independently combined a core two-clause
frame with optional locative PP, relative, and adjunct slots. It maintained
complete unmatched prefix/suffix buffers while choosing each word. The base,
PP, relative, and combined frames visited 1,938, 1,938, 816, and 24 live
states respectively, with zero exact candidates. Twenty unique intact
contemporary-English controls were independently audited; no repair, reversal,
mirrored unit, catalogue text, or fragment was admitted. This result motivated
the larger authored two-clause lexical bank rather than another discourse
toggle. Artifact:
`runs/live-buffer-optional-grammar-search-20260920.json`.

The richer authored two-clause live-buffer lane expanded the lexical bank to
71,280 complete SVO/SVO+PP/SVO+relative clause realizations. It indexed 894
endpoint buckets and examined 57,644 live paired states after enforcing
determiner-number, subject-verb agreement, and transitive-object valency. No
exact candidate above 38 letters emerged. Twenty unique intact
contemporary-English controls were independently audited (maximum 40 letters),
including “the artist admires the apple at the harbor.” The initial draft's
bare common-noun surfaces was corrected before this run; only the corrected
artifact is counted. No repair, reversal, mirrored unit, catalogue text, or
fragment was admitted. Artifact:
`runs/richer-two-clause-live-buffer-20260920.json`.

The bidirectional phrase-trie join was a separate whole-sentence construction:
148 independently authored intact sentences were indexed by their exposed
character prefixes and queried against compatible opposite-end phrase paths.
It indexed 1,262 prefix states and produced no complementary joins and no
exact palindrome above 38 letters. The run retains intact prose controls and
audits every emitted diagnostic with an independent mismatch scan and
forward/reverse SHA-256. It does not reverse or repair a finished tape, reuse
catalogue text, or mirror a token list. The result is a constructive inventory
diagnostic: the phrase bank has ordinary prose but insufficient reverse
character compatibility. Artifact:
`runs/phrase-trie-bidirectional-join-20260920.json`.

The typed whole-sentence scene lattice selected both clause-role inventories
simultaneously while carrying exposed character buffers. Its 17,634 live
states and 77,050 pruned states yielded no exact palindrome above 38 letters;
40 partial-lattice witnesses were retained for debugging, and the 38-letter
seed remained a regression control only. Every state was generated before
rendering, with independent pointer/SHA auditing and explicit rejection of
finished-tape reversal, word-order symmetry, repeated units, catalogue text,
and post-hoc repair. Artifact:
`runs/whole-sentence-scene-lattice-20260920.json`.

As a deliberately rejected diagnostic, the reverse-stream grammar parser
enumerated 12,000 complete authored SVO/PP/relative clauses and attempted to
parse each required opposite-end character stream under an independent
grammar. It found zero reverse parses and zero exact candidates above 38
letters. Because this diagnostic materializes a completed clause before
forming its reverse stream, it is not the admissible generation method and no
row is promoted to the reader package. The artifact is retained to prevent
repeating the same dead end; the next constructive method must carry both
grammar states and the shared character obligation simultaneously. Artifact:
`runs/reverse-grammar-parser-20260920.json`.

The simultaneous multi-constituent orbit lane selected independently authored
multiword subjects, predicates, objects, PPs, relatives, and coordinations on
both sides while consuming their shared character obligation. It visited 168
live states and produced zero exact closures above 38 letters. Six intact
contemporary-English controls were retained with independent pointer/SHA
audits; no row was promoted without the anti-shortcut checks and a future
reader gate. Artifact:
`runs/multiconstituent-orbit-20260920.json`.

The center-out CFG frontier lane grew one ordinary-English parse tree from a
selected center terminal, alternating nonterminal expansion on the two exposed
frontiers. It visited 259 live states, pruned 73, and rendered 22 contemporary
controls, with zero exact candidates above 38 letters. The lane is distinct
from clause-pair products and endpoint sweeps: grammar state and character
obligation advance in one parse tree, and no completed tape is reversed or
repaired. Artifact:
`runs/centerout-cfg-frontier-20260920.json`.

The live recursive grammar-frontier lane then carried the same obligation
through paired slot expansion for simple, locative, and coordinated sentence
shapes. It performed 144 hole-level obligation checks, pruning all 144 before
tree completion because the current lexical inventory had no compatible outer
character classes. It rendered no generated candidate and found no exact item
above 38 letters. This is a true online construction diagnostic: unlike the
recursive-tree enumeration above, it never audits a finished tree as a
candidate. The next lexical step is to index ordinary words by compatible
outer characters and expand the grammar with relative-clause constituents.
Artifact: `runs/live-recursive-grammar-frontier-20260920.json`.

## Reader evidence and API gate

### Live contextual prefix--suffix infilling (2026-09-20)

This lane keeps both surfaces in ordinary reading order: the left surface is
extended by appending words and the right surface by prepending words.  Each
state carries the two word buffers, the unmatched character residual and its
side, the number of characters paired so far, and the authored continuation
provenance.  Newly exposed characters are compared immediately, including
across word boundaries.  A deterministic 16-start, 32-state, 12-round beam
preserves residual-side diversity; it uses authored sentence continuations,
not a local model, reward scoring, finished-tape reversal, or repair.

The reproducible run is `runs/live-context-infilling-20260920.json` and the
implementation is `experiments/live_context_infilling_20260920.py`.  It
starts from grammatical noun-final suffixes (for example, *idea* and *night*)
and carries an explicit final lexical category while prepending ordinary words
toward a clause.  Both edges now carry finite grammar states (subject,
finite-verb/object, and clause-final), and repeated content-word cycles are
rejected before beam ranking.  Beam ranking now prioritizes short residuals,
balanced edge progress, paired characters, grammar completion, then length and
lexical diversity; a bounded per-round progress window prevents one-sided
growth from monopolizing the beam.  Initialization now consumes the complete
opening/final lexical words; the earlier run's implicit one-character
initialization is withdrawn because it discarded residual characters.  The
corrected nonterminal transitions require both
edges to add a word; empty-side transitions are permitted only after clause
completion.  The rerun retained no one-word right suffix as a nonterminal
state, and records 32 initialization/character conflicts and 32 one-sided-stall
rejections before the stricter gate exhausts the corrected beam.  No prior
deepest state from the invalid initialization is promoted.
it produced no exact candidate above 38 letters.  The result is useful as a
constructive discriminator: residuals can remain live across word boundaries,
but the current continuation inventory needs relative/appositive constructions
whose exposed characters can satisfy those residuals.  The next run therefore
holds out those constructions and indexes them by residual prefix rather than
increasing the beam or applying repair.  The reader gate remains closed.

`experiments/reader_package_v4_20260919.py` creates six deterministic blinded
pairs: each exact frontier item and each intact prose control is paired with a
word-shuffled control. A fixed seed randomizes A/B order, while the answer key
is held separately from the rater form. The package is ready, but human ratings
are pending. Therefore `/api/v4/generate` remains fail-closed; v4 exposes
evidence and diagnostics only.

The next reader-facing test is a randomized blinded intact-prose versus
shuffled-control rating with independent raters and explicit exclusions.
The reverse-lexicon typed-clause prototype was a distinct construction attempt, not a repair pass, but it is rejected as evidence. Although it constructed a reversed lexicon, the implementation did not query that index, and its residual-buffer invariant has not been independently established. Its 1,978,436-prune/0-closure result and SHA-256 artifact are retained for debugging only; they supply no readability or search-frontier claim.

The recursive CFG/orbit lane tested genuine relative-clause and coordination
productions in authored parse trees, then independently audited their character
orbits. This implementation is explicitly diagnostic rather than a live
frontier generator, so it is not claimed as simultaneous construction. A
bounded run visited 6,000 recursive-tree states and retained 24 intact
contemporary controls (maximum 38 letters), but found no exact candidate above
38 letters. Pointer, SHA, and proper-span checks are retained in
`runs/recursive-cfg-orbit-20260920.json`; this result does not certify
readability and is not promoted to the reader package.

## Outside-in typed two-clause character CSP

The outside-in typed two-clause CSP selects words from independent
contemporary-English clause derivations while maintaining a live character
obligation between exposed prefix and suffix. Its grammar includes simple
transitive clauses plus optional prepositional and relative-clause structure;
agreement and lightweight valency checks run on the derivation. No completed
tape is reversed, no seed is parsed backward, and no repair pass is applied.
The run visited 923 live states for each simple/quantity/PP plan and 208 for
each relative plan, pruning 918 or 207 incompatible states respectively. It
found zero exact candidates above 38 letters. Two authored controls and partial
prose witnesses are retained in
`runs/two-clause-character-csp-20260920.json`, with independent two-pointer
and SHA-256 audits. This is a construction result, not readability evidence;
the blinded reader gate remains closed.

The obligation-relative CSP is a separate lexical-index construction. It keeps subject-gap and object-gap relative constituents as typed roles, indexes each role inventory by exposed endpoint characters, and queries those inventories while both clause obligations are live. The bounded run indexed 59 endpoint keys, visited 9 live states, and pruned 2,460 incompatible pair expansions. It rendered three authored intact controls and found zero exact candidates above 38 letters. The endpoint index is queried during construction; this is not a completed-clause reverse lookup or a repair pass. Proper-span, repeated-content, catalogue, and finished-tape checks are recorded in `runs/obligation-relative-csp-20260920.json`; no row is promoted to the reader package.

## Live CFG intersection with an n-gram frontier prior

The live CFG/n-gram lane grows two independent contemporary-English slot
derivations one word at a time. After each pair of word choices, the exposed
letters are compared immediately; obligations may cross word boundaries, and
incompatible states are discarded before the next grammar slot. A small word
bigram prior orders the bounded frontier only. It is not an admission test,
does not certify readability, and performs no repair or finished-tape reverse
lookup. The grammar includes simple transitive clauses, PP adjuncts, and
relative expansions with authored role inventories.

The bounded run visited 9,244 live states and pruned 9,192 incompatible
obligations. It retained four intact contemporary-English controls, including
“Some dancers admire the lantern under the bridge.” (41 letters), and found no
completed exact candidate above 38 letters. Every control and any future
candidate is independently checked by a two-pointer mismatch audit and
forward/reverse SHA-256 digest; the n-gram score is diagnostic ordering only.
The result is recorded in `runs/char-cfg-ngram-intersection-20260920.json` and
is not reader evidence; the blinded human gate remains closed.

The inflectional/clitic boundary CSP changed the construction variables rather
than enlarging the clause bank. It carried explicit inflectional and clitic
choices while consuming the opposite-end character obligation online with a
correct residual operation. Three bounded plans visited 3 live states and
pruned 75 incompatible obligations; three authored intact controls were
rendered and no exact candidate above 38 letters emerged. Independent
two-pointer and forward/reverse SHA-256 audits, provenance, and anti-shortcut
flags are retained in `runs/inflectional-clitic-boundary-csp-20260920.json`.
This is diagnostic construction evidence only; the human reader gate remains
closed.

## Semantic valency-frame product

This lane pairs complete contemporary argument-frame derivations rather than
repairing a nearly-palindromic sentence. Transitive and intransitive frames
carry subject-number agreement, verb valency, object selection, and optional
PP expansion on both sides. The product checks exposed character obligations
during construction; no completed tape is reversed or resegmented. A proper
embedded palindrome-span check rejects structural shortcuts.

The bounded inventory tested 132,634 paired frame products, making 141,966 live
opposite-end character comparisons before pruning 132,634 incompatible
products. It yielded zero fresh exact closures above 38 letters. The exact
38-letter aide/memos/Diana sentence is retained only as an independent
calibration (`generated: false`), not as a result of this lane.
The artifact records frame provenance, pointer and forward/reverse SHA-256
audits, and the next constructive step: enlarge the vivid role lexicon while
indexing endpoint character classes before optional PP expansion. This is
construction evidence, not readability evidence; no output enters the reader
package.

## Recipient/theme and locative attachment product

The endpoint construction was extended with typed ditransitive recipient/theme
roles and locative attachment roles. These features are part of the endpoint
bucket key and state, so a candidate is selected from a grammatical attachment
class before live opposite-character matching; this is not an SVO/PP sweep or
a repair pass. The bounded run visited 58 buckets and 972 probes, rendered an
intact ditransitive control, and found zero exact closures above 38 letters.
Independent pointer/SHA audits and provenance are retained in
`runs/role-attachment-endpoint-product-20260926.json`. The next construction
adds semantic recipient/theme compatibility and locative scene roles inside
the same live state before any adjunct expansion.

## Frame-yield endpoint classes and live obligation buckets

As the next bounded construction, typed transitive argument frames were
indexed by their yielded tape endpoints plus agreement number and valency.
The opposite frame bucket was queried before paired expansion, while every
candidate obligation still required immediate character matching. This differs
from phrase-envelope endpoint sweeps because the key is a semantic frame yield
and its grammatical bundle, not a completed lexical phrase. The run visited
36 endpoint buckets and 540 bounded bucket probes, rendered one intact
contemporary control, and found no exact closure above 38 letters. The control
and independent pointer/SHA audit are in
`runs/frame-yield-endpoint-buckets-20260926.json`; no row is reader evidence.
The next constructive step is to add vivid ditransitive and locative role
frames to the same endpoint buckets before optional adjunct expansion.

## Direct narrative-beat lattice authoring

To test a construction that does not begin from clause products or a tape,
we wrote two independent banks of complete contemporary scene beats and
joined them with ordinary narrative seams.  The 5-by-5-by-3 lattice produced
75 intact rendered scenes (the longest was 117 letters).  Every row was
audited by a fresh two-pointer scan and forward/reverse SHA-256 comparison;
zero rows were exact above 38 letters.  The best rendered control was:

> At noon, the village teacher told a kind story to the restless class; then the river saw the lantern while a careful sailor watched by moonlight.

This is a deliberate near-miss, not a claimed palindrome or readability
result.  It is retained because its provenance is fully authored and its
failure is concrete: the next construction chooses the opening letters of a
second complete beat bank from live seam obligations, rather than repairing a
finished sentence.  Artifact: `experiments/direct_scene_lattice_authoring_20260920.py`;
run: `runs/direct-scene-lattice-authoring-20260920.json`.

## Non-nested compositional grammar

We next composed two independent three-beat contemporary scenes (subject,
verb-object, and adjunct) while consuming opposite-end character obligations
before rendering. This is a whole-sentence construction: it does not nest a
palindrome inside a larger sentence, reverse a completed tape, or repair a
finished string. The bounded lattice visited 14,400 component pairs and
pruned all 14,400 at their first live mismatch. It produced zero fresh exact
candidates above 38 letters. An intact diagnostic near-miss was:

> The calm poet reads a letter by the river; the calm poet reads a letter in the garden.

The row is not promoted because it is not exact and repeats its subject; it is
retained only as provenance for the next construction. The next operator is
to add a third independently authored beat bank and solve component terminal
classes jointly before lexical expansion. Artifact:
`experiments/compositional_non_nested_20260920.py`; run:
`runs/compositional-non-nested-20260920.json`.

## Edge-first lexical compatibility

The next construction moved lexical selection to the outside of the sentence.
Rather than choosing a grammar shape and filling it later, it compared 33
authored natural opening phrases against 33 authored natural sentence-final
phrases by their complete exposed character streams. Every compatible pair
retained both residual strings; no unmatched suffix was discarded. Only those
survivors entered a bounded cross-product of ordinary clause continuations.
This is distinct from an endpoint bucket: the opening and closing lexical
phrases are selected jointly before grammatical growth.

The run examined 1,089 edge pairings and found one compatible calibration pair,
the known seed split as `An aide rips nine memos` / `men inspire Diana`. The
online state retained the four-character residual `emos`, rather than claiming
that the edge was closed. All 144 continuation combinations then conflicted
with that residual; there were zero exact candidates above 38 letters. The
rendered seed remains a control, not a fresh result. Independent pointer and
forward/reverse SHA-256 audits are retained in
`runs/edge-lexical-compatibility-20260920.json`; the next construction expands
the edge inventory by semantic scene while preserving residual-indexed growth.

## Language-first heteropalindrome lattice

The next construction made readable language the primary search object: two
independently authored complete clauses were drawn from a small scene lattice,
and their exposed character buffers were compared before a rendering was
accepted. The lattice contains cross-word boundary joins, so a word is never
required to mirror a word; the only hard equation is the letter stream. A
character trie was not used as a post-hoc repair step, and no candidate was
formed by reversing an already finished clause.

The 46,656 bounded clause combinations produced a strongest intact prose
near-miss, retained here because it is concrete evidence about the next search
space rather than a claimed palindrome:

> **A careful gardener describes a distant harbor, and finds the lantern burning. The old lighthouse keeper as morning enters the garden, and holds the blue umbrella.**

This rendering has 133 letters. Its independent pointer audit first conflicts
at normalized character 1 (`c` versus `l`); forward and reverse SHA-256 hashes
therefore differ (`b029c6719559639936da0989a7e907d425286857bbb8846fdcae176e4dc2c92b`
versus `b885f6ba6131a7219b9441d43fbd92c8494b5b9bc114145f76b6eea843874564`).
The next construction is a larger *boundary-conditioned lexical lattice*: it
adds alternative natural clause realizations whose first two exposed letters
are indexed before the semantic continuation is selected. This is a new
generation space, not an edit of the displayed near-miss. The run artifact is
`runs/language-first-heteropalindrome-20260920.json`.

## Boundary-conditioned lexical lattice

The boundary-conditioned lane indexed two-, three-, and four-letter classes of
fresh opening noun phrases and closing clauses before selecting an independent
typed middle. This makes the first lexical seam part of construction while
keeping both sides in ordinary forward order; it does not edit the 133-letter
language-first near miss. The 10-by-10-by-10 lattice generated 1,000 complete
prose renderings. Its longest control was:

> **The young cartographer describes the clear route, and records a measured answer.**

The control has 67 letters and fails at normalized character 0 (`t` versus
`r`); its forward and reverse SHA-256 digests are
`9979d0a786ce8be5a3c5f9c3d98f2083390e2d8f116bdeaa1d725e5fdc2c46d2` and
`a9ef86f41e09bcc3e68d89af51b9fcb0986a832c864eda260f012f2b115cbe53`.
The best boundary-class trace shared two characters before the first conflict:

> **The young cartographer asks a simple question, and keeps the lantern lit.**

It has 60 letters and fails at character 1 (`h` versus `i`). Both sides were
fresh hand-authored clause/NP realizations, with independent attachment state;
no row used reversal, catalogue text, mirrored token units, or post-hoc repair.
No exact candidate above 38 letters appeared, so the reader gate remains
closed. The next construction expands the lexical classes with held-out
outer-letter alternatives rather than widening this same bank.

## Mixed speech-act clauses

To test whether ordinary English syntax was too narrowly restricted to
declaratives, the next lane paired independently authored declarative,
imperative, question, copular, and short dialogue clauses. Agreement,
attachment, and tense metadata were carried through the pair, and the complete
rendering was audited by an independent two-pointer scan and forward/reverse
SHA-256 comparison. The longest intact control was:

> **The patient archivist labels the weathered charts by lamplight. Those careful gardeners carried fresh water through winter.**

It has 105 letters. Its first mismatch is normalized character 1 (`t` versus
`r`), with SHA-256 digests
`ff7bb426b1575331fd2c4722a5f20d821e772d68cc0555b96593fd3a193ac380` and
`8bf5e0d6e264a5d16644fced4e31f7233dbb32345404fa0b26da64f61bb7a51a`.
The 56 complete pairs produced zero exact candidates above 38 letters. These
are broad contemporary-English readability controls, not a literal
Shakespeare-style constraint; no output enters the reader package without exact
closure and human ratings. The next construction adds subordinate and
reported-speech variants while retaining independent forward derivations.

## Typed lexical-graph walks

The lexical-graph lane generated each side as an independent typed POS/valency
walk through a frequency-ranked graph, then memoized the exposed character
obligation. Because a connector is added between the two complete clauses, the
graph intersection is explicitly only a seam diagnostic; admission uses the
full rendered-tape audit. Among 992 rendered candidates, the longest was:

> **The teacher keeps the quiet garden, and the teacher keeps the quiet lesson.**

It has 61 letters and fails at character 0 (`t` versus `n`), with forward and
reverse SHA-256 digests
`3872c31d77874a5dd6b116978cec0fbe3f295e08a30cdfd7f85ab5649e8e20ff` and
`7d9f4c835ba1886a9eedb774df7133652d549a629a4e716b47b314177ef65c52`.
No exact candidate above 38 letters appeared. This lane is retained because
its graph state is genuinely different from the phrase trie and boundary
lattice; its next construction expands the graph from a corpus bigram table and
filters for complete English walks before the same two-ended intersection.

## Corpus phrase-pair dynamic program

The corpus phrase-pair lane assembled both sides left-to-right from independent
contemporary-English phrase banks. A paired dynamic program ranked states by
the number of currently agreeing outer characters, but this score was never an
admission test. The 2,832 transitions retained 160 complete prose controls;
the strongest displayed row was:

> **A young cartographer records a difficult question, and a thoughtful gardener carries the morning train.**

It has 87 letters (the lattice maximum was 90), first mismatching at character
0 (`a` versus `n`). Its forward and reverse SHA-256 digests are
`6ff58d7c684e06ce3e7fc22dfc533b40ee5f64906ae4666bf8c00c38b72942de` and
`20c780e4be2581d6f3206eb8100f60e75e7055502403ed7e2f4d7df0e8474ceb`.
Both sides are independently forward-generated; neither is a reversal or a
repair of the other. The exact count above 38 is zero. The next reader-facing
test is to expand the phrase banks by exposed outer-letter class while keeping
the same complete-prose and independent-audit contract.

## Symmetric endpoint-class grammar

The endpoint-class lane made the top-level boundary equation explicit before
interior search: a two-letter prefix class of a forward-generated left clause
was matched to the reversed two-letter suffix class of an independently
generated right clause. Only compatible endpoint classes then entered the SVO
by-clause grammar product. This differs from the preceding boundary lattice,
which indexed internal opening/closing classes but did not gate the whole
sentence's first and last letters first.

The 30 retained complete-prose candidates reached 64 letters. The longest was:

> **The young cartographer names the distant harbor, and waits beneath the night.**

Its independent full-tape audit fails at normalized character 2 (`e` versus
`g`). The forward and reverse SHA-256 digests are
`7d4f33ff610411e6ce85b97f38e5611a22fdc22a538ed2f3d6df518d841993c9` and
`8ab6dd9f7b4e2f899fe2b9cb3c4080442f6ca64082be91824a2f94e999960cea`.
Endpoint conditioning therefore improved the construction boundary but did not
close the interior character equations: zero exact candidates above 38 letters
appeared. Both clauses were generated forward from fresh hand-authored banks,
with no reversal, repair, catalogue text, mirrored units, or fragments. The
next test increases endpoint width only after adding a second independent
interior grammar, so endpoint compatibility cannot be mistaken for a full
palindrome certificate.

## Manual bilateral scene authoring

As a direct language-first check, a separate lane authored two disjoint banks
of complete scene clauses and crossed every forward realization before
rendering. The left bank had 252 clauses and the right bank had 252 clauses;
63,504 complete candidates were audited. The longest retained prose was:

> **The morning gardener carries warm bread home when the first stars appear; the careful doctor opens the weathered gate and remembers the promise.**

It has 120 letters and fails at normalized character 0 (`t` versus `e`). Its
forward and reverse SHA-256 digests are
`b1a5c1459677b27dfb834e2388117227000c392c194515d648deb1f88e92be5f` and
`a5c96f0e2925d0d1521af3a1e4f383ae61ce6743c57c28ab22266d1b647dafb8`.
The clauses are fresh, complete, and independently forward-authored; no
catalogue borrowing, finished-tape reversal, mirrored units, repeated units,
or repair was used. The full exact count above 38 is zero. This lane confirms
that readability can be present in long candidates before exact closure, but
the next construction must condition the clauses' outer letters before
crossing the interior banks.

## Endpoint-conditioned clause authoring

The follow-up applied the top-level endpoint equation before any interior
clause pair was rendered. Six independently authored heads, bodies, and tails
per side produced 216 left and 216 right clauses; a one-letter prefix/suffix
index retained 2,592 compatible pairs for full rendering. The longest complete
prose control was:

> **A calm archivist reads the morning paper after the evening bell; our thoughtful friend holds the faded photograph near a quiet sea.**

It has 108 letters and fails at normalized character 1 (`c` versus `e`). Its
forward and reverse SHA-256 digests are
`a0edb719e701865c57c1961c1c6331efd68266de75e5cf448484a0091bdc41ee` and
`604590712648ff037abebfd830421001d274bf7c700c48e9d5b5acf2497eeefe`.
The endpoint equation was a pruning condition, not a palindrome certificate:
all 2,592 rows still received a fresh full-tape audit, and zero exact candidates
above 38 letters resulted. Both clause interiors were authored forward and
independently; no row used reversal, repair, catalogue borrowing, mirrored
units, or repeated units. The next construction increases endpoint width only
with a fresh compatible bank, preserving this full-audit contract.

## Synchronized slot audit (shortcut correction)

The next endpoint follow-up synchronized four grammatical roles (subject,
verb, object, and adjunct) after a productive one-letter endpoint filter. It
visited 8,192 complete forward clause pairs from fresh 256-by-320 banks and
rendered 8,192 controls. The longest raw row was 108 letters, but the audit
found repeated content words in 3,872 rows; those rows are diagnostic only and
cannot enter a reader packet. Among the eight disjoint-content controls, the
longest was:

> **A careful sailor carries warm bread near the window before dusk; the harbor keeper finds a lantern after the rain near a marina.**

It has 104 letters and fails at normalized character 1 (`c` versus `n`). Its
forward and reverse SHA-256 digests are
`8215f0b47bc250489a9dbd3d5d04a91d9c357e747c708d35f51e43bf8ba2add4` and
`d3ae7160fa971e40938538d833b877abaf5d6c2c8a5654fa43c16eeceb5ea317`.
The role synchronizer was a grammar constraint, not a character closure—the
record explicitly reports `interior_width=0`—and the full tape was independently
audited. No exact candidate above 38 letters appeared. The next construction
must add a real width-two character equation with a disjoint fresh bank; this
correction prevents slot alignment from being mistaken for palindrome progress.

## Width-two endpoint equation

The next pass implemented the recorded character construction directly. Fresh
three-by-three-by-three-by-three clause banks were first filtered by the
one-letter endpoint equation (4,374 survivors), then by a real two-character
prefix/suffix equation (1,458 survivors). The longest complete forward prose
control was:

> **A calm baker carries warm bread near the window in spring; our thoughtful friend lights the hall near the shore toward America.**

It has 104 letters. The full rendered tape first mismatches at normalized
character 2 (`a` versus `i`); the endpoint equation itself matches `ac` on both
sides. Independent SHA-256 digests are
`8ab52d84d473194feae96d47ca585fb4e89df80f57e68f6b383b24a63e65d363` and
`8266e53e85f9e32f76bf53b74107d129de07a9e57d03cb8ec4340486a44951f0`.
The 1,458 rows are complete prose controls, not palindrome claims; the exact
count above 38 is zero. Both clauses were generated forward from disjoint
banks, with no repair, reversal, catalogue borrowing, mirrored units, or
repeated units. The next construction increases the equation to width three
only with a fresh bank whose endpoint classes are not preselected to be
identical.

## Width-three endpoint equation

The width-three pass varied the third endpoint class instead of preselecting
every bank item to share the same prefix. The 729 width-one survivors remained
729 at width two, then fell to 243 at width three. The longest complete control
was:

> **An old teacher beside the river near the garden; the patient artist draws the distant road near Verona.**

It has 84 letters and fails at normalized character 3 (`l` versus `r`). Its
forward and reverse SHA-256 digests are
`32a3c2703af76e197d074dab31df751dbe8c47a870bc6c18d78533505327d5c5` and
`c86a1f4162576379e7764e741e5959c9381ad51bf83282a42e7519cc64df2f1f`.
The matched endpoint class is `ano`, but the interior tape immediately
diverges; all 243 rows are complete forward clauses and zero are exact above
38 letters. No row uses reversal, repair, catalogue text, mirrored units, or
repeated units. This run is the first endpoint-width experiment in which a
newly varied character actually prunes the bank; the next step is to carry the
same class state into an interior word-boundary equation rather than widening
the endpoint alone.

## Width-three endpoint plus interior boundary equation

The next lane carried the width-three endpoint state into two real interior
word-boundary equations. From 12,288 width-one endpoint survivors, the varied
third class retained 3,072 pairs; requiring both the first character of the
left verb phrase to match the reverse-facing last character of the right verb
phrase and the object/tail onset equation left 128 complete forward renders.
The longest disjoint-content control was:

> **An old teacher carries warm bread near the garden; the patient artist makes music near Verona.**

It has 77 letters and fails at normalized character 3 (`l` versus `r`). Its
forward and reverse SHA-256 digests are
`c493b13b1094ccd7e9a6dcfb27c324abc05c1045824abdc47e3b957d962ba05f` and
`1198b3fb13788705807194a14c01b8c7be7d0efd3d83fda192da7ee6ec82a2b2`.
The endpoint class `ano`, the interior verb-boundary class `c`, and the
object/tail onset class `n` all match their independently generated
counterparts, but the complete tape still diverges at the next character. All
128 rows have zero repeated content, no shortcut or repair provenance, and
independent full-tape audits; exact candidates above 38 letters remain zero.
The next construction carries both equations into a third interior slot with a
new disjoint bank.

## Forward event-frame readability baseline

An orthogonal language-first baseline generated complete event frames before
any character comparison. Each frame carried subject-number agreement,
transitivity, semantic scope, and adjunct attachment into surface realization;
the two sides were independently authored and neither side was copied or
reversed. Across six typed frames it produced 22 complete prose candidates.
The longest rendered control was:

> **The patient nurses carry a quiet message through town, while a young pilot checks the clear signal before rain.**

It has 91 normalized letters and fails immediately at character 0 (`t` versus
`n`). Its independent forward and reverse SHA-256 digests are
`577e42b5ee2b7b961a1c1ab649f1ff5a4ec441663933fd5dbed4ef3c1e256b41` and
`33b7fa1832f78d8958b55132b84dc89c3d088b02583d8ca8f2420756e001fd8b`.
The run has zero exact candidates above 38 letters and no reader packet is
claimed. The next reader-facing test remains a randomized blinded comparison
of any future exact intact-prose row against a word-shuffled control.

The four-slot carried-character follow-up was corrected before promotion:
its first implementation labeled a post-hoc mismatch check as “carried” and
rendered a 128-letter control, but did not prune during slot growth. The
corrected implementation propagates the obligation after every independently
generated slot and prunes all 6,144 states before rendering. It therefore has
no rendered candidate to promote; the earlier 128-letter row is withdrawn as
method evidence. This correction keeps the ledger honest and identifies the
next construction requirement: widen the slot banks while retaining a
non-empty prefix-compatible state after the first live obligation.

## Relation and connector state

The next forward-language probe added a semantic relation state rather than
another lexical sweep. Contrast (`although`), cause (`because`), and sequence
(`after`) were selected together with two independently generated,
agreement-valid event frames. The relation and the live outer-character
agreement score were fixed before punctuation or final surface rendering.
Twenty-four complete prose candidates were produced; the strongest was:

> **Patient nurses carry a quiet message, although the careful teacher marks the new route.**

It has 72 normalized letters, first mismatching at character 0 (`p` versus
`e`), with forward/reverse SHA-256 digests
`af9030c54b83d3c637c62e02ca5e98f4e648e63bab5a7370adcd8d74cf4e7073` and
`86403e191a29d15ca2b9b248309b789237dd8221906d29b0c1b7d7f5a6d3a8d0`.
The maximum pre-render outer agreement was five characters, but no exact
candidate above 38 letters appeared. The relation state is therefore a real
construction discriminator and a readable baseline, not a palindrome claim;
the next reader-facing test remains a blinded intact-versus-shuffled study
for any exact closure.

## Typed semordnilap phrase graph preflight

Because semordnilap edges can look promising while producing fragments, a
separate preflight treated `data/mirror_pairs.json` only as a search index.
Each edge was assigned a phrase role, inserted into a complete clause shell
only when both sides had compatible roles, and checked for ordinary `a`/`an`
agreement before rendering. The index contained 4,656 pairs, but no edge
survived the combined role and article gates. Thus the earlier malformed rows
(including “answers emits a” and “a eta”) are quarantined diagnostics, not
English candidates, and the lane contributes zero rendered or exact rows.
The next construction must add a fresh, typed phrase bank whose reversed
surface also has valid determiner phonotactics before any graph edge is used.

## Held-out inner subject/object character classes

The next constructive lane carried two small character-class obligations
inside the event frame rather than widening the already-tested outer endpoint:
the left and right subject boundaries had to share a vowel/consonant class,
and the corresponding object boundaries had to share the opposite class. The
classes were checked before surface rendering over independently authored
frames. Eight of nine frame pairs were rejected at this live inner gate; one
survived as complete ordinary English:

> **A quiet curator keeps the archive guide, while every careful guide maps a distant plaza.**

It has 72 normalized letters and fails at character 1 (`q` versus `z`). Its
independent forward and reverse SHA-256 digests are
`c63ab93d450d5c8171a52b0f55606e243c91ae2ce37775ceb981456b857b17b9` and
`89fcfe6d07ce8326dc13a78548a419c70b43ef9c01bb269c54cd23703111cb11`.
There were no exact candidates above 38 letters. The survivor is retained as a
readability control, not promoted as a palindrome or certified by the class
score; the next reader-facing test is a randomized blinded intact-versus-word-
shuffled comparison if a future exact row reaches the gate.

## Semantic relation-frame orbit

An independently authored relation-frame lane paired agent, relation, object,
and setting realizations before rendering. A live outer lexical orbit rejected
eight of nine pairings; the one complete ordinary-English control was:

> **The baker greets the traveler at noon; the traveler thanks the baker at sunset.**

It has 64 normalized letters and first mismatches at character 1 (`h` versus
`e`). Its independent forward and reverse SHA-256 digests are
`e1c26b01062309c31e23bc4016011ee077450a8937ca9098e73d03422f0c88e2` and
`dd4d0b2073bb96facee984b24617e92290570563663ee3d7368d0656de23e738`.
There were no exact candidates above 38 letters, so no reader packet is
claimed. The lane's concrete next test is to carry residual character debt
through relation and setting slots while preserving independent forward
realizations.

## Semantic relation realization lattice

A separate, broader lattice varied three authored relation families and their
active, passive, or locative realizations. It retained only compatible
attachment signatures before surface rendering: 24 of 36 relation-pair states
were rejected, leaving 12 complete prose controls. The longest control was:

> **The cartographer maps the shore, and at the harbor, the shore appears.**

It has 59 normalized letters and fails at character 0 (`t` versus `s`). The
full run recorded independent pointer and forward/reverse SHA audits, no exact
candidate above 38 letters, and no catalogue/reversal provenance. This is a
construction discriminator rather than a palindrome claim; if an exact row is
found, the specified next reader test is a randomized relation-family versus
word-shuffled naturalness study.

## Live residual relation slots

The smallest follow-up carried width-one and width-two character obligations
through relation, object, and setting slots before any sentence was rendered.
All nine independently authored frame pairings failed the first width-one
gate, leaving zero rendered candidates and zero exact closures. This is a
useful over-constraint result: the next construction must widen the ordinary
surface-realization banks before adding another equation, rather than treating
the empty frontier as a readable output.

Two subsequent probes are retained as diagnostics, not as constructive
advances. The cross-word boundary DP recorded 63 per-word transition states
and nine complete controls, the longest 77 letters:

> **The steady pilot charts a hidden island, while each quiet sailor studies the open map.**

Its first mismatch is at character 0 (`t` versus `p`); the run explicitly
records `global_residual_enforced: false`. The relation/setting debt trace
retained four complete controls to 90 letters, including:

> **The cartographer maps the shore by the window, and the shore is mapped by the cartographer through the window.**

Its first mismatch is at character 0 (`t` versus `w`). That lane's filter was
arbitrary, so neither diagnostic can enter a reader packet or be described as
an exact-generation method.

## Unequal center and explicit-buffer frontiers

To test the unequal-partition possibility directly, three fresh typed-
grammar lanes carried actual center buffers before rendering. The first
two-character center check examined 16 clause pairings and retained none. A
wider subject/verb variant propagated the unmatched buffer through 243 states
and pruned all of them. Finally, a two-sided word-trie DP stored unmatched
characters on both streams rather than a scalar similarity score: 18
transitions were attempted, 17 were rejected by an actual buffer mismatch, and
the final live frontier was empty. An orientation fixture separately confirms
that a forward right-side word is inserted reverse-facing before comparison.
These are precise zero-frontier results,
not failed reader candidates; the next construction widens only the
buffer-compatible subject/agent classes before adding adjuncts.

The first such expansion added variable-length word-trie transitions and
independently authored adjunct slots. It attempted eight slot positions, made
four transitions, and pruned all four on actual buffer mismatch. The live
frontier again reached zero before any prose was rendered; this is a concrete
construction boundary, not a reader result.

The same invariant was then applied to three complete event scenes carrying
semantic attachment state. All three scene transitions were rejected at the
live character mismatch, leaving no rendered prose and no reader packet. This
confirms that the buffer state is active while showing that the scene bank
needs compatible reverse-facing openings before adjunct growth can begin.

The reverse-trie typed grammar then widened the search to 12,001 nodes with a
deque residual and unequal word boundaries. Its only longest diagnostic row
was malformed and repeated content (“a artist” / “this captain”), so it is
quarantined rather than counted as a candidate; the corrected run has zero
rendered and zero reader-eligible exact rows. A separate prosodic-skeleton
probe used only aggregate word-length shapes from the quarantined catalogue to
rank 16 freshly authored controls (maximum 79 letters). Because that score is
computed after rendering and never enforces a character equation, it remains a
prose diagnostic, not generation evidence.

Two compact live-CFG checks followed. The optional-adjunct reverse CFG tried
two clause productions with three adjunct branches and pruned both transitions
before rendering. The live boundary-shift grammar tried nine semantic-slot
transitions and found no complete boundary closure. Both preserve the same
independent-forward/right-reverse-facing invariant and therefore contribute
zero reader candidates.

Endpoint-conditioned authoring was then tested in two forms. The scene lane
seeded two compatible endpoints but pruned both before interior expansion. The
broader semantic-frame lane retained nine endpoint seeds, applied article and
surface-grammar gates, and rejected every one at the next interior character;
zero interior closures remained. This separates the useful endpoint prior
from the still-unsolved interior construction.

## Three fresh constructive lanes

The next batch kept the target broad: ordinary, vivid English with a coherent
scene, not literal Shakespearean diction. Each lane selected words in normal
left-to-right order, enforced character obligations during construction, and
ran an independent outside-in audit plus forward/reverse SHA-256 check. None
was allowed to turn a finished string into its reverse or to repair a failed
candidate after rendering.

The bidirectional scene-authoring lattice fixed semantic roles and argument
structure first, then admitted lexical choices only when their exposed
characters matched the live equation. It produced three complete broad-English
controls but no live closure or exact candidate above 38 letters. The semantic
obligation automaton carried scene obligations across a simultaneous transition;
its one attempted transition was rejected at the first character mismatch, so
it rendered no candidate. These are zero-frontier construction results, not
readability failures.

The typed-clause meet-in-the-middle lane split complete short clauses into
typed halves and joined left/right halves only when seam characters, agreement,
and semantic types were compatible. It evaluated 800 halves on each side and
60,000 joins, retaining 50 complete prose controls but no exact closure. The
longest controls are deliberately kept out of the reader packet because they
are not palindromes. The concrete next construction is to carry a two-character
seam buffer and inflection agreement across the half boundary rather than
performing another duplicate sweep.

A follow-up consequence grammar increased the semantic scope rather than the
lexical sweep: each side was an independently generated agent/action/object/
setting clause followed by a consequence beat. Prefix indexing made only three
live probes, and no compatible join survived. Because the frontier was empty
before rendering, this lane contributes no reader candidate and no claim of
progress in length; its concrete next step is to widen the consequence-frame
bank while preserving the same live prefix equation.

Finally, an equation-first manual lane authored two fresh clause pairs and
required an eight-character outer equation before opening any interior slot.
Both pairs failed before inward expansion. The strongest readable diagnostic
was:

> **A careful teacher opens a bright window in winter; our kind neighbor carries warm bread to the station.**

It has 84 normalized letters, fails immediately at offset 0 (`a` versus `n`),
and carries independent pointer/SHA-256 evidence. It is retained as a prose
control only; no exact candidate or reader packet is claimed. The next
construction should author endpoint-compatible role choices before extending
the equation width, rather than trying to repair this rendered sentence.

The follow-up role-seeded scene lane made that next construction explicit:
semantic roles and endpoint classes were chosen before inward expansion, with a
global residual buffer carried through the scene. Its single fresh transition
was pruned immediately by an endpoint mismatch, leaving zero rendered prose.
This confirms that endpoint conditioning is active but the current role bank
does not yet contain a compatible start; the next construction must widen the
fresh role bank, not reopen the failed sentence.

Widening that bank to two independently authored choices per side still found
zero endpoint-compatible pairs in the indexed preflight. The inward residual
search therefore performed no transitions and rendered no prose. This is an
early rejection showing that the endpoint index is active, not a readability
result or a reason to promote a non-palindrome.

The local-minimum reset then tested two different constructions. A
reverse-conditioned semantic transducer authored three fresh left scenes and
parsed the required reverse character stream online with a 40-word
common-English lexicon. It found zero right-grammar parses, so no finished
string was rendered. A compositional center lane varied semantic frame and
terminal-class choices across nine live combinations; all nine were rejected
by the residual equation before rendering. These failures rule out the current
lexicon/frame banks, not the broad-English objective, and each records a next
expansion rather than a relaxed gate.

The asynchronous typed-clause buffer DP changed the scheduling geometry: it
expands whichever forward-authored clause has an empty residual buffer, so
word boundaries can cross while characters are consumed online. Its nine typed
template pairs were structurally valid. After adding a small authored bank of
natural endpoint-compatible nouns and places, the run evaluated 348,335
indexed joins and 2,107,452 memoized residual states. It produced no exact
candidate above 38 letters. The retained controls are grammatical but not
palindromic, and the next construction widens grammar roles only after this
non-empty buffer frontier rather than repeating the same endpoint sweep.

The first grammar expansion added plural agreement and one relative/locative
frame to the asynchronous scheduler. Two fresh frame pairs reached the live
transition check; both were pruned by residual mismatch before rendering. The
lane therefore contributes zero candidates, but it tests a real syntactic
dimension that the singular base grammar could not express.

The corrected weighted grammar automaton also received a beam ablation. With
the Brown-backed NOUN/VERB/PREP/ADJ banks, beam 180 explored 6,185 states and
beam 1,000 explored 28,596; the two arms pruned 48,734 and 221,510 residuals
respectively, with 0 exact closures in either arm. Complete controls such as
“The quiet poet sees a young keeper while the old sailor follows a bright
child.” are retained separately; partial grammar fragments are quarantined and
never treated as readable evidence.

A separate relative-clause beam kept the construction fully forward-authored:
two finite who/that clauses were crossed with two independently complete
matrix-clause tails. The four rendered controls are grammatical and retain
their punctuation as presentation only (61, 61, 62, and 62 normalized
letters); the run produced no quarantined fragment and no exact candidate
above 38 letters. For example, it rendered “The gardener who works in the
garden opens the gate; the school stays quiet.” and “A teacher that lives near
the school writes a letter; a garden holds flowers.” These are controls, not
palindromes, and therefore are not presented as reader-study candidates.

Two orthogonal zero-frontier checks followed. A forward phrase-equation lane
kept both sides independently authored and consumed a live residual character
stream; 10 of 12 transitions failed at the first incompatible character, so it
rendered no candidate. An agreement/valency-aware clause-final WFSA then
required complete subject–verb–object or intransitive frames on both sides;
its fresh Brown-derived domains produced zero exact closures at the 40-letter
gate. Both runs preserve their concrete next expansion rather than relaxing
the exactness or intact-prose gates.

The next seam-first construction enumerated four complete authored clauses on
each side and compared their exposed cross-word characters before any claim of
closure. It rendered 16 intact broad-English controls, including a 76-letter
cartographer/harbor scene, but no exact candidate. The known 38-letter anchor
was held out of the enumeration and never emitted. This is a new forward
construction frontier, not a repair of a failed rendered sentence.

The exact frontier is therefore unchanged: the best independently verified
reader-plausible output remains the 38-letter anchor above. No human reader
study is claimed for any non-exact control; a future exact intact-prose row
must still be tested against randomized word-shuffled controls in blinded order.

The clitic-boundary follow-up changed the construction state space rather than
repairing a failed sentence. Eighteen independently authored agreement frames
included legal contractions and clitic boundaries, and each of the 324 paired
transitions carried the full residual-vector comparison before completion. It
retained two complete ordinary-English controls (71 and 72 normalized letters),
with independent outside-in and forward/reverse SHA-256 audits, but no exact
candidate above 38 letters. The strongest controls were “The poet can't forget
a small vow in the hall; keepers don't lose an old song at first light.” and
“The poet can't forget the blue key in the hall; keepers don't lose a small vow
at first light.” They are prose controls, not palindrome claims. The next
construction composes two independent clitic frames through a typed comma or
relative boundary while retaining the residual vector; no human rating is
claimed until an exact candidate clears the 38-letter gate.

The typed-central CSP then changed the grammar topology again. It inserted an
independently authored observation sentence between complete agentive and
eventive clauses, and indexed the residual boundary state while constructing
the three-part sentence. The bounded bank yielded 18 typed assignments, two
indexed residual states, and 18 character transitions. It retained complete
controls at 99 and 118 letters, including “A careful baker carries warm bread
to the village; the evening bell sounds, while the village welcomes a careful
baker.”, but no exact candidate above 38 letters. Every row has independent
outside-in and forward/reverse SHA-256 audits and explicit anti-shortcut
provenance. The next construction replaces the fixed center with two
independently authored typed centers and requires seam-type agreement before
interior expansion; no reader evidence is claimed for these non-exact
controls.

The chart-composed phrase-path lane widened the searchable grammar without
copying corpus sentences. Independently authored subject, verb, object,
complement, and adjunct chunks were composed through optional grammar edges,
then joined with unequal word-boundary states while the opposing characters
were consumed live. The held-out relative-, passive-complement,
temporal-adjunct, instrumental-adjunct, and causal-adjunct expansions reached
256 chart paths (40, 40, 32, 32, and 32 new paths respectively) and 65,280
complete states,
retaining controls up to 120 normalized letters, but found zero exact
palindromes above 38 letters. Its audits are independent outside-in scans and
forward/reverse SHA-256 hashes; the reader gate stayed closed. The next
construction adds a held-out concessive adjunct edge rather than repeating the
same phrase bank.

The dependency-frame center-seam lane changed the semantic state space. Three
independently authored event frames on each side carried attachment, valency,
and agreement state across a live complement seam. The held-out ditransitive
expansion reached 24 states (three ditransitive, three benefactive, three
agreement-sensitive relative, three passive-relative, and three modal-passive
held-out frames) and
rendered complete ordinary-English controls from 94 to 119 letters; for
example, “The patient keeper guards a narrow bridge beside the orchard, while
the tools are used by the careful mason with steady hands.” None was exact.
Each row has the independent two-pointer and forward/reverse SHA-256 audits,
with no reversal, repair, mirrored units, or catalogue text. The next
construction is a causative frame with infinitival complement and
subject-control agreement.

The coordination/apposition lane changed only the sentence-boundary grammar,
not the lexical bank. Four independently authored complete clauses were joined
through comma, conjunction, or appositive seams, and each opposing character
was checked before the next state was admitted. It rendered 48 complete
controls, with no live closure and no exact candidate above 38 letters; the
longest causal fourth-clause control was 144 letters. Every row carries
independent two-pointer and forward/reverse SHA-256 audits plus anti-shortcut
provenance. The next construction adds a concessive appositive seam with a
sixth independent clause bank, and no reader evidence is claimed for these
non-exact controls.

The architecture review then exposed a more important correction. The first
forward-lexicalized prototype generated 64 tiny atomic S-to-NP-VP derivations
and independently found “Anna sees Anna.” and “Ava sees Ava.”, but its
implementation still materialized each complete derivation before checking
the palindrome equation. Those short exact diagnostics are retained only as a
differential soundness test, not as scalable search evidence. The corrective
construction is a fixed-length lexical-edge chart: shared character variables
are tied across the whole sentence, word boundaries remain free, and grammar,
agreement, and valency choices are pruned before a complete string exists. It
must first pass exhaustive tiny-grammar differential tests and rediscover the
38-letter anchor from atomic vocabulary without injecting the phrase. The
corrected implementation now passes that regression: with only atomic
entries, it visits 3,240 states, prunes 6,082 mismatches before completion, and
recovers two exact 38-letter witnesses, including the anchor, with the
independent SHA/pointer audits. The witness is rendered as “An aide rips nine
memos; some men inspire Diana.”; no shortcut witness is admitted. This validates
the global boundary semantics, but it is not yet a >38 result. The first remote
Brown inventory run (500 deterministic common-POS entries, lengths 39–64, and a
5,000-node cap) timed out at 5,000 nodes and 800,966 live character mismatches,
with zero complete closures and zero rendered candidates. This is a genuine
larger-search result, not a readability result: it says the current
grammar/lexicon envelope needs a new constructive expansion, not that readable
palindromes are impossible. The next method must change the construction
dynamics rather than add a post-hoc repair.

The follow-up bilateral grammar CSP changes the search dynamics directly. It
expands the first clause from its left edge and an independently parsed second
clause from its right edge; each right lexical edge contributes reversed
characters to a live residual before the next edge is chosen. The two sides
are not token mirrors and no completed tape is reversed. On the nine-word
atomic inventory it visited 159 states, pruned 690 residual mismatches, and
reached 22 complete grammar states. It recovered the same 38-letter anchor and
its clause-order variant, with independent SHA/pointer audits; it produced no
>38 candidate. A bounded remote run over 200 Brown-derived entries exhausted
the current two-`CLAUSE` grammar envelope at 2,413 states and 9,385,984 live
pair prunes, with zero complete closures and zero exact candidates. This is a
constructive baseline for adding typed adjunct and relative productions, not a
repair pass and not a readability claim.

Two bounded extensions were evaluated against that baseline. The observed
n-gram lattice used 100,000 intact 3--6-gram rows to constrain adjacent lexical
transitions (2,000 Brown POS entries); its bilateral grammar CSP visited 428
states and pruned 13,652,945 character and 320,313,552 transition attempts,
with zero complete closures. A separate curated agent/theme grammar visited
65,841 states and reached 56,430 complete states, but its longest exact output
was initially 30 letters (for example, “deer deliver leon; noel reviled reed.”), an
exact telegraphic control rather than reader prose. Neither lane is promoted to
the reader gate. These results narrow the next construction: keep lexical
transitions as hard search constraints, but add richer discourse/valency
realizations before seeking another exact closure.

We also tested two non-repair construction boundaries. A variable-length
word-path lattice over 30,000 observed bigram rows (100-word vocabulary,
eight-word cap) visited 376 states, rejected 47,771 character mismatches and
3,091 repeats, and reached no exact closure. A whole-sentence vocative grammar
(`VOC+CLAUSE` on the left and `CLAUSE+VOC` on the right) did produce 512 exact
closures above 38 letters in a remote run (400,087 states; 347,360 complete
states), but every closure contained a proper word-aligned palindromic span.
The longest rejected rows are retained verbatim in the run artifact; the
nested-span gate leaves zero reader candidates. This is a useful boundary
diagnostic, not a promoted output: the next construction must make discourse
attachment part of the grammar while preventing a seed clause from being
wrapped by names.

Finally, the variable-path search was made genuinely bidirectional: when one
side carried unmatched characters, only the opposite side advanced, and an
observed-bigram beam ordered the live states. At 300 vocabulary words and
100,000 intact bigram rows (5,000-state beam, 12-word cap), it visited 1,631
states, pruned 126,222 character mismatches and 4,984 repeats, and reached no
exact closure. Exact states would have been terminal, so this lane cannot hide
a seed under an outer wrapper. The next branch is a typed clause gate over
this residual scheduler, not a larger beam or a post-hoc repair.

The typed residual scheduler then restored hard subject/object/preposition
frontiers while retaining one-sided character-debt advancement. With 70
Brown-derived entries per coarse part of speech plus names and numerals, it
visited 6,620 states, pruned 295,080 character mismatches and 2,268 repeats,
and reached 344 complete grammar states; no exact candidate above 38 letters
survived. The known 38-letter anchor is recovered as an internal control when
the anchor vocabulary is supplied, but is not counted as progress. This closes
the residual-scheduler branch and points to a genuinely different next step:
add a typed relative complement to the grammar, rather than widening the same
SVO/PP beam.

The first topology change added a subject-gap relative complement (`DET N
REL`, with `REL` realized as a relative pronoun, verb, and object). The remote
run visited 9,341 states, pruned 413,528 character mismatches and 2,821
repeats, and reached 344 complete states, but produced no exact candidate
above 38 letters. The relative branch is therefore closed as a construction
family; the next method must change the sentence topology itself rather than
add another adjunct to this clause envelope.

We then inverted the construction order without importing text: 5,913 intact
left paths were generated from 100,000 observed bigram rows (fanout 20), and
each required reverse character stream was independently parsed against typed
clause patterns. No exact closure survived. This closes the observed-corpus
path family at the tested bound; increasing fanout would repeat the same
evidence rather than improve the reader-facing objective.

A fresh remote Qwen phrase bank supplied 76 ordinary-English proposals to the
exact outside-in unit beam (the model never supplied acceptance labels). Ten
seeds yielded nine exact rows from 39 to 125 letters and three rows passing the
mechanical repeated/nested-span gate. The rendered rows were still visibly
fragmentary (“Tons its operations a hotel basis able to. Has no it are post is
not.”); no row is reader-eligible, and no model score is treated as readability
evidence. This branch supplies a reusable proposal interface but not the
working generation method.

We then tested direct whole-sentence drafting as a separate proposal topology.
Twelve remote Qwen batches produced 158 extracted drafts and 21 exact rows,
but every exact row was a short, already-famous palindrome (“Rats live on no
evil star.”, maximum 20 letters); none passed the repeated/nested structural
gate. This is retained as a model-recall control, not as generated progress:
the model did not produce a new long sentence and no catalogue text is claimed.

Finally, a broad two-sided PCFG sampler drew 11,814 unique ordinary clause
tapes and independently segmented each reversed character stream with a
lexical dynamic program. It produced zero exact closures in 30,000 draws.
The lane has no finished-tape reversal or repair, but its random one-sided
sampling is not competitive with live bilateral construction; it is closed at
this bound and remains a negative control for the next constructive design.

The next construction changed the state representation rather than widening
that sampler. A memoized weighted-CFG hypergraph merged equivalent grammar
stacks, live character residuals, semantic-role state, and ordinary-order
frontiers before lexical expansion. Its bounded Brown-derived run visited
3,825 unique chart items, traversed 92 grammar hyperedges and 3,732 terminal
edges, and recorded 34,509 character-obligation prunes. It reached 3,535
complete parses but no exact closure above 38 letters. Because no rendered row
survived the exact gate, there is no reader candidate; the next constructive
direction is a semantic-frame hyperedge representation, not a larger lexical
sweep. Artifact:
`runs/cfg-hypergraph-obligation-20260920.json`.

The semantic-frame hyperedge follow-up made that direction explicit. Each
hyperedge carried an animate agent, concrete patient, action, and setting, so
selectional constraints were fixed before lexical emission. Twenty-five
independently paired event hyperedges entered the live two-frontier product;
all 25 were pruned at the first character boundary, with zero complete
closures. Intact controls ranged from 41 to 48 letters and were independently
audited, but none is an exact candidate. This closes the four-role frame at
its outer boundary and specifies the next topology: add a typed recipient
relation edge, not a lexical substitution or a larger sweep. Artifact:
`runs/semantic-frame-hyperedges-20260920.json`.


A distinct dialogue-act product tested six independently authored conversational acts on each side: questions, requests, reports, warnings, and promises. The product retained complete utterances rather than token mirrors and generated 36 transitions, with intact controls from 45 to 59 letters. Independent two-pointer and forward/reverse SHA audits found zero exact candidates above 38 letters. Because this bounded product checks complete acts after selection, it is recorded as a topology diagnostic rather than a live character constructor; the next construction is a three-turn acknowledgement/clarification grammar with jointly solved act lengths. Artifact: `runs/dialogue-act-product-20260920.json`.

The lexicalized-constituent interior CSP then changed the unit of generation.
Instead of joining completed phrases or emitting isolated POS words, it
expanded NP/VP/PP interiors one typed terminal at a time while carrying
agreement, semantic role, and pending word-boundary state on both ordinary
orders. In the bounded authored inventory it visited 41,280 live states,
matched 4,524 characters, and pruned 25,622 character conflicts. Its deepest
frontier reached 23 emitted letters, but no complete closure or exact candidate
above 38 letters survived. This is a distinct zero-frontier result: the next
construction must add a genuinely new lexicalized relation constituent rather
than enlarge the same terminal bank. Artifact:
`runs/lexicalized-constituent-interior-csp-20260920.json`.

The terminal-closure audit subsequently found an implementation restriction:
both solvers required an empty residual after the two complete derivations.
Let $U$ be the character stream already cancelled by the live matcher.  If
the remaining stream on one side is $C$, the rendered tape has the form
$U C U^{\mathrm{rev}}$ (or the symmetric case with $C$ on the other side),
so exact closure requires $C=C^{\mathrm{rev}}$, not $C=\epsilon$.
Correcting this condition admits odd lengths and unequal clause-length centers.
An exhaustive differential check of 900 synthetic string pairs recovered all
126 oracle palindromes, compared with 30 under the old rule, with no false
accepts. These synthetic strings are test fixtures, not candidate material.
An unchanged-bank remote replay produced 456 exact semantic-bank outputs,
including 200 additional outputs with nonempty centers, with a maximum of
31 letters and none above 38. For example, “deer deliver rats; a star reviled
reed.” is exact but does not constitute ordinary readable prose. The
constituent bank remained at 41,280 states with no closure. Every retained
exact output has an independent pointer and SHA audit in
`runs/palindromic-residual-closure-regression-20260920.json`; no reader-evidence
or availability claim changes.

The next representation change was a typed dependency-edge yield algebra.
Rather than pairing complete dependency trees, it first composed four typed
semantic edges (agent, action, object, and setting) through attachment
interfaces, then applied the bilateral character equation to the composed
yields. The authored bank produced 81 typed paths and 6,561 bilateral states;
all 6,561 were rejected at the first character comparison, leaving no
compatible frontier or exact candidate. The two intact controls (44 and 47
letters) were independently pointer/SHA audited and remain diagnostic only.
This closes that four-edge inventory at its outer interface; a future
construction must change the edge algebra or lexical language, not repeat the
same path product. Artifact:
`runs/typed-edge-yield-algebra-20260920.json`.

To change the search state rather than its vocabulary, we then applied a
residual-equivalence quotient to typed edge continuations. A canonical key
contained the remaining typed interfaces, semantic obligations, live
character debt, and content-word exclusion state; paths with the same future
continuation were merged while one full provenance trace was retained. The
existing semantic-role bank reduced 33 explored states to 9 canonical states
(24 merges), while a fresh authored two-frame bank reduced 12 to 9 (3 merges).
Both banks reached zero compatible frontiers and zero exact candidates above
38. Their 45- and 53-letter intact controls have independent pointer/SHA
audits. This is a state-space result, not a readability claim, and its
artifact is `runs/residual-equivalence-edge-quotient-20260920.json`.

The next closure change kept that quotient but gave the live residual a small,
typed center language: ordinary connective/complement continuations such as
“while the bells ring,” “and the quiet tide turns,” “because the old harbor
waits,” and “near dawn.” These center edges were selected as grammatical
continuations and intersected with the outstanding character debt before any
rendered row was accepted; they were not a finished-tape reversal, a
resegmentation pass, or a repair operator. The existing bank again visited 33
states (24 quotient merges and 8 character prunes), and the fresh authored
edge bank visited 12 (3 merges and 8 prunes). Neither bank admitted a
compatible center frontier or an exact candidate above 38. The intact controls
“The patient scribe marks the old letters by the harbor.” and “A careful
archivist copies a faded map near the quay.” are retained with independent
pointer/SHA audits. Since no exact line reached the reader gate, the concrete
next construction is a held-out two-edge typed-center grammar (complementizer
plus finite clause) whose attachment state and characters are solved online;
it will be novelty-preflighted before execution rather than enlarging this
finite center list. Artifact:
`runs/residual-language-center-automaton-20260920.json`.

That boundary was then tested directly rather than by enlarging the same
center list. We composed 20 ordinary typed centers from a complementizer and a
finite subject/verb clause, attached each to an outer edge path, and emitted
the center characters into the bilateral residual while the opposing outer
path was still unfinished. The existing semantic-role bank produced 100 left
derivations and 511 live states (499 quotient merges and 11 prunes); the fresh
authored bank produced 40 left derivations and 91 states (79 merges and 11
prunes). Neither bank reached a complete rendering or an exact candidate above
38. The same two intact prose controls remain independently pointer/SHA
audited. This closes the complementizer-plus-finite-center topology at this
bound; the next construction is a novelty-preflighted held-out center valency
frame with explicit subject/object attachment interfaces. Artifact:
`runs/composed-typed-center-grammar-20260920.json`.

Three parallel Luna probes then reopened the endpoint hypothesis space. The
endpoint-conditioned compositional decoder jointly paired fresh lexical
endpoints before expanding independent interiors, so the outer character
equation selected the endpoint schemas themselves. It rendered 72 intact
ordinary-English candidates, with a maximum of 79 letters; for example,
“A patient cartographer sketches distant hills; answers a folded letter beside
a small veranda.” The endpoint `a`/`a` equation passed, but the next character
failed (`p` versus `d`), and no fresh exact candidate above 38 survived. The
candidate carries fresh endpoint/interior provenance and independent
two-pointer/SHA audits; no reversal, repair, mirrored unit, or catalogue text
is involved. The next construction is an agreement-typed endpoint schema with
held-out interiors. Artifact:
`runs/endpoint-conditioned-compositional-decoder-20260920.json`.

The center-first scene grammar chose a semantic center before lexical outward
growth and then expanded paired scene arms under live obligations. It tested
three centers across nine outward transitions, retaining three complete prose
controls at 58--66 letters. The longest rendered control was “In autumn Nell
entered; the apples ripened; while the keeper counted the baskets.” Its first
outer comparison failed (`i` versus `s`) before any outward transition could
close, so there were zero exact candidates above 38. This is a fresh semantic
center topology, not an appended center or a finished-tape operation. The next
construction is a two-clause event bridge with an obligation-indexed semantic
phrase bank. Artifact:
`runs/center-first-scene-grammar-20260920.json`.

Finally, a character-level five-gram model was used only to order emissions
inside independently authored typed SVO clauses while a live cross-boundary
residual remained hard. It tested 16 clause pairs and pruned all 16 at their
first live transition; there were zero live closures and zero exact candidates
above 38. A representative intact control is “A careful sailor studies the
northern current; the northern current tests a careful sailor.” (76 letters).
The model neither validates nor repairs a tape, and the output remains
ordinary prose provenance rather than generated palindrome evidence. The next
construction is a held-out typed valency bank with the character model kept
strictly as an ordering signal. Artifact:
`runs/character-lm-grammar-constrained-20260920.json`.

The endpoint lane then carried number and attachment type in the endpoint
schema while holding out the interior clause choices. Thirty-two fresh
ordinary-English renderings reached 89--92 letters; for example, “Our careful
surveyors collect pressed leaves near the station; mark the river crossings
near distant gardens.” The agreement schema did not close the first outer
character (`o` versus `s`), and no exact candidate above 38 survived. The
held-out interiors changed the live character state, so this is not a rerun of
the earlier endpoint pair. Artifact:
`runs/agreement-typed-endpoint-decoder-20260920.json`.

The center-first follow-up selected a two-clause event bridge and indexed scene
phrases by the exposed obligation class before rendering. It tested three
bridges across six live transitions and retained six diagnostic renderings up
to 94 letters. These rows are not reader candidates: their outer comparison
failed immediately, and some bridge tails are intentionally logged as
fragmentary controls (for example, “At first light, through the garden the
bell rang; the keeper opened the chapel door then the traveler by dusk.”).
There were zero exact candidates above 38. The next construction adds a third
bridge clause and two-character obligation-prefix indexing; it will retain the
fragment filter. Artifact:
`runs/center-bridge-obligation-bank-20260920.json`.

The third-bridge follow-up changed the center topology again: it selected a
three-clause event bridge first, indexed scene-arm choices by a two-character
obligation prefix, and applied the hard complete-prose filter before admission.
Two fresh bridges produced four rendered diagnostics up to 113 letters; all
four failed at the first outer character and none was reader-eligible. For
example, “At first light, through reeds the bell rang; the keeper opened the
chapel door; the choir began the hymn then the ferryman smiled.” is preserved
with its pointer mismatch and forward/reverse SHA values, not promoted as a
palindrome or readability evidence. The next construction conditions a fourth
bridge clause on the residual two-character prefix and requires finite-verb
agreement before rendering. Artifact:
`runs/third-bridge-prefix-index-20260920.json`.

In parallel, the tense/aspect endpoint decoder carried tense and aspect in the
jointly selected endpoints while keeping finite-verb interiors held out from
the earlier endpoint banks. It rendered 32 diagnostics, the longest 99
letters, with zero exact candidates above 38. The longest row,
“The careful mechanic was checking the weather log beside the old runway; had
saved the final chart beside the old runway.”, is ordinary-word material but
not a reader candidate: its first outer comparison is `t` versus `y`, and the
semicolon/auxiliary construction is retained only as a construction trace.
The next operator is a polarity-conditioned auxiliary/negation interior with
a complete-clause admission check. Artifact:
`runs/tense-aspect-endpoint-decoder-20260920.json`.

Three additional Luna lanes changed the construction state in different ways.
The synchronous scene-CFG lane paired semantic-scene nonterminals and consumed
role terminals concurrently from opposite grammar edges. It recorded 64
states, 64 seam prunes, and 16 intact prose diagnostics up to 118 letters;
none closed exactly above 38. A representative rendering is “The patient
sailor studies the northern chart beside the quiet harbor; a quiet keeper
guards the narrow gate before the winter dawn.” Its first outer comparison is
`t` versus `n`. The next construction adds an optional relative-clause
nonterminal with shared attachment state. Artifact:
`runs/synchronous-scene-cfg-20260920.json`.

The joint scene-orbit lane selected a complete authored event schedule—left
scene arm, center relation, and right scene arm—before applying the live
character equation. It produced 24 complete-prose diagnostics up to 83
letters, all pruned by an outer mismatch and none exact above 38. For example,
“At dawn, the watchman opened the gate; as the eastern sky paled, and the boats
left the quay.” is retained with its `a`/`y` first mismatch and independent
hashes. The next operator is residual-prefix-keyed alternate scene-arm
expansion. Artifact: `runs/scene-orbit-joint-equation-20260920.json`.

The productive morphology lane selected number, tense, and verb paradigms
before the outside-in zipper, rather than editing an already formed sentence.
It enumerated 324 grammatical frames and 104,976 paradigm transitions; zero
pairs were live-compatible, so two intact controls at 84--85 letters are
diagnostics only. One is “The lantern keeper marks the old letter by the river;
the young poets read the quiet poem at first light.” Its first mismatch is
`e` versus `g`. The next construction carries residual suffix-class vectors
through a three-clause paradigm grammar with held-out auxiliaries. Artifact:
`runs/morphological-paradigm-zipper-csp-20260920.json`.

The residual-prefix scene-arm follow-up used a two-character obligation from
an authored opening as a lookup key for the next scene arm, then retained the
new residual rather than restarting the sentence. Three live lookups produced
six complete-prose diagnostics up to 114 letters; all failed at the first
outer character and none was exact above 38. The longest rendering was “After
rain, the gardener gathered the apples; while the dark branches dripped, after
the storm, the bright meadow opened beneath the clouds.” It is preserved as a
diagnostic with its `a`/`s` mismatch, not as a palindrome. The next operator is
a second residual-key transition with semantic-role and agreement preflight.
Artifact: `runs/residual-prefix-scene-arm-20260920.json`.

The synchronous relative-clause follow-up added an optional object-relative
nonterminal and shared attachment state to the lockstep transducer. It tested
36 synchronous states and 36 seam prunes, retaining nine intact diagnostics
up to 127 letters and zero exact candidates above 38. Every row carries an
independent pointer audit and forward/reverse SHA-256 values. The next
construction is a distinct temporal-adjunct transducer, not another relative
clause sweep. Artifact:
`runs/synchronous-clause-relative-transducer-20260920.json`.

The three-clause morphology follow-up carried a two-character residual
suffix-class vector across two independent outer clauses before admitting a
held-out auxiliary center. It enumerated 64 frames and 4,096 outer
transitions; no pair was outer-compatible. Two 102-letter controls remain
fully rendered and independently audited, including “The quiet keeper marks
the old letter by the river; while the bell has sounded; patient poets kept a
silver map near the garden.” The next operator conditions a fourth clause on
the full residual vector and finite-verb agreement. Artifact:
`runs/three-clause-residual-suffix-paradigm-20260920.json`.

The endpoint-indexed common grammar then used a fresh compact typed clause
bank and queried a right-edge index by the character required by the left
clause before any interior residual walk. It produced 50 endpoint-compatible
complete-prose diagnostics up to 114 letters, but zero exact candidates above
38. A non-repeated example is “The patient sailor studies the chart beside
market; the young scholar copies the letter under market.” The endpoint gate
therefore improves the first-character state but still leaves an interior
mismatch; the next construction is a two-character endpoint trie with
held-out nouns. Artifact:
`runs/endpoint-indexed-common-grammar-20260920.json`.

Astra's architecture audit identified that many earlier lanes enumerated small
finished sentence pairs rather than solving one global language. The next
implementation corrected that: a single forward grammar factored lexical
alternatives, carried agreement/transitivity/attachment features, allowed
variable word boundaries, and propagated exact position factors
`x[i] = x[N-1-i]` during each word emission. At target lengths 44, 52, and
60 it explored 5,485 bounded states and made 30,746 live factor prunes, with
zero complete palindrome parses and zero exact candidates above 38. The
ordinary controls (“The patient sailor studies the chart beside the harbor.”
and “A careful gardener carries a silver lantern through the orchard.”) were
independently pointer/SHA audited. This is an implementation correction and
scale test, not a readability claim; the next construction applies
position-domain lexical arc consistency over held-out alternatives before
choosing the next grammar factor. Artifact:
`runs/global-forward-sentence-csp-20260920.json`.

The position-domain arc-consistency implementation then made that correction
explicit and testable. A 32-case exhaustive differential test passed for both
odd and even centers and unequal word boundaries. On the 44/52/60 target
lengths it reproduced 5,485 states and 30,746 live prunes, with zero complete
parses and zero exact candidates above 38; the two intact controls retained
independent pointer/SHA audits. Because the bounded search is numerically the
same frontier, this result is a correctness/representation result rather than
a claimed generation gain. The next construction orders grammar factors by
minimum remaining mirrored-support domain and propagates support across
adjacent factors. Artifact:
`runs/position-domain-arc-consistency-csp-20260920.json`.

Minimum-domain grammar-factor branching was then tested as a controlled search
ordering change. It passed the same 32-case differential suite and recorded
the number of forward sentences represented by each target-length grammar
frontier. The 44/52/60 traversals still visited 5,485 states and made 30,746
live prunes, with zero complete parses and zero exact candidates above 38.
Thus ordering alone did not change the reachable language on this inventory;
the next operator is two-factor lookahead support propagation before either
factor is assigned. Artifact:
`runs/min-domain-factor-csp-20260920.json`.
