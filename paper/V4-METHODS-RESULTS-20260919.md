# v4 construction and evaluation ledger

This note is the current evidence-led core for the paper. The working claim
is constructive: choose grammatical lexical paths while satisfying character
seams during search. Exactness is independently checked; automatic language
scores and AI feedback only diagnose historical lanes. The active search is
exact-by-construction, with grammar boundaries and mirrored character orbits
chosen together. No output below is human certified yet.

## Anchor and frontier

The strongest reader-plausible exact output remains:

> **An aide rips nine memos; some men inspire Diana.**

It has 38 ASCII letters, normalized tape
`anaideripsninememossomemeninspirediana`, and forward/reverse SHA-256
`ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6`.
The independent outside-in pointer audit and the fail-closed mechanical gate
both pass. It has not yet been rated by blinded humans.

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
| Semantic slot/orbit product | Chooses valency, attachment, agreement, and center-out character equations jointly over authored Shakespearean scene frames | 12 agreement-valid scene pairs; intact controls to 75 letters; no exact closure | 0 / 0 |
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

## Reader evidence and API gate

`experiments/reader_package_v4_20260919.py` creates six deterministic blinded
pairs: each exact frontier item and each intact prose control is paired with a
word-shuffled control. A fixed seed randomizes A/B order, while the answer key
is held separately from the rater form. The package is ready, but human ratings
are pending. Therefore `/api/v4/generate` remains fail-closed; v4 exposes
evidence and diagnostics only.

The next reader-facing test is a randomized blinded intact-prose versus
shuffled-control rating with independent raters and explicit exclusions.
The reverse-lexicon typed-clause prototype was a distinct construction attempt, not a repair pass, but it is rejected as evidence. Although it constructed a reversed lexicon, the implementation did not query that index, and its residual-buffer invariant has not been independently established. Its 1,978,436-prune/0-closure result and SHA-256 artifact are retained for debugging only; they supply no readability or search-frontier claim.
