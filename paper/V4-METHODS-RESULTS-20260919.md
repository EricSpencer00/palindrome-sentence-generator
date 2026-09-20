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
singular+VBZ, plural+VBP, and past+VBD frames. It searched two complete
templates per feature (64/64, 58/53, and 64/64 subject/verb inventories),
visited 59 bounded states, and pruned all 59 before an exact closure. No
candidate reached the reader gate; the next construction adds frame-attested
adjunct and prepositional banks without relaxing the character obligation.

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

## Reader evidence and API gate

`experiments/reader_package_v4_20260919.py` creates five deterministic blinded
pairs: each exact frontier item and each intact prose control is paired with a
word-shuffled control. A fixed seed randomizes A/B order, while the answer key
is held separately from the rater form. The package is ready, but human ratings
are pending. Therefore `/api/v4/generate` remains fail-closed; v4 exposes
evidence and diagnostics only.

The next reader-facing test is a randomized blinded intact-prose versus
shuffled-control rating with independent raters and explicit exclusions.
