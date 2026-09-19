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

The longest mechanically admitted repair is exact but not readable:

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

Every lane records a literal two-pointer audit, forward/reverse SHA-256, source
provenance, novelty preflight, and a concrete next repair. None reverses a
finished sentence, imports catalogue text, or scores each search state with an
LM/RLAIF reward.

The constructive result is therefore specific, not a proxy claim: the z2
half-tape representation independently recovers the 38-letter readable anchor,
but the bounded 38--45-letter pilot does not yet improve its length. The next
repair exposes relative-clause internals as separate character-constrained
edges; its 194,216-node run also closes at zero, so it remains a repair record,
not a promoted example.

An orthogonal dictionary-DP check also returned zero above 38. Its failure is
not folded into a general sparsity claim: it chooses a complete seed before
reverse segmentation, so the concrete repair is a POS/inflection-aware
character trie that inserts boundaries during half-tape search.

The character-trie decoder now implements that repair and independently
recovers the anchor across individual word boundaries. Its 69-run pilot still
has no >38 closure, so the representation is a verified construction step,
not a claim that the length target has been met.

The next character-trie repair made the relative clause internal rather than
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

These repair and frontier rows now motivate a strategy reset rather than more
residual patching. The active construction policy is exact-by-construction:
grammar boundaries and mirrored character orbits must be selected together,
so an off-tape prose draft is never promoted into a repair queue. The
the paired-slot clause DFS, two-sided semantic orbit product, its semantic-slot
extension, and the semantic-role character FSM are the working generation
claim; earlier
repair runs remain auditable evidence and controls, but they are no longer the
paper's proposed route to a readable palindrome.

## Reader evidence and API gate

`experiments/reader_package_v4_20260919.py` creates five deterministic blinded
pairs: each exact frontier item and each intact prose control is paired with a
word-shuffled control. A fixed seed randomizes A/B order, while the answer key
is held separately from the rater form. The package is ready, but human ratings
are pending. Therefore `/api/v4/generate` remains fail-closed; v4 exposes
evidence and diagnostics only.

The next reader-facing test is a randomized blinded intact-prose versus
shuffled-control rating with independent raters and explicit exclusions.
