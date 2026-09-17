# Ten-Luna lane evidence

This is the reader-facing audit index for the ten requested construction
lanes. Every row below preserves an actual rendered string, its letter count,
independent exact checks, provenance, novelty status, and the next repair
operator. These are diagnostic outputs, not readability certification. A
probe that repeats a unit, fails the word-form gate, or is an exact control is
labelled as such and is not promoted.

| lane | rendered prose (intact diagnostic) | letters | independent exact result | provenance / novelty | concrete next repair |
|---:|---|---:|---|---|---|
| 1 | “A baker marks the button near a bright field. A caller answers the message behind a brief plaza.” | 77 | direct tape `False`; two-pointer `False` (35 mismatches); SHA-256 replay `False`; admission `False` | `runs/live-gpt2-character-decoder-preflight-20260916.json`; local GPT-2, copied sentences `False`, fixed tape `False`, reverse emission `False`; preflight passed with no overlaps | Replace the first-mismatch word with a held-out same-POS/sense item, re-query the live decoder, and replay both audits |
| 2 | “A baker carried a candle near the garden.” *(complete control only)* | 33 | immutable source tape is 46-letter exact, but the grammatical control is not exact and is below the promotion floor; two-pointer/hash/mechanical gates agree | `runs/typed-cfg-exact-tape-resegmentation-20260916.json`; fresh lexical-chunk constructor, no copied sentences; typed-CFG/valency preflight passed | Add one held-out number/tense variant at the highest-scoring dead CFG edge without changing the immutable letters |
| 3 | “The baker reads the letter in the quiet room. The caller answers the answer at the brief plaza.” | 76 | seam DP closed `False`; two-pointer `False`; hash `False`; admission `False` | `runs/dependency-tree-seam-solver-20260916.json`; normal word order, copied sentences/catalogue `False`; novelty preflight ran before search | Substitute a held-out POS/agreement/valency lexeme or switch only the mismatching adjunct frontier, then rerun seam DP |
| 4 | “A nerd carries a candle in our gardens; a nerd carries a candle in the arena.” *(diagnostic probe; repeated-unit gate fails)* | 60 | normalized tape `False`; independent two-pointer `False`; SHA-256 replay `False`; admission `False` | `runs/weighted-morphology-fst-lockstep-20260916.json`; authored inflectional banks, catalogue excluded; weighted-FST preflight passed | Add one held-out inflectional variant with the same lemma/role and rerun character-lockstep optimization |
| 5 | “A nerd carries a candle at our gardens; a nerd carries a candle beside the arena.” *(diagnostic probe; repeated-unit gate fails)* | 64 | normalized tape `False`; independent two-pointer `False`; SHA-256 replay `False`; admission `False` | `runs/earley-finite-state-grammar-intersection-20260916.json`; authored finite lexical banks, no copied sentences/catalogue; Earley-product preflight passed | Add one held-out lexical transition licensed by the same event-role and agreement registers at the deepest dead state |
| 6 | “After the recital, the musician places a dark case beside the steps, shows it to her teacher, and waits until the last listeners go home.” | 109 | 216/216 independent exact agreements and 216/216 admission agreements; exact count `0` | `runs/reader-first-discourse-scene-lattice-20260916.json`; two authored scenes, no corpus/catalogue import or wrapping; novelty preflight passed | Change exactly one sense-compatible clause realization at each first mismatch and recompute the complete scene equation |
| 7 | “After rain, the platform porter leaves a wet bundle beside the bench — requests the weary passenger's name, then waits as the evening train passes the bridge.” | 127 | 384/384 independent exact agreements and 384/384 admission agreements; exact count `0` | `runs/semantic-valency-boundary-csp-20260916.json`; human-authored station scene, no copied/catalogue text or word-order symmetry; preflight passed | Substitute a complete sense-compatible frame at the first mismatch and recompute all variable boundaries globally |
| 8 | “At sunrise, the harbor pilot checks the warning lights, marks the tide in the crew's notebook, and alerts the waiting passengers that the ferry will depart.” | 126 | 192/192 independent exact agreements; exact count `0`; admission count `0` | `runs/inflectional-clitic-boundary-csp-20260916.json`; authored harbor scene, no catalogue import/wrapping/symmetry; preflight passed | Change one held-out agreement, tense, clitic boundary, or attachment choice at the first mismatch |
| 9 | “At first light, the gardener unlocks the old shed, labels the seed trays, writes a note for the neighbor, mends the loose gate.” | 100 | independent slice/two-pointer/hash and mechanical checks agree; exact count `0` across 36 probes | `runs/scalable-compositional-clause-grammar-20260916.json`; fresh ordinary increments, proper spans and repeated units excluded; preflight passed | Append one held-out ordinary increment at the first obligation mismatch and recompute the complete tape |
| 10 | “An aide rips nine memos; Two aides inspire Diana.” *(39-letter near miss; not an exact candidate)* | 39 | source-tape preservation `False`; independent exact agreement `4134/4134` for failures; admission `0` | `runs/exact-tape-semantic-slot-repair-20260916.json`; source tape independently checked, no wrapper/catalogue/symmetry; preflight passed | Try one held-out typed subject/verb/object/adjunct substitution at the first mismatch, accepting only unchanged source tape |

Before the follow-up repairs, the ten-lane aggregate contained 3484 rendered
rows across 81 route phases, 73 exact-but-rejected tapes, and zero mechanically
admitted rows. No row was reader-eligible. The required next
reader-facing test remains an intact-prose versus shuffled-control study with
randomized blinded order, but it cannot be run honestly until a candidate
passes the exact and mechanical gates.

## Follow-up construction repairs

The assumption-core solver (`runs/assumption-core-scene-solver-20260916.json`)
searched 240 complete multi-clause scenes. A representative 69-letter probe
was:

> the careful courier delivered the letter and the patient watchman waited the gate

Its grammar audit passed, but exact tape equality and the independent audit both
failed; the minimal conflict core was `all_different`, and a held-out one-slot
repair was recorded. The run spans 61--74 letters and has no exact closures.

The repaired masked-character search explored three genuine semantic
alternatives per scene (`runs/masked-character-scene-gibbs-20260916.json`). Its
best intact probe was:

> After rain, the station porter carries a wet parcel to the bench.

It is 52 letters, with forward/reverse SHA-256 mismatch and independent
two-pointer mismatch; the held-out alternatives, complete-assignment scores,
and reject-and-remask repair are preserved. No exact or reader-eligible output
was promoted. The phrase-lattice automata run is retained separately as an
explicit preflight exclusion because it duplicates existing phrase-FST state.

After these repairs the aggregate is 3727 rendered rows across 83 route phases,
73 exact-but-rejected tapes, and 0 mechanically admitted rows.

## Additional constructive wave

The min-cost-flow lane retained this complete 76-letter pair:

> the pilot charts harbor. the keeper opens gate. the farmer tends garden. the writer marks page.

Its residual flow matched only 27 character obligations; the independent tape
and hash audits both reported non-exact, and its held-out 77-letter lexical
repair also failed. Provenance records independent typed clause supplies with
no catalogue or repeated units.

The encoder lane retained three complete probes, including:

> After rain, the station porter carries a wet parcel to the bench.

This is 52 letters and fails direct tape, two-pointer, and reverse-hash checks.
The bidirectional checkpoint was unavailable as PyTorch weights in the local
cache, so the run explicitly used a fail-closed inspection fallback; no model
evidence is claimed.

The recursive stack route produced an exact 62-letter semordnilap chain, but
the shared mechanical gate rejects it for word-order symmetry and a proper
multiword palindrome span. It is preserved as rejected failure evidence, not
as prose or a candidate.

The current aggregate is 4282 audit-compatible rendered rows across 100 route
phases, with 73 exact-but-rejected tapes and 0 mechanically admitted rows.

## Orthogonal continuation wave

The next three Luna lanes were executed as distinct state representations:

| route | rendered prose | length | exact / admission | provenance and next repair |
|---|---|---:|---|---|
| compound/derivational scene CSP | “the gardener carries the lighthouse carefully; the raincoat carries carefully our gardener” | 78 | direct, two-pointer, hash, and mechanical exact `False`; admission `False` | hand-authored modifier–head compounds and scene frames; replace only the right compound at the first mismatch while preserving role and derivational path |
| paraphrase graph debt paths | “At dawn, the harbor pilot checks the warning lamps and marks the tide. At dusk, the station keeper locks the gate and tells the waiting travelers.” | 117 | direct, two-pointer, SHA-256, and mechanical exact `False`; admission `False` | complete authored scene nodes joined by labeled meaning-preserving edges; repair the held-out `r2→r3` paraphrase edge and replay the debt |
| parse-tree exact cover | “the young bakers repair the letters in the gardens. the quiet sailors observe the ledgers by the harbors.” | 86 | direct, two-pointer, hash, and mechanical exact `False`; admission `False` | authored constituency terminal spans with side/role/agreement columns; change one held-out tree lexical/agreement index at the first dead column |

The compound run contains 80 complete probes, the graph run 81 probes plus one
repair, and the parse-tree run 24 base probes plus 8 held-out repairs. The
common audit report retains 48, 82, and 6 normalized rows respectively; this
normalization difference is recorded rather than hidden. None is reader
eligible, so no intact-versus-shuffled human study is claimed yet.

## Rater-package contract (all ten lanes)

Every lane row above is eligible to become the source item for the same
reproducible rater package once it passes the mechanical gate. The builder
`experiments/build_blinded_reader_package_20260916.py` freezes the intact
rendering and provenance, creates a length-matched shuffled control, rejects
unchanged or palindromic controls, randomizes blinded order, and keeps the
item key separate. Thus programmatic readability is diagnostic only; human
ratings of Englishness, coherence, and grammaticality remain the criterion.
At this checkpoint the package is intentionally not emitted because all ten
lanes have zero mechanically admitted candidates.

The subsequent repair wave adds three actual prose diagnostics:

| route | rendered prose | letters | exact / admission | concrete repair |
|---|---|---:|---|---|
| bilateral semantic CFG | “the quiet clerk records the parcel. the courier carries the letter. the young baker mixes the dough. the patient guard checks the seal. the keeper opens the gate. the old sailor mends the sail” | 154 | direct, two-pointer, SHA-256, mechanical `False` / `False` | held-out semantic substitution `gate → lock` |
| human scene equation frames | “The patient courier delivered the sealed letter to the quiet office. The waiting clerk opened the letter beside the window. The grateful clerk thanked the courier before noon.” | 145 | two independent exact audits `False` / `False`; admission `False` | author a response place phrase against the live outer-character debt |
| typed edit-program repair | “The careful nurse carries a warm blanket while birds settle; carefully, the clerk checks the ledger.” | 82 | two-pointer and SHA-256 agree `False`; admission `False` | held-out typed edit on certified agreement/attachment/sense dimension |

The next repair wave adds three more complete-prose diagnostics:

| route | rendered prose | letters | exact / admission | concrete repair |
|---|---|---:|---|---|
| event-graph character SAT | “At by the fire, nora writes a brief note. At near shore, eli mends the torn sail.” | 61 | two-pointer and hash `False`; admission `False` | replace one held-out event slot and re-propagate position equations |
| syntax-stack semantic-role decoder | “the station porter; carries a wet parcel; beside the quiet bench; then the night guard locks the gate.” | 81 | two-pointer and hash `False`; admission `False` | replace the complete semantic role frame containing the first debt mismatch |
| phrase-equation inventory solver | “the careful porter carries the sealed parcel beside the quiet gate for the waiting child.” | 74 | two-pointer and SHA-256 `False`; admission `False` | author a held-out place/purpose phrase against global character debt |

The latest repair wave adds:

| route | rendered prose | letters | exact / admission | concrete repair |
|---|---|---:|---|---|
| semantic-center SAT | “The careful nurse carried a sealed letter. At dusk the nurse closed the market ledger.” | 70 | pointer and SHA-256 `False`; admission `False` | expand held-out verb/object paradigms while preserving the fixed center event |
| reverse-segmentation CFG/valency | “The patient courier delivered the sealed letter before noon.” | 51 | direct and two-pointer `False`; admission `False` | held-out tail-lexeme repair under the joint boundary/valency chart |
| dependency mirror-pair constructor | “The patient courier delivers the sealed letter for the waiting child before dusk.” | 68 | pointer and SHA-256 `False`; admission `False` | held-out subject/adjunct repair while preserving dependency order |

The held-out continuation repairs are also in the shared audit:

| repair route | rendered prose | letters | exact / admission | provenance and concrete next repair |
|---|---|---:|---|---|
| phrase-equation inventory repair | “the careful porter carries the sealed parcel beside the lantern-lit school for the waiting child.” | 81 | two-pointer `False`; SHA-256 `False`; admission `False` | 48 whole-phrase held-out trials from 12 authored scenes; author a second complete phrase against the residual equation debt |
| semantic-center SAT repair | “The careful nurse opened a sealed letter. At dusk the nurse closed the market ledger.” | 69 | independent pointer `False`; SHA-256 `False`; admission `False` | frozen authored center with verb/object coordinate descent; replace the least-satisfied verb–object seam with a held-out valency-compatible pair |
| dependency mirror repair | “The careful scholar delivers the sealed letter for the waiting child before dusk.” | 68 | independent pointer `False`; SHA-256 `False`; admission `False` | exactly 18 held-out subject/adjunct substitutions per pair; try a held-out agreement-preserving verb–object substitution |

These are real complete-prose repairs, not duplicate sweeps. They add no exact
closure, so the next reader-facing test remains the preregistered
intact-prose versus shuffled-control package with randomized blinded order,
to be run only after an exact mechanically admitted candidate is frozen.

The next orthogonal continuation wave adds three state representations:

| route | rendered prose | letters | exact / admission | provenance and concrete next repair |
|---|---|---:|---|---|
| online grammar-state character decoder | “The careful porter carries the sealed parcel beside the quiet gate.” | 56 | two-pointer `False`; SHA-256 `False`; admission `False` | two ordinary-order grammar cursors with live obligations; replace the first mismatching semantic slot and replay the ledger |
| live slot-equation CFG resegmentation | “The patient courier delivers sealed note at sunrise. The quiet teacher opens marked parcel beside the gate.” | 89 | direct and SHA-256 `False`; admission `False` | two bounded object/adjunct repairs re-solve obligations before charting; author a new finite verb/adjunct pair against the remaining debt |
| clause-growth semantic frame repair | “the patient courier delivered the sealed letter before dusk. the quiet clerk opened the wooden parcel by the window. the careful gardener watered the young cedar after rain. the weary sailor mended the canvas sail near harbor. the old farmer gathered the ripe apples before sunset.” | 231 | two-pointer and SHA-256 `False`; admission `False` | incremental ordinary-order scene growth with a fifth typed frame; author a sixth semantic frame against the updated frontier |
| online grammar-state slot repair | “The careful porter carries the sealed package beside the quiet gate.” | 57 | two-pointer `False`; SHA-256 `False`; admission `False` | three held-out semantic-slot substitutions replayed the live ledger; author a new outer-edge-compatible frame |
| online grammar-state outer-frame repair | “The patient keeper carries a sealed letter beside the quiet harbor.” | 56 | two-pointer `False`; SHA-256 `False`; admission `False` | two outer-frame probes authored before interior slots; solve a paired outer-edge equation before filling interior slots |

The aggregate after this wave is 4301 normalized rows across 105 route phases,
with 73 exact-but-rejected tapes and 0 mechanically admitted rows. These
outputs remain diagnostic; none is reader-eligible.

The subsequent closure attempts changed the construction state rather than
 widening an existing sweep:

| route | rendered prose | letters | exact / admission | provenance and concrete next repair |
|---|---|---:|---|---|
| reverse-lexicon synthesis | “The careful porter carries the sealed parcel beside the lantern-lit school for the waiting child.” | 81 | two-pointer `False`; SHA-256 `False`; admission `False` | left-to-right semantic clause plus reverse-lexicon chart; replace the fixed lexical chart with held-out inflectional variants |
| center-out grammar boundary DP | “The careful porter delivers the sealed parcel. Beside the lamplit school for a waiting child.” | 77 | two-pointer `False`; SHA-256 `False`; admission `False` | center-out whole-word emissions with grammar/boundary state; replace the first mismatching boundary with a held-out agreement-compatible adjunct |
| authored clause-template SAT | “The observant nurse carries a sealed parcel near the station. The careful nurse carries a warm parcel beside the lamp.” | 97 | two-pointer `False`; SHA-256 `False`; admission `False` | authored valency/template variables solved under a full character equation; choose a fresh held-out valency frame at the first debt conflict |

The common audit now contains 4328 normalized rows across 108 route phases,
with 73 exact-but-rejected tapes and 0 mechanically admitted rows. These
lanes produced readable diagnostics but no exact closure, so the blinded
intact-versus-shuffled reader package remains gated.

Their bounded repairs are retained separately:

| repair route | rendered prose | letters | exact / admission | concrete next repair |
|---|---|---:|---|---|
| reverse-lexicon inflection repair | “The patient courier delivered the sealed letter for the waiting child before dusk.” | 69 | two-pointer `False`; SHA-256 `False`; admission `False` | jointly solve determiner/adjective boundary seams rather than varying only the verb |
| center-out grammar boundary repair | “The careful porter delivers the sealed parcel. Near the lamplit school for a waiting child.” | 75 | two-pointer `False`; SHA-256 `False`; admission `False` | author one semantically compatible adjunct targeting the boundary mismatch |
| authored clause-template SAT repair | “the observant usher guides a young visitor toward the quiet gallery before noon. The quiet guide checks the gallery before noon.” | 106 | two-pointer `False`; SHA-256 `False`; admission `False` | add one fresh usher–guide–visitor frame whose adjunct targets the unmatched outer character |

The aggregate after these repairs is 4336 normalized rows across 111 route
phases, with 73 exact-but-rejected tapes and 0 mechanically admitted rows.

The next directed repair wave retained four reverse-lexicon seam candidates at
65--68 letters, a second center-out adjunct repair at 75 letters, and a second
SAT outer-character repair at 102 letters. The independent pointer/SHA checks
and mechanical gate reject all of them. The aggregate is now 4342 normalized
rows across 114 route phases, with 73 exact-but-rejected tapes and 0 admitted.

The follow-up repairs retained three shared-agreement seam candidates at
61--68 letters, a two-boundary center-out repair at 66 letters, and a fresh
t-initial SAT subject repair at 103 letters. Each has independent pointer and
SHA-256 rejection, provenance, and a concrete next operator. The aggregate is
now 4347 normalized rows across 117 route phases, with 0 mechanically admitted
rows.

The next frontier repairs retained a centered-complement reverse-lexicon scene
at 87 letters, a center-out time-adjunct repair at 70 letters, and a guide-
clause SAT complement at 103 letters. All fail independent exact/hash and
mechanical gates. The aggregate is now 4350 normalized rows across 120 route
phases, with 0 mechanically admitted rows.

The next targeted repairs retained a typed-complement reverse-lexicon scene at
93 letters, a residual center-out adjunct at 75 letters, and a same-valency SAT
guide-verb repair at 102 letters. Each fails independent exact/hash and
mechanical gates. The aggregate is now 4353 normalized rows across 123 route
phases, with 0 mechanically admitted rows.

The next construction step retained a semantic-role noun-boundary scene at 93
letters, a paired subject/adjunct boundary-CSP scene at 72 letters, and a
fresh SAT subject/object pair at 101 letters. All fail independent exact/hash
and mechanical gates. The aggregate is now 4356 normalized rows across 126
route phases, with 0 mechanically admitted rows.

The following bounded repairs retained an adjacent-PP reverse-lexicon scene at
106 letters, a fresh verb-frame center-out scene at 57 letters, and a
recipient-role SAT scene at 99 letters. All fail independent exact/hash and
mechanical gates. The aggregate is now 4359 normalized rows across 129 route
phases, with 0 mechanically admitted rows.

## Current authoritative snapshot (2026-09-16)

The ten requested Luna lanes are now represented by append-only artifacts with
actual rendered prose, independent tape/pointer/hash checks, provenance,
novelty preflight, and a concrete repair operator. The common audit currently
contains **4,390 rows across 133 route phases**, including **76 exact tapes**;
all 76 are rejected by at least one hard quality gate, so **0 are mechanically
admitted and no reader study is claimed**.

| lanes | actual rendered evidence | letters | independent result | next repair |
|---|---|---:|---|---|
| 1--2 character LM + exact-tape resegmentation | “the careful teacher helps a pupil” (plus four other fresh proposals) | 28 | non-exact; the 42-letter exact control is the catalogue “a man a plan a canal panama” and is rejected for self/repeated-unit provenance | held-out ordinary-prose LM and new lexical boundaries |
| 3--5 dependency seam CSP + agreement morphology + CFG/Earley intersection | “Ava saw radar level civic; civic level radar was Ava” | 42 | pointer/hash exact, but rejected for self-palindromic words, repeated content, and word-order symmetry | seam-directed feature swap with fresh non-palindromic roles |
| 6 scene lattice | “At dawn, the patient guide carries a red map to the waiting child.” | 52 | pointer/hash non-exact | replace one sense-compatible object slot |
| 7 valency/attachment | “By noon, the careful nurse labels the sealed vial for the quiet ward.” | 55 | pointer/hash non-exact | replace one complete valency-compatible object/attachment |
| 8 inflection/clitic boundaries | “After rain, the young keeper opens the old gate beside the garden.” | 53 | pointer/hash non-exact | held-out inflectional/clitic boundary substitution |
| 9 flat compositional grammar | the same three-clause scene lattice, assembled without nested palindrome spans | 52--55 | pointer/hash non-exact | add one new flat clause frame against the residual |
| 10 semantic slot repair | “At dawn, the patient guide carries a blue chart to the waiting child.” | 55 | pointer/hash non-exact | use the first mismatch as a global character obligation |

The 20-probe multi-clause character-LM continuation adds intact 66--72-letter
scenes such as “the careful teacher guides a young visitor; a patient sailor
carries the red lantern,” all non-exact under both audits. These rows are
reader-facing controls only; they do not certify readability. Every future
repair must change the construction state or a typed semantic slot, never just
rerun a larger duplicate sweep.

## Targeted seam repair update (2026-09-16)

The next seam repair is recorded in
`runs/seam-feature-slot-repair-20260916.json`. Novelty preflight passed and 11
single-slot substitutions were rendered against the exact 42-letter seam
witness. Every attempt has independent pointer/hash checks and provenance; no
substitution remained exact or mechanically admissible. The next operator is
typed boundary resegmentation at the first failing seam, allowing adjacent
short words to absorb reflected suffixes. With this artifact included, the
authoritative aggregate is **4,539 rows, 142 route phases, 78 exact tapes, and
0 mechanically admitted outputs**.

## Typed-boundary follow-up (2026-09-16)

The next repair is now a distinct artifact,
`runs/typed-boundary-resegment-shortwords-20260916.json`. It passed novelty
preflight and rendered 23 fresh adjacent-short-word boundary variants at the
first failing seam. All have independent pointer and SHA-256 checks; none is
exact or mechanically admissible. Its next operator is a fresh
non-palindromic subject/verb/object frame with typed agreement, rather than
another substitution sweep.

## Additional joint construction lanes (2026-09-16)

Three distinct lanes now have persisted evidence. The central-pivot clause CSP
retains the 113-letter intact pair “The archivist records the carefully folded
winter map at dawn. A patient sailor observes a weathered cedar rescue boat
beside the river.”; pointer and hash audits agree it is non-exact at the first
character. The semantic-slot lattice renders 108 jointly selected states and
prunes all 108 against the global equation. The bidirectional scene decoder
reproduces the 51-letter “Doc, note: I dissent. A fast never prevents a
fatness. I diet on cod.” exactly, but novelty/provenance identifies it as a
known catalogue palindrome and excludes it. Their artifacts record the next
repairs and remain out of any reader claim.

## Fresh typed-frame follow-up (2026-09-16)

`runs/fresh-typed-frame-live-seam-20260916.json` renders “The baker carries a
letter near the quiet harbor” (40 letters), a new singular-agreement frame
with eight live obligations. It passes novelty and every mechanical quality
check except exactness. The next repair targets only the adjunct boundary
lexeme selected by the first residual obligation.

## Adjunct-boundary targeted repair (2026-09-16)

`runs/adjunct-boundary-targeted-repair-20260916.json` performs exactly one
fresh repair: `near` to `by` in “The baker carries a letter by the quiet
harbor.” The typed SVO frame and singular agreement are preserved. Independent
pointer/hash validation remains non-exact (38 letters, below the 39-letter
mechanical floor), and the next repair carries the residual into the determiner
slot.

## Additional reverse-segmentation evidence (2026-09-16)

The corpus-backed reverse-segmentation lane retains two fresh authored clauses
at 103--104 letters. Its weighted boundary DP finds no valid reverse
segmentation, so both remain intact prose controls with exactness false and a
concrete next repair to expand the held-out lexical/POS inventory.

## Authoritative continuation snapshot (2026-09-16, latest)

The common audit now contains **4,564 normalized rows across 159 route phases,
79 exact tapes, and 0 mechanically admitted outputs**. The registry validator
reports **261 unique artifacts, 26 preflight exclusions, and 239 run
artifacts**. The latest orthogonal additions are preserved as reader-facing
prose evidence: a 164-letter word-pair graph frontier (no closure), two fresh
61-letter paired semantic mutations (no closure), a 113-letter central-pivot
two-clause scene (first-character mismatch), a 108-state semantic-slot lattice
(all pruned), and two fresh 103--104-letter reverse-segmentation clauses (no
legal segmentation). Each row has independent pointer/hash validation,
provenance, novelty preflight, and a concrete next repair; none is promoted as
readability evidence.

The latest lane-specific renderings are also frozen here: the character-LM
decoder emits “a patient child carries the river trail at dusk. the keeper
opens old wooden gates before dawn.” (77 letters); the dependency seam CSP emits
“The patient archivist carefully maps a quiet museum archive before dawn.
Curious visitors study faded stars beside the river.” (105); the morphology /
clitic transducer emits “When the careful archivist restores the damaged ledger,
she records each witness name and sends the sealed copy to the harbor office
before winter.” (122); and the joint CFG/Earley chart emits “The patient
gardener waters the cedar seedlings beside the schoolhouse before sunrise The
teacher labels every seedling and stores the tools beneath the quiet porch.”
(140). All four are intact, independently audited, novelty-checked, non-exact,
and paired with a concrete next repair. The 729-row repeated-clause CFG sweep
is explicitly excluded as non-progress.

The newest hand-authored scene lattice keeps five fresh complete clauses intact
and enforces all-different content words before alignment. It renders a
308-letter scene: “Mara carries warm bread to the river. The careful pilot
studies cloud maps. Children gather bright shells by moonlight. A gardener
shelters young cedar shoots. Old friends share stories beside fire. By the
hearth, new tales begin. Near the grove, small finches settle. At twilight, the
patient tide returns. Beyond the hills, a quiet engine waits. At the shore, Owen
listens for bells.” Its independent pointer audit mismatches at index 0 and its
forward/reverse SHA digests differ; the mechanical gate therefore remains
closed. Provenance records the five-clause bank, cross-word residual ledger,
and a concrete semordnilap-compatible verb-object slot repair.

Three genuinely new follow-up constructions are also frozen. Boundary-
conditioned finite-state resegmentation emits a 270-letter six-clause scene;
live dependency character CSP emits a 127-letter marine-biologist scene with
127 explicit boundary equations; and a semantic residual-slot lattice emits a
122-letter observatory scene. Each has independent pointer/hash rejection,
fresh provenance, novelty preflight, and a concrete local repair: boundary
lexical targeting, typed adjunct substitution, or joint instrument/purpose
replacement respectively.

The outside-in scene grammar CSP was repaired to use distinct held-out right
frames rather than reflecting the left frame. Its longest fresh pair is “At
first light, the marine biologist records patient observations beside the
sheltered tide pool. At dusk, the coastal engineer maps hidden channels.” (124
letters); six fresh 102--124-letter prose pairs survive the all-different
content-word filter. All fail exactness at the outer seam and carry unequal
forward/reverse hashes plus a distinct-frame transitive repair. The exact-tape-first
route is kept outside the aggregate because it preserves the known 16-letter
“Able was I; I saw Elba” catalogue sentence, below the long-prose floor, and a
fresh 72-letter exact tape whose reflected half is visibly gibberish.

A ten-clause residual-equation solver adds a 355-letter clinic scene with ten
complete coordinated clauses: “At first light Mara opens the clinic, checks the
quiet generators, greets the two nurses, and records the medicine count. She
carries clean water to the waiting room, labels each parcel, phones the mountain
driver, updates the weather board, thanks the volunteers, and closes the ledger
before dusk. After supper she inventories the blankets, answers the radio, repairs
a torn notice, and leaves clear instructions for the morning shift.” It jointly
substitutes verb/object spans against the first twenty residual positions while
locking agreement and content-word uniqueness; independent pointer/hash checks
reject exactness and the next paired verb/object repair is recorded.

Three further orthogonal lanes are frozen. The finite clause automaton emits
“The patient archivist records quiet observations beside the river. Curious
visitors study faded stars beyond the winter station.” (109 letters), with a
live word-boundary state trace and no exact closure. The semordnilap typed lane
keeps the exact 16-letter “Stressed desserts.” control plus a fresh 100-letter
near miss: “The tired baker served stressed desserts, then repaired a drawer
while a quiet traveler delivered bread to the old river.” The exact witness is
not admitted as prose; the near miss fails at the first character. Finally, the
two-sided discourse equation lane emits a distinct 156-letter scene: “Nora
briefs the harbor crew before sunrise, then files the weather charts for the
evening watch. Later, the coastal pilot studies fresh signals and stores a
sealed map beside the lighthouse.” Its repeated-unit and content-word checks
pass, but the character equation remains open. All three carry provenance,
novelty preflight, independent audits, and concrete repairs.

## Fresh ten-lane audit (2026-09-16)

The ten requested dimensions were rerun as separate state representations. At
that ten-lane checkpoint, the aggregate contained **4,605 rendered rows across
177 route phases, 79 exact but rejected tapes, and 0 mechanically admitted
outputs**. At that checkpoint, the novelty registry had **279 unique retained
artifacts, 30 explicit exclusions, and 257 run
artifacts**. Every retained lane below has an intact rendering, an independent
pointer/hash audit, provenance, novelty preflight, and a concrete next repair.
The lane-8 exact result is intentionally excluded because it repeats a
canonical catalogue clause six times.

| lane | fresh rendered evidence | letters | exact audit | next reader-facing construction |
|---:|---|---:|---|---|
| 1 character LM | “At first light, the patient archivist opened the cedar cabinet and read each map aloud. Outside, a small river carried leaves past the quiet bridge while neighbors planned a careful repair. By noon the room was warm, orderly, and full of useful stories for the returning children.” | 227 | pointer `False` at 0 (`a`/`n`); forward/reverse SHA differ | Rewrite the first failing clause boundary under the character beam, preserving the scene and rerun both audits. |
| 2 exact-tape resegmentation | “The baker carries a letter to the quiet garden, and the teacher reads the message in the room.” | 75 | immutable 119-letter tape: 0 dictionary paths and 0 grammatical paths; fallback prose pointer/hash `False` | Author a held-out center-bearing clause and rerun the boundary DP without mutating the tape. |
| 3 dependency seam CSP | “The patient curator carries a brass compass through the quiet archive while the young cartographer records each turning near the northern window.” | 123 | pointer `False`, 59 mismatches; forward/reverse replay unequal | Substitute a held-out same-role lexicalization at the first unsatisfied seam. |
| 4 agreement morphology | “At dawn the patient cartographer unfolds a salt-stained map beside the quiet pier and marks each shoal where the returning boats find shelter before the weather turns cold over the inlet.” | 155 | pointer `False`; SHA pair unequal | Add irregular agreement transitions (`go→goes`, `carry→carries`) and re-solve the finite-state emission. |
| 5 CFG/Earley intersection | “At evening the patient keeper closes the garden gate. Beyond the hill a silver river carries moonlit leaves. Quiet readers gather stories beside the warm fire. Before sunrise the watchful traveler checks the old bridge. Across the valley distant bells answer a waking village. Careful hands arrange fresh maps beneath a window!” | 270 | pointer/hash `False` | Repair the terminal frontier in the intersected forward/reverse Earley charts while preserving clause roles. |
| 6 scene lattice | “At first light, Mara carried the brass key across the flooded courtyard, unlocked the archive door, and waited while the rescued records dried.” | 117 | two-pointer and ASCII-reverse `False`; hashes unequal | Re-author the purpose frame to satisfy the outstanding outer debt, then rerun the live slot equation. |
| 7 valency/attachment | “The careful archivist files the brittle maps before dusk. The patient curator labels the sealed boxes after steady rain.” | 100 | pointer `False` (49 mismatches); SHA pair unequal | Replace one sense-compatible adjunct at the first residual and recompute both attachments. |
| 8 inflection/clitic boundary | “At dusk, the harbor pilots checked the mooring lights, logged the tide in the crews' ledger, and warned each waiting sailor that boats would leave before dawn.” | 128 | pointer/hash `False` (57 mismatches); distinct-unit gate passes | Replace only the exposed suffix/clitic boundary, preserving plural/past-tense registers. |
| 9 scalable flat grammar | “At first light, the gardener unlocks the old shed, checks the water barrel, trims the apple tree, sweeps the stone path, labels the seed trays, carries the spare hose, mends the loose gate, folds the canvas tarp, writes a note for the neighbor, waters the herb bed.” | 208 | pointer/hash `False` (95 mismatches); proper-span and repetition gates pass | Append one held-out typed action increment and recompute the full tape. |
| 10 semantic slot repair | “After rain, the patient gardener carries a wrapped bundle beside the quiet greenhouse, records its arrival in the weather ledger, and waits for the evening porter to wheel the cart toward the dry storehouse.” | 170 | pointer/hash `False`; all anti-shortcut mechanical checks pass | Change only the typed adjunct at the first mismatch, then rerun the exact audit. |

Lane 8's distinct repair replaces the rejected repeated unit with a fresh
128-letter harbor scene: “At dusk, the harbor pilots checked the mooring lights,
logged the tide in the crews' ledger, and warned each waiting sailor that boats
would leave before dawn.” Its plural, past-tense, and possessive-clitic
obligations are independently audited; exactness remains false and the next
repair targets only the exposed suffix/clitic boundary.

These are construction results, not a readability certificate. The next
reader-facing test is still an intact-prose versus shuffled-control packet with
randomized blinded order; no row enters it until exactness and all mechanical
anti-shortcut checks pass.

The reversible lexical-shell follow-up is deliberately excluded: both of its
near misses embed the known 38-letter seed verbatim. That is a seed-wrapping
shortcut, not a generated long palindrome, so its evidence stays in the
registry's excluded section with its first-mismatch repair.

## Post-ten-lane constructive continuations

Three additional Luna state representations were run after the ten-lane
audit. A free-center semantic state machine produced a 138-letter scene; a
phrase-level reverse parser retained nine independently parsed 116--118-letter
clause pairs; and a finite-domain clause-equation SAT probe retained a
116-letter two-clause scene. All are fresh intact prose with independent
pointer/hash rejection, provenance, novelty preflight, and concrete repairs.
They add **11 rows and 3 route phases** to the aggregate; at that intermediate
checkpoint the common report contained **4,605 rows, 177 route phases, 79 exact
but rejected tapes, and 0 mechanically admitted outputs**. None
is promoted as reader evidence.

The next constructive probes add three distinct states: a 130-letter dialogue
question/answer grammar, a 68-letter typed morpheme-compound realization, and
a 91-letter first-token equation grammar. Their ordinary prose, independent
audits, provenance, novelty records, and concrete repairs are retained; the
morpheme probe is below the long-prose floor and fails its lexicon gate, so no
shortcut is promoted.

The A* typed word-boundary lane adds a 140-letter intact garden/schoolhouse
scene, and a human two-sentence single-slot repair adds seven deduplicated
92--97-letter scene variants. Both remain non-exact with independent audits and
named repairs. An attested phrase-bridge preflight found zero complete-clause
bridges and is excluded rather than counted as a fabricated candidate.

Three further Luna continuations are now retained as distinct constructive
states. The paired lexical grammar renders six fresh 75--82-letter SVO clause
pairs and the free-center semantic bridge renders a 109-letter scene; each has
independent two-pointer/SHA rejection, provenance, novelty preflight, and a
first-residual repair. A productive grammar-pair composition then grows four
complete paired-production states from 55 to 223 letters (base and targeted
right-arm repair), again with zero exact closures. These are bounded growth
states, not larger duplicate sweeps. The common audit is now **4,824 rows
across 223 route phases, 79 exact rejected tapes, and 0 mechanically admitted
outputs**; the novelty registry now records **324 retained artifacts, 31
explicit exclusions, and 302 retained run artifacts**, with the aggregate
report itself listed as an audit report rather than a construction lane. None
is eligible for readers until an exact anti-shortcut survivor exists.

The latest three Luna continuations are retained as genuinely new state
representations, not larger duplicate sweeps. Endpoint-aware bilateral seam
decoding reserves the outer lexical terminals before interior character
decoding and renders six 100--105-letter scenes. Fresh-scene tape/CFG
resegmentation keeps word boundaries and inflections mutable and records a
126-letter complete-prose repair. A typed semantic-slot neighborhood changes
one role-compatible slot at the first residual in six 109--160-letter scenes.
Each row includes the rendered text, independent two-pointer and
forward/reverse SHA checks, provenance, novelty preflight, and a concrete next
repair; none is exact or mechanically admitted.

The next bounded wave adds a typed onset/rime grapheme constructor with 12
37--47-letter complete clauses and a reader-first function/inflection editor
with five fresh 96--104-letter scenes. The grapheme constructor was repaired
after its first probe omitted object determiners; the corrected renderings are
ordinary prose and remain non-exact. An entailment-preserving active/passive
rewrite was preflighted but overlaps existing voice/information-structure
families, so it is retained only as excluded failure evidence.

Representative corrected grapheme output is “The baker repairs the gate; the
pilot maps a gate.” (39 letters; normalized SHA forward
`db308c0c...`, reverse `911b6228...`, first mismatch 0; exact `False`). The
longest reader-first function-edit output is “At dusk, Mara carries the blue
lantern across the quiet bridge, and Jonah records each rescued name so the
town archive can open.” (104 letters; normalized SHA forward
`49d8b841...`, reverse `443a004b...`, first mismatch 0; exact `False`). Both
are fresh authored prose with no catalogue import or symmetry shortcut; the
next reader-facing test remains blocked until an exact anti-shortcut survivor
is independently admitted.

The next repair wave adds three distinct lanes. A typed semantic center-out
solver renders “The patient curator shelters the fragile maps during the storm;
the careful teacher copies the final field notes beside the window.” (109
letters; first mismatch 0; normalized SHA prefixes `f73f2178...` and
`4bab5944...`). A lexical word-equation intersection renders “The curator labels
the fragile map before the archivist stores the ledger, while rain gathers
softly against the western windows and visitors wait beside the reading room.”
(143 letters; first mismatch 0; SHA prefixes `ad660754...` and `3f372d81...`).
A joint slot/boundary repair emits “The quiet curator labels the old faded map
before dawn.” (45 letters; first mismatch 0; SHA prefixes `032e4c56...` and
`60a5d269...`). All are complete ordinary prose with fresh provenance,
independent pointer/SHA validation, novelty preflight, and concrete repairs;
none is exact or reader-eligible.

Two additional Luna lanes are retained as distinct constructive states. A
character-level beam over typed semantic frames emits “The young botanist
studies the silver seed cases beside the greenhouse; a careful pilot marks the
distant landing lights through the mist.” (115 letters; first mismatch 1; SHA
prefixes `e57cebaf...` and `99cbaec3...`). A fresh CFG/Earley character
intersection emits “At first light, the surveyor records the river current
while the baker warms bread for the waiting crew, and the harbor keeper checks
the lamps before opening the gate.” (137 letters; first mismatch 0; SHA
prefixes `a24dca02...` and `8b577593...`). Both are complete prose with independent
pointer/SHA checks, provenance, novelty preflight, and typed next repairs;
neither is exact or reader-eligible.

Three additional Luna constructions are retained as fresh states. A paired
semantic CFG emits “A patient gardener waters the young cedar trees after the
rain; the careful pilot maps the distant landing lights through the mist.”
(108 letters; first mismatch 0; SHA prefixes `48d4b48c...` and
`61adb2f6...`). A joint semantic/inflection word-boundary DP emits “The
curator records the eastern seedlings before the evening archive closes. A
young pilot carries fresh charts to the lighthouse.” (109 letters; first
mismatch 0; SHA prefixes `1891fdf8...` and `025253bc...`). An online clause-order
scene lattice emits “At dawn, the archivist opened the cedar cabinet. The patient
apprentice copied each date into a clean ledger. The curator sorted four
journals for the river school. Evening bells faded.” (151 letters; first
mismatch 0; SHA prefixes `a8ac451b...` and `66d84b33...`). All have independent
pointer/SHA validation, fresh provenance, novelty preflight, and concrete
repairs; none is exact or reader-eligible.

## Three orthogonal Luna continuations (2026-09-16)

The next queue pass added three distinct construction states rather than a
larger duplicate sweep. The constrained edit program rendered the fresh scene
“At dawn, the careful cartographer marked the northern trail, while a patient
ranger checked the bridge and recorded the weather.” (106 letters; 51
mirrored-character mismatches; normalized SHA forward
`b4ca1de6...`, reverse `d92f8d11...`). Its accepted substitutions lower debt
monotonically to 44 in a 97-letter final state while preserving the two-clause
parse and semantic roles. The next repair is a held-out role-preserving
inflection lexicon.

The append-algebra probe emits complete, independently authored clauses, with
the longest rendering “Mara observes the harbor lantern. Jon repairs the
western gate. Iris records the morning tide. Noah carries a copper compass.”
(102 letters; first mismatch 0; normalized SHA forward `2ae18869...`, reverse
`746851d5...`). It records the failed append-preserving invariant and the
concrete repair of selecting a typed macro against the live suffix obligation.

The discourse-relation involution lane renders “The rain cooled the garden
because the seedlings survived the heat.” (56 letters; first mismatch 1;
normalized SHA forward `f7687fb8...`, reverse `9d847a67...`) among 12 complete
cause/effect and contrast propositions. Its next repair changes only the
connective and subordinate attachment at the first residual while preserving
relation polarity. All three runs have fresh generator hashes, explicit
anti-shortcut flags, novelty preflights with no signature collisions, and
independent pointer/hash replays. All remain non-exact and reader-ineligible;
the next reader-facing test is still the randomized intact-versus-shuffled
packet, gated on a mechanically admitted exact survivor.

## Five orthogonal Luna continuations (2026-09-16)

The current continuation adds five distinct construction states: nested-free
clause-boundary DP (six 98--107-letter scenes), semantic phrase-edge joining
(three 114--124-letter scenes), coupled object/attachment repair (four
108--109-letter scenes), role-typed semordnilap clause products (four
grammatical 92--106-letter scenes), and a typed reversible-clause composer with
appendable growth (two 107--111-letter scenes). Every row is rendered intact,
independently pointer/SHA audited, provenance-backed, novelty-preflighted, and
paired with a concrete next repair. None closes exactly; all remain outside the
reader packet.

Two further fresh Luna states are retained in the common audit. Seed-free
authored-frame insertion produces “At dawn, the archivist opens the cedar
cabinet and records the harbor map, before dusk.” (83 letters), and an
interrogative/relative-template solver produces “Was the quiet curator sure that
the young pilot had seen the chart I filed beside the harbor ledger?” (87
letters). Each row carries independent pointer/SHA checks, provenance,
novelty preflight, anti-shortcut flags, and a named next repair; neither closes
exactly or enters the reader packet.

The following queue pass adds three boundary-aware states: an independently
resegmented reflected-tape grammar (two 103--107-letter scenes), a semantic
scene equation lattice with asymmetric token counts (two 111--112-letter
scenes), and a finite-state semantic boundary-macro grammar (three growth
states up to 74 letters). Each row has independent pointer/SHA replay,
provenance, novelty preflight, anti-shortcut checks, and a concrete residual
repair; none closes exactly or enters the reader packet.

The newest queue pass adds three genuinely distinct constructive states: a live
mirrored-tape terminal decoder (six 88--103-letter scenes), a fresh
human-authored attachment CSP (two 105--107-letter museum scenes), and an
unbounded bilateral semantic-growth grammar (three non-repeating growth states
up to 140 letters). Every row has independent pointer/SHA replay, provenance,
novelty preflight, anti-shortcut checks, and a concrete first-residual repair;
none closes exactly or enters the reader packet.

The latest Luna audit adds four orthogonal states: a past-tense dependency
transducer (six intact 94--104-letter scenes), an interrogative/relative
reverse-tape resegmenter (two 109--112-letter questions), a causal
scene-semordnilap graph (two 100--111-letter event scenes), and a center-out
open-word seam decoder (two 107--110-letter scenes). Every row has rendered
prose, independent two-pointer and forward/reverse SHA-256 checks, provenance,
novelty preflight, and a first-residual repair. All four have zero exact
closures and remain outside the reader packet.

The next three Luna lanes are retained separately: a Brown-attested
phrase-pair seam reconciler (six 91--96-letter scenes), a bilateral role
lattice coupling first-residual substitutions (four 70--72-letter scenes),
and a feature-carrying character CFG with tense and clitic states (two
69--75-letter scenes). Every row includes intact rendered prose, independent
two-pointer and forward/reverse SHA checks, provenance, novelty preflight,
anti-shortcut flags, and a concrete repair. None closes exactly or enters the
reader packet.

The newest orthogonal continuations extend the same evidence contract. A
cross-boundary phrase-block grammar renders three linked 109--112-letter
scenes, a typed boundary-block scene grammar renders sixteen 80--93-letter
imperative scenes, and a cross-POS semordnilap scene CFG renders two connected
79--83-letter scenes. Each row is shown in its run artifact with independent
two-pointer and forward/reverse SHA checks, provenance, novelty preflight,
anti-shortcut flags, and a named first-residual repair. All three lanes have
zero exact closures and remain outside the reader packet; they are retained as
new construction states, not duplicate sweeps.

Three further Luna continuations change the state again. An outside-in
semantic scene compositor renders three 152--163-letter archive scenes with
held-out phrase-pair seams; a joint semantic-slot and cross-word resegmentation
lane renders four 85--94-letter supply instructions; and a valency/clitic
lexicalizer renders two 78--89-letter transfer scenes with live inflectional
equations. Each row carries actual prose, independent pointer/SHA validation,
provenance, novelty preflight, anti-shortcut flags, and a concrete next repair.
All three lanes have zero exact closures and remain outside the reader packet.
