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

The current aggregate is 4224 audit-compatible rendered rows across 95 route
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
