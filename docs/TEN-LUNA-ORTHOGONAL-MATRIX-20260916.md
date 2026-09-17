# Ten orthogonal Luna lanes: evidence matrix

This matrix is a contract check, not a claim of readable-palindrome success. Each lane is a distinct construction state space, not a larger sweep of a previous one. The rendered text below is the actual output retained by the run. Exactness is independently recomputed by `experiments/validate_ten_luna_lane_contract_20260916.py` with a fresh letter normalizer, two-pointer comparison, and forward/reverse SHA-256. Programmatic diagnostics do not certify human readability.

| lane | method and retained run | actual rendered prose | letters | independent exact check | provenance / novelty | concrete next repair |
|---:|---|---|---:|---|---|---|
| 1 | Character-level LM-constrained decoding — [`live-gpt2-character-decoder-preflight-20260916.json`](../runs/live-gpt2-character-decoder-preflight-20260916.json) | “A baker marks the button near a bright field. A caller answers the message behind a brief plaza.” | 77 | `false`; 35 two-pointer mismatches; forward/reverse SHA differ | Local GPT-2 grammar-prefix decoding; no copied source, fixed tape, or reverse emission; novelty preflight passed | At the first live seam mismatch, replace one held-out same-POS/sense word, then replay character probabilities and both exact audits |
| 2 | Exact-tape grammatical resegmentation — [`exact-tape-grammatical-resegmentation-20260916-luna.json`](../runs/exact-tape-grammatical-resegmentation-20260916-luna.json) | “The baker carries a letter to the quiet garden, and the teacher reads the message in the room.” | 75 | `false`; independent two-pointer and SHA replay disagree | Fresh intact prose; immutable tape DP and typed phrase grammar; catalogue and borrowed-text flags are false | Author a center-bearing clause with paired letters, then rerun the same DP without mutating the authored tape |
| 3 | Dependency-tree seam CSP — [`dependency-seam-csp-20260916-luna.json`](../runs/dependency-seam-csp-20260916-luna.json) | “The patient curator carries a brass compass through the quiet archive while the young cartographer records each turning near the northern window.” | 123 | `false`; 59 mismatches; independent reverse SHA supplied by the audit overlay | Fresh typed dependency scene; catalogue import and fragment flags are false; novelty signature is collision-free | Replace the first-residual lexical item with a held-out same-role/same-feature item and rerun both audits |
| 4 | Agreement-carrying morphology transducer — [`morphology-crossword-transducer-20260916.json`](../runs/morphology-crossword-transducer-20260916.json) | “The curator opens the sealed cabinet, and she carefully files its maps before the evening visitors arrive.” | 88 | `false`; independent two-pointer and forward/reverse SHA checks | Held-out verb/noun/clitic choices with agreement-state replay; fresh cabinet/inlet scenes; anti-shortcut checks pass | Change one held-out inflection or clitic seam while preserving dependency features at the first residual |
| 5 | CFG/Earley character intersection — [`cfg-earley-character-intersection-fresh-20260916.json`](../runs/cfg-earley-character-intersection-fresh-20260916.json) | “At first light, the surveyor records the river current while the baker warms bread for the waiting crew.” | 85 | `false`; independent two-pointer and SHA fields agree; no exact closure | Fresh authored scene CFG; complete parse, no fixed tape, no catalogue import, and anti-shortcut gates pass | Add one typed adjunct production whose terminals satisfy the first unresolved chart debt while preserving a complete SVO parse |
| 6 | Human-authored scene lattice with live equations — [`human-scene-lattice-live-equations-20260916.json`](../runs/human-scene-lattice-live-equations-20260916.json) | “At first light, Mara carried the brass key across the flooded courtyard, unlocked the archive door, and waited while the rescued records dried.” | 117 | `false`; two independent audits disagree at the outer pair; SHA digests differ | Human-authored archive scene; slot trace records live equations; catalogue import and known-palindrome reuse are false | Author a second purpose-frame realization whose opening characters discharge the outstanding outer debt, then rerun the complete slot equation |
| 7 | Semantic valency/attachment solver — [`semantic-valency-attachment-solver-20260916-luna.json`](../runs/semantic-valency-attachment-solver-20260916-luna.json) | “the careful archivist files the brittle maps before dusk. the patient curator labels the sealed boxes after steady rain.” | 100 | `false`; 49 mismatches; forward/reverse SHA differ | Fresh event-frame lexicon with attachment replay; no catalogue import or known-palindrome reuse | Replace only the first-residual subject adjective with the held-out same-sense choice, preserving both event attachments |
| 8 | Inflectional and clitic boundary search — [`inflection-clitic-distinct-repair-20260916-luna.json`](../runs/inflection-clitic-distinct-repair-20260916-luna.json) | “At dusk, the harbor pilots checked the mooring lights, logged the tide in the crews' ledger, and warned each waiting sailor that boats would leave before dawn.” | 128 | `false`; 57 mismatches; independent pointer/SHA replay agrees | Fresh harbor scene with plural agreement, past tense, possessive clitic, and live suffix obligations; no repeated content units and no catalogue text | Replace only the exposed suffix or clitic boundary at the first mismatch; preserve the distinct clause inventory and reject catalogue-family closures |
| 9 | Scalable compositional grammar without nested palindrome spans — [`scalable-compositional-grammar-20260916-luna.json`](../runs/scalable-compositional-grammar-20260916-luna.json) | “At first light, the gardener unlocks the old shed, checks the water barrel, trims the apple tree, sweeps the stone path, labels the seed trays, carries the spare hose, mends the loose gate, folds the canvas tarp, writes a note for the neighbor, waters the herb bed.” | 208 | `false`; 95 mismatches; independent pointer/SHA replay agrees | Unbounded flat action grammar, no nested palindrome units, no word-order mirror, and fresh garden seed | Append one held-out typed action increment and recompute the complete tape, retaining the flat grammar state |
| 10 | Exact-candidate repair with semantic slot substitutions — [`semantic-slot-substitution-repair-20260916-luna.json`](../runs/semantic-slot-substitution-repair-20260916-luna.json) | “After rain, the patient gardener carries a wrapped bundle beside the quiet greenhouse, records its arrival in the weather ledger, and waits for the evening porter to wheel the cart toward the dry storehouse.” | 170 | `false`; 160 mismatches; independent pointer/SHA replay agrees | Fresh authored scene; one held-out typed object substitution; catalogue, copied-clause, wrapper, and word-order flags are false | Change only the first-mismatch typed adjunct (locative or purpose), never two slots, then rerun both exact audits |

## Rejected control and reader-facing test

The earlier lane-8 artifact also rendered a 126-letter exact surface by repeating “A man, a plan, a canal, Panama.” six times. The audit records that output as an exact control but rejects it for repeated canonical units and catalogue-control status; it is not a generated result and is not in this matrix.

No row above is reader-certified. The next reader-facing test is a reproducible, randomized blinded packet containing only an exact anti-shortcut survivor plus intact-prose and shuffled controls, with a rater package that keeps method provenance hidden. Until an exact survivor exists, these rows remain construction evidence and repair targets, not claims of success.

## Three held-out repairs executed after the matrix

The first repair pass was deliberately one state per lane, not a resweep:

- CFG/Earley adjunct repair: “At first light, the surveyor records the river current near the dock while the baker warms bread for the waiting crew.” (96 letters; exact `false`; next repair adds one opposing adjunct terminal.)
- Semantic-slot adjunct repair: “After rain, the patient gardener carries a wrapped bundle near the glasshouse, records its arrival in the weather ledger, and waits for the evening porter to wheel the cart toward the dry storehouse.” (163 letters; exact `false`; next repair changes only the final locative attachment.)
- Inflection/clitic suffix repair: “At dusk, the harbor pilots checked the mooring lights, logged the tide in the crews' ledger, and warned each waiting sailor that boats would leave before dusk.” (128 letters; exact `false`; next repair changes only `crews'` to `crew's`.)

Each has its own JSON provenance, novelty preflight, independent pointer/SHA audit, and anti-shortcut decision. The aggregate now contains 4,952 rows across 271 routes; none is mechanically admitted or reader-certified.

A second one-state repair pass then produced three more intact candidates:

- CFG/Earley opposing adjunct: “At first light, the surveyor records the river current near the dock while the baker warms bread for the waiting crew beside the quay.” (109 letters; exact `false`; next repair changes one lexical terminal.)
- Inflection/clitic possessive boundary: “At dusk, the harbor pilots checked the mooring lights, logged the tide in the crew’s ledger, and warned each waiting sailor that boats would leave before dusk.” (127 letters; exact `false`; next repair changes only the complementizer boundary.)
- Semantic-slot final locative: “After rain, the patient gardener carries a wrapped bundle near the glasshouse, records its arrival in the weather ledger, and waits for the evening porter to wheel the cart into the dry storehouse.” (161 letters; exact `false`; next repair changes only the temporal adjunct.)

These are single-state repairs with fresh provenance, novelty preflight, independent exact audits, and no catalogue or repetition shortcut. The aggregate is now 4,955 rows across 274 routes.

A third one-state continuation kept the same discipline:

- CFG/Earley lexical terminal: “At first light, the navigator records the river current near the dock while the baker warms bread for the waiting crew beside the quay.” (110 letters; exact `false`; next repair changes one object terminal.)
- Inflection/clitic complementizer: “At dusk, the harbor pilots checked the mooring lights, logged the tide in the crew’s ledger, and warned each waiting sailor when boats would leave before dusk.” (127 letters; exact `false`; next repair changes only the directional verb.)
- Semantic-slot temporal adjunct: “Before dusk, the patient gardener carries a wrapped bundle near the glasshouse, records its arrival in the weather ledger, and waits for the evening porter to wheel the cart into the dry storehouse.” (162 letters; exact `false`; next repair changes only the porter action verb.)

The aggregate is now 4,958 rows across 277 routes; no candidate in these repair queues is mechanically admitted or reader-certified.

## State-space pivots after the repair plateau

Rather than resampling the same local seams, three bounded pivots were run:

- Simultaneous phrase-pair construction: “The orchard keeper gathers ripe apples beside the stone wall, while a patient engineer tests the quiet turbine under the glass roof.” (109 letters; exact `false`; next pivot changes the typed role pair.)
- Center-out observatory scene lattice: “The astronomer calibrates the telescope; the observatory clock chimed, and the assistant records the readings.” (93 letters; exact `false`; next operator authors a held-out telescope-side verb/object pair.)
- Immutable-scene exact-tape diagnostic: “At dawn, the archivist opens the west gallery, and the curator labels each recovered map while visitors wait quietly beside the old stair.” (113 letters; exact `false`; reverse grammar found no complete path; next operator chooses seam terminals before tape freeze.)

These pivots keep complete prose, independent exact checks, provenance, novelty preflight, and anti-shortcut decisions. They add six rows and move the aggregate to 4,964 rows across 280 routes; none is mechanically admitted or reader-certified.

## Character-equation structural lanes

The next bounded trio enforced the mirrored-character invariant during lexical
selection itself:

- Live paired lexical graph: “The miller repairs the wooden wheel beside the creek, and a patient chemist measures the clear solution inside the glass room.” (104 letters; exact `false`; next repair changes the subject edge.)
- Grammar-first phrase intersection: “The patient surveyor records the northern channel while the careful deckhand repairs a torn sail before the evening tide.” (113 letters; exact `false`; next repair replaces the first failing phrase.)
- Semantic word-pair event graph: “The pilot carries a sealed parcel to the garden, where the keeper waters the basil before sunset.” (79 letters; exact `false`; next operator adds one delivery/care event edge.)

All three enforce their character checks before word commitment, preserve intact prose, and retain independent audits, provenance, novelty, and anti-shortcut decisions. They add three rows, bringing the aggregate to 4,967 rows across 283 routes; none is mechanically admitted or reader-certified.

## Final exact-focused attempt and duplicate rejection

The next bounded cycle used three different construction states and rejected a
larger duplicate sweep explicitly:

- Exact-focused lexical-edge DP emitted “The miller repairs the wooden wheel beside the creek, and a patient chemist measures the clear solution inside the glass room.” (104 letters; independent two-pointer `false`; forward/reverse SHA differ). Its precommit character-pair logic is preserved, but the rendered candidate is byte-for-byte the retained paired-lexical-graph output. Novelty status is `excluded_duplicate`, so it is not added to the aggregate. The next repair is a fresh same-role lexical edge at residual zero.
- Center-terminal clause family emitted eight jointly selected complete states; the best is “The singer carries the melody at noon; The quiet clerk files the record.” (58 letters; independent two-pointer `false`; SHA digests differ). The run records human-authored semantic slots, a live center equation, no nested span or catalogue import, and a new center connective as the next repair.
- Grammar-first matching-boundary intersection emitted “The quiet curator records a coastal chart while the patient guide describes an old harbor at evening.” (84 letters; independent two-pointer `false`; forward/reverse SHA differ). All five phrase boundaries passed their local equations before commitment; novelty preflight found no fixed tape, reversal, catalogue, or word-order mirror. The next repair adds a typed center production for the remaining global residual.

The two nonduplicate lanes add nine rows, moving the authoritative aggregate to
4,976 rows across 285 routes (79 exact tapes, 0 mechanically admitted). The
registry now records 386 retained artifacts, 32 explicit exclusions, and 364
retained run artifacts. None is reader-certified; the intact-versus-shuffled
packet remains gated on an exact anti-shortcut survivor.

## Three further orthogonal Luna states

The next frontier reopened three distinct construction representations:

- Semantic slot/attachment repair: “At first light, the patient archivist carries a weathered register through the west room toward the reading table, checks its brittle clasp, and leaves the record beside a quiet lamp for the evening clerk.” (168 letters; independent two-pointer `false`; forward/reverse SHA differ). One carried-record substitution and one path-attachment rewrite were selected before realization; novelty passed, all anti-shortcut flags are clean, and the next repair changes one held-out verb-frame inflection.
- Finite feature-center grammar: “the modest teacher packed blue notebooks after class for the reading group, and the alert ranger guided hikers toward camp beneath stars” (114 letters; independent two-pointer `false`; SHA digests differ). An explicit `CENTER` production and feature-unification stack kept the clauses ordinary; novelty passed with no fixed tape, reversal, catalogue, or word-order mirror. The next repair replaces one right-clause production at the first pending obligation.
- Joint complete-constituent equation solver: “The harbor clerk records cargo manifests before sunrise. The museum guide opens the west gallery after visitors arrive. The waiting curator locks glass cabinets after evening lectures. The watchful pilot secures fishing vessels beside stone piers.” (208 letters; independent two-pointer `false`; forward/reverse SHA differ). Thirty-six jointly selected complete constituents were audited; no shortcut was used. The next repair replaces the paired complete constituent at the first frontier mismatch.

These lanes add fourteen rows, moving the authoritative aggregate to 4,990
rows across 288 routes (79 exact tapes, 0 mechanically admitted). The registry
now records 389 retained artifacts, 32 explicit exclusions, and 367 retained
run artifacts. None is reader-certified.

## Word-internal seam and residual-center continuation

Two more constructive states were retained after novelty preflight:

- Word-internal seam equation: “a teacher guided the patient apprentice through the winter archive; the recorder preserved a precise account beside the weathered map.” (113 letters; independent two-pointer `false`; forward/reverse SHA differ). The `e == e` morpheme-boundary equation was satisfied inside independently authored words, without a fixed tape or broad morphology sweep. The next repair replaces only the two seam-adjacent morphemes at the first global mismatch.
- Minimal residual grammar: “The pilot checks the engine. A quiet gardener waters the roses. The mason measures the arched window. A patient teacher guides the new reader. Meanwhile, The sailor marks the distant buoy. A careful nurse carries fresh water. The curator labels the painted vessel. A young driver parks beside the mill.” (244 letters; independent two-pointer `false`; SHA digests differ). An atomic center was committed before feature-agreement SVO growth; the next repair changes one right-side verb/object pair while preserving agreement.

A third relation-plan probe rendered ordinary-looking clauses and was initially
misread. Its anti-shortcut fields are positive checks: every state is *not* a
word-order mirror, has distinct content, and lacks catalogue scaffolding. For
example, “the patient keeper opened the garden gate in the morning because the
harbor crew secured the boat before dusk.” (91 letters; exact `false`) is now
retained in `runs/semantic-relation-plan-solver-20260916.json`; the next repair
replaces the active relation-compatible event at the first residual.

Three orthogonal targeted repairs are also retained: a 117-letter internal-seam
repair changing only two morphemes, a 59-letter atomic-center repair changing
one right verb/object pair, and a 214-letter complete-constituent repair. Each
has independent exact validation, provenance, novelty preflight, and a concrete
next repair. The authoritative aggregate before this continuation was 5,010
rows across 294 routes (79 exact tapes, 0 mechanically admitted); the registry
was 395 retained artifacts,
32 exclusions, and 373 retained run artifacts.

The next orthogonal continuation adds three bounded states: a 72-letter
agreement/clitic transducer scene (“The patient pilot checks the engine, notes
its gauge, and tells the crew it starts at dawn.”), a 108-letter whole-word
semantic-slot expansion (“The careful archivist stores weathered maps beside
the north window. A patient gardener waters young cedars near the school gate.”),
and a 154-letter semantic-valency lattice witness (“The quiet gardener waters
young seedlings after steady rain. The watchful sailor repairs loose rigging
near the harbor. The young porter carries sealed parcels toward records
offices.”). Each is exact `false` under independent pointer/SHA checks, has
fresh provenance and novelty preflight, and names a concrete clause/terminal
repair. The aggregate before the next exact-closure attempts was 5,018 rows
across 297 routes; the registry was 398 retained artifacts, 32 exclusions, and
376 retained run artifacts.

The exact-closure attempts add three nonduplicate routes: nine 70--81-letter
agreement/clitic terminal states, six 155--159-letter clause-equation states,
and one 106-letter 59,049-pair semantic-slot frontier witness. Their best
rendered texts are shown in the evidence ledger; all independent exact checks
remain `false`, and each run records provenance, novelty, anti-shortcut flags,
and its next repair. The aggregate is now 5,195 rows across 309 routes; the
registry is 410 retained artifacts, 32 exclusions, and 388 retained run
artifacts.

The latest orthogonal-wave checkpoint adds finite-state boundary decoding,
agreement-carrying morphology, a human-authored scene lattice, memoized
POS/valency intersection, semantic phrase-chunk synchronization, and a
held-out exact-survivor repair. The representative intact rendering is “The
careful archivist stores weathered maps beside the north window. A quiet
teacher reviews marked field notes near the harbor office.” (113 letters;
independent exact checks both false). Every row carries provenance, novelty
preflight, anti-shortcut flags, and a concrete next repair; the exact
101-letter near-survivors are visibly unreadable and not reader-eligible. The
authoritative aggregate is now **5,427 rows across 332 routes** with **84
exact and 0 mechanically admitted**, and the registry is **431 retained, 35
excluded, 409 retained run artifacts**.
