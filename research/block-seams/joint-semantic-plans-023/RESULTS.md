# Joint semantic plans 023

Two exact short request/reply constructions were found; no editing narrative or convincing long paragraph was produced. Neither is a claimed new best or human-approved result. The prior 35-letter roti construction remains a quality rejection.

## Finite language and accounting

Meaning was specified before letter screening: document revision enables a recipient to use the same document, or a requested drawing is acknowledged by its bound addressee. Eight editing relations each had three sentence orders (24 proposals). Drawing choices jointly varied six command verbs, six response verbs, two aspects, two actors, eight modifiers, eight ward types, and six reference forms: 55,296 virtual paths. Twelve command/tail-head checks admitted only draw/ward; 47,616 drawing paths were pruned before materialization. The complete 7,704 materialized proposals comprise 24 editing and 7,680 drawing paths, with duplicates and failures retained. Of these 7,700 fail fixed-edge compatibility; four drawing paths compile to 16 chunks. Two are exact and two nonexact. This is an authored finite language, not a natural-language yield estimate.

Both algorithms agree with the exhaustive exact set (two unique texts, four method occurrences), and every output is 100% ASCII-letter exact. Editing searches were skipped with endpoint proofs. Drawing center-out search completed 90 states/46 failures; outside-in completed 25 states/24 failures and recorded four duplicate states. All engine failures remain in engine-results.json. Total experiment time: 0.233827 seconds, below the predefined 30-second CPU cap. Seed 19034; 10,000 states, 100 outputs, four steps, 5,000 seeds and two seconds per search. No paid calls or remote compute. No speedup claim.

## Individual quality review

| Text | Letters / words / distinct | Grammar | Readability | Coherence | Overall |
| --- | --- | --- | --- | --- | --- |
| Draw a ward. I draw a ward. | 19 / 7 / 4 | 5 | 4 | 3.5 | 4.5/10 |
| Draw a mock coma ward. I draw a mock coma ward. | 35 / 11 / 6 | 5 | 3.5 | 3.5 | 5/10 |

Scores are owner model judgments using the frozen 019 rubric (three dimensions 0–5 and holistic overall 0–10); human labels are null. Short confirmation is semantically linked but the simple-present reply is less natural than an immediate progressive acknowledgment. The longer mock coma ward compound is interpretable as a fictional design yet awkward; no lexical attestation is asserted. It repeats five tokens twice (5/11 repeat excess), including the entire action phrase. These are transparent recompositions of the known draw a ward component, not novel paragraph claims. Neither exceeds the prior 019 owner overall 5.5/10. Repetition is reported as a feature, never a hard rejection rule.

Sampling is ceil(10% × output count) independently per method/arm: zero of zero editing; one of two drawing outputs per method. Overall this gives 2/4 method occurrences (50%, not a literal 10%) and one unique sampled text, the 19-letter control. Each sampled occurrence is scored in sample-reviews.json. The 35-letter unsampled unique output also receives a separate owner review. Neither silence nor these model scores is human acceptance.

## Seam diagnostic and representation limit

The longer construction uses intact grammatical chunks: Draw / a mock coma ward. / I draw / a mock coma ward. A center inside the I draw predicate extends across whole noun phrases. Letter debt is paid asymmetrically and can change side; there is no association/no i tai filler or token-deletion patch. However, its noun phrase has a palindromic modifier and its entire command is already palindromic. This demonstrates licensed recomposition, not a productive multi-event narrative family.

The 24 editing plans all have immutable endpoint mismatches. Internal synonym expansion cannot fix them. The current authored phrase compiler also does not induce new lexical completions: it searches only the finite surface choices supplied to it. Semantic bindings enforce declared identity; they do not independently establish natural discourse, and draw/design/outline are approximate depiction paraphrases rather than equivalent actions in every context.

A concrete alternative is a plan-conditioned bidirectional phrase lattice: choose a small everyday two-event semantic plan, enumerate independently licensed clause orders (fronting, passive, explicit and pronominal references), and jointly select initial and final lexical phrases before freezing endpoints. Maintain lemma/argument/coreference links throughout; allow dictionary-backed partial-word boundary completions only if the resulting full phrase has a licensed parse and the original semantic role. Predeclare that finite lattice and its baseline, rather than enlarging the failed editor/document vocabulary under fixed edges. This remains proposed future work, not implemented or measured here.

## Evidence and resumability

plan.json freezes bounds and provenance; endpoint-roots.json retains all 12 root decisions; proposals.json retains every materialized proposal; inventory.json contains the compiled language; engine-results.json retains outputs, traces and failures; sample-occurrences.json preserves the deterministic sample. Raw generation receipts are unchanged by reviews. checkpoint.json records sole ownership and no remaining active lock. The merged paper and prior human judgments remain unchanged.

Independent ASCII-letter verification agrees with the exhaustive proposal set for both engines. Eleven existing focused engine/endpoint/grammar tests passed in 0.39 seconds. An initial test invocation named a nonexistent file and ran no tests; the corrected invocation passed. This is research validation, not production qualification.
