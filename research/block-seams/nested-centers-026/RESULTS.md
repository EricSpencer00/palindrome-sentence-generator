# Nested center and seam expansion 026

Center replacement can produce new grammatical seam interpretations. Four possessive-roti constructions reach 38–40 letters, a modest gain over the 35-letter seed. No long paragraph or rich multi-event narrative has been established. The best owner-reviewed case is “Some men, if editors tap Pat's roti, define memos.” New human labels remain pending; the seed and the exact ten previously liked examples remain positively human-labeled.

## Correct center and both seams

The verified seed is L + sus + reverse(L), where L = somemenifeditors (16 letters) and reverse(L) = srotidefinememos (16 letters). The center is sus, not the complete word suss: the verb's last s belongs to the mirrored right side.

Expanding the editable radius by one gives fixed outer left somemenifeditor (15 letters), old center ssuss (5 letters), and fixed outer right rotidefinememos (15 letters). Replace ssuss with s + tapPat + s. The inner tapPat is a six-letter palindrome, and the wrapped center stappats is an eight-letter palindrome. Resegmentation licenses three interfaces in this radius-one representation: the first s completes editors plural; tap is its base-form transitive predicate; Pat plus the final s makes Pat's possessive owner of roti. The main men/define/memos clause remains intact. The resulting 38-letter text is exact and has nine distinct words, no repeated tokens and zero finite repetition cost. The minimal equivalent form is L(16) + tapPat(6) + reverse(L)(16): simply replacing sus with tapPat already yields the38-letter text. Expanding to radius1 is an equivalent representation, not a necessary change or additional expressive power. The left editors plural s is preserved; only the right fixed s changes its lexical role, from the final letter of suss to the possessive marker. Both interfaces are checked. The s...s wrapper itself earns no progress credit.

The same finite rule yields top/pots' roti (38), tip/pits' roti (38), and draw/wards' roti (40). These use plural owner morphology at the new right seam. Their readability differs, and the awkward cases remain retained rather than being promoted by exactness alone. The original human-positive seed survives as a baseline.

## Bounded construction and parse audit

Predeclared at source commit b9dd89de: 14 palindromic phrase components across editable radii 0–9 (140 proposals), plus 13 frozen food/documentation action lemmas tested as a nested s + verb + reverse(verb) + s center at radius1. The word-pair method queries existing frozen Brown POS and dictionary support before full seam recognition. The deliberately symmetric tapes are then recognized by a declared finite grammar for the seed: plural editor predicates, transitive roti objects, proper-name or plural-possessor NPs, memo arguments, and separately licensed rot/I clause splits. Singular Ed/see is a shortening diagnostic. It is not a general English parser.

The documentation scenario, owner classes (person/container/location/institution), lexical actions and parse interfaces were fixed before the bounded run. Several positive fixtures were discovered during prototype design; the subsequent finite enumeration is an audit of that designed family, not held-out or unbiased creativity evidence. Dictionary/POS membership does not independently certify our semantic roles. The constructive food-documentation interpretation is declared, not entailed by the short wording.

Actual resource use:153 proposals,30-second CPU cap,256-attempt cap; complete in0.036986 CPU seconds. All153 normalized tapes are exact by symmetry, but only8 have a licensed full-text parse. The145 other proposals comprise137 finite-grammar misses and8 unsupported lexical pairs. Each attempt, failure and parse is retained. A grammar miss does not prove that no broader English parse exists. The lexically supported read/daer pair, for example, has no owner role in this frozen grammar. No paid API, HPC, GPU or remote job was invoked; no speedup claim.

## Individual owner review

| Text | Letters | Grammar | Readability | Coherence | Overall |
| --- | --- | --- | --- | --- | --- |
| Some men, if editors tap Pat's roti, define memos. |38|5|4|3.5|5.5/10|
| Some men, if editors top pots' roti, define memos. |38|5|3.5|3|5/10|
| Some men, if editors draw wards' roti, define memos. |40|5|3.5|3|5/10|
| Some men, if editors tip pits' roti, define memos. |38|5|2.5|2|4/10|
| Some men, if editors draw awards, rot; I define memos. |41|5|3|2.5|4.5/10|
| Some men, if editors live on no evils, rot; I define memos. |44|5|2.5|2|4/10|
| Some men, if Ed sees, define memos. |26|4.5|3.5|2.5|4/10|

These are owner model scores under the frozen 019 rubric, not human scores or labels. The tap/Pat candidate adds a concrete referent and clearer physical action, yet the memo connection still needs the declared documentation context. Container/institution possession is more awkward; tip/pits is particularly strained. The41/44-letter cases resegment roti as rot + I and introduce a speaker clause; their extra length does not make the conditional relation cohesive. Ed/sees is shorter and receives no length-progress credit.

The seed's historical022 model score remains4/2.5/1.5 and3/10 only as historical evidence. The user's subsequent explicit positive label takes precedence for taste; it is not a current rejection. No new candidate is called human BAD or accepted without feedback. Soft repetition cost remains a reported feature, not a veto. All8 parsed outputs have zero token repetition.

## Sampling, originality scope and independent checks

Sampling ceil(10% per method/radius stratum) selects3 of8 licensed occurrences:one of3 phrase-center radius0 outputs;one of1 radius5 output;one of4 nested word-pair outputs. Overall37.5% after rounding, not literal10%. Each sampled text is individually reviewed in sample-reviews.json, and all8 unique texts have owner review records. The seven nonseed texts are distinct from the seed and absent from a limited cache consisting of v3_bank, the prior positive human-preference labels and022 conditional-followup summary; this is not world-originality evidence. The raw unique_new_texts field means nonseed, as interpretation.json clarifies.

Independent rendered-letter regex and two-pointer checks verify all8 full texts;153 deliberately palindromic tapes are separately verified. Twenty-two focused center/generator/compiler/engine/endpoint/grammar tests passed in1.01 seconds. Tests demonstrate both-seam morphology and that exact insertions can still fail argument/agreement licensing, including reward a drawer, step on no pets and Don't nod when inserted without correct seam treatment.

## Review provenance and deliverables

A separate proposal-stage model review reported no useful richer ordinary-English center, only doubled-clause/wrapper controls, and rejected the sass variant as no gain. Those cases were not repeated. proposal-stage-review.json preserves that reported scope; no raw receipt or independent ratings for the current new seam candidates were supplied. review-sample-export.json provides three new modest candidates (tap/Pat, top/pots, draw/wards) for independent review. These are fresh predicate/object seam parses, not copied whole-clause wrapper controls.

The finite positive finding is that a palindromic center replacement can use the unchanged mirrored right overhang to form possessive morphology while preserving the left plural subject. Equivalent radius-one wrapping is not counted as additional progress. The unresolved goal is a richer long cohesive paragraph: current gains are only3–5 letters and one word in the food arm. Source, tests, bounds, all raw proposals, failed parses, output derivations, model/sample reviews and safe checkpoint are preserved. Main, merged paper, prior receipts and exact human labels remain unchanged.

A separately reported exactness check confirmed the tap/Pat candidate as38 letters and exact. This is verification evidence, not new human approval. minimal-center-clarification.json preserves the equivalent minimal and expanded decompositions for all four possessive candidates.
