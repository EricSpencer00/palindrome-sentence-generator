# Compound derivational scene CSP (2026-09-16)

This lane maps to the compound/semantic-scene family: lexical choices carry an
explicit modifier–head decomposition (`rain+coat`, `sun+flower`,
`light+house`) and a scene role, while suffixes such as `-er`, `-or`, `-ful`,
and `-ly` carry derivational meaning. It is orthogonal to the retained
inflectional/clitic FST lanes because the state is a compound's internal
semantic segmentation, not a new agreement or clitic surface form.

The preflight read 193 registry entries and found no non-self signature or
artifact collision. The run generated 80 complete ordinary-order clause-pair
probes (all at least 39 letters). None closed exactly; the longest intact
diagnostic prose was:

> the gardener carries the lighthouse carefully; the raincoat carries carefully our gardener

Each probe includes an independent two-pointer comparison, SHA-256 forward /
reverse tape audit, and the project's mechanical admission checks. Provenance
records the generator digest and states that all compounds and scene frames
were authored for this run, with no catalogue lookup.

The held-out repair operator is concrete: at the first mirrored mismatch,
replace only the right compound slot with a different modifier–head compound
while preserving its scene role and derivational path. This is recorded for
every probe in the run artifact; it is a repair proposal, not a claimed exact
closure.
