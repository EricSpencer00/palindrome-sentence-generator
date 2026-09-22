# Reverse-compatible SVO transition beam (2026-09-16)

This route mines a broad lexical transition bank from the shipped frequency
bigrams, keyed by reverse-edge character compatibility. Independent left and
right lexical banks are then realized as complete
`DET SUBJ VERB DET ADJ OBJ PREP OBJ` clauses using a deterministic
frequency-scored beam/DP. Complete clause tapes are joined through an exact
reverse index and passed through the shared mechanical admission gate.

Novelty preflight found no exact signature collision against the 107 prior
registry entries. The nearest prior was `neural-dual-prefix-beam-v2` at
Jaccard 0.12; the new state is mined reverse-compatible transitions followed
by bilateral full-clause beam/DP realization.

The run considered 6,000 lexical bigrams (61 transition keys), retained 256
complete clauses per side, and rendered three complete probes of 63–65 letters.
All probes were non-palindromic; there were zero exact closures and zero
mechanically admitted rows. The concrete held-out repair is to expand only
lexical alternatives keyed by the first reverse-edge mismatches in these full
probes, preserving the same grammar and exact join.

Artifact: [`experiments/reverse_transition_svo_beam_20260916.py`](../experiments/reverse_transition_svo_beam_20260916.py)

Evidence: [`reverse-transition-svo-beam-20260916.json`](../runs/reverse-transition-svo-beam-20260916.json)
