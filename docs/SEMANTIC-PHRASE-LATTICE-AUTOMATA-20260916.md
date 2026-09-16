# Semantic phrase-lattice automata preflight (2026-09-16)

This bounded probe builds two independent, complete semantic phrase lattices. Each weighted finite-state path emits ordinary prose in normal word order. An online product consumes the left character yield against the reverse residual of the independent right yield; it never reverses words, wraps a known palindrome, or repeats a unit.

Novelty preflight inspected 187 registry entries and failed closed: `corpus-sentence-gram-fst`, `wordnet-synonym-frame-csp-20260915`, and related phrase-lattice/FST families already cover this method. The run is therefore diagnostic, not a retained lane.

The 64 cross-product candidates are complete garden prose (the first is “At dawn, the patient gardener waters a young fig tree; and records its first green leaf; while the quiet birds cross the yard.”), with 100 letters per side. Online character intersection produced 0 closures; the first candidate matched 0 leading residual characters. Independent two-pointer and SHA-256 forward/reverse audits agree on every side (all non-palindromic). Mechanical admission is false because no exact closure exists; readability is a mechanical complete-prose diagnostic, not human evidence.

Artifact: `experiments/semantic_phrase_lattice_automata_20260916.py`  
Evidence: `runs/semantic-phrase-lattice-automata-20260916.json`

Next repair if the family is reopened: replace one held-out semantic end-phrase arc in each independent lattice and rerun the complete online product, preserving ordinary word order and the two exact audits.
