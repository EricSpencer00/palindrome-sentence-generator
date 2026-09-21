# Construction queue: 2026-09-21

This ledger records the fresh search-space operators added after the live API
audit. A row is not a reader claim: only an independently exact, novel,
intact-prose output can enter the reader package.

| operator | evidence | result | concrete next construction |
|---|---|---|---|
| Boundary-geometry crossword | `experiments/boundary_geometry_crossword_20260921.py`; `runs/boundary-geometry-crossword-20260921.json`; commit `0a5a7ebd` | Six fixed 41/43/47-letter graphs; 0 exact; two intact prose controls; all six fail lexical propagation with explicit overlap witnesses. | Freeze geometries from supported grammatical endpoint/center overlaps, not random role lengths. |
| Authored geometry + role-aware domains | `experiments/authored_geometry_domain_search_20260921.py`; `runs/authored-geometry-domain-search-20260921.json`; commit `176e457c` | Two intact-prose geometries (39, 41 letters); 0 exact; role pools still have singleton/empty domains. | Add a larger independently authored role-annotated prose bank while preserving source geometry. |
| Full-slot center-aware CSP + agreement/valency | `experiments/center_aware_slot_csp_20260921.py`; `runs/center-aware-slot-csp-20260921.json`; commits `f4b13c34`, `1c0d2ff6` | Six complete templates; 19 states; the 38-letter seed survives as a control; 0 novel exact >38. | Expand valency-compatible slot families without returning to clause-pair joins. |
| CFG forest character trie DP | `experiments/english_cfg_forest_char_trie_dp_20260921.py`; `runs/english-cfg-forest-char-trie-dp-20260921.json`; commit `ad43dff3` | 108 generated CFG sentences, 11,664 forest pairs; all live-pruned; 0 exact. | Add dependency/role-conditioned templates rather than a larger undifferentiated forest. |
| Dependency-template recombination | `experiments/dependency_template_recombination_20260921.py`; `runs/dependency-template-recombination-20260921.json`; commits `a053b4b2`, `21f22583`, `f9d64e85` | 24 extracted frames; 487,872 six-clause recombinations; maximum 91 letters; 0 exact. Boundary comparison is retained as a post-render diagnostic, not called online decoding. | Replace the diagnostic with a true left/right role-emission queue carrying typed valency and live character domains. |

Every row has an independent normalized pointer scan and forward/reverse
SHA-256 audit. No row claims human readability; the blinded intact-versus-
shuffled reader gate remains closed until a novel exact output exists.
