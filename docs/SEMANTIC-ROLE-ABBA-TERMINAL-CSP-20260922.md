# Semantic-role ABBA terminal CSP (2026-09-22)

This lane treats a paragraph as an A/B/C/D discourse scaffold: an authored
museum-restoration scene, a question, its answer, and a return scene. ABBA is
only a semantic order; admission still requires one global letter-level
palindrome.

Unlike fixed-bank or endpoint-only probes, the solver first selects complete
typed A and D scenes jointly. It requires equal tape lengths and mirrors every
character of the two outer scenes before it instantiates the B/C question and
answer roles. Each rendered row receives an independent two-pointer audit and
forward/reverse SHA-256 replay, plus provenance and shortcut gates.

Run: `python3 experiments/semantic_role_abba_terminal_csp_20260922.py`

The fresh museum-restoration lattice produced 0 compatible A/D pairs, so no
interior realization was admitted and no exact candidate exists. This is a
constructive failure: the outer role domains are the blocking state, not a
claim that readable paragraphs are impossible. The next repair is a held-out
museum scene with a complete mirrored A/D tape, followed by live B/C residual
realization; merely enlarging the same bank is rejected as non-progress.
