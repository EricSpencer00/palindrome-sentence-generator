# Character-paired grammar search — 2026-09-19

This is a small, fresh experiment. Each side is typed from a lexical state
(`subject`, `verb`, `object`, or `adverb`); the next character is checked against
the outstanding character obligation immediately. Spaces and punctuation are
not obligations, so word boundaries may differ.

The rendered controls were `Live on time, emit no evil`, `Live on time, emit
no evil; live`, and `Live on time, no evil, emit no evil`. The first is a
readable exact 20-letter control, but none is a 39+ letter result. The longest
candidate is 26 letters and fails at its first recorded mismatch. No complete
clause was reversed, no catalogue text or RLAIF was used, and no repeated
self-palindromic unit was admitted.

The independent audit uses a fresh two-pointer comparison and SHA-256 of the
normalized forward and reverse strings. Results, per-character states,
provenance, novelty preflight, and the first failure are in
`runs/character-paired-grammar-20260919.json`; rerun with:

    python experiments/character_paired_grammar_20260919.py

Stats: 3 bounded templates, 30 online character states, 1 exact control,
longest 26 letters. The next repair is one newly authored transitive-event
state targeting the failed boundary while preserving finite-verb agreement.
This run is not reader-eligible and does not claim a new >38-letter solution.
