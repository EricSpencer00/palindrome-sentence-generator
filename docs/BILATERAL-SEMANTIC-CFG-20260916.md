# Bilateral semantic nonterminal CFG (2026-09-16)

This lane expands three paired semantic nonterminals. Each side is a separately authored ordinary-order S–V–O realization; the constructor carries a live character frontier while the pair is expanded. It does not reverse-decode a tape, resegment an existing palindrome, or replay a parse-tree exact-cover solution.

The generated candidate is 154 letters after normalization:

> the quiet clerk records the parcel. the courier carries the letter. the young baker mixes the dough. the patient guard checks the seal. the keeper opens the gate. the old sailor mends the sail

It is intact six-clause prose, but it is not an exact palindrome. Independent direct comparison, two-pointer comparison, and forward/reverse SHA-256 all agree on rejection. The run artifact preserves the complete pair meanings, frontier trace, generator hash, and mechanical admission result.

The held-out repair substitutes `lock` for `gate` in one semantic object slot and recomputes both sides; it does not copy or reverse characters. The concrete next repair is to expand held-out object domains jointly with verb valency while retaining the live frontier and complete-clause gate.
