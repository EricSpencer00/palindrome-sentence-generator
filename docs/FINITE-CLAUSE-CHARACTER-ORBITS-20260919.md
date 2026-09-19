# Finite SVO character-orbit search — 2026-09-19

This lane is a bounded construction diagnostic for the 39–60-letter target.
It compiles two independent, hand-authored finite languages of complete
subject–finite-verb–object clauses into character tries.  At each live product
transition it assigns one mirrored character orbit: the next character from
the left clause and the next character from the final end of the right clause.
The right clause is still rendered in ordinary order.  A closure is possible
only when both tries are complete SVO terminals; partial fragments cannot be
promoted.

The novelty survey rejected the adjacent `typed-clause-zipper-20260919`
whole-word-debt lane and the `typed-word-boundary-clause-automaton-20260918`
lane as duplicates.  This experiment changes the state representation to a
character-trie product with explicit orbit assignments.  It does not use
reversible lexical-pair tables, word-order symmetry, catalogue text,
finished-tape reversal, or RLAIF.

The bounded run was executed on the remote `hst-bench` host with:

    cd /home/eric/cloud/palindrome-sentence-generator
    python3 experiments/finite_clause_character_orbits_20260919.py

Remote run SHA-256: `6160fe349e0485ef14ff19bf7e35c284b105ee2d7aadc184e87b2fce2292a7a9`.
The local replay has the same construction and audit results and additionally
loaded two prior exact tapes for duplicate preflight.

Results:

| measure | value |
| --- | ---: |
| expanded orbit states | 8 |
| matched live orbit transitions | 7 |
| rejected live orbit transitions | 5 |
| complete-SVO exact closures | 0 |
| mechanically admitted candidates | 0 |
| intact controls | 3 |
| control length range | 40–48 letters |

The retained intact controls are:

* “An aide writes nine memos; Some men inspire Diana.” — 40 letters; first
  orbit failure 6; independent pointer and SHA audits agree that it is not
  exact.
* “A patient keeper guards charts; The baker records a sonnet.” — 48 letters;
  first orbit failure 0.
* “A careful pilot maps rivers; A singer carries parcels.” — 44 letters; first
  orbit failure 0.

These are complete finite SVO clauses, but they are controls, not palindrome
claims and not reader evidence.  The next repair is one held-out subject or
object noun bundle chosen against the first live orbit frontier while keeping
the finite SVO terminal gate, ordinary rendering order, and both independent
audits unchanged.

Rerun with:

    python experiments/finite_clause_character_orbits_20260919.py
