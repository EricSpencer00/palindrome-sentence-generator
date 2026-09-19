# Exhaustive grammar reachability

This architecture compiles typed subject/verb/object choices into a character
automaton. A pair of states tracks simultaneous consumption from the beginning
and end of one grammatical text. Each transition consumes equal letters; equal
states or a connecting edge close the midpoint. Reachability is exhaustive over
the finite pair graph, with no length bound or beam. A productive-cycle test
decides whether the grammar admits arbitrarily long exact tapes. It does not
certify novelty, avoidance of repetition, or human readability.

Unlike the existing `typed_grammar_character_nfa_20260917.py` prototype, lexical
path identity is retained between characters. That older prototype selects a
fresh lexical alternative at each offset; this implementation has a regression
test against inventing `aba` from the inventory `abc`, `xba`.

| Grammar | Character states | Reachable pairs | Productive pairs | Nonempty exact language |
|---|---:|---:|---:|---|
| Typed core | 99 | 49 | 37 | Yes; seed calibration |
| Past tense and pronoun repair | 147 | 61 | 37 | Yes; same calibration |
| Repair with seed object withheld | 139 | 33 | 1 | No, at any length |

The only shortest witnesses are “an aide rips nine memos; some men inspire
Diana.” and its clause-order rearrangement. Each is 38 letters. They are seed
calibration, not original results. Forward and reverse SHA-256 for the first:
`ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6`.
Independent pointer comparison also passes. The unrestricted grammar has
productive cycles because repetition is possible; this is explicitly rejected
as a route to the requested output.

Fresh intact control: “A reader files some notes; the aides admire Nora.”
(39 letters, not exact). No reader evidence exists for this run.

The concrete next construction repair is to use the dead pair frontier to
request new role-compatible lexical paths spanning multiple word boundaries.
The evidence says that extending runtime or maximum length on this grammar
cannot produce a solution after withholding the seed object. New grammar
paths are required. The past-tense/pronoun repair was run immediately and did
not add productive pairs; do not repeat it as a larger sweep.

Reproduce with `python3 experiments/palindrome_grammar_reachability_20260918.py`.
Three tests compare against independent finite enumeration, reject lexical path
switching, and check the seed-withheld all-length result.
