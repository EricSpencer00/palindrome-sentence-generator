# Typed word-boundary clause automaton (2026-09-18)

This lane repairs the whole-phrase failure by expanding two independently
typed clauses one lexical token at a time.  The left clause is built in
subject–verb–object order; the right clause is built from its outer edge in
reverse order.  When a token is shorter than the live character obligation,
the next token on the other side is allowed to continue the same residual.
Agreement and optional adjunct presence remain part of the state.

The run expanded 1,856 live lexical witnesses at its widest point and found
82 exact terminal tapes.  Two are retained as smoke controls for the known
38-letter seed (including its clause-order rearrangement).  The other 80 are
short proper-name/``asks`` constructions of 22–26 letters; all fail the
length, uniqueness, or proper-span gates.  Therefore:

- independent exact terminals: 82;
- exact seed controls: 2;
- new exact terminals: 80;
- mechanically admitted new candidates: 0;
- reader-eligible candidates: 0.

The important positive result is algorithmic: the live residual now crosses
determiner, noun, and verb boundaries, and the seed is recovered without a
finished-tape reversal.  The important negative result is that the current
lexical inventory collapses into short name loops once the seed is excluded.
The next repair is a hand-authored event-frame inventory with semantic
valency and a live uniqueness/proper-span filter before closure; simply adding
more names or more seeds is explicitly excluded.

Run artifact: `runs/typed-word-boundary-clause-automaton-20260918.json`.
