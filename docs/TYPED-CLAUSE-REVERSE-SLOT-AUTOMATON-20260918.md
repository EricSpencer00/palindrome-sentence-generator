# Reverse-slot typed clause automaton (2026-09-18)

This is a new repair of the failed same-slot typed automaton.  The left
clause is expanded in grammatical order (`subject`, `verb`, `object`), while
the independently authored right clause is expanded from its outer edge in
reverse grammatical order (`object`, `verb`, `subject`).  Agreement features
are carried when the verb is selected before its subject on the right.  The
live state compares the character stream from each outside edge and retains
only the unmatched residual.

The first run used 11 complete adjuncts chosen to meet the left determiner
boundary, plus typed subject/verb/object inventories.  It still produced zero
terminal exact tapes: 462 character obligations conflict before the first
slot pair can be completed.  There is no rendered candidate to promote and no
reader evidence.  This is a distinct structural exclusion, not a vocabulary
or beam sweep.

The next repair is to expand each clause phrase into independently typed word
tokens so a residual can cross a phrase boundary (as in the 38-letter seed),
while retaining reverse grammatical slot order and agreement.  The current
run is preserved as evidence and is not presented as progress toward
readability.

Run artifact: `runs/typed-clause-reverse-slot-automaton-20260918.json`.
