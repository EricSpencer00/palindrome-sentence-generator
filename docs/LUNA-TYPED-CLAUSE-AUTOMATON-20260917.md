# Luna typed-clause automaton (2026-09-17)

This lane uses a hand-authored typed clause grammar rather than corpus-shaped
sentence plans. Each side independently chooses a determiner/subject,
subject-agreeing transitive verb, determiner/object, and adjunct. Subject
number, verb valency, and object number are carried in the automaton state.
After every slot pair, the newly chosen left tape and reversed right tape are
compared immediately; incompatible character obligations are discarded before
later slots expand.

The run expanded the first slot and rejected all 1,764 lexical pairings at the
first character obligation. It produced zero terminal tapes and therefore no
rendered candidate or reader-eligible output. This is a concrete exclusion,
not evidence that readable palindromes are impossible: the current outer
lexical domains have no compatible first seam. The next repair is to introduce
an explicit boundary-conditioned lexicon (words grouped by first/last
character and grammatical role), while retaining agreement and live debt, not
to widen this same inventory or run another duplicate beam.

Run artifact: `runs/luna-typed-clause-automaton-20260917.json`.
