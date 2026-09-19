# Reversible predicate center extension (2026-09-18)

This lane tested one narrow construction operator: insert a small, authored
reversible predicate/answer pair at the live center of the 44-letter
polar-question diagnostic. It is not a catalogue import, word-order mirror,
or broad lexical sweep. The pair is inserted before the question mark and at
the beginning of the answer, then audited independently on the normalized
letter tape.

## Actual outputs

The longest exact diagnostic is:

> **Was Noel an era, a gas, an item smart? Trams met in a, saga, arena, Leon saw.**

It has 54 letters and passes the two-pointer and forward/reverse SHA-256
checks. The more natural-looking member of the same construction is:

> **Was Noel an era, a gas, an item raw? War met in a, saga, arena, Leon saw.**

It has 50 letters and is also exactly palindromic.

## Admission result

All five authored pair probes close exactly (50--54 letters), but all five
fail the strict mechanical gate because the inserted word pair itself is a
proper multiword palindrome (`raw war`, `smart trams`, `mad dam`, `live evil`,
or `stop pots`). Their answer remains fragmentary as well. The independent
audit therefore records 5 exact diagnostics, 0 mechanically admitted rows,
and 0 reader-eligible rows. No readability claim is made and no reader study
is justified for these rows.

The result is still useful: it demonstrates that length can be increased by a
center operation, while exposing the exact shortcut that must be removed. The
next repair replaces the direct reversible pair with a complete finite
predicate complement whose semantic arguments cross the same boundary without
creating a self-palindromic span.
