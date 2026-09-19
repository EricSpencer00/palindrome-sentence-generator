# Polar-question token-boundary CSP (2026-09-18)

This is the next repair after the phrase-level boundary graph. The answer is
not committed as one phrase: the graph independently chooses a fronted object,
subject, and verb, or a normal subject/verb/object path, while matching
incoming and outgoing characters at every edge. The 44-letter discourse path is
still present only as a diagnostic control.

## Result

The graph contains 11,171 lexical states and 3,118 reachable pair states. It
produced one independently exact row:

> Was Noel an era, a gas, an item? Met in a, saga, arena, Leon saw.

That row is unchanged, fails the proper-palindromic-subspan gate, and is
fragmentary. A second reconstructed path, `Was Eva a gas? a saga saw Ave.`,
was retained as a non-exact reconstruction diagnostic and was not exposed as a
candidate. After the independent audit there are 0 new exact closures, 0
mechanically admitted rows, and 0 reader-eligible rows.

## Concrete next repair

The next graph change must carry a finite predicate complement across the first
dead token-boundary pair, preserving subject/verb/object state and lexical
history. The non-exact reconstruction is a regression fixture: no path may be
promoted without the independent two-pointer and SHA checks.

Human readability remains unmeasured; programmatic checks diagnose and filter,
but never certify English prose.
