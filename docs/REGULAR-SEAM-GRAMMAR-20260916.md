# Length-indexed regular seam grammar (2026-09-16)

This experiment tests a fresh scalable construction state: a finite-state
grammar, not a recursive tree or a reverse sentence lookup. Each side emits
an independently typed clause. A small adjunct cycle (`q0 -> q1 -> q2`) makes
length a first-class state, and an ordinary connective (`while`, `because`, or
`although`) joins the clauses. The connective is deliberately a
non-palindromic seam unit; the audit treats its letters as part of the global
character ledger rather than smuggling in a preassembled palindrome.

The repair operator is concrete: when a target surface does not close, it
greedily tries typed subject, predicate, and object substitutions, retaining a
complete compound sentence after every edit and accepting only a lower
two-pointer mismatch count. Content words remain distinct and no catalogue
text is imported.

## Replay

```text
python experiments/regular_seam_grammar_20260916.py
```

The run enumerated 11,997 fresh grammatical surfaces and instantiated three
prose outputs at 95, 105, and 115 letters. All three are complete ordinary
sentences, and all three exceed the 38-letter floor. The finite-state bank had
zero exact closures; the best repair reduced the mismatch count from 44 to 39
on the selected surfaces. This is retained as a useful scalability and repair
diagnostic, not as a readable-palindrome claim.

The run records an independent indexed two-pointer validator, seam diagnostics,
mechanical admission results, and a permanently closed reader gate. A future
exact result requires expanding the typed lexical inventory or seam choices,
then passing the same exact and blinded-reader checks; increasing the target
length does not change the construction algorithm.
