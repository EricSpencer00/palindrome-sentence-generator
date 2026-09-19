# Role-labeled grammar CSP (2026-09-18)

This lane kept subject, finite transitive verb, object, and adjunct roles live
on both sides of a two-clause construction while checking opposite characters
before any candidate could be admitted. Its inventory is fresh and
hand-authored, independent of the semantic-relay phrase bank.

## Result

The four complete-clause candidates range from 73 to 76 letters. The longest
representative is:

> **The singer carries the compass near the orchard; one sailor guides a basket beside the river.**

Its independent two-pointer audit is false at the outer character (`t` versus
`r`); forward and reverse SHA-256 values also differ. The strict lexical,
provenance, repeated-unit, word-order, and hidden-span checks pass, but exact
closure fails. The lane therefore produced 0 exact closures, 0 strict
admissions, and no reader-eligible output.

The next repair is a typed adjunct substitution that preserves subject/verb/
object valency while carrying the residual farther inward.

Reproduce with `experiments/role_labeled_grammar_csp_20260918.py`; the saved
artifact is `artifacts/role-labeled-grammar-csp-20260918.json`.
