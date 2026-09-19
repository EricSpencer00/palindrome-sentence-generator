# Weighted grammar/CSP probe (2026-09-18)

This bounded construction test uses a typed weighted SVO grammar (`DET SUBJ
VERB OBJ`) and an online mirrored-edge character constraint. Left and right
lexical choices are paired while grammar slots are expanded; incompatible
boundary characters are rejected before a complete sentence exists. It does
not materialize completed sentence products, reverse-decode a finished tape, or
use RLAIF. A candidate is admitted only after an independent normalized
two-pointer audit and forward/reverse SHA-256 comparison.

Replay:

```text
python experiments/weighted_grammar_csp_probe_20260918.py
```

The run expanded 2 chart states under a 2,000-state budget, rejected 12
lexical pairings at the first mirrored character, and produced 0 exact paths.
No readability claim is made. Provenance and the complete output are in
`runs/weighted-grammar-csp-probe-20260918.json`.

The immediate repair is to carry a residual character offset through each
lexeme (rather than comparing only the first/last boundary) and to add a
small, independently sourced boundary inventory for each role. The next run
must preserve typed valency and retain the same exact audit; increasing the
beam alone is not evidence of scalability.
