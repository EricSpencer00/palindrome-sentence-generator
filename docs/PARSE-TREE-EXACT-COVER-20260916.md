# Parse-tree exact-cover solver (2026-09-16)

This lane starts from independently authored constituency trees. Their terminal
spans become exact-cover columns (`side × semantic role` plus agreement), and a
bounded search assigns lexical realizations while carrying word-boundary and
bilateral character obligations. It does not replay dependency seams, Earley,
CFG intersection, or min-cost-flow state.

The preflight read the live novelty registry and found no exact signature
collision. The run generated complete ordinary-order probes, each at least 39
letters, and recorded terminal-span provenance. Independent two-pointer,
hash, and mechanical audits were run in addition to the exact tape check.

The base run produced 24 complete candidates (24 at 39+ letters) and the
held-out number/tree lexical repair produced 8 more. No candidate closed the
mirror equation. The repair changes a held-out tree agreement/lexical index at
the first mismatch; it never resegments or reverse-emits a tape.

Artifact: [`parse_tree_exact_cover_20260916.py`](../experiments/parse_tree_exact_cover_20260916.py)

Evidence: [`parse-tree-exact-cover-20260916.json`](../runs/parse-tree-exact-cover-20260916.json)
