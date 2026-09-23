# Locative inversion plus active SVO: bounded seam probe

This experiment loaded the pinned 568-letter parent
`runs/incumbent-560-outer-causal-scene-20261002.json` (normalized SHA-256
`6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380`) and
targeted its clean replacement seam `[148,163]` / `[405,420]`. It pre-registered
complementary word-boundary masks, then compared an inverted locative event
with a transitive SVO observation. No child was admitted.

The five bounded clause-pair probes were:

| Left clause | Right clause | Matched letters | First mismatch |
|---|---|---:|---|
| `Among the reeds rose a heron.` | `The tern soars over islands.` | 0 | `a / s` at 0 |
| `At dawn stood one watchful guard.` | `Rangers read current field data.` | 2 | `d / a` at 2 |
| `At a dam stood one watchful guard.` | `Rangers read current field data.` | 4 | `a / d` at 4 |
| `At a dam stood one watchful guard.` | `Rangers read current magma data.` | 6 | `s / g` at 6 |
| `At a dam stood one watchful guard.` | `Rangers read current llama data.` | 6 | `s / a` at 6 |

The best local control therefore stops with residuals
`stoodonewatchfulguard` versus `gamtnerrucdaersregnar`; `llama` changes the
right residual prefix to `allam...` but still disagrees at the same cursor.
The two 27-letter clauses would have yielded a 592-letter parent replacement
if they had closed. They did not: there are zero exact local equations, zero
children, and no readability or reader claim. The wetland control is also
retired: the 5-letter animate subject would have to reverse to a 3-letter
determiner, with no ordinary grammatical slot filler found.

The mask repair `dawn → a dam` and the one-slot substitutions `field → magma`
and `field → llama` are preserved as residual-directed attempts, not as
readability improvements. Phrase-bank expansion within this topology is
retired at cursor 6. The run artifact records rendered controls, normalized
tapes, word masks, exact residuals, parent provenance, and the zero-child
result. The source phrases were checked against the tracked corpus before
registration; no prior exact phrase collision was found.

The next constructive action is not another modifier sweep. The repository
also contains longer exact descendants, including a 626-letter comparison
with SHA-256
`96c9e136da7a8a4f78a469f8ac2f701238ccc89a674499ff6ec1cce920c2a234`; that
text has its own insertion/readability debt and is not the active parent under
the current user directive. Continue from a different actual seam of the
pinned 568 tape, while preserving those descendants as comparison artifacts
and retaining the 560/558/556 frontier unchanged.
