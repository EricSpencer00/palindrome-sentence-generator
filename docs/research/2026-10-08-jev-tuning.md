# Jev parameter advice — 2026-10-08

Eric requested Jev tuning. An automatic approval review rejected the detailed
private experiment summary. The executed request contains only general search
principles and hypothetical parameter choices, without private results, texts,
or source. The request and raw reply are preserved in
`runs/jev-tuning-20261008/`.

The pinned response reports `jev-1.13.0`:

| Question | Selected option | Reported confidence |
|---|---|---:|
| Search strategy | Grammar-preserving residual edits | 0.93 |
| Development penalty grid | Paired 0, 1, 2, 4 | 0.94 |
| Readability validation | Held-out blinded humans | 1.00 |
| Word cap | Prove feasibility first | 1.00 |

These are choices among author-supplied options. Their probabilities are not
empirical performance estimates or evidence that these values are optimal.
The model did not inspect the experiment results or author parameter values.
The numeric sweep is development only, with a zero-penalty baseline and new
seeds. Its outcomes cannot validate reader quality. Jev is a structured-choice
advisor, not an optimizer that executed a search.

Official request contract: https://docs.typesafe.ai/primitives/choice

## Executed development follow-up

The32-job paired sweep completed on the NUC with fresh seeds9100/9101 and
feasible cap256. Every strength returned4/8 exact outputs; all16 selected
texts failed the strict screen. The longest observed candidate has824 letters
at strength1, but this tiny, selected set does not justify changing defaults
or calling that strength optimal. Full evidence and separately archived setup
failures: `runs/fragment-penalty-tuning-20261008/REPORT.md`. Root independently
reproduced every selected tape reversal and complete admission map. No Brown,
grammar-model or human evaluation was used in this sweep.
