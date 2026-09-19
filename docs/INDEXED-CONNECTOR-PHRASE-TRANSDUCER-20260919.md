# Indexed connector phrase transducer (2026-09-19)

This is a new constructive lane, not a larger sweep of the prior phrase product.
It carries the unmatched normalized residual between transitions and indexes an
authored connector bank by the character needed at the residual boundary. Each
transition therefore changes which grammatical scene connector can be tried.
No completed candidate is reversed, and no catalogue sentence or language-model
reward is used.

## Result

The run produced 640 rendered clause candidates across four bounded depths and
256 retained residual states. The longest rendered candidate is 92 letters:

> The patient astronomer measures the western shore; a small boat drifts, the courier waits before the quiet bell.

Its independent outside-in audit is not exact: the first mismatch is normalized
position 0 (`t`) versus 84 (`l`). The run found 0 exact closures, so this lane is
not reader-eligible. The first transition already leaves residual
`smidnretnala`; the concrete next repair is to add a role-compatible connector
indexed by that observed residual's closing character, then repeat the same
bounded transition. This preserves a changed construction method rather than
restarting the failed Cartesian sweep.

The complete machine-readable ledger is in
`runs/indexed-connector-phrase-transducer-20260919.json`. It records every
rendered candidate retained by the ledger, chunks, transition key, residual,
provenance, shortcut flags, and two independent exact checks (outside-in
two-pointer comparison and forward/reverse SHA-256 equality). The generator is
reproducible with:

```text
python3 experiments/indexed_connector_phrase_transducer_20260919.py
```

The phrase banks are small authored role phrases. They are construction inputs,
not borrowed palindrome text. Programmatic scores are diagnostic only; no human
readability claim is made for this failed closure lane.
