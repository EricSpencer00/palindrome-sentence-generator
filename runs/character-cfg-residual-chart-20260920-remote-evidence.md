# Character CFG residual chart — remote evidence

Decision: does a character-level residual chart over complete CFG constituents
produce a clean exact palindrome in this grammar envelope?

Acceptance gate: at least one exact candidate passing two-pointer comparison,
forward/reverse SHA-256 equality, and the hard exclusions; otherwise the
grammar envelope is killed and the next construction is recorded in the JSON.

Fixed controls: 256 independently expanded complete `S -> NP VP PP`
derivations, all 65,536 ordered pairs, no reversal/editing of rendered text,
and the same audit on every row. The independent prose controls are the
non-palindromic complete derivations retained in the JSON.

Host/toolchain: `hst-bench`, Python 3, command
`python3 /home/eric/character_cfg_residual_chart_20260920.py`.

Observed output:

```text
{"grammar_derivations": 256, "chart_pairs": 65536, "character_steps": 69632,
 "closed_pairs": 0, "exact_clean": 0, "max_letters": 49}
```

Verdict: killed for the current grammar envelope (zero closed pairs), not a
readability claim. Next repair is nullable complement and relative-clause
productions while preserving independent residual chart state.
