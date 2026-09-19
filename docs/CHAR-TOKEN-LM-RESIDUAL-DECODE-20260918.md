# Character/token LM residual decoding (18 September 2026)

This lane tests a distinct construction idea: choose independent grammatical
tokens on both sides while emitting their characters jointly from the outer
edges. A transparent add-one character-bigram prior ranks states, but it is
not a readability certificate. Every character conflict is rejected before a
completed sentence exists; the finished-tape reversal shortcut is disabled.

The bounded run used five typed slots (`DET`, `SUBJ`, `VERB`, `OBJ`, `ADV`),
50,000 states, and 64 character-obligation conflicts. It reached zero states
after the second slot, so it produced **zero exact closures** and **zero
mechanically admitted candidates**. The output still records two intact prose
controls, their provenance, and independent audits. This is a useful failure:
the same-slot pairing is too restrictive, and is not the target method.

Run:

```text
python3 experiments/char_token_lm_residual_decode_20260918.py \
  --out runs/char-token-lm-residual-decode-20260918.json
```

The next repair is to retain live residuals while allowing heterogeneous slot
pairs and a character trie over inflected phrase tokens, then evaluate intact
prose controls in randomized blinded order. Human readers—not the LM score—
must decide whether any surviving output reads as English.
