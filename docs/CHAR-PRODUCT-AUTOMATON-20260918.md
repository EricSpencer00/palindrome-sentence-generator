# Character product automaton (18 September 2026)

This bounded experiment tests a new search geometry: a typed clause DFA is
crossed with a live character-obligation machine. The left and right clauses
advance through *heterogeneous* grammatical slots, while each newly emitted
character is compared immediately with the mirrored obligation. This differs
from same-slot residual decoding: the product state carries independent DFA
positions, token histories, residual characters, and a transparent character
bigram prior.

Command:

```text
python3 experiments/char_product_automaton_20260918.py \
  --out runs/char-product-automaton-20260918.json
```

The 100,000-state cap was not reached. After depth one only 2 states survived;
depth two reached 0 (1,370 character conflicts), yielding 0 exact closures and
0 candidate rows. The run nevertheless stores two intact English controls and
independent two-pointer plus forward/reverse SHA-256 audits. The controls are
not claimed as palindromes or readability evidence. No catalogue text was
imported and no finished tape was reversed. A real closure must additionally
pass the repository novelty preflight against `data/known_palindromes.json`.

The scaling repair is structural: replace the tiny slot vocabulary with a
held-out character trie of inflected phrase tokens and add an explicit center
closure state. This preserves the product geometry while preventing the
current sparse lexicon from collapsing at the second transition.
