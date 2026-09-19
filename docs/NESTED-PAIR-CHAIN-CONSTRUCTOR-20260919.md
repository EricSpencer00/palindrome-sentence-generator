# Nested mirror-pair chain constructor (2026-09-19)

This is a fresh exact-by-construction lane. It composes distinct units from
`data/novel_pairs.json` as `L1 ... Lk C Rk ... R1`, with `k ∈ {3,4}` and a
one-word center. The right side is emitted in ordinary inventory order; no
finished candidate is reversed, no catalogue unit is loaded, and no clause is
repeated. A corpus-style bigram proxy is used only to rank proposals.

## Remote result

The bounded run was executed on `hst-bench`:

```text
cd /home/eric/cloud/palindrome-sentence-generator
python3 experiments/nested_pair_chain_constructor_20260919.py
```

It enumerated 281,712 exact chains above 38 normalized letters. Every retained
row passed an independent outside-in two-pointer audit and forward/reverse
SHA-256 equality. However, zero are reader-eligible: the pair inventory is a
list of semordnilap fragments, so concatenation produces word salad rather than
two readable English clauses. The proposal score is explicitly not a
readability claim.

Representative rendered outputs (all exact, 65 letters):

* “pat notes no cotton stolen in draw a pit a tip award nine lots not to con set on tap”
* “stolen in draw a pit pat notes no cotton a not to con set on tap tip award nine lots”

The first is structurally valid but neither side is a grammatical clause; this
is the clean discriminator. The next experiment must replace pair fragments
with independently authored clause-shaped mirror units (or add a grammar layer)
while preserving the live nesting construction and the same two independent
audits. More permutation or score search cannot repair this lexical failure.

Artifact: `runs/nested-pair-chain-constructor-20260919.json`.
Generator SHA-256 on `hst-bench`:
`65926f0245f4f9427a6434063dad53f461a381dfd75b007a3350be8d31e8b256`.
