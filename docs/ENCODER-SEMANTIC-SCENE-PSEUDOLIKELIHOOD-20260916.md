# Bidirectional encoder semantic-scene lane

The novelty preflight inspected the registry before execution (102 entries at
run time) and found no exact signature collision. The retained family is
distinct from `masked-character-scene-gibbs-20260916`: it enumerates complete,
hand-authored semantic alternatives first, carries every mirrored character
obligation in the assignment state, and only then scores each complete scene by
masking each token with a bidirectional encoder. No next-character probability
or character Gibbs proposal is used.

Three complete English probes were 50, 52, and 50 letters. The best actual
prose was “After rain, the station porter brings a wet parcel to the bench.”
All three passed the mechanical readability/novelty checks, but none closed:
the independent direct tape, two-pointer, and SHA-256 reverse-hash audits all
reported false. The concrete repair is mismatch-directed semantic-slot
replacement followed by rebuilding the complete assignment; one-sided edits
are rejected because they break the scene meaning.

The cache contains ModernBERT metadata and a multilingual-BERT tokenizer, but
the installed Transformers runtime sees the cached BERT snapshot as TF-only
without PyTorch weights. The run therefore records a fail-closed cache
inspection fallback rather than downloading weights or substituting a causal
LM; the scorer is ready to use the cached bidirectional checkpoint when its
weights are available.
