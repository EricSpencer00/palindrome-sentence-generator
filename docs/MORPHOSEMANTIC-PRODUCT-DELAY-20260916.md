# Morphosemantic product delay pilot (16 September 2026)

## Preflight

The proposal was checked against `docs/experiment-novelty-registry.json`
before execution.  The formal preflight found no exact id, signature, or
artifact collision across 92 retained families and 6 excluded routes.  The
nearest conceptual families were reviewed manually:

- `character-clause-fst-joint-emission` also emits mirrored characters, but it
  compiles finite complete arithmetic clause banks into tries.  This pilot
  generates lexical arcs on demand from looping semantic states and never
  materializes complete clauses.
- `morphological-derivational-seam` realizes independent inflectional forms
  around a reverse character seam.  This pilot uses a persistent
  number/tense agreement environment inside a semantic product and a general
  output-delay monoid; there is no seam inventory or fixed tape.
- `discourse-graph-walk-palindrome-20260915` walks a typed event graph with a
  character-balance carry.  This pilot's changed dimension is productive
  coordination depth plus morphology-conditioned lexical realization, not an
  event graph walk.

The proposed state-space signature is:

```text
morphosemantic-product-delay|looping-feature-automata|on-demand-morphological-realization|single-output-delay-monoid|no-complete-clause-materialization
```

The proposal is therefore retained as a new, manually reviewable family, with
the conceptual overlaps above explicitly disclosed.

## Construction

The left and right sides are independent semantic automata for a simple
subject–finite-verb–(optional modifier)–object frame.  The `COORD -> START`
edge is a real productive loop, so a target of 160 letters and a target of
several thousand letters use the same transition system.  Lexical arcs carry
subject number and verb tense; the right automaton walks the reverse semantic
presentation so its edge output reverses back to normal English word order.

The product state is `(left semantic state, right semantic state, output
delay, delay owner, word counts)`.  A side may advance one complete lexical
arc.  Equal output prefixes cancel; if the same side continues, its output is
appended to its existing delay.  This avoids a precomputed clause cross
product and avoids reading or resegmenting an immutable tape.

Replay:

```text
python3 experiments/morphosemantic_product_delay_20260916.py
```

Observed output:

```text
states=100000, transitions=480563, character_rejects=183140,
exact_closures=0, mechanically_admitted=0, reader_eligible=0
```

The bounded frontier exhausted its 100,000-state budget without an exact
closure.  This is a failed construction attempt, not evidence that readable
arbitrary-length palindromes are impossible.  No output entered a reader
packet.  The focused regression tests cover delay cancellation, right-edge
rendering, and the no-complete-clause-product invariant.
