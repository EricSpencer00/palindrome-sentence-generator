# Broad-POS exact geometries: grammar-aware reauthoring

The two exact 39-letter rows were re-read as construction targets, not as
sentences to be edited after the fact. Their normalized tapes are independently
exact (forward and reverse SHA-256 agree), but the live word geometry imposes
hard grammatical conflicts:

| exact geometry | blocking conflict |
|---|---|
| `an aide rips nine metal; a late men inspire diana` | `a late men` has determiner/number disagreement; `nine metal` is a mass noun used as a count object; `Diana` is a proper name but has no compatible subject/object role in this frame |
| `an aide kill right rowan; a worth girl like diana` | singular `an aide` requires `kills`, not `kill`; `a worth girl` is not a normal determiner/adjective combination; `girl like Diana` is an incomplete/ambiguous clause |

Fresh grammar-aware attempts around the same geometry were deliberately
authored rather than repaired in place:

1. `An aide rips nine medals; a late man inspires Diana.` — grammatical enough
   as prose, but changing `metal/men/inspire` breaks the mirrored character
   equations; normalized audit is not exact.
2. `An aide kills right Rowan; a worthy girl likes Diana.` — grammatical
   inflections and adjective choice, but changing `kill/worth/like` breaks the
   exact tape; normalized audit is not exact.
3. `An aide rips nine metal; a late man inspires Diana.` — fixes agreement but
   still changes the required `men` and `inspire` geometry; not exact.

Independent audit of each reauthoring is the ordinary normalized pointer scan
(`t == reverse(t)`), not a tape edit or catalogue lookup. No fresh grammatical
sentence over 38 letters closed. The two exact rows therefore demonstrate a
lexical-boundary coincidence, not a viable semantic frame.

## Concrete next semantic frame

Use two independently authored transitive event clauses with fixed singular
agreement and countable objects, e.g. “An aide carries nine medals; a late man
guides Diana.” Carry the typed features `(singular subject, transitive verb,
count object, proper-name recipient)` into the character orbit before lexical
selection. This changes the semantic frame and agreement state rather than
trying to rescue the incompatible `a late men` / bare-verb geometries.
