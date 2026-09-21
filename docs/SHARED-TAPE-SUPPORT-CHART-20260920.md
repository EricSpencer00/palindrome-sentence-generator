# Shared-tape support chart

This implementation packs one complete forward CFG derivation across a shared
letter tape. Lexical arcs have free word boundaries. Bottom-up chart reachability
is followed by top-down support from the sentence root; unsupported lexical arcs
cannot contribute letter domains. Mirrored domains intersect, and the complete
chart is rebuilt until a fixpoint. Search branches on the smallest unresolved
character orbit, including interior positions. The center may occur within a
word, and spaces never need to align.

This is an implementation distinction, not a claim that grammar CSPs are new.
The previous bounded Earley lane enumerated complete authored controls. The
fixed-slot character CSP did not propagate full-root parse support. The packed
single-sentence lane additionally required cursor meeting at an accepting node,
synchronized whitespace, and imposed unjustified mirrored grammatical roles.
This lane uses none of those restrictions.

The grammar gives singular people finite reading/writing predicates and text
objects. Recursive `and`/`while` clauses can extend a derivation to the selected
length. Grammatical selection happens before any candidate is emitted; repeated
words and self-palindromic lexical units are rejected during reconstruction.
Semantic typing is deliberately coarse and cannot establish coherence.

On `hst-bench`, four tests cover independent brute-force equivalence for a tiny
language, asymmetrical boundaries with an odd center, unreachable lexical-support
removal, and a complete recursive English clause. All pass. The eight target
lengths 39, 40, 44, 48, 52, 60, 72, and 100 each become UNSAT after two root
propagation rounds, without branching. This is a proof only for this small
grammar and each tested length, not an English impossibility claim.

The recorded non-palindromic prose controls are “The guard reads a letter.” and
“Diana writes a poem while Leon studies a map.” Both parse in the actual grammar.
There are no exact candidates and no reader-facing output. Independent pointer
and SHA-256 audits are in the JSON evidence.

The conflict is structural and specific: the only compatible outer letter is
`m`, forcing initial `Mira` and terminal `poem`; their next letters `i` and `e`
conflict. The next repair should add a typed person-object predicate frame with
named objects, changing the final argument category. Increasing the current
text-noun product does not address this conflict. The chart is a reusable
correctness foundation, not evidence of readable-palindrome progress.

## Separate calibration and argument-frame repair

The `--calibrate` mode supplies lexical categories for the known 38-letter
benchmark, without storing its letter tape in the solver. It recovers “an aide
rips nine memos; some men inspire Diana.” in three search nodes and seven
propagation rounds. This recovery is explicitly calibration, never promotion.
The independent regression test verifies its normalized tape.

The follow-up then adds person-object predicates, singular/plural agreement
frames, quantified plural objects, alternate people and names, and the original
recursive reading/writing grammar. Target lengths 39, 40, 44, 48, and 60 each
become UNSAT at the root in three to five propagation rounds. The repair therefore
removes the initial text-object bottleneck but does not produce a longer result.
Five tests now pass on `hst-bench`. The separate evidence is
`runs/shared-tape-support-chart-calibrated-20260920.json`.

The next structural addition should be an object-relative clause with a bound
object gap. That permits the terminal word to be a finite predicate, rather than
another noun or name, while retaining an explicit argument dependency. It should
be accompanied by positive grammatical and negative missing-argument tests,
then subjected to the same root-support analysis.

Files: `shared_tape_support_chart_20260920.py`,
`tests/test_shared_tape_support_chart_20260920.py`, and
`runs/shared-tape-support-chart-20260920.json`.
