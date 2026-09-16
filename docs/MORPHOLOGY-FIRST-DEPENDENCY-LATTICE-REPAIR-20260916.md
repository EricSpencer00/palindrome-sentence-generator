# Morphology-first dependency lattice repair (2026-09-16)

## Decision

Retain the morphology-first state space as a negative-result experiment, but do
not promote a readable palindrome. The repair produced no exact closure after
expanding the morphology inventory and dependency topologies.

## Preflight and method

The run read the current novelty registry (95 entries) before generation. The
base family `morphology-first-dependency-lattice` was the only matching family;
foreign signature and artifact collisions were empty. The repair selected a
typed dependency topology first, then enumerated node-local paths of the form
lemma → derivation → inflection → surface. It carried number and valency
attributes through agreement checks and rendered every survivor in ordinary
reading order. It did not import catalogue strings or emit from a reversed
tape, and it excluded center-out, reverse-segmentation, grammar-template,
MCTS, beam, and clause-product routes.

## Run evidence

Artifact: `runs/morphology-first-dependency-lattice-derivational-repair-20260916.json`

| measure | result |
| --- | ---: |
| dependency topologies | 6 |
| raw morphology/tree yields | 24,570 |
| exact letter-palindrome yields | 0 |
| catalogue rejections | 0 |
| symmetry rejections | 0 |
| mechanically admitted | 0 |
| reader studies | not run |

The artifact keeps a bounded near-miss ledger with independent mismatch
positions and morphology provenance. Zero exact closures means there are no
rendered candidates to certify; no readability claim is made.

## Next repair

If this family is continued, add independently authored derivational paradigms
with corpus-backed argument frames and a blinded intact-vs-shuffled reader
study. Do not relax catalogue or symmetry gates to convert a near miss into a
candidate.
