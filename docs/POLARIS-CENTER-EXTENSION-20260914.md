# Polaris fixed-center extension evaluation — 2026-09-14

This run asks a constructive question: can the human-selected 38-letter
palindrome

    An aide rips nine memos; some men inspire Diana.

be extended while preserving exact letter symmetry? The center is fixed, and a
center-out solver adds new left/right words. The extension gate rejects every
self-palindromic or one-letter unit, forbids reuse of center words and added
words, and independently rechecks the final tape. Brown tags and bigrams are
diagnostics only; they are not a readability certificate.

## Queue jobs and artifacts

All jobs ran in the Polaris `debug` queue under project `EVITA`, with 32 ranks
and the frozen 30,000-word vocabulary.

| job | method | result artifact |
|---|---|---|
| `7618961` | fixed center, score-max and length-max arms | `runs/center_extension_debug_20260914_194609/aggregate.json` |
| `7618964` | free exact-search control | `runs/search_debug_20260914_194659/aggregate.json` |
| `7618966` | sentence-bank/Brown-shape control | `runs/sentence_bank_20260914_194731/aggregate.json` |
| `7618975` | grammar-constrained fixed-center extension | `runs/grammar_extension_debug_20260914_195405/aggregate.json` |

The independent local audit is
`runs/polaris-eval-20260914.json`, produced by
`experiments/evaluate_polaris_debug_20260914.py` with 32 deterministic own-word
shuffles per item.

## Exact results

| arm | searches | exact closures | length range | mean length |
|---|---:|---:|---:|---:|
| fixed-center, score objective | 256 | 256 | 100–116 | 104.1 |
| fixed-center, length objective | 256 | 256 | 272–308 | 298.2 |
| grammar-constrained center | 2,304 shape pairs | 2 | 72–80 | 76.0 |
| free sentence-bank control | 1,024 | 123 | 28–83 | 41.4 |

The free exact-search control independently closed 256/256 searches at 80–81
letters. Every reported closure passed both the repository validator and an
independent normalized-tape equality check.

## Rendered candidates

The longest score-objective surface was:

    May be many meh to still are here we not as in or of an aide rips nine memos some men inspire Diana for on is at one were her all it so the my name by am.

It is 116 letters and exactly palindromic, but it is not reader-worthy English.
The length objective reached 308 letters, for example:

    Della clear side end date los law onto go get new film them its use last call up and was Eva how teh those not never even made by me but one were her at ah to still as in or of an aide rips nine memos some men inspire Diana for on is all it so that are here we no tube my be damn ever event ones oh the two have saw DNA pull act sales us time HTML if went ego got now also let add need Israel called.

The grammar-constrained arm produced only:

    One ma get over care pro do an aide rips nine memos some men inspire Diana odor per acre vote game no.

All three are exact outputs with provenance in the JSON artifacts above. None
has been represented as readable evidence or sent to readers.

## Programmatic evaluation

The Brown observed-minus-own-shuffle order gain was:

| group | mean order gain | repeated-word rate |
|---|---:|---:|
| human seed | 0.510 | 0.000 |
| score extension | 0.848 | 0.000 |
| length extension | 0.647 | 0.000 |
| grammar extension | 0.669 | 0.000 |
| free sentence-bank control | 0.440 | 0.194 |

The score arm therefore beats the human seed on this local-order diagnostic
while plainly reading worse. This is a direct evaluation of the method: the
proxy ranks common function-word scaffolds above meaningful prose. The paper
must report these values only as diagnostics and must not call them readability
scores.

## Decision

The fixed center is a useful exact extension operator and can reliably make the
surface longer. It does not yet produce a reader-worthy long palindrome. The
grammar restriction is too sparse at the current lexical boundary, while the
length objective is explicitly unusable for prose selection. A reader study is
not claimed for this batch; it remains gated on a manually inspected candidate
that survives the exact, provenance, and anti-shortcut checks.
