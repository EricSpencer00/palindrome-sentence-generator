# Dialogue speech-act residual grammar (2026-09-16)

This route uses three independently authored dialogue pairings: request/answer,
greeting/acknowledgment, and report/response. Each side is a complete utterance
with its own speech-act label; lexical choices are emitted online by a
character residual automaton that compares the left tape with the reversed
right tape. Fragments, echoed utterances, and catalogue text are prohibited.

The novelty preflight read the registry before execution and registered the
new signature `dialogue-speech-act-grammar|...|online-lexical-realization`.
The bounded run examined 25 pairs (9 act-compatible), reached 49 letters, and
found no exact closure at the 39-letter floor. Exactness was independently
checked by normalized reversal, ASCII reversal, a two-pointer audit, and the
residual ledger. The concrete next repair is speech-act-preserving lexical
substitution: replace one content slot in each complete utterance with a
same-act alternative and replay from the first changed residual position.

Run evidence: `runs/dialogue-speech-act-residual-20260916.json`.
