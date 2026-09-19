# v4 construction and evaluation ledger

This note is the current evidence-led core for the paper. The working claim
is constructive: choose grammatical lexical paths while satisfying character
seams during search. Exactness is independently checked; automatic language
scores and AI feedback only select the next repair. No output below is human
certified yet.

## Anchor and frontier

The strongest reader-plausible exact output remains:

> **An aide rips nine memos; some men inspire Diana.**

It has 38 ASCII letters, normalized tape
`anaideripsninememossomemeninspirediana`, and forward/reverse SHA-256
`ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6`.
The independent outside-in pointer audit and the fail-closed mechanical gate
both pass. It has not yet been rated by blinded humans.

The longest mechanically admitted repair is exact but not readable:

> **To new one post is an evening. Is sign in even as its open owe. Not.**

It has 50 letters and SHA-256
`855d51fa2b5cb8b4f63b9e043494f066702f8abbb329671ee5c68b82ca7788e3` in both
directions. A post-hoc `gpt-oss:20b` AI-feedback pass scored intact English,
scene coherence, and Shakespearean cadence 0/3, 0/3, 0/3. This is diagnostic,
not a readability certificate and not a search reward.

The 66-letter exact row is rejected mechanically for a hidden proper
multiword palindrome span:

> **Erased on forever event is an evening. Is sign in even as it never ever. Of nodes are.**

## Constructive methods

| Method | Search-space change | Rendered evidence | Exact / admitted |
|---|---|---|---:|
| Fixed half-tape grammar CSP | Character aliases are assigned before word boundaries; agreement and valency are live state | 38-letter anchor | 2 / 2 at 38; none above |
| Phrase-valued half-tape CSP (z2) | Phrase edges keep determiner/noun boundaries live while assigning mirrored character variables | **An aide rips nine memos; some men inspire Diana.** (38 letters; 24 target runs; 48,999 nodes) | 1 / 1 at 38; no >38 closure |
| Dictionary reverse-segmentation DP | 51,129 authored SVO seeds are matched against a 52,927-headword POS-neutral reverse segmentation | No exact closure above 38; no candidate promoted | 0 / 0 |
| Indexed half-tape path CSP | Inverted character-slot seam index over typed grammar paths | Anchor recovered from 1,205 nodes | 2 / 2; longest 38 |
| Connector character product | Typed clauses and connectors cross an outside-in character product | 100 frontier witnesses; longest 65 | 0 / 0 |
| Residual-seam scene lattice | Complete grammatical clauses indexed by remaining length, required character, agreement, and valency | 246 grammatical renderings; longest 66 | 0 / 0 |
| Semantic shell growth | Held-out event pairs require live edge overlap at each incremental shell step | 7,272 renderings; longest 183 | 0 / 0 |
| Held-out terminal repair | Same-valency terminals selected by residual character and typed state | 252 grammatical renderings; longest 67 | 0 / 0 |
| Relative indexed boundary CSP | Agreement-carrying relative complement with indexed word-boundary offsets | 12,588 search nodes; longest exact 0 | 0 / 0 |
| Explicit relative-slot CSP (z2) | `who`/`that`, finite relative verb, and object are separate live edges rather than one opaque clause | 194,216 nodes across 39–70 letters; no exact closure | 0 / 0 |
| Shared relative complement | Shared participant plus finite `that`/`and` markers across the relative seam | 4,042 indexed nodes; no exact closure | 0 / 0 |
| Residual-prefix attachment lattice | Two-character seam-prefix state with attachment, agreement, and valency | 258 candidates; longest 77 | 0 / 0 |
| One-time typed lexical bank | Local model authors lexical alternatives once; deterministic CSP searches them without per-candidate feedback | 703,172 nodes; 390 grammatical frontier controls | 0 / 0 |
| Three-beat alias grammar | Three finite SVO beats with live boundary aliases and conjunction state | 1,344 renderings; longest frontier 81 | 0 / 0 |

Every lane records a literal two-pointer audit, forward/reverse SHA-256, source
provenance, novelty preflight, and a concrete next repair. None reverses a
finished sentence, imports catalogue text, or scores each search state with an
LM/RLAIF reward.

The constructive result is therefore specific, not a proxy claim: the z2
half-tape representation independently recovers the 38-letter readable anchor,
but the bounded 38--45-letter pilot does not yet improve its length. The next
repair exposes relative-clause internals as separate character-constrained
edges; its 194,216-node run also closes at zero, so it remains a repair record,
not a promoted example.

An orthogonal dictionary-DP check also returned zero above 38. Its failure is
not folded into a general sparsity claim: it chooses a complete seed before
reverse segmentation, so the concrete repair is a POS/inflection-aware
character trie that inserts boundaries during half-tape search.

## Reader evidence and API gate

`experiments/reader_package_v4_20260919.py` creates five deterministic blinded
pairs: each exact frontier item and each intact prose control is paired with a
word-shuffled control. A fixed seed randomizes A/B order, while the answer key
is held separately from the rater form. The package is ready, but human ratings
are pending. Therefore `/api/v4/generate` remains fail-closed; v4 exposes
evidence and diagnostics only.

The next reader-facing test is a randomized blinded intact-prose versus
shuffled-control rating with independent raters and explicit exclusions.
