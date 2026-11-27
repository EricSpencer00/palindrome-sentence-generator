# Central-shell dual-parse protocol — 2026-10-08

## Question

Can the 650-letter exact parent replace its repeated 36-letter-per-side
central shell with a longer character-exact pair of distinct, connected event
parses, without whole-word reversal shortcuts?

This is a seam-local construction experiment. The parent already has inherited
global construction debt, so a successful local result is not a readable
palindrome claim or reader-study item.

## Frozen input and seam

- Parent artifact: `runs/incumbent-608-repeated-shell-repair-20261002.json`
- Row: `double-event-shell-repair-650`
- Normalized letters: 650
- Normalized SHA-256: `2bd92686cbd01945be3869fbeee7ae9415cfcdae5cd616b26d89ec4f0acc54a9`
- Left shell: normalized `[289,325)`, `Nadia delivers maps. Leon. Ari delivers maps.`
- Right shell: normalized `[325,361)`, `Spam's reviled, Ira. Noel; spam's reviled, Aidan.`
- Each old side has 36 normalized letters. Their concatenated tape is an exact
  palindrome shell, but reflected word boundaries collide at offsets 5, 17,
  21, and 24. It also contains repeated `delivers maps` / `spam's reviled`
  and direct word reversals (`Nadia/Aidan`, `delivers/reviled`, `maps/spam's`,
  `Leon/Noel`, and `Ari/Ira`).

The driver checks the artifact id, normalized length and hash, exactness, exact
old rendering, shell spans, and old local equation before starting. It stops
with an error on drift; do not update these values from a candidate.

## Search representation and frozen domains

`experiments/central_shell_dual_parse_20261008.py` uses the existing
`llm_palindrome.dual_parse.word_residual_search`. Each side has three clause
slots, and each slot is a complete active transitive sentence. The engine
matches the selected clause tapes against the live reverse character residual
before closure. It rejects intermediate complete closures.

Frozen actors: `Mara, Nora, Aidan, Aron, Tara, Leon`.

Frozen verb/patient frame pairs:

| Verb | Patients |
|---|---|
| carries | lamp, letter |
| cleans | boat, window |
| finds | key, coin |
| folds | map, cloth |
| guides | child, guest |
| marks | page, wall |
| opens | box, door |
| paints | boat, wall |
| reads | book, note |
| repairs | roof, gate |
| saves | bird, guest |
| writes | report, letter |

Determiners are `a` and `the`. The frame list is the grammar for this bounded
probe, not an open-ended lexical model. Each side must keep one agent across
its three events. Across both parses, event frames, verbs, and patients must
be unique. Participant names can repeat only as the explicit discourse link.
The online gate rejects self-palindromic content words and any two emitted
content words that are exact reversals.

Before constructing the residual chart, the driver exhaustively checks all
120 outer endpoint combinations: each frozen left actor's letter tape against
the reversed tape of every right final patient. Prefix consumption is a
necessary condition for the first chart transition. If zero pairs are
compatible, the driver returns `infeasible_before_search`, records all pair
checks and mismatch positions, and does not launch the 500,000-state chart.
The independent preflight found 0/120 compatible pairs: 115 first-character
mismatches, 4 second-character mismatches, and 1 third-character mismatch.
The closest pair is `Tara` / `boat`: `tara` versus reverse(`boat`) = `taob`,
which diverges at character 3. This frozen endpoint grammar is rejected as
infeasible before search, so a no-candidate result is not attributed to an
exhausted 500k-state search. The preflight-only run is stored at
`runs/central-shell-dual-parse-20261008.json`; it records all 120 pairs,
`status: infeasible_before_search`, and `search_launched: false`. Source SHA-256
for that result is `c5d1c09e029fb465c684e853e6c8abeefc2dbadffdec332b3d134933f3339b90`.

## Frozen objective and stop rules

Objective: return the longest exact full-parent child whose new shell is an
exact local equation, grows the whole normalized tape by at least 20 letters,
uses different word-boundary segmentation on the two parses, has no shared
reflected internal word boundary, has no reversed content-word pair, has no
repeated multiword unit, and has no proper palindromic multiword span inside
the replacement shell. The full parent must remain exact. Maximum: 500,000
visited states, 5,000 exact local closures, fixed seed field `null` because
this deterministic finite-domain search is not randomized.

If endpoint preflight passes in a future protocol revision, stop when the queue
empties, a cap is reached, or the exact-closure cap is reached. Preserve every
closure including rejected candidates, all stored dead frontiers, gate
rejection counts, cap status, the parent shell diagnostic, and the full
normalized hash for each exact child. Do not relax gates or rerun with a
changed domain after seeing outcomes; a changed domain is a new protocol.

The unchanged parent fails three whole-text screens:
distinct-content words, no repeated nontrivial unit, and no proper
self-palindromic multiword span. This is a known obstruction to global
mechanical promotion while the rest of the tape remains immutable. Report
local seam yield separately from global parent eligibility.

## Endpoint topology extension proposal

Do not enlarge the same actor/patient bank. A plausible next experiment
changes the endpoint grammar: expose the right final patient as a determiner
plus noun phrase whose characters can be consumed across the left subject
boundary. For example, left outer word `Mara` and right final phrase `a ram`
have `mara == reverse("a ram")`; no content token is the reverse of another,
and the right phrase has a word boundary that does not reflect a left word
boundary. A possible clause ending is `Tara sees a ram.` This is an endpoint
compatibility witness only, not a complete palindrome or readability result.

The concrete extension should replace whole-clause endpoint slots with a
grammar-state chart that exposes the rightmost determiner and noun separately,
carries residual ownership into the next left word, and checks this witness
before expanding inner events. Freeze a new event inventory and rerun the
whole-parent debt audit. Do not report the current domain as a substantive
negative search.

## Invocation on the NUC checkout

After copying the source file, this protocol, and the exact parent artifact
into `/home/eric/palindrome-sentence-generator` on the NUC:

```sh
cd /home/eric/palindrome-sentence-generator && python3 experiments/central_shell_dual_parse_20261008.py --parent runs/incumbent-608-repeated-shell-repair-20261002.json --out runs/central-shell-dual-parse-20261008.json --max-states 500000 --max-results 5000
```

Because the current endpoints fail preflight, this invocation returns the
certificate without running the chart. The local preflight result above is
not a remote run; if the remote copy is run for artifact parity, append its
host/runtime details and verify the source hash before comparing its JSON.
