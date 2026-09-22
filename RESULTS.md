# Results

[README](README.md) · **Results** · [Full results ledger](docs/RESULTS-LEDGER.md) · [Goal record](docs/READABLE-PALINDROME-GOAL.md) · [Design](DESIGN.md)

This is the short version of the experiment record. The repository contains many failed searches and repair attempts, so this page keeps the useful progression in view and links to the complete committed ledger when the details matter.

## Current generation

The current promoted readability frontier is **666 letters**, independently exact after normalizing case and punctuation. It is a connected central mini-scene over the existing working track; it still carries repeated vocabulary and rough discourse as construction debt.

```text
Leon won. Wolf spots Nora. Nora sees Nadia. Nadia sees Ira. Sara saw God. Ari saw Dog. Mara stops rats. Nora spots a ram. Mara sees rats. Nora stops rats. Nora sees Aram. Mara sees Nadia. Nadia saw Noel live. Mara stops Nadia. Nora sees Aram. Sara saw Noel live. Now, Noel, did I live? Nora saw Noel live. Noel, I sit. Pat notes. Mara saw God. Sara, did I live? Nora, I saw desserts. Leon stops Noel. Noel spots Nadia. Nadia stops Aidan. Ari sees God. Dog sees Ira. Nadia spots Aidan. Aidan stops Leon. Leon spots Noel. Stressed was I, Aron. Evil I did, Aras. Dog was Aram. Seton, tap. 'Tis I, Leon. “Evil Leon” was Aron. Evil I did, Leon won. Evil Leon was Aras. Mara sees Aron. Aidan spots Aram. Evil Leon was Aidan. Aidan sees Aram. Mara sees Aron; star spots Aron. Star sees Aram. Mara stops Aron. Star spots Aram. God was Ira. Dog was Aras. Ari sees Aidan. Aidan sees Aron. Aron stops flow now, Noel.
```

- Artifact: [`runs/incumbent-666-central-mini-scene-comparison-20260922.json`](runs/incumbent-666-central-mini-scene-comparison-20260922.json)
- SHA-256: `3951b9449ed3ab28f55d9798e344dfaf3123035f5f07c9dffdf0047bed0e1d79`
- Audit: two-pointer exact, byte-pointer exact, and forward/reverse hashes agree

The separate **568-letter** causal-scene row remains the working incumbent because it has the clearest construction lineage. The 666-letter result is promoted as the readability frontier, not presented as finished prose.

## A longer earlier generation

The 650-letter generation below is the earlier event-shell repair that made the length increase concrete while keeping the residual equation closed. It is exact, but its repeated names and event frames are visible.

```text
A ram saw Nadia. Leon won. Wolf spots Nora. Nadia stops, so Tara rewards Nadia. Nora spots a rat. Nadia spots a ram. Mara spots rats. Leon maps Nora. Draw no maps. Aidan spots a rat. Nora spots a ram. Mara stops rats. A tub? He maps Nora. Deliver no evil. Now, Noel, did I live? Nora saw Noel live. Noel, I sit. Pat notes. Mara saw God. Sara, did I live? Nora, I saw desserts. Noel, was I stressed? Nadia delivers maps. Leon. Ari delivers maps. Spam's reviled, Ira. Noel; spam's reviled, Aidan. Desserts I saw, Leon. Stressed was I, Aron. Evil I did, Aras. Dog was Aram. Seton, tap. 'Tis I, Leon. “Evil Leon” was Aron. Evil I did, Leon won. Live on, reviled. Aron, spam. Eh, but a star spots Aram. Mara stops Aron. Tara stops Nadia. Spam onward. Aron, spam Noel. Star stops Aram. Mara stops Aidan. Tara stops Aron. Aidan's drawer, Aratos, spots Aidan. Aron stops flow now, Noel. Aidan was Mara.
```

- Artifact: [`runs/incumbent-608-repeated-shell-repair-20261002.json`](runs/incumbent-608-repeated-shell-repair-20261002.json)
- SHA-256: `2bd92686cbd01945be3869fbeee7ae9415cfcdae5cd616b26d89ec4f0acc54a9`
- Audit: independently normalized exact reversal

## Generation history

| Length | What changed | Status |
|---:|---|---|
| 38 | Diana seed and the initial exact benchmark | calibration |
| 44 | Reconstructed “Now, an aide rips nine memos; some men inspire. Diana won.” | exact, discourse debt |
| 54 | Noel / stressed-desserts seam diagnostics | exact diagnostic |
| 106 | Reversed-word insertion around the Diana seed | exact, word-salad control |
| 236–240 | Immutable-tape and paired-seam growth | exact, formulaic debt |
| 498 | Content-bearing center-out growth | exact working parent |
| 568 | Causal-scene outer-shell construction | working incumbent |
| 650 | Event-shell substitution with a closed live equation | exact frontier |
| 666 | Connected central mini-scene comparison | promoted readability frontier |

The 38–240 rows are historical diagnostics. They remain useful because they show where construction debt entered, but they are not reader-validated prose claims.

## How to read the results

Exactness is mechanical: remove case, spaces, and punctuation, then compare the character stream with its reverse. Readability is a separate claim. A result can be a real long palindrome and still need work on discourse, repetition, or syntax. That distinction is the point of keeping the debt visible.

The [full ledger](docs/RESULTS-LEDGER.md) indexes every committed JSON artifact by method, rendered length, independently exact length, and audit signal. The [goal record](docs/READABLE-PALINDROME-GOAL.md) explains the active incumbent, preserved frontier, and next repair seam.
