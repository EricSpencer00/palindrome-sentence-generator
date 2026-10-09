# Eric's development ratings — October 9, 2026

`ratings.json` preserves the pasted export verbatim: 183 scalar ratings from
one local rater, study `palindrome-readability-v1`. Each scale runs from 0 to 3.
These are genuine reader labels, not yet a joined passage dataset or a trained
RLHF model. They are development feedback, not independent held-out evidence.

- 118/183 (64.5%) have meaning 1: “Word associations with unclear relations.”
- 111/183 (60.7%) have both grammar and meaning 1.
- 16/183 (8.7%) have both grammar and meaning 3.
- 9/183 have both scores 0.
- 46/183 have both scores at least 2.

The dominant failure is recognizable phrases without clear sentence relations.
This is the rubric's **association** category; it does not establish heavy use
of the literal word “association.” Eric reports that shorter palindromes were
interesting. Length is absent from the export, so that pattern remains a reader
observation until the passages are joined. High ratings alone cannot establish
palindrome validity: intact prose controls could also score highly.

## Missing passage packet

The export contains item IDs, positions, ratings, and times, but no texts or
control labels. Local searches found no matching packet. The matching public
artifact is https://claude.ai/artifact/NKoBoSFc8PbkxuToAfnAdh ; its completed
review UI exposed only item 1, “May a moody baby doom a yam?”. Its Next button
returned directly to completion and refreshed that item's browser timestamp;
this repository uses the original pasted timestamp, not the modified browser
export. The artifact has no accessible source-download control in this session.

All 183 rows are therefore `unresolved` in the generated audit. Empty exact and
non-palindrome files mean **not checked**, not “none invalid.” No weights,
search defaults, paper results, or existing training sets were changed based
on unknown texts. The original responses are retained rather than deleted.

To finish the audit, save the original ID-to-text packet as `packet.json` here:

```json
{"study":"palindrome-readability-v1","items":[{"item":"id0d43d30f8","text":"May a moody baby doom a yam?"}]}
```

Include all 183 original item IDs and exact displayed texts. Optional source or
control metadata should remain in the original packet for interpretation.

```sh
python3 -m training.audit_reader_ratings \
  data/reader-feedback/2026-10-09/ratings.json \
  --packet data/reader-feedback/2026-10-09/packet.json \
  --out data/reader-feedback/2026-10-09/audit
```

The importer matches by ID, rejects duplicate labels and mismatched study IDs,
checks A–Z letters against their full reversal using the project's validator,
and writes separate exact, non-palindrome, and unresolved datasets. Invalid
rows retain the first mismatching letters for inspection. It summarizes exact
rows by length and separately counts literal “association” occurrences.
Non-palindromes are excluded from exact-only training data; preserve them as
possible controls. Scalar scores do not become invented A/B preference pairs.
Use fresh passages and independent raters to validate any later tuning.
