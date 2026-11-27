# Eric's development ratings — October 9, 2026

`ratings.json` preserves the pasted export verbatim: 183 scalar ratings from
one local rater, study `palindrome-readability-v1`. Each scale runs from 0 to 3.
These are reader labels, not yet a joined passage dataset or a trained
RLHF model. They are development feedback, not independent held-out evidence.

- 118/183 (64.5%) have meaning 1: “Word associations with unclear relations.”
- 111/183 (60.7%) have both grammar and meaning 1.
- 16/183 (8.7%) have both grammar and meaning 3.
- 9/183 have both scores 0.
- 46/183 have both scores at least 2.

The 118 meaning-1 ratings count the rubric's “Word associations with unclear
relations” category. They do not count occurrences of the word “association.”
The earlier interpretation that Eric meant only the rubric was unsupported:
the repository independently confirms heavy literal use of that word.

- The 52 frozen texts in `review-site/dist/index.html` include 44 containing
  the word “association.” All 52 pass the exact normalized-letter check.
- The historical [directional-search report](../../../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md)
  records “one poem association” in 36 of 66 selected escalated outputs.

These are different counts and cohorts from the 183 reader ratings; the 52
review texts deduplicate historical outputs and must not be treated as an
independent sample. Their `P-...` IDs do not match this export's `i...` IDs, so
there is no valid join assigning these ratings to those passages. Repetition
is established, but these counts do not prove that the word caused poor
readability or that removing it would improve scores.

Eric reports that shorter palindromes were interesting. Length is absent from
the export, so that pattern remains a reader observation until the passages
are joined. High ratings alone cannot establish palindrome validity: intact
prose controls could also score highly.

## Missing passage packet

The export contains item IDs, positions, ratings, and times, but no texts or
control labels. Local searches found no matching packet. The matching public
artifact is https://claude.ai/artifact/NKoBoSFc8PbkxuToAfnAdh ; its completed
review UI exposed only item 1, “May a moody baby doom a yam?”. Its Next button
returned directly to completion and refreshed that item's browser timestamp;
this repository uses the original pasted timestamp, not the modified browser
export. The artifact has no accessible source-download control in this session.
The displayed first text is not tied to a verified item ID; matching its screen
position to an export row would be an unsupported join.

All 183 rows are therefore `unresolved` in the generated audit. Empty exact and
non-palindrome files mean **not checked**, not “none invalid.” No weights,
search defaults, paper results, or existing training sets were changed based
on unknown texts. The original responses are retained rather than deleted.

The audit accepts an ID-to-text packet at `packet.json` with this shape
(illustrative ID and text, not a recovered mapping):

```json
{"study":"palindrome-readability-v1","items":[{"item":"example-item-id","text":"Madam, I'm Adam."}]}
```

Each recovered mapping must retain the original item ID and exact displayed
text. Optional source or control metadata remains useful for interpretation.

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
