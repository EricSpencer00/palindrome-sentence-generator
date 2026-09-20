# Remote overhang search: 2026-09-19

This run searched for a palindrome longer than 500,000 normalized letters.
It did not reach that target. The best finished strict run is preserved under
`artifacts/remote-norvig-500k-20260919/custom-word-3600/`.

## Candidate

- 283,049 letters
- 53,280 word tokens
- 46,737 distinct normalized word units
- 2,016-second remote run on `hst-bench`
- 89,623,773 search steps and 19,681 closures
- maximum three uses of any non-function word
- no adjacent repeated words

The search is the Norvig/Hoey two-sided overhang construction. It used a
370,108-entry English-word inventory, a dynamic unused-unit index, and the
feasible inventory guard. The inventory snapshot is saved as
`artifacts/remote-norvig-500k-20260919/custom-word-1800/dictionary.txt`.

## Independent audit

The rendered text passes a fresh outside-in character comparison. Forward and
reverse normalized SHA-256 both equal
`53baf39b2f2f1067bd9ac72b9f531bb7cc79d87c39ca184717cf6923a87d1545`.
Phrase keys are unique, adjacent word repeats are zero, and the maximum
non-function-word count is three. The complete audit is in `audit.json`.

This is a very long exact English-word construction, not intact prose. It is
therefore not reader-eligible under `docs/READABLE-PALINDROME-GOAL.md`, and it
does not claim a 500,000-letter win.

For benchmark context, Norvig's published version-3 construction is 90,439
letters; the repository's earlier controlled run reached 90,937. The famous
`50,000,000` figure associated with Norvig's program is a search-step budget,
not a palindrome length.
