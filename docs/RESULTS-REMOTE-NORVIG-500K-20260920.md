# Remote overhang search: 2026-09-20

This run tried to pass 500,000 normalized letters in the Panama-style
Norvig/Hoey exact-letter category. It did not reach that target. The best
finished candidate is preserved under
`artifacts/remote-norvig-500k-20260920/fresh-nocap/`.

## Best candidate

- 286,561 normalized letters
- 54,097 word tokens
- 47,354 unique normalized phrase units
- 1,800-second remote run on `hst-bench`
- 105,623,005 search steps and 19,534 closures
- no artificial content-word reuse cap; the maximum observed content-word
  count is 14
- no adjacent repeated words

The search used the two-sided overhang algorithm, a 476,757-entry dictionary,
dynamic unused-unit indexing, and feasible-inventory pruning. The exact
dictionary snapshot is saved beside the candidate.

## Independent audit

The fresh standard-library audit reports:

- exact outside-in letter palindrome: true
- every phrase key present in the recorded dictionary: true
- unique phrase keys: 47,354
- adjacent repeated words: 0
- normalized SHA-256: `cc127b134ba945ed6405b3b27da257d2032bd64577ddc3be5aef8c43ecc212e6`

This is an exact English-word construction, not intact readable prose. It is
not reader-eligible under `docs/READABLE-PALINDROME-GOAL.md`, and it is not an
official world-record claim. It beats Norvig's published 90,439-letter
construction, but it does not beat the unverified 500k figure mentioned in
the conversation.

For the official category distinction, Guinness currently lists the longest
known palindrome as a 19-letter Finnish word; that is not the same category as
these long sentence-style constructions.
