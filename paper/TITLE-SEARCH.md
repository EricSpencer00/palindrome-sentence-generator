# Palindromic title search — 10 September 2026

The manuscript now uses the non-palindromic title **Generating Long Palindromes
with Syntactic Pruning and Inventory-Aware Search**. The search below is
retained as a record of the earlier request for a palindromic title; none of
these candidates is the current title.

The strongest method-aligned palindromic candidate found in this search was:

> Test On: Not Set

The complete title normalizes to `testonnotset` (12 letters), which reads
identically in reverse. It is a compact technical phrase: test the states that
are not yet set (closed). That is what the incremental arm does when it applies
the sentence-pattern feasibility gate to openings and children before they enter
the stack. The colon only makes the operation legible; it contributes no letters.

The title-search driver produced `test on not set` directly, with zero
unattested joins under its saved bigram table. The same project vocabulary
contains every word in the title, and the existing word-level search records the
candidate in both orientations. Punctuation is editorial and does not alter the
letter check.

The longer sentence `Are We Not Drawn Onward, We Few, Drawn Onward to New Era?`
remains a valid catalogue palindrome and a good literary alternative, but it
does not say what the experiment changes as directly as the chosen title. It is
kept here as a rejected alternative, not as a subtitle.

## Shortlist

| Candidate | Normalized letters | Editorial assessment |
| --- | --- | --- |
| **Test On: Not Set** | `testonnotset` | Rejected palindrome candidate. It directly describes testing partial, unclosed states, but is less natural than the current title. |
| **Are We Not Drawn Onward, We Few, Drawn Onward to New Era?** | `arewenotdrawnonwardwefewdrawnonwardtonewera` | Best literary full-sentence option. A direct allusion to outward growth, but less explicit about the sentence-pattern gate. |
| **Not Set? Test On.** | `notsetteston` | Shorter full-title option. A readable question and response suggesting continued testing, though more clipped. |
| **Test One Not Set** | `testonenotset` | Can mean “test one that is not set.” Closer to an instruction about an unfinished candidate, but less natural. |
| **Word Row** | `wordrow` | A readable name for a word sequence. Concise and related to the object being built; says little about the algorithm. |
| **Drawn Onward** | `drawnonward` | Concise palindrome that suits outward construction, but does not identify the subject by itself. |

Other valid outputs included “Test it. Set.” and “Test: Is it set?” The first
is clipped; the second suggests a different test from the paper's actual
pattern-feasibility check. “Stack cats” and “Parse yes rap” fit the letters but
do not describe this paper. No novelty claim is made for any candidate.

## What ran

The driver is `experiments/title_hunt.py`. It uses the existing `WordTries`,
`State`, and `_expand` operations from `llm_palindrome/search.py`; it does not
invent a new letter-matching algorithm. It starts from a requested word or
phrase and searches inward for palindromic closure. Frequency and the existing
bigram counts prioritize branches. Their scores are search heuristics, not
readability measurements; the shortlist above is an editorial judgment.

The two runs searched 156 topic-word anchors and 85 phrase anchors, with 230
distinct anchors across both runs. They used 18,751 vocabulary entries:
dictionary-filtered words from the top 60,000 frequency entries, with a Zipf
floor of 3.2, plus the topic words explicitly restored. Each anchor received at
most 6,000 popped states or approximately 1.5 seconds. The word/letter limits
were 12 words and 60 letters. Words could occur once, except `a` and `i`, which
could occur twice. Frontier and empty-debt expansion caps also bound the search.

The runs examined 475,012 states and produced 49,316 distinct word sequences
after combining and deduplicating their results. All candidates passed an exact
normalized-letter reversal check. This is a bounded search, not an exhaustive
survey of English titles. These are title-writing results, not new evidence for
the manuscript's experimental claims. In particular, short title fragments do
not have to meet the experiment's minimum of three words in each half.

Raw results, including per-anchor stopping statistics, configuration, vocabulary
hash, and every candidate, are in:

- `runs/title_hunt/2026-09-10.json`
- `runs/title_hunt/phrases-2026-09-10.json`

## Literal topic words

A separate trie check asked whether a word's reverse could occur anywhere in
any concatenation of the search vocabulary. It allows both end boundaries to
fall inside words, so it does not require the reversed string to be one word.

| Requested word | Required reverse | Result within this vocabulary |
| --- | --- | --- |
| palindrome | `emordnilap` | No spelling exists. |
| palindromes | `semordnilap` | No spelling exists. |
| search | `hcraes` | No spelling exists. |
| pattern | `nrettap` | No spelling exists. |
| patterns | `snrettap` | No spelling exists. |
| text | `txet` | No spelling exists. |
| sentence | `ecnetnes` | Letter-compatible, for example inside `sec net nest`. No readable title was found. |
| sentences | `secnetnes` | Letter-compatible. No readable title was found. |

These negative spelling results apply to this finite vocabulary, not to all
English, names, abbreviations, or invented words. The required terms themselves
were included even when they fell below the frequency floor. For “sentence,”
the phrase search produced results such as “last sentence sec net nest sal”:
valid letters, without a coherent meaning.

## Reproduction with the paper's outward construction

The existing `enumerate_palindromes` also produced `not set test on` from the
four-word vocabulary `{not, set, test, on}`. Its outward transitions are:

| Left half | Right half | Letters still owed |
| --- | --- | --- |
| set | | Right owes `tes`. |
| set | test | Left owes `t`. |
| not set | test | Right owes `on`. |
| not set | test on | None: the pair closes. |

This confirms that the full-title candidate can be built by the same outward
letter operation explained in the manuscript. Punctuation is editorial and
does not alter the letter check.

To rerun the initial word search from the repository root:

```sh
.venv-v3/bin/python experiments/title_hunt.py --seconds-per-anchor 1.5 --nodes-per-anchor 6000
```

For the phrase search, write the saved JSON's `anchors` list to a text file, one
phrase per line, and pass that file with `--anchors-file` and a separate `--out`.
Time limits make exact candidate counts machine-dependent.
