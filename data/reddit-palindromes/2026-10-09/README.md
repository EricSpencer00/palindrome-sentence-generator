# Credited r/palindromes titles

Collected **196 distinct public posts** on October 9, 2026. Each record keeps
the observed title, Reddit poster, profile link, post permalink, and collection
source. Credit names the poster; original authorship is not independently
verified. These examples are references, not project-generated originals.

## Exactness audit

| Result | Count |
|---|---:|
| Distinct posts | 196 |
| Distinct exact ASCII-letter palindromes | 189 |
| Numbers-only title, excluded | 1 |
| Non-ASCII letters, held separately | 6 |
| Repeated exact letter tapes across distinct posts | 0 |
| Longest exact title, in letters | 137 |

The checker ignores spaces, punctuation, case, and digits, following the
repository's letter-only convention. A title with no letters is excluded.
Non-ASCII letters are held for a separate normalization decision rather than
silently discarded or replaced. Those six titles are **not labeled invalid**.
The only other exclusion is `6/26/26`, which has no letters. All 189 supported
nonempty titles pass full reversal; exactness does not establish readability.

- [All original titles and provenance](posts.json)
- [Readable titles and credits, including exclusions](audit/CREDITS.md)
- [Exactness results and mirrored word spans](audit/audit.json)

## Collection evidence

The combined corpus has 225 observations of 196 posts. Observations are grouped
by Reddit post ID only after checking that the title and poster agree.
Overlapping captures keep every observation; the Atom feed supplies publication
timestamps where available. The three original captures remain available:

- [100 Atom-feed posts](feed-posts.json), with the [source snapshot](feed-snapshots/page-01.xml).
- [50 rendered browser titles](browser-posts.json), extracted with adjacent author links.
- [75 indexed listing titles](indexed-posts.json), with permalinks resolved from the actual title links.

The indexed root, hot, and rising listings were cached at different times;
they are not a live chronological sample. Profile links in that capture use
the observed username. This is neither an exhaustive crawl nor a random sample.
It covers titles, not comment threads or image transcription.

The collection requested five feed pages of 100. The first page succeeded;
the second returned HTTP 429. Collection stopped without retry, keeping the
100 successful feed records. The combined set uses those records and the
earlier captures. A public JSON request had returned 403; the public Atom feed
was accessible without login.

## What long words constrain

For every word, the audit reflects its letter span onto the other side of the
palindrome and records the words that overlap it. Word boundaries need not
match. A count includes partial words and any self-overlap at the center.

In [“No IT, a Zin, a grocery rec organization.”](https://www.reddit.com/r/palindromes/comments/1wxwi6m/no_it_a_zin_a_grocery_rec_organization/)
by [u/Waterpark_Enthusiast](https://www.reddit.com/user/Waterpark_Enthusiast/),
the 12-letter `organization` constrains six words: `No`, `IT`, `a`, `Zin`, `a`,
and part of `grocery`.

In [“Put rats at a startup”](https://www.reddit.com/r/palindromes/comments/1wpt1zb/put_rats_at_a_startup/)
by [u/whenindrime](https://www.reddit.com/user/whenindrime/), `startup` mirrors
`Put rats`. This is why word length alone is a poor penalty. Judge the forced
phrase in context. The overlap count measures spelling constraints, not
readability or a proven cause of awkward writing.

## Reproduce or extend

Rebuild the audit offline:

```sh
python3 tools/audit_credited_palindromes.py \
  data/reddit-palindromes/2026-10-09/posts.json \
  --out data/reddit-palindromes/2026-10-09/audit
```

Collect into a **new directory**; stop on server errors and retain partial data:

```sh
python3 tools/collect_reddit_palindromes.py --out /tmp/palindrome-feed-new --pages 5
python3 tools/audit_credited_palindromes.py /tmp/palindrome-feed-new/feed-posts.json \
  --out /tmp/palindrome-feed-new/audit
```

Before using examples to tune a generator, group authors, source threads and
related constructions across splits. Keep credits attached during export and
deduplication. Exclude retrieved or adapted examples from independent-novelty
claims. These titles have no grammar or meaning ratings and are not joined to
[Eric's 183 ratings](../../reader-feedback/2026-10-09/README.md).
