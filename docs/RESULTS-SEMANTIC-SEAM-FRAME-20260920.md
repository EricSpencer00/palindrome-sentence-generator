# Semantic seam frame search — 2026-09-20

This experiment widened the live overhang search from the 38-letter benchmark
to complete subject–predicate frame pairs, then added initial-bearing and
longer literary frame shapes. Both sides were emitted in forward reading
order while their exposed characters were matched online. No finished string
was reversed and no exact row was repaired afterward.

The strict complete-clause lane used 18 frames, a 1,000-word lexical limit,
and a 50,000-state beam. It found 112 exact closures; the longest was 46
letters:

> ten nose lose no set on an aid; diana notes one sole sonnet

The two-pointer and forward/reverse SHA-256 audits pass, but the row contains
a hidden palindromic span and its first clause is not English prose. The
longest row without that mechanical hidden-span diagnostic was 38 letters,
and the only reader-admitted original remains:

> An aide rips nine memos; some men inspire Diana.

The separate poetic/initial-bearing lane found exact rows up to 48 letters,
but all were malformed phrase material with hidden spans. A broader 30-frame
diagnostic reached 60 exact letters and 42 hidden-span-free letters; human
inspection rejects those rows as word salad or clause fragments. Those counts
are diagnostics, not readability results.

The complete machine-readable record is
`artifacts/semantic-seam-frame-search-20260920.json`. The executable is
`experiments/semantic_seam_frame_search_20260920.py`.

Conclusion: this search changes the evidence about coverage, not the readable
record. Exactness is cheap once grammar is relaxed; coherent English is the
remaining constraint.
