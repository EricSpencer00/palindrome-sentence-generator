# The existing realtime chunk constructor

Eric explicitly permits construction from the defined corpus, including ABCBA
nesting and useful word reuse. The sentence-template experiments did not replace
that construction. This replay restores a runnable entry point to its existing
core; it does not create another vocabulary or sentence grammar.

## Construction

Load and validate the saved material once. Keep a normalized letter tape and
all original source records for each distinct chunk. Invalid, empty and
unsupported-alphabet inputs remain in an exclusion report. Punctuation, spaces
and case do not contribute to the mirror.

For complete self-palindromic chunks, choose a theme, order questions before
statements using the existing helper, and reflect the selected sequence:

```
A B C B A
A B C D E D C B A
```

Each outer chunk returns once; the turn appears once. It is an explicit refrain.
No chunk is copied fifty times to fill a request. Repeated words inside different
chunks remain eligible. Calling a repeated chunk a return describes its
structural role; it does not automatically prove a new semantic contribution.

For verified paired chunks, the right text can differ from the left text:

```
A B C B′ A′
letters(A′) = reverse(letters(A))
letters(B′) = reverse(letters(B))
```

This uses the existing hierarchy and paragraph renderer. Both halves retain
their sentence boundaries. The current paired replay retains the historical
2/2/3 bigram/template/content-word limits as a named comparison policy. Those
limits are not a definition of useful repetition and are not applied to ABCBA.
Whole-sentence POS shapes alone have not established meaning in the paired bank.

Finally, verify the complete rendered tape, target ceiling, chunk use counts
and lineage. The batch adds a separate raw-surface two-pointer check. Partial
chunks are never clipped to force a length, and missing material is reported.

## Corpus and selection

Inputs are the existing 540-row v3 bank, 49 curated centres, 71 catalogue
spellings, 196 credited Reddit posts and the frozen Brown payload. The index
contains 766 distinct exact ASCII-letter tapes after deduplication, preserving
all original credits. Seven input rows are excluded. Generated material retains
its original Polaris lineage; catalogue or Reddit material is not relabelled
as newly authored text.

The refrain preference is explicit: use curated centres matching the requested
anchor when available, otherwise use matching indexed chunks. The result reports
the full eligible count, curated eligible count and selected pool. This is a
quality preference, not an exhaustive corpus-wide search or a meaning score.
An explicit anchor with no matching material returns `no_construction`.

## Run it

From the repository root, these require only Python's standard library:

```sh
python3 tools/realtime_chunk_construct.py --mode refrain --anchor saw --units 3 --target-letters 1200
python3 tools/realtime_chunk_construct.py --mode refrain --anchor saw --units 31 --target-letters 1200
python3 tools/realtime_chunk_construct.py --mode paired --units 64 --target-letters 1200 --seed 9
python3 tools/realtime_chunk_construct.py --mode refrain --anchor elated --target-letters 500
```

The last request demonstrates original Reddit text and credit. Each response
contains text, complete chunk layout, source metadata, hashes, actual length and
undershoot. Reuse `ChunkConstructor` in a process to avoid loading the corpus on
every request. The retired product API is not used by this runner.

The [Argonne result record](../../runs/argonne-realtime-chunks-20261009/README.md)
includes all requests, actual timings, sources and exact passages. The retrieved
ABCBA method is the existing whole-sentence route; random paired assembly is a
comparison, not a demonstrated improvement in meaning.
