# Recovered realtime chunk construction

Polaris job **7731661** ran the existing corpus-backed construction on October
9, 2026. All **22 compositions** were exact, including **11 distinct tapes**.
Another **200 warmed requests** passed separate exactness checks.

This restores the chunk construction Eric requested. The previous finite
sentence-template experiments were separate methods and did not establish a
limit on this construction or the defined corpus.

## The algorithm

Compile the verified corpus once, keeping original text, credits and hashes.
Select complete chunks around a topic. Use the existing theme ordering and
nesting core:

```
A B C B A                 complete self-palindromic chunks
A B C B′ A′               different paired halves with reversed letter tapes
```

The character mirror is guaranteed by those chunk equations, independently of
sentence spacing and punctuation. The request step does not search for every
word from scratch. It keeps whole chunk boundaries and returns each selected
outer refrain chunk once. Useful repeated words remain available in ABCBA.

The current refrain preference chooses matching curated centres when available.
The response discloses this preference and the wider matching inventory count.
The paired comparison keeps the historical hierarchy guards: bigrams and
word-length templates at most twice, content words at most three times. Those
guards are not a general test of useful reuse or meaning.

## Measured run

| Measure | Result |
|---|---:|
| One-time corpus compilation | 136.3 ms |
| Themed ABCBA, 100 warm calls, median / p95 | 4.24 / 4.33 ms |
| Paired form, 100 warm calls, median / p95 | 7.95 / 8.03 ms |
| Longest tested thematic refrain | 253 letters, 101 words |
| Longest tested paired construction | 868 letters, 249 words |
| Total driver time | 1.52 seconds |
| Scheduler wall time | 2 seconds |
| Exit status | 0 |

These are in-process request times, including response metadata and credits;
they exclude corpus compilation and network round trips. Warm calls repeated
the same seeded request, not 200 distinct generations. The complete matrix
contains 22 requests: three theme anchors at several ceilings/unit limits, a
fourth anchor, and three paired seeds at 400/1,200/2,500-letter ceilings.

The larger ceilings do not authorize padding to reach them. The selected
`saw` pool has eight curated chunks, yielding 253 letters once all are used.
No whole-text meaning score was generated in this run. The visibly weak paired
output is a capacity comparison, not an improvement over the themed refrain.

## Actual ABCBA output: C001

**103 letters, 43 words**, assembled from three existing catalogue chunks:

> Was it a bar or a bat I saw?
> Was it a car or a cat I saw?
> Stressed was I ere I saw desserts.
> Was it a car or a cat I saw?
> Was it a bar or a bat I saw?

The centre appears once; each outer chunk returns once. This is explicit
refrain structure. Assigning a return slot does not prove a distinct semantic
contribution or independently authored sentences.

## Actual larger refrain: C002

**253 letters, 101 words**:

> Was it a bar or a bat I saw? Was it a car or a cat I saw? Was it a rat I saw?
> Was it a cat I saw? Delia saw I was ailed. God saw I was dog. Able was I ere
> I saw elba. Stressed was I ere I saw desserts. Able was I ere I saw elba.
> God saw I was dog. Delia saw I was ailed. Was it a cat I saw? Was it a rat
> I saw? Was it a car or a cat I saw? Was it a bar or a bat I saw?

All literal returned texts and per-occurrence source records are in
[examples.json](examples.json). The paired 868-letter output is also preserved
there without promoting its weak wording as a best passage.

## Corpus, credits and provenance

The existing inputs are unchanged: 540 v3 bank rows (499 generated), 49 curated
centres, 71 catalogue spellings, 196 Reddit posts and the frozen Brown table.
They compile to 766 distinct exact ASCII-letter chunks and 306 sentence-shaped
generated mirror pairs. Seven input rows are excluded; the inventory preserves
their exclusion reasons and original records. Exactness is not a readability
label.

Deduplication merges sources without dropping credit. Every output occurrence
keeps its original source text, metadata, input row locator and SHA-256.
Catalogue reuse is disclosed; generated bank material keeps its Polaris origin.
The extra [credit verification](local-credit-verification.json) exercises the
actual title posted by u/Kind-Photograph6926, retaining its literal wording and
permalink. That is a local lightweight provenance check, separate from the
Argonne measurements. An explicit `roman` request in the main batch instead
selected the existing curated line; it was not a Reddit result.

The job ran on `x3001c0s13b1n0`, Python 3.11.15, one node/32 reserved CPUs under
EVITA/debug; the constructor used one Python process and no inference.
Queue/start/finish UTC: 19:19:03 / 19:19:24 / 19:19:32. Scheduler overhead is
separate from the measured driver time. No training, private rating transfer,
AWS resource or paid model endpoint was used.

## Reproduction and verification

Use [the runnable constructor](../../tools/realtime_chunk_construct.py), or
[the batch driver](../../tools/polaris/realtime_chunk_batch.py) and
[PBS script](../../tools/polaris/realtime_chunk_batch.pbs).
[Construction notes](../../docs/research/2026-10-09-realtime-chunk-construction.md)
give standalone commands and explain selection scope.

All 16 staged code/input hashes agree with the returned manifest and local
sources. The original results, scheduler record and standard output are under
`returned/`. Local verification repeated normalized reversal and an independent
raw-surface two-pointer check for every complete output, checked target ceilings
and every occurrence's original source hashes. The product API remains separate
from this standalone requested run.

Validation: **60 relevant tests pass**. Task-owned remote scratch was removed
after results and hashes were verified; unrelated jobs and data were preserved.
