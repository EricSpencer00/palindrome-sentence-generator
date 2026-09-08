# Chunk inventory smoke benchmark

> **Retracted.** The benchmark passed multi-word atomic units directly to a
> single-word Brown-tag lookup. Any result containing a phrase was therefore
> discarded before it could become a hit. Its throughput measurement remains a
> real trie-cost observation; its phrase-yield conclusion does not. The script
> now flattens units before the grammar check and records phrase use explicitly.

Run 3 September 2026. `experiments/chunk_scaling.py`, local MPS-capable
machine; every arm received eight seconds per seed. This is a screening result,
not a quality claim: `sentence_like` is the same permissive structural filter
used by the Polaris yield study, and no blind reader judged these outputs.

## Question

Can multi-word chunks buy enough shallower search to offset their larger trie?
They would not reduce seams in the output: a successful result here must still
be one chunk. The only thing under test is whether chunks make a longer single
find more reachable.

## Results

Inventory: 2,000 frequency-ranked words and 5,000 attested bigrams. A
**core** is the fourteen-letter middle cluster, so rewordings of one pattern do
not count as independent findings.

| band | seed | words cores | chunks cores | chunks-only cores |
|---|---:|---:|---:|---:|
| 12--16 | 0 | 356 | 33 | 0 |
| 22--26 | 0 | 0 | 1 | 0 |
| 22--26 | 1 | 1 | 0 | 0 |
| 27--31 | 0 | 0 | 0 | 0 |
| 32--36 | 0 | 0 | 0 | 0 |

At the abundant 12--16 control band, words searched at 421 draws/s and chunks
at 89 draws/s: adding the bigrams costs about 4.7x throughput. At 22--26, each
of words and chunks found one core across two seeds; the chunk seed's six hits
were all variants of `talk law walk late`, while the word seed's two hits were
variants of `off ill a feel lee fall if`. There is therefore no replicated
evidence that the chunk inventory improves independent-find yield.

The two longer-band cells are all zero at this budget. They are explicitly
uninformative, not evidence of no yield.

## Decision

Do not promote phrase chunks as the route to longer coherent palindromes. They
pay a large, measured throughput cost, still produce template families, and
cannot solve the seam problem: a composition of multiple chunks has already
lost at its first seam (`RESULTS-seams.md`).

### Addendum, 4 September

The directional experiment is no longer prospective: the backward scorer beat
its matched forward control on the per-token proxy at every tested weight and
beat Zipf by +0.423 while preserving 24/24 closures (`docs/training.md`). That
does **not** reverse this chunk decision. It has not yet improved a blinded
quality outcome, and the graph sampler has since shown that long valid single
chunks are abundant but almost entirely score at the coherence floor. The next
test is therefore a paired, blinded proposal evaluation -- not a larger phrase
inventory or a full dual-head training run.

## Benchmarking change

`chunk_scaling.py` now accepts `--seeds` and records the seed in every row.
The earlier version could only report a single shuffled walk, which made the
apparently promising 22--26 chunk result impossible to distinguish from luck.

## Sources checked

- `experiments/RESULTS-seams.md` — the first seam loses all 14 blinded
  comparisons.
- `experiments/RESULTS-polaris-yield.md` — wide vocabulary has the best core
  yield, but yield falls roughly 1.7x per letter.
- [POINTER](https://aclanthology.org/2020.emnlp-main.698/) — progressive,
  hard-constrained insertion is a relevant model family, but requires a
  trained insertion model.
- [PPL-MCTS](https://aclanthology.org/2022.naacl-main.215/) — guided tree
  search can apply a learned constraint at decode time, but its discriminator
  would still need calibration against the project's blind evaluation.
