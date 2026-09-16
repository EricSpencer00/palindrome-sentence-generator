# Brown attested residual lattice (16 September 2026)

## Preflight

Registered as `brown-attested-residual-lattice-20260916` with signature
`brown-attested-sentence-pairs|independent-residual-segmentation|pos-shape-filter|cross-boundary-word-lattice|repair-by-attested-span-expansion`.
The route does not reuse the palindrome catalogue or any generated sentence.

## Method

The left author is an intact, POS-filtered Brown sentence (39–140 letters).
Its normalized character tape is reversed and independently segmented through
a frequency-pruned Brown word lattice; segmentation may cross original word
boundaries. A candidate must cover the residual exactly and retain a
sentence-like subject/verb POS shape. The repair pass expands the author pool
to intact adjacent two-sentence spans while preserving source indices.

## Replay and audit

```text
python experiments/brown_attested_residual_lattice_20260916.py
```

Observed: 57,340 Brown sentences; 3,076 eligible authors; 0 lattice
candidates; 0 exact closures; 0 readable closures over 38 letters; 0 repair
candidates. No text was rendered as generated output (`displayed: []`). The
independent tape audit, intact-span provenance, and catalogue-reuse gate all
passed. This is a failed construction attempt: the corpus word lattice could
not spell even one complete reverse residual under the grammar filter.

The concrete repair was executed (adjacent intact two-sentence span expansion)
and also produced zero candidates, so no misleading probe is preserved.
