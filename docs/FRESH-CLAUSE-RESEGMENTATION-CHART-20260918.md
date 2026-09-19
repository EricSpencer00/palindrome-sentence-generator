# Fresh clause reverse-tape grammar chart (18 September 2026)

This lane uses six freshly authored scene clauses as left arms. For each arm,
the chart consumes the *live reverse character tape* using typed slots
(`DET → ADJ → N → V → P`), with epsilon transitions for optional phrase
slots. It does not reverse a completed sentence, copy a catalogue row, or
search repeated word units. The run is replayable with:

```bash
python3 experiments/fresh_clause_resegmentation_chart_20260918.py
```

The independent two-pointer audit and forward/reverse SHA-256 tape hashes are
stored in `runs/fresh-clause-resegmentation-chart-20260918.json`.

| authored rendered text | letters | chart reach / tape | result |
|---|---:|---:|---|
| At sunrise, the careful keeper opens a quiet archive. | 43 | 0 / 43 | no right chart path |
| By dusk, the patient baker carries warm loaves home. | 42 | 0 / 42 | no right chart path |
| After rain, the young teacher gathers bright paper models. | 48 | 0 / 48 | no right chart path |
| At first light, the harbor pilot studies a folded chart. | 45 | 0 / 45 | no right chart path |
| In spring, the kind gardener waters small tomato plants. | 46 | 0 / 46 | no right chart path |
| Before winter, the village mason repairs an old stone wall. | 48 | 0 / 48 | no right chart path |

The rendered rows above are intact prose controls (not claimed palindromes).
There were zero exact candidates and zero shortcut-gate survivors. The best
control was **“By dusk, the patient baker carries warm loaves home.”** (42
letters; 20 mismatched mirrored pairs; first mismatch `b/e`). The reader gate
therefore remains closed.

The concrete next repair is to add typed plural and inflectional variants at
the first residual reverse-tape character, then rerun these same authored
clauses. This keeps the construction auditable instead of widening into an
inventory sweep.
