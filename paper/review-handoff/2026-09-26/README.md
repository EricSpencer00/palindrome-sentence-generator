# NAACL draft review handoff

This folder holds the anonymous four-page PDF, a short reviewer brief, the full local-judge results report, the current readiness plan, an offline copy of the one-author review page and its instructions, and the frozen reproducibility archive for the September 26, 2026 handoff. The PDF is built from `paper/naacl2027.tex` and `paper/refs.bib` with the repository's ACL style files. `SHA256SUMS` records the handoff-file hashes.

To rebuild in a clean output directory from the repository root:

```sh
tectonic -X compile --outdir /tmp/naacl2027-review paper/naacl2027.tex
```

`SHA256SUMS` records the PDF digest. The manuscript is an exploratory resource-sensitivity report. The attached reviewer brief states the claim boundaries and the three decisions on which feedback is most useful.

The separate one-author review is ready at the private Tailscale link supplied with the handoff. The same self-contained page and instructions are included here if opening it directly is easier. It contains 52 unique items, autosaves in the current browser, supports pause/resume and checkpoint export, and does not transmit ratings. The item instructions explain the letter-level versus word-order distinction. Its export is a first-person pilot, not independent confirmatory evidence.
