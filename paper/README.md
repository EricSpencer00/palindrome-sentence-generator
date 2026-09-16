# Manuscript workbench -- constructive palindrome generation

The target paper is specified in [TARGET-PAPER-SPEC.md](TARGET-PAPER-SPEC.md).
The manuscript centers a working independent construction loop: live
character obligations coupled to typed grammar and semantic repair, exact
independent validation, rendered long examples, and a blinded reader protocol.
The current snapshot is still a development result: no candidate has yet
passed every gate, so the paper reports the actual frontiers and their next
repairs without calling them readable successes.

[naacl2027.tex](naacl2027.tex) is the method-centered draft. Catalogue-family
and repeated-unit materials remain rejected controls, not paper results. The
historical 2026-09-11 release is quarantined because its bundled claims and
source no longer meet the acceptance standard.

## Build and verify

From the repository root:

```sh
tectonic --keep-logs -o output/pdf paper/naacl2027.tex
python3 experiments/possessive_name_relexicalizer.py --out RUN.json
python3 experiments/verify_possessive_name_candidates.py --input RUN.json --out INDEPENDENT.json
```

All legacy release builders and validators now fail closed. A new release may
be enabled only after an independent candidate passes the hard provenance gate
and the corrected blinded human study described in the draft.

## Current evidence status

The possessive-name run is retained solely as an auditable rejection: its
separate verifier re-enumerates all 100 frozen derivations, recomputes every
gate, requires the rejected-run contract and matching source/provenance hashes,
and finds four exact but zero promotion-eligible closures. The public catalogue
control provenance is recorded in
[`data/catalogue_provenance.json`](../data/catalogue_provenance.json).

The active construction is a ten-lane workbench: character-level decoding,
exact-tape resegmentation, dependency seams, agreement morphology, CFG
intersection, scene lattices, valency/attachment, inflectional boundaries, flat
composition, and semantic slot repair. Each lane writes an intact rendered
surface, independent exact checks, provenance, novelty status, and a concrete
next repair. The authoritative 2026-09-16 ledger is summarized in
`docs/TEN-LUNA-LANE-EVIDENCE-20260916.md` and the aggregate run is
`runs/parallel-luna-readability-diagnostics-20260916.json`.
