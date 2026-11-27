# Packet checksum and validation report

- Packet: `palindrome-v4-pilot-9229cdfdf761dace`
- Frozen source: `runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json`
- Source SHA-256: `9229cdfdf761dace9824686bd2f9ed285b30a12c798045995d6fa834e3130bf0`
- Generator SHA-256: `e156e9daf396d5bb1c126e4141aab765d3e8218875ad57b722b86d7d1038b085`
- HTML template SHA-256: `47bdf55b38711a0103a54d3a39eb0d2969ab4de6e6daf7a5c0bcf2d92ea2685c`
- Answer validator SHA-256: `d6b57cdac7dc97d744ce65a4d4038fa4a8cf479d87e4f0ffef8db7e9fb9a5a24`
- Candidate rows: 66; unique texts embedded once: 52.
- Exact letter-palindrome check: passed for all 52 embedded candidates.
- Same-word shuffle controls archived separately: 12 (maximum 12); each is distinct, is not a candidate, and preserves its source word multiset, punctuation, and separators.
- Reviewer page: single local HTML file with embedded content and code; no third-party assets, server, or network calls.
- Scope: one-author first-person exploratory pilot; no independent-rater or confirmatory claim.
- Prior-score exposure: the reviewer has already seen aggregate Qwen results and one example score; ratings are not blind to prior model outputs.
- Determinism: packet files omit build timestamps; sample inclusion, opaque IDs, display order, and control selection use frozen hash/seed rules independent of method labels and ratings.

## Generated-file SHA-256

| File | SHA-256 |
| --- | --- |
| `answer.schema.json` | `4365eb0ea804d5c397f708d2ce0cb20b79e1e9b87e998c51053c45e595e49992` |
| `instructions.md` | `476e09eec69cd9e9afdff72d24954145bba277af5b9ce907e8630b38443cef04` |
| `reviewer.html` | `c6ed5e81701909668789586372c3c4bbfc72011cab807c803d8fca564831fc3f` |
| `sealed_key.json` | `efeb367df0eace41ce797ef0e7843537182abf44d6bdc74a88f2e09c2ffd1312` |
| `shuffle_controls_key.json` | `a2fd0a43cae683f30c75d593f7d88255dd14a61111e514742549ea54e87c6644` |
| `generate_packet.py` | `e156e9daf396d5bb1c126e4141aab765d3e8218875ad57b722b86d7d1038b085` |
| `validate_answers.py` | `d6b57cdac7dc97d744ce65a4d4038fa4a8cf479d87e4f0ffef8db7e9fb9a5a24` |

Generated-file hashes are repeated in `packet_manifest.json`; the manifest intentionally does not hash itself.
