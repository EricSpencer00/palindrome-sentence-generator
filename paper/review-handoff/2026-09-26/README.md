# NAACL 2027 professor review handoff

This bundle contains the anonymous manuscript PDF and source, reviewer brief,
readiness plan, reproducible diagnostic reports, and a separate first-pass
rating packet. The paper reports a bundled resource-sensitivity result; it
does not claim the generated strings are readable. The packet explicitly
defines letter-level versus word-order palindromes.

## Read the paper

Start with `naacl2027.pdf`. The included TeX source, bibliography, and ACL
style files can rebuild it from this folder with:

```sh
tectonic -X compile --outdir /tmp/naacl2027-review naacl2027.tex
```

## Optional first-pass rating

Extract `first-person-review-packet.zip` and follow `START-HERE.txt`, or open
the included `reviewer.html` directly. It is a one-file offline page. It
supports pause/resume, saves locally when browser storage is available, and
exports the ratings to JSON. No ratings are transmitted. The export validator
and schema are included beside the page in the ZIP.

The private Tailscale route is https://erics-mac-mini.taild34f34.ts.net/review. Tailscale Serve is limited
to the tailnet; see the [Tailscale Serve documentation](https://tailscale.com/docs/features/tailscale-serve).
HTTPS returned 200 with certificate validation on the checked hosts. The
custom name `review.espencer.me` is not configured. Use the offline page if
the reviewer is not on the tailnet.

The ratings are one author's exploratory pilot, not independent or
confirmatory evidence. The intended reviewer has already seen both Qwen
judges' aggregate scores and one example score, so these ratings are not blind to prior
model results. Rate from the displayed wording alone. Keep the raw export with
the packet and do not open the sealed provenance key before the first rating
pass is complete.

## Evidence archive

The full `v4-reproducibility.zip` is maintained separately in the repository.
It is deliberately not embedded here because it includes the sealed mapping
from opaque review IDs to method and task provenance. Keep it closed until
first-pass ratings are finalized. `SHA256SUMS` covers files in this handoff
folder except itself.
