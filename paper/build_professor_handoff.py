#!/usr/bin/env python3
"""Build the method-label-masked first-pass review packet and professor handoff."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MANUSCRIPT_PDF = ROOT / "output" / "pdf" / "palindrome-evaluation" / "naacl2027.pdf"
HANDOFF = PAPER / "review-handoff" / "2026-09-26"
PACKET_SOURCE = PAPER / "human_eval_v4_packet"
REVIEW_PACKET = HANDOFF / "first-person-review-packet"
REVIEW_PACKET_ZIP = HANDOFF / "first-person-review-packet.zip"
HANDOFF_ZIP = PAPER / "review-handoff" / "NAACL2027-PROFESSOR-HANDOFF-2026-09-26.zip"

PACKET_FILES = (
    "reviewer.html",
    "instructions.md",
    "validate_answers.py",
    "answer.schema.json",
    "packet_manifest.json",
    "validation_report.md",
)

HANDOFF_FILES = (
    "README.md",
    "naacl2027.pdf",
    "naacl2027.tex",
    "refs.bib",
    "acl.sty",
    "acl_natbib.bst",
    "REVIEWER-BRIEF.md",
    "FRAMING-REVISION-20260926.md",
    "LOCAL-LLM-READABILITY-AUDIT-RESULTS.md",
    "AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md",
    "READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md",
    "analyze_readability_length_construction_20260926.py",
    "NAACL-READINESS-PLAN.md",
    "READABILITY-SCORE-AUDIT.md",
    "DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md",
    "START-HERE.txt",
    "REVIEW-INSTRUCTIONS.md",
    "reviewer.html",
    "validate_answers.py",
    "answer.schema.json",
    "packet_manifest.json",
    "validation_report.md",
    "first-person-review-packet.zip",
)

TAILNET_REVIEW_URL = "https://erics-mac-mini.taild34f34.ts.net/review"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_zip(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def start_here() -> bytes:
    return f"""FIRST-PASS REVIEW — START HERE

1. Open `reviewer.html` in a desktop browser.
2. Read `instructions.md` for the full 1–5 scoring anchors. Select **Start / Resume**.
   Rate each of the 52 strings for grammaticality
   and understandability from 1 to 5. The one-line paraphrase is optional.
3. **Pause** any time. Reopen the same file in the same browser to continue;
   answers autosave locally when browser storage is available.
4. At the end, select **Export answers** and keep the downloaded JSON file.
   If local autosave is unavailable, confirm the file appears in Downloads
   before selecting **I saved the checkpoint**.
5. To validate the exported JSON, run
   `python3 validate_answers.py /path/to/exported-answers.json` in this folder.

This is one author's exploratory pilot, not independent or confirmatory human
evidence. The strings are letter-level palindromes; none reverses word order.
The reviewer has already seen both Qwen judges' aggregate scores and one
example score, so this pass is not blind to prior model results. Rate from the displayed
wording alone; do not treat the ratings as model validation.
The page works offline and does not send ratings anywhere.

Optional private route for devices on the same Tailscale network:
{TAILNET_REVIEW_URL}
""".encode()


def build() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MANUSCRIPT_PDF, HANDOFF / "naacl2027.pdf")
    for name in ("REVIEWER-BRIEF.md", "FRAMING-REVISION-20260926.md"):
        shutil.copyfile(PAPER / name, HANDOFF / name)
    packet_contents = {name: (PACKET_SOURCE / name).read_bytes() for name in PACKET_FILES}
    packet_contents["START-HERE.txt"] = start_here()
    REVIEW_PACKET.mkdir(parents=True, exist_ok=True)
    for name, data in packet_contents.items():
        (REVIEW_PACKET / name).write_bytes(data)
    deterministic_zip(REVIEW_PACKET_ZIP, packet_contents)

    # Keep convenient loose copies current for reviewers who do not want to
    # extract the rating packet ZIP.
    shutil.copyfile(PACKET_SOURCE / "reviewer.html", HANDOFF / "reviewer.html")
    shutil.copyfile(PACKET_SOURCE / "instructions.md", HANDOFF / "REVIEW-INSTRUCTIONS.md")
    for name in ("validate_answers.py", "answer.schema.json", "packet_manifest.json", "validation_report.md"):
        shutil.copyfile(PACKET_SOURCE / name, HANDOFF / name)
    (HANDOFF / "START-HERE.txt").write_bytes(start_here())
    shutil.copyfile(PAPER / "NAACL-READINESS-PLAN.md", HANDOFF / "NAACL-READINESS-PLAN.md")
    readiness = (HANDOFF / "NAACL-READINESS-PLAN.md").read_text()
    readiness = readiness.replace("../output/pdf/palindrome-evaluation/naacl2027.pdf", "naacl2027.pdf")
    readiness = readiness.replace("../experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md", "DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md")
    readiness = readiness.replace("../experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md", "AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md")
    readiness = readiness.replace("../experiments/READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md", "READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md")
    (HANDOFF / "NAACL-READINESS-PLAN.md").write_text(readiness)
    shutil.copyfile(ROOT / "experiments" / "LOCAL-LLM-READABILITY-AUDIT-RESULTS-20260926.md", HANDOFF / "LOCAL-LLM-READABILITY-AUDIT-RESULTS.md")
    qwen3_report = (ROOT / "experiments" / "AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md").read_text()
    with zipfile.ZipFile(PAPER / "v4-reproducibility.zip") as archive:
        archived_run_hash = sha256(archive.read("runs/aws-qwen3-32b-readability-audit-20260926.json"))
    if archived_run_hash not in qwen3_report:
        raise ValueError("Qwen3 results report does not identify the archived run record hash")
    (HANDOFF / "AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md").write_text(qwen3_report)
    shutil.copyfile(ROOT / "experiments" / "READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md", HANDOFF / "READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md")
    shutil.copyfile(ROOT / "experiments" / "analyze_readability_length_construction_20260926.py", HANDOFF / "analyze_readability_length_construction_20260926.py")
    shutil.copyfile(ROOT / "experiments" / "DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md", HANDOFF / "DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md")
    shutil.copyfile(PAPER / "READABILITY-SCORE-AUDIT.md", HANDOFF / "READABILITY-SCORE-AUDIT.md")
    for name in ("naacl2027.tex", "refs.bib", "acl.sty", "acl_natbib.bst"):
        data = (PAPER / name).read_text()
        if name.endswith(".bst"):
            data = "\n".join(line.rstrip() for line in data.splitlines()) + "\n"
        (HANDOFF / name).write_text(data)

    readme = f"""# NAACL 2027 professor review handoff

This bundle contains the anonymous manuscript PDF and source, reviewer brief,
readiness plan, reproducible diagnostic reports, and a separate first-pass
rating packet. The paper is titled *Evaluating Exact Palindrome Generation
Beyond Character Validity*. It examines how positive shuffle-relative
likelihood coexists with floor-level absolute model ratings, with construction
and length providing context. This framing revision uses saved evidence and
contains no new experiments or model scoring calls. No human ratings have
been collected. The packet defines letter-level versus word-order palindromes.

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

The private Tailscale route is {TAILNET_REVIEW_URL}. Tailscale Serve is limited
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
"""
    (HANDOFF / "README.md").write_text(readme)

    contents = {name: (HANDOFF / name).read_bytes() for name in HANDOFF_FILES}
    sums = "".join(f"{sha256(data)}  {name}\n" for name, data in sorted(contents.items()))
    (HANDOFF / "SHA256SUMS").write_text(sums)
    contents["SHA256SUMS"] = sums.encode()
    deterministic_zip(HANDOFF_ZIP, contents)

    # Fail closed if either reviewer-facing ZIP accidentally contains an
    # unblinding key, shuffle labels, or the complete evidence archive.
    forbidden = ("sealed_key.json", "shuffle_controls_key.json", "v4-reproducibility.zip")
    for archive_path in (REVIEW_PACKET_ZIP, HANDOFF_ZIP):
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if any(any(token in name for token in forbidden) for name in names):
                raise RuntimeError(f"review ZIP contains a provenance-revealing file: {archive_path}")
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"corrupt ZIP member {bad!r} in {archive_path}")

    print(f"rating packet: {REVIEW_PACKET_ZIP} ({sha256(REVIEW_PACKET_ZIP.read_bytes())})")
    print(f"professor handoff: {HANDOFF_ZIP} ({sha256(HANDOFF_ZIP.read_bytes())})")
    print(f"handoff files: {len(contents)}; sealed keys and full repro archive excluded")


if __name__ == "__main__":
    build()
