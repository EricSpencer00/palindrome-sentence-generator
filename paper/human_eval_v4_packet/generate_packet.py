#!/usr/bin/env python3
"""Build a blinded, offline first-person review packet from frozen v4 input.

Only the saved v4 static-verifier input is read. The HTML contains no method,
seed, band, or control metadata. The key and same-word shuffle controls are
written to separate files and should remain unopened until the review is done.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PACKET_VERSION = "v4-first-person-pilot-1"
EXPECTED_EXPERIMENT = "directional-search-static-verifier-20260925"
EXPECTED_INPUT_STATUS = "frozen_confirmatory_outputs_and_deterministic_controls"
EXPECTED_CANDIDATE_COUNT = 66
EXPECTED_UNIQUE_COUNT = 52
MAX_CONTROLS = 12
ORDER_SEED = 20260926
CONTROL_TARGET_SEED = 20260927
CONTROL_SHUFFLE_SEED = 20260928
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / Path(
    "runs/directional-search-resource-sensitivity-20260925/"
    "static-verifier-input-aws.json"
)
EXPECTED_INPUT_ARTIFACT_LABEL = DEFAULT_INPUT.relative_to(REPO_ROOT).as_posix()

HERE = Path(__file__).resolve().parent


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def letter_tape(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text)


def opaque_id(prefix: str, text_sha256: str) -> str:
    # IDs depend on content only. They disclose no method or task fields.
    suffix = sha256_bytes(f"{PACKET_VERSION}\0{prefix}\0{text_sha256}".encode())[:12]
    return f"{prefix}-{suffix.upper()}"


def load_frozen_input(path: Path) -> tuple[dict[str, Any], str, bytes]:
    raw = path.read_bytes()
    source_hash = sha256_bytes(raw)
    data = json.loads(raw)
    if data.get("experiment") != EXPECTED_EXPERIMENT:
        raise ValueError(f"unexpected source experiment: {data.get('experiment')!r}")
    if data.get("status") != EXPECTED_INPUT_STATUS:
        raise ValueError(f"unexpected source status: {data.get('status')!r}")
    if data.get("candidate_count") != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("frozen input candidate_count is not 66")
    if data.get("shuffle_count_per_candidate") != 32:
        raise ValueError("frozen input does not contain the expected 32 shuffles per candidate")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("frozen input has no items list")
    if len([row for row in items if row.get("kind") == "candidate"]) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("expected exactly 66 candidate rows")
    return data, source_hash, raw


def collect_candidates(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in data["items"]:
        if row.get("kind") != "candidate":
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"candidate {row.get('id')!r} has empty/non-text content")
        text_hash = sha256_bytes(text.encode("utf-8"))
        if row.get("text_sha256") != text_hash:
            raise ValueError(f"candidate text hash mismatch: {row.get('id')!r}")
        tape = letter_tape(text)
        if not tape or tape != tape[::-1]:
            raise ValueError(f"candidate is not an exact letter palindrome: {row.get('id')!r}")
        source = row.get("source", {})
        if source.get("text_sha256") != text_hash:
            raise ValueError(f"candidate source hash mismatch: {row.get('id')!r}")
        groups[text].append(row)

    if len(groups) != EXPECTED_UNIQUE_COUNT:
        raise ValueError(f"expected 52 unique candidate texts, found {len(groups)}")

    rows: list[dict[str, Any]] = []
    for text, origins in groups.items():
        text_hash = sha256_bytes(text.encode("utf-8"))
        rows.append({
            "item_id": opaque_id("P", text_hash),
            "text": text,
            "text_sha256": text_hash,
            "origins": sorted(origins, key=lambda row: row["id"]),
        })
    # First establish a content-only order, then apply the frozen seed. Neither
    # source IDs nor method labels enter sample inclusion or display order.
    rows.sort(key=lambda row: sha256_bytes(
        f"{ORDER_SEED}\0{row['text_sha256']}".encode()
    ))
    if len({row["item_id"] for row in rows}) != EXPECTED_UNIQUE_COUNT:
        raise ValueError("opaque ID collision")
    return rows, groups


def choose_shuffle_controls(
    data: dict[str, Any],
    candidates: list[dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidate_texts = set(groups)
    candidate_hashes = {sha256_bytes(text.encode("utf-8")) for text in candidate_texts}
    by_candidate_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in data["items"]:
        if row.get("kind") != "candidate_same_word_shuffle":
            continue
        source = row.get("source", {})
        candidate_id = source.get("candidate_id")
        if candidate_id:
            by_candidate_id[candidate_id].append(row)

    # Candidate targets are chosen by hashes of text only, before method labels
    # are joined from the key. Each control is an archived exact shuffle whose
    # provenance says it preserves the candidate's words and formatting.
    target_order = sorted(
        candidates,
        key=lambda row: sha256_bytes(
            f"{CONTROL_TARGET_SEED}\0{row['text_sha256']}".encode()
        ),
    )
    controls: list[dict[str, Any]] = []
    used_control_hashes: set[str] = set()
    for target in target_order:
        origins = groups[target["text"]]
        origin_ids = sorted({origin["id"] for origin in origins})
        options_by_hash: dict[str, dict[str, Any]] = {}
        for candidate_id in origin_ids:
            for row in by_candidate_id.get(candidate_id, []):
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                text_hash = sha256_bytes(text.encode("utf-8"))
                if row.get("text_sha256") != text_hash:
                    raise ValueError(f"shuffle text hash mismatch: {row.get('id')!r}")
                if text in candidate_texts or text_hash in candidate_hashes:
                    continue
                if collections.Counter(word_tokens(text)) != collections.Counter(word_tokens(target["text"])):
                    raise ValueError(f"same-word shuffle does not preserve tokens: {row.get('id')!r}")
                if re.findall(r"[^A-Za-z]+", text) != re.findall(r"[^A-Za-z]+", target["text"]):
                    raise ValueError(f"same-word shuffle changed punctuation/spacing: {row.get('id')!r}")
                source = row.get("source", {})
                if source.get("preserves") != "word multiset, punctuation, whitespace, and each word's original casing":
                    continue
                option = options_by_hash.setdefault(text_hash, {"row": row, "source_rows": []})
                option["source_rows"].append({
                    "candidate_id": source["candidate_id"],
                    "replicate": source.get("replicate"),
                    "shuffle_seed": source.get("seed"),
                })
        options = sorted(
            options_by_hash.values(),
            key=lambda row: (
                sha256_bytes(
                    f"{CONTROL_SHUFFLE_SEED}\0{target['text_sha256']}\0{row['row']['text_sha256']}".encode()
                ),
                row["row"]["text_sha256"],
            ),
        )
        for option in options:
            row = option["row"]
            text_hash = row["text_sha256"]
            if text_hash in used_control_hashes:
                continue
            controls.append({
                "control_id": opaque_id("S", text_hash),
                "text": row["text"],
                "text_sha256": text_hash,
                "target_item_id": target["item_id"],
                "source_rows": sorted(option["source_rows"], key=lambda x: (x["candidate_id"], x["replicate"])),
                "preserves": row["source"].get("preserves"),
            })
            used_control_hashes.add(text_hash)
            break
        if len(controls) == MAX_CONTROLS:
            break
    return controls


def instructions_text() -> str:
    return """# Offline first-person review packet: v4 palindrome outputs

## What this review is

This is one author's blinded first-pass pilot of 52 unique texts that the v4 search had already selected. It is exploratory feedback, not an independent or confirmatory human evaluation. The sample is conditioned on search success and the system's existing selection rule. It cannot estimate generation coverage, show superiority over an external method, or support population-level readability claims.

The texts were taken from the frozen v4 static-verifier input. Repeated identical outputs were shown once. All 52 retained the source study's exact-letter palindrome check. A separate key preserves the original method and task provenance; keep `sealed_key.json` and `shuffle_controls_key.json` closed until this first pass is complete.

## How to use the review page

Open `reviewer.html` directly in a current desktop browser. It uses no server, external service, network request, or library. Press **Start / Resume** to begin. Rate the text as shown. Your work autosaves in that browser on this device after every answer. **Pause** disables editing and preserves the current position. Reopen the same file in the same browser and press **Start / Resume** to continue. If the page reports that local autosave is unavailable, export a checkpoint before closing and restore it after reopening. Use **Export answers** regularly to keep a portable checkpoint. **Import checkpoint** restores a previously exported file for this packet.

The export contains your ratings and anonymized item IDs. Keep a copy separate from the sealed key. Ratings remain on this device unless you explicitly export them; the page sends nothing anywhere. To validate an exported file and print descriptive means, run `python3 paper/human_eval_v4_packet/validate_answers.py /path/to/palindrome-v4-first-person-pilot-answers.json`. The summary does not unblind methods or run significance tests.

## Rate each text on its own

Judge only what the sentence says as written. Do not try to repair it, infer intended wording, or reward it for satisfying a palindrome constraint. Do not compare it with other items. If you cannot recover a coherent reading, say so in the understandability score and leave the paraphrase blank.

**Grammaticality**

1. Not a well-formed sentence.
2. Major grammatical problems; only a fragment or strained reading is possible.
3. Some problems, but a sentence-level reading is possible.
4. Mostly well formed, with at most a minor problem.
5. Well formed and natural as written.

**Understandability**

1. No coherent meaning is recoverable.
2. Meaning is mostly unclear; a reading requires substantial guessing.
3. A plausible gist is recoverable, but important relations or meaning are uncertain.
4. The meaning is mostly clear.
5. The sentence expresses a clear, coherent meaning.

The optional paraphrase is one short line in your own words. It is a check on the reading you actually took from the sentence, not a rewrite or correction.

## Frozen packet provenance

- To regenerate the packet from the saved source: `python3 paper/human_eval_v4_packet/generate_packet.py`.
- Source: `runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json`.
- That saved verifier input contains 66 selected v4 candidates and 32 deterministic same-word shuffles per candidate.
- This review shows all 52 unique candidate texts once, in one fixed randomized order. Same-word shuffles are not shown in the review page; a deterministic set of up to 12 is archived separately for optional later calibration.
- Selection and display order were fixed without reading method labels: include every unique candidate text, derive opaque item IDs from text hashes, then order by the hash of the preregistered constant seed `20260926` plus each text hash. No labels, ratings, or wall-clock timestamps are used to select or order the items.
- `packet_manifest.json` records the source, generator, and output hashes. Regenerating from the same input and generator should reproduce the same packet files byte for byte.

## Interpretation limit

This is one author's first-person pilot. Do not describe it as three-rater agreement, a blinded independent study, or confirmatory evidence. Use it to inspect the items and refine a later evaluation. Preserve the raw export and the frozen packet if you report this pilot.
"""


def answer_schema(packet_id: str, source_hash: str) -> dict[str, Any]:
    rating = {"anyOf": [{"type": "integer", "minimum": 1, "maximum": 5}, {"type": "null"}]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Palindrome v4 first-person pilot answer export",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "packet_id", "source_input_sha256", "exported_at_utc", "review_scope", "item_count", "completed_count", "current_index", "started", "paused", "answers"],
        "properties": {
            "schema_version": {"const": "1.0"},
            "packet_id": {"const": packet_id},
            "source_input_sha256": {"const": source_hash},
            "exported_at_utc": {"type": "string", "format": "date-time"},
            "review_scope": {"const": "one_author_first_person_pilot_exploratory_not_confirmatory"},
            "item_count": {"const": EXPECTED_UNIQUE_COUNT},
            "completed_count": {"type": "integer", "minimum": 0, "maximum": EXPECTED_UNIQUE_COUNT},
            "current_index": {"type": "integer", "minimum": 0, "maximum": EXPECTED_UNIQUE_COUNT - 1},
            "started": {"type": "boolean"},
            "paused": {"type": "boolean"},
            "answers": {
                "type": "array",
                "minItems": EXPECTED_UNIQUE_COUNT,
                "maxItems": EXPECTED_UNIQUE_COUNT,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["item_id", "item_sha256", "grammaticality", "understandability", "paraphrase"],
                    "properties": {
                        "item_id": {"type": "string", "pattern": "^P-[A-F0-9]{12}$"},
                        "item_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                        "grammaticality": rating,
                        "understandability": rating,
                        "paraphrase": {"anyOf": [{"type": "string", "maxLength": 500}, {"type": "null"}]},
                    },
                },
            },
        },
    }


def render_html(packet_id: str, source_hash: str, candidates: list[dict[str, Any]]) -> str:
    safe_items = [{"id": row["item_id"], "text": row["text"], "sha256": row["text_sha256"]} for row in candidates]
    payload = json.dumps(safe_items, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    boot = json.dumps({"packet_id": packet_id, "source_sha256": source_hash}, separators=(",", ":"))
    boot = boot.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (HERE / "reviewer.template.html").read_text()
    if template.count("__ITEMS__") != 1 or template.count("__META__") != 1:
        raise ValueError("reviewer template must contain one item and one metadata placeholder")
    return template.replace("__ITEMS__", payload).replace("__META__", boot)

def build(input_path: Path, output_dir: Path) -> dict[str, Any]:
    data, source_hash, _raw = load_frozen_input(input_path)
    candidates, groups = collect_candidates(data)
    controls = choose_shuffle_controls(data, candidates, groups)
    if len(controls) > MAX_CONTROLS:
        raise AssertionError("control cap exceeded")
    packet_id = f"palindrome-v4-pilot-{source_hash[:16]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    reviewer = render_html(packet_id, source_hash, candidates).encode("utf-8")
    key = {
        "packet_id": packet_id,
        "review_scope": "one_author_first_person_pilot_exploratory_not_confirmatory",
        "source_input_sha256": source_hash,
        "input_manifest_sha256": data["input_manifest_sha256"],
        "search_result_sha256": data["search_result_sha256"],
        "candidates": [
            {
                "item_id": row["item_id"],
                "text": row["text"],
                "text_sha256": row["text_sha256"],
                "source_rows": [
                    {
                        "source_item_id": origin["id"],
                        "pair_id": origin.get("pair_id"),
                        **origin.get("source", {}),
                    }
                    for origin in sorted(row["origins"], key=lambda x: x["id"])
                ],
            }
            for row in candidates
        ],
    }
    controls_key = {
        "description": "Archived deterministic same-word shuffle controls; not embedded in reviewer.html and not part of this first-pass candidate sample.",
        "source_input_sha256": source_hash,
        "selection_protocol": {
            "maximum_controls": MAX_CONTROLS,
            "target_order_seed": CONTROL_TARGET_SEED,
            "shuffle_choice_seed": CONTROL_SHUFFLE_SEED,
            "basis": "hash-ranked unique candidate texts; controls chosen only from saved candidate_same_word_shuffle rows whose provenance states preserved word multiset, punctuation, whitespace, and casing",
        },
        "controls": controls,
    }
    schema = answer_schema(packet_id, source_hash)

    files = {
        "reviewer.html": reviewer,
        "sealed_key.json": (json.dumps(key, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
        "shuffle_controls_key.json": (json.dumps(controls_key, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
        "instructions.md": instructions_text().encode(),
        "answer.schema.json": (json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    }
    validation_report = [
        "# Packet checksum and validation report",
        "",
        f"- Packet: `{packet_id}`",
        f"- Frozen source: `{EXPECTED_INPUT_ARTIFACT_LABEL}`",
        f"- Source SHA-256: `{source_hash}`",
        f"- Generator SHA-256: `{sha256_file(Path(__file__).resolve())}`",
        f"- HTML template SHA-256: `{sha256_file(HERE / 'reviewer.template.html')}`",
        f"- Answer validator SHA-256: `{sha256_file(HERE / 'validate_answers.py')}`",
        f"- Candidate rows: {EXPECTED_CANDIDATE_COUNT}; unique texts embedded once: {len(candidates)}.",
        f"- Exact letter-palindrome check: passed for all {len(candidates)} embedded candidates.",
        f"- Same-word shuffle controls archived separately: {len(controls)} (maximum {MAX_CONTROLS}); each is distinct, is not a candidate, and preserves its source word multiset, punctuation, and separators.",
        "- Reviewer page: single local HTML file with embedded content and code; no third-party assets, server, or network calls.",
        "- Scope: one-author first-person exploratory pilot; no independent-rater or confirmatory claim.",
        "- Determinism: packet files omit build timestamps; sample inclusion, opaque IDs, display order, and control selection use frozen hash/seed rules independent of method labels and ratings.",
        "",
        "## Generated-file SHA-256",
        "",
        "| File | SHA-256 |",
        "| --- | --- |",
    ]
    for name, payload in sorted(files.items()):
        validation_report.append(f"| `{name}` | `{sha256_bytes(payload)}` |")
    validation_report.extend([
        f"| `generate_packet.py` | `{sha256_file(Path(__file__).resolve())}` |",
        f"| `validate_answers.py` | `{sha256_file(HERE / 'validate_answers.py')}` |",
        "",
        "Generated-file hashes are repeated in `packet_manifest.json`; the manifest intentionally does not hash itself.",
        "",
    ])
    files["validation_report.md"] = ("\n".join(validation_report)).encode()
    for name, payload in files.items():
        (output_dir / name).write_bytes(payload)

    manifest = {
        "packet_version": PACKET_VERSION,
        "packet_id": packet_id,
        "review_scope": "one_author_first_person_pilot_exploratory_not_confirmatory",
        "source_artifact": EXPECTED_INPUT_ARTIFACT_LABEL,
        "source_input_sha256": source_hash,
        "source_input_experiment": data["experiment"],
        "source_input_status": data["status"],
        "source_input_manifest_sha256": data["input_manifest_sha256"],
        "source_search_result_sha256": data["search_result_sha256"],
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "template_sha256": sha256_file(HERE / "reviewer.template.html"),
        "answer_validator_sha256": sha256_file(HERE / "validate_answers.py"),
        "candidate_rows": EXPECTED_CANDIDATE_COUNT,
        "unique_candidate_texts": len(candidates),
        "palindrome_exactness_checked": True,
        "controls_archived": len(controls),
        "validation_counts": {
            "frozen_input_items": len(data["items"]),
            "candidate_rows": EXPECTED_CANDIDATE_COUNT,
            "unique_candidate_texts": len(candidates),
            "unique_opaque_item_ids": len({row["item_id"] for row in candidates}),
            "exact_palindromes_checked": len(candidates),
            "archived_shuffle_controls": len(controls),
        },
        "randomization_protocol": {
            "sample": "all unique selected candidate text values from the 66 candidate rows; exact duplicates appear once",
            "method_blinding": "source method/task labels are never read for sample inclusion, item IDs, or display order",
            "opaque_item_ids": "derived from packet version and SHA-256 of text only",
            "display_order": f"sort by SHA-256 of fixed seed {ORDER_SEED} plus each text SHA-256; this hash permutation is independent of method labels and ratings",
            "timestamp": "not used for inclusion, IDs, controls, or order; deterministic artifacts omit build timestamps",
        },
        "generated_files": {name: sha256_bytes(payload) for name, payload in files.items()},
        "manifest_self_hash": "omitted",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    (output_dir / "packet_manifest.json").write_bytes(manifest_bytes)
    validate_output(output_dir, data, source_hash)
    return manifest


def validate_output(output_dir: Path, data: dict[str, Any], source_hash: str) -> None:
    manifest = json.loads((output_dir / "packet_manifest.json").read_text())
    if manifest["source_input_sha256"] != source_hash:
        raise ValueError("manifest source hash mismatch")
    html_text = (output_dir / "reviewer.html").read_text()
    match = re.search(r'<script id="packet-items" type="application/json">(.*?)</script>', html_text, re.S)
    if not match:
        raise ValueError("reviewer.html has no embedded item payload")
    embedded = json.loads(match.group(1))
    if len(embedded) != EXPECTED_UNIQUE_COUNT:
        raise ValueError("reviewer.html does not contain exactly 52 items")
    if len({item["id"] for item in embedded}) != EXPECTED_UNIQUE_COUNT:
        raise ValueError("reviewer.html item IDs are not unique")
    candidate_texts = [row["text"] for row in data["items"] if row.get("kind") == "candidate"]
    expected_unique = set(candidate_texts)
    if {item["text"] for item in embedded} != expected_unique:
        raise ValueError("embedded candidate sample differs from frozen v4 unique candidate texts")
    for item in embedded:
        if sha256_bytes(item["text"].encode()) != item["sha256"]:
            raise ValueError(f"embedded text hash mismatch: {item['id']}")
        tape = letter_tape(item["text"])
        if not tape or tape != tape[::-1]:
            raise ValueError(f"embedded candidate fails exact palindrome check: {item['id']}")
    if any(set(item) != {"id", "text", "sha256"} for item in embedded):
        raise ValueError("reviewer payload contains fields outside opaque ID, text, and text hash")
    for forbidden in ("center_out_project", "outside_in_norvig_hoey_adaptation"):
        if forbidden in match.group(1):
            raise ValueError(f"method metadata leaked into reviewer item payload ({forbidden})")
    key = json.loads((output_dir / "sealed_key.json").read_text())
    if len(key["candidates"]) != EXPECTED_UNIQUE_COUNT:
        raise ValueError("sealed key does not map exactly 52 candidate texts")
    controls = json.loads((output_dir / "shuffle_controls_key.json").read_text())["controls"]
    if len(controls) > MAX_CONTROLS:
        raise ValueError("too many controls")
    control_hashes = [item["text_sha256"] for item in controls]
    if len(control_hashes) != len(set(control_hashes)):
        raise ValueError("control texts are duplicated")
    for name, expected_hash in manifest["generated_files"].items():
        if sha256_file(output_dir / name) != expected_hash:
            raise ValueError(f"generated file hash mismatch: {name}")
    if "localStorage" not in html_text or "Export answers" not in html_text or "Import checkpoint" not in html_text:
        raise ValueError("reviewer page is missing local autosave/export/import functionality")
    for required in ("Start / Resume", "Pause", "current_index", "normalizeState", "URL.createObjectURL", "localStorage.setItem"):
        if required not in html_text:
            raise ValueError(f"reviewer page is missing required offline interaction support: {required}")
    if re.search(r'<script\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=|\bfetch\s*\(|XMLHttpRequest|WebSocket|sendBeacon', html_text, re.I):
        raise ValueError("reviewer page contains an external asset or network API")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    try:
        manifest = build(args.input, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"packet generation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "packet_id": manifest["packet_id"],
        "source_input_sha256": manifest["source_input_sha256"],
        "candidate_rows": manifest["candidate_rows"],
        "unique_candidate_texts": manifest["unique_candidate_texts"],
        "palindrome_exactness_checked": manifest["palindrome_exactness_checked"],
        "controls_archived": manifest["controls_archived"],
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
