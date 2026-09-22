"""Inventory exact rendered outputs across run artifacts.

This is an audit, not a generator.  It deliberately reports all mechanically
exact rendered strings over the 38-letter seed, then applies only provenance
flags available in the artifact itself.  Readability is left for human review;
the report never promotes a string on a language-model or lexical score.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "exact-artifact-inventory-20260928.json"
SEED_LENGTH = 38


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def exact(text: str) -> bool:
    tape = letters(text)
    return bool(tape) and tape == tape[::-1]


def walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        rendered = value.get("rendered")
        if isinstance(rendered, str) and exact(rendered) and len(letters(rendered)) > SEED_LENGTH:
            yield rendered, path, value
        for key, child in value.items():
            yield from walk(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}/{index}")


def classify(run: dict[str, Any], row: dict[str, Any], rendered: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    prov = run.get("provenance", {})
    novelty = run.get("novelty_preflight", {})
    if not isinstance(prov, dict):
        prov = {}
    if not isinstance(novelty, dict):
        novelty = {}
    status = str(row.get("status", "")) + " " + str(run.get("status", ""))
    blob = json.dumps({"run": run, "row": row}, ensure_ascii=False).lower()
    if "catalogue" in blob or "borrowed" in blob or "fixture" in blob:
        flags.append("catalogue_or_fixture")
    if "repeated" in blob or "self_palindromic" in blob:
        flags.append("repetition_gate_or_warning")
    if "word salad" in blob or "gibberish" in blob:
        flags.append("gibberish_warning")
    if prov.get("human_readability_certified") is False or "reader_gate" in blob:
        flags.append("no_human_readability_evidence")
    if prov.get("finished_tape_reversed") is True or "reverse" in status.lower() and "tape" in status.lower():
        flags.append("reverse_tape_warning")
    if novelty.get("status") in {"closed_borrowed_controls_only", "failed"}:
        flags.append("novelty_not_passed")
    if not flags:
        flags.append("requires_manual_review")
    return ("not_admissible" if any(x in flags for x in ("catalogue_or_fixture", "gibberish_warning", "reverse_tape_warning")) else "manual_review", flags)


def main() -> None:
    records: dict[str, dict[str, Any]] = {}
    skipped_large: list[str] = []
    for path in sorted(RUNS.glob("*.json")):
        # Some historical trace dumps are hundreds of MB.  They are retained
        # evidence, but loading them would make this lightweight audit unsafe
        # on the coordinator; their names are reported for follow-up.
        if path.stat().st_size > 1_000_000:
            skipped_large.append(path.name)
            continue
        try:
            run = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for rendered, pointer, row in walk(run):
            tape = letters(rendered)
            digest = hashlib.sha256(tape.encode()).hexdigest()
            disposition, flags = classify(run if isinstance(run, dict) else {}, row, rendered)
            records.setdefault(digest, {"rendered": rendered, "letters": len(tape), "sha256": digest,
                                        "sources": [], "disposition": disposition, "flags": flags})
            records[digest]["sources"].append({"run": path.name, "pointer": pointer})
    rows = sorted(records.values(), key=lambda row: (-row["letters"], row["sha256"]))
    report = {
        "experiment_id": "exact-artifact-inventory-20260928",
        "purpose": "reconcile all exact >38 outputs before making a best-result claim",
        "seed_letters": SEED_LENGTH,
        "mechanical_rule": "letters-only normalized tape equals its reverse; independent SHA recorded",
        "readability_rule": "programmatic inventory does not certify English; human review remains required",
        "unique_exact_outputs": len(rows),
        "skipped_large_artifacts_over_5mb": skipped_large,
        "rows": rows,
        "best_admissible_readable_claim": "38-letter seed remains the only reader-admitted result in the current artifacts",
        "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"unique_exact_outputs": len(rows), "longest_letters": rows[0]["letters"] if rows else 0,
                      "output": str(OUT)}))


if __name__ == "__main__":
    main()
