"""Build a reproducible blinded reader package for a promoted candidate.

This builder is intentionally fail-closed.  It cannot turn a diagnostic row,
catalogue sentence, or automatic score into reader evidence: every candidate
must already pass the independent exact tape check and shared mechanical gate.
Each event is assigned one of four conditions per rater (candidate, candidate
shuffle, intact prose control, or its shuffle), so no rater sees competing
versions of the same event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from llm_palindrome.admission import mechanical_admission_checks

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CONDITIONS = ("candidate", "candidate_shuffle", "intact_prose", "intact_shuffle")
MAX_CONTROL_LENGTH_DELTA = 5
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def letters(text: str) -> str:
    return "".join(WORD_RE.findall(text.lower())).replace("'", "")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def shuffled(text: str, seed: int) -> str:
    tokens = words(text)
    if len(tokens) < 3:
        raise ValueError("reader controls need at least three words")
    source = letters(text)
    identity = list(range(len(tokens)))
    for attempt in range(256):
        order = identity[:]
        random.Random(seed + attempt).shuffle(order)
        if order == identity:
            order[0], order[1] = order[1], order[0]
        control = " ".join(tokens[i] for i in order) + "."
        control_tape = letters(control)
        # A control must preserve the word multiset while visibly changing
        # order, and it must not accidentally remain a letter palindrome.
        if control == text or control_tape == source:
            continue
        if control_tape == control_tape[::-1]:
            continue
        return control
    raise ValueError("could not construct a changed, non-palindromic shuffle")


def _validated_provenance(row: dict) -> dict:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("candidate provenance must be a structured record")
    required_false = (
        "source_sentences_copied",
        "catalogue_imported",
        "borrowed_text",
        "reversed_finished_sentence",
        "word_order_symmetry",
        "repeated_self_palindromic_unit",
    )
    missing = [key for key in required_false if key not in provenance]
    if missing:
        raise ValueError(f"candidate provenance is missing explicit flags: {', '.join(missing)}")
    if any(provenance[key] is not False for key in required_false):
        raise ValueError("candidate provenance contains a disallowed source or shortcut flag")
    for key in ("generator_sha256", "source_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value.lower()):
            raise ValueError(f"candidate provenance requires a 64-hex {key}")
    return provenance


def _require_length_match(candidate: str, control: str, index: int,
                          max_delta: int = MAX_CONTROL_LENGTH_DELTA) -> None:
    candidate_letters = len(letters(candidate))
    control_letters = len(letters(control))
    if abs(candidate_letters - control_letters) > max_delta:
        raise ValueError(
            f"intact control {index} is not length-matched: "
            f"candidate={candidate_letters}, control={control_letters}, "
            f"max_delta={max_delta}"
        )


def candidate_text(row: dict) -> str:
    text = row.get("rendered") or row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("candidate row has no rendered text")
    tape = letters(text)
    if len(tape) < 100 or tape != tape[::-1]:
        raise ValueError("candidate must be an exact palindrome of at least 100 letters")
    checks = mechanical_admission_checks(text, min_letters=100, max_letters=20_000)
    if not all(checks.values()):
        failed = ", ".join(sorted(k for k, value in checks.items() if not value))
        raise ValueError(f"candidate failed mechanical admission: {failed}")
    _validated_provenance(row)
    return text


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "candidates", "rendered_probes", "rendered_candidates"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
    raise ValueError("candidate file must contain a list or a recognized row list")


def build(candidate_file: Path, prose_file: Path, out_dir: Path,
          raters: int = 24, seed: int = 20260916) -> dict:
    candidates = load_rows(candidate_file)
    controls = load_rows(prose_file)
    if not candidates:
        raise ValueError("no candidate rows supplied")
    if len(candidates) != len(controls):
        raise ValueError("one intact-prose control is required per candidate event")
    events = []
    for index, (candidate, control) in enumerate(zip(candidates, controls), 1):
        ctext = candidate_text(candidate)
        ptext = control.get("rendered") or control.get("text")
        if not isinstance(ptext, str) or len(letters(ptext)) < 20:
            raise ValueError(f"intact control {index} is too short or missing")
        _require_length_match(ctext, ptext, index)
        event = f"E{index:03d}"
        texts = {
            "candidate": ctext,
            "candidate_shuffle": shuffled(ctext, seed + index),
            "intact_prose": ptext,
            "intact_shuffle": shuffled(ptext, seed + 10_000 + index),
        }
        for condition, text in texts.items():
            opaque = hashlib.sha256(f"{seed}:{event}:{condition}".encode()).hexdigest()[:16]
            events.append({"event": event, "condition": condition,
                           "opaque_id": opaque, "text": text,
                           "letters": len(letters(text))})

    internal = out_dir / "internal"
    raters_dir = out_dir / "rater-package"
    internal.mkdir(parents=True, exist_ok=False)
    raters_dir.mkdir()
    key = [{k: row[k] for k in ("event", "condition", "opaque_id", "letters")}
           for row in events]
    (internal / "condition-key.json").write_text(json.dumps(key, indent=2) + "\n")
    instructions = {
        "title": "Blinded English reading study",
        "instructions": (
            "Read each item once as ordinary English. Do not search for the item, "
            "look for palindromes, or edit the wording before answering."
        ),
        "questions": [
            {"id": "grammaticality", "scale": [1, 5]},
            {"id": "meaning", "scale": [1, 5]},
            {"id": "paraphrase", "type": "free_text"},
        ],
    }
    by_event = {event: [row for row in events if row["event"] == event]
                for event in sorted({row["event"] for row in events})}
    for rater in range(1, raters + 1):
        selected = []
        for event_index, (event, rows) in enumerate(by_event.items()):
            # Rotate by event as well as rater so the four conditions are
            # counterbalanced even when a packet contains several events.
            selected.append(rows[(rater - 1 + event_index) % len(CONDITIONS)])
        random.Random(seed + 100_000 + rater).shuffle(selected)
        payload = instructions | {"rater_id": f"R{rater:03d}",
                                  "items": [{"opaque_id": row["opaque_id"],
                                             "text": row["text"]} for row in selected]}
        (raters_dir / f"R{rater:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (raters_dir / "README.md").write_text(
        "Complete one form in the supplied order. Do not discuss or search the items.\n"
    )
    (internal / "analysis-plan.md").write_text(
        "Primary endpoints are grammaticality, meaning, and free-text paraphrase. "
        "Controls are matched to each candidate within five normalized letters. "
        "Report every response by opaque item and condition after the key is opened. "
        "Automated scores are not used as readability evidence.\n"
    )
    manifest = {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out_dir.rglob("*")) if path.is_file()}
    (out_dir / "MANIFEST-SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"events": len(by_event), "raters": raters, "items_per_rater": len(by_event),
            "conditions": list(CONDITIONS), "status": "ready_for_human_collection"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--intact-controls", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raters", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.raters < 4 or args.out_dir.exists():
        raise SystemExit("need at least four raters and a new output directory")
    print(json.dumps(build(args.candidates, args.intact_controls, args.out_dir,
                           args.raters, args.seed), indent=2))


if __name__ == "__main__":
    main()
