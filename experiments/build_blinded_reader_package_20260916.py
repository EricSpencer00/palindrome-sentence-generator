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


def letters(text: str) -> str:
    return "".join(WORD_RE.findall(text.lower())).replace("'", "")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def shuffled(text: str, seed: int) -> str:
    tokens = words(text)
    if len(tokens) < 3:
        raise ValueError("reader controls need at least three words")
    order = list(range(len(tokens)))
    random.Random(seed).shuffle(order)
    if order == list(range(len(tokens))):
        order[0], order[1] = order[1], order[0]
    return " ".join(tokens[i] for i in order) + "."


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
    provenance = row.get("provenance") or row.get("source")
    if not provenance or "catalogue" in str(provenance).casefold():
        raise ValueError("candidate provenance must be independent and non-catalogue")
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
