#!/usr/bin/env python3
"""Validate a first-person review export and print descriptive summaries only."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SCOPE = "one_author_first_person_pilot_exploratory_not_confirmatory"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_embedded_items() -> tuple[dict[str, str], dict[str, Any]]:
    html = (HERE / "reviewer.html").read_text()
    match = re.search(r'<script id="packet-meta" type="application/json">(.*?)</script>', html, re.S)
    items_match = re.search(r'<script id="packet-items" type="application/json">(.*?)</script>', html, re.S)
    if not match or not items_match:
        raise ValueError("reviewer.html is missing its embedded packet metadata")
    meta = json.loads(match.group(1))
    items = json.loads(items_match.group(1))
    expected = {item["id"]: item["sha256"] for item in items}
    if len(expected) != 52:
        raise ValueError(f"expected 52 unique packet IDs, found {len(expected)}")
    return expected, meta


def validate(payload: dict[str, Any], expected: dict[str, str], meta: dict[str, Any]) -> list[dict[str, Any]]:
    required = {
        "schema_version", "packet_id", "source_input_sha256", "exported_at_utc",
        "review_scope", "item_count", "completed_count", "current_index",
        "started", "paused", "answers",
    }
    if set(payload) != required:
        raise ValueError(f"answer export fields differ from schema: {sorted(set(payload) ^ required)}")
    if payload["schema_version"] != "1.0" or payload["review_scope"] != SCOPE:
        raise ValueError("answer export schema or review scope is unsupported")
    if payload["packet_id"] != meta["packet_id"] or payload["source_input_sha256"] != meta["source_sha256"]:
        raise ValueError("answer export belongs to another packet/source")
    if payload["item_count"] != 52 or not isinstance(payload["answers"], list) or len(payload["answers"]) != 52:
        raise ValueError("answer export must contain exactly 52 item records")
    if not isinstance(payload["current_index"], int) or isinstance(payload["current_index"], bool) or not 0 <= payload["current_index"] < 52:
        raise ValueError("current_index must be an integer from 0 to 51")
    if not isinstance(payload["started"], bool) or not isinstance(payload["paused"], bool):
        raise ValueError("started and paused must be booleans")
    try:
        timestamp = datetime.fromisoformat(payload["exported_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("exported_at_utc must be ISO-8601") from error
    if timestamp.tzinfo is None:
        raise ValueError("exported_at_utc must include a timezone")

    seen: set[str] = set()
    completed = 0
    sums = {"grammaticality": 0, "understandability": 0}
    counts = {"grammaticality": 0, "understandability": 0}
    for answer in payload["answers"]:
        if not isinstance(answer, dict) or set(answer) != {
            "item_id", "item_sha256", "grammaticality", "understandability", "paraphrase"
        }:
            raise ValueError("an answer record has missing or unexpected fields")
        item_id = answer["item_id"]
        if item_id not in expected or item_id in seen or answer["item_sha256"] != expected[item_id]:
            raise ValueError(f"unknown, duplicate, or hash-mismatched item: {item_id!r}")
        seen.add(item_id)
        for field in ("grammaticality", "understandability"):
            value = answer[field]
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5):
                raise ValueError(f"{field} must be null or an integer from 1 to 5")
            if value is not None:
                sums[field] += value
                counts[field] += 1
        if answer["grammaticality"] is not None and answer["understandability"] is not None:
            completed += 1
        paraphrase = answer["paraphrase"]
        if paraphrase is not None and (not isinstance(paraphrase, str) or len(paraphrase) > 500):
            raise ValueError("paraphrase must be null or a string of at most 500 characters")
    if seen != set(expected):
        raise ValueError("answer export does not cover the exact packet item set")
    if payload["completed_count"] != completed:
        raise ValueError("completed_count does not match the two rating fields")
    return [
        {"measure": field, "n_rated": counts[field], "mean": round(sums[field] / counts[field], 3) if counts[field] else None}
        for field in ("grammaticality", "understandability")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path, help="JSON exported by reviewer.html")
    args = parser.parse_args()
    try:
        payload = json.loads(args.answers.read_text())
        expected, meta = load_embedded_items()
        summary = validate(payload, expected, meta)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"invalid review export: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "valid": True,
        "answer_file_sha256": sha256_file(args.answers),
        "review_scope": SCOPE,
        "n_items": 52,
        "n_completed": payload["completed_count"],
        "descriptive_ratings": summary,
        "independent_raters": 1,
        "confirmatory": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
