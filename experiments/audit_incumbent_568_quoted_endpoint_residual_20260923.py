#!/usr/bin/env python3
"""Reconstruct and reject a quoted-pair endpoint proposal on the 568 tape."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-quoted-endpoint-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "be9191c4"
LEFT_SPAN = (7, 148)
RIGHT_SPAN = (420, 561)
LEFT_RAW = (10, 195)
RIGHT_RAW = (580, 775)
LEFT_PREFIX = "Leon won. "
RIGHT_SUFFIX = " now, Noel."
RIGHT_ENDPOINT = "A liar draws a ward"
LEFT_ENDPOINT_PREFIX = "Draw a"


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def outside_in(tape: str) -> dict[str, Any]:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return {"exact": False, "first_mismatch": [left, tape[left], right, tape[right]]}
        left += 1
        right -= 1
    return {"exact": True, "first_mismatch": None}


def phrase_preflight(phrase: str) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "grep", "-n", "-F", "-i", PREFLIGHT_REVISION, "--", phrase],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"phrase preflight failed: {result.stderr}")
    hits = result.stdout.splitlines()
    return {
        "revision": PREFLIGHT_REVISION,
        "phrase": phrase,
        "scope": "tracked repository paths at the preflight revision",
        "hits": hits,
        "status": "no_literal_hits" if not hits else "literal_hit",
    }


def build_payload() -> dict[str, Any]:
    source = json.loads(PARENT_PATH.read_text())
    row = next(item for item in source["rows"] if item["working_status"] == "working_length_incumbent")
    parent = str(row["rendered"])
    parent_tape = normalize(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")
    if parent[LEFT_RAW[0] : LEFT_RAW[1]] != "Wolf spots Nora. Nadia stops, so Tara rewards Nadia. Nora delivers maps. Mara stops rats. A tub? He maps Aron. Aidan delivers maps. Mara stops rats. A tub? He maps Nora. Deliver no evil":
        raise AssertionError("left raw seam differs from the pinned parent")
    if normalize(parent[RIGHT_RAW[0] : RIGHT_RAW[1]]) != parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]]:
        raise AssertionError("right raw span does not normalize to the reported seam")
    left_source = parent_tape[LEFT_SPAN[0] : LEFT_SPAN[1]]
    right_source = parent_tape[RIGHT_SPAN[0] : RIGHT_SPAN[1]]
    if len(left_source) != 141 or left_source != right_source[::-1]:
        raise AssertionError("reported 141-letter source seam is not an exact reflected pair")
    if parent[: len(LEFT_PREFIX)] != LEFT_PREFIX or parent[RIGHT_RAW[1] :] != RIGHT_SUFFIX:
        raise AssertionError("retained endpoint context differs from the pinned parent")

    right_tape = normalize(RIGHT_ENDPOINT)
    required_left = right_tape[::-1]
    left_prefix_tape = normalize(LEFT_ENDPOINT_PREFIX)
    if not required_left.startswith(left_prefix_tape):
        raise AssertionError("endpoint prefix does not match the reverse-right tape")
    residual = required_left[len(left_prefix_tape) :]

    # The proposed segmentation is deliberately exposed: all four internal
    # right word boundaries reflect onto left word boundaries. This proves the
    # otherwise tempting continuation is just whole-word reversal.
    right_words = ["a", "liar", "draws", "a", "ward"]
    left_words = ["draw", "a", "sward", "rail", "a"]
    right_tape_parts = [normalize(word) for word in right_words]
    left_tape_parts = [normalize(word) for word in left_words]
    right_boundaries: list[int] = []
    left_boundaries: list[int] = []
    cursor = 0
    for part in right_tape_parts[:-1]:
        cursor += len(part)
        right_boundaries.append(cursor)
    cursor = 0
    for part in left_tape_parts[:-1]:
        cursor += len(part)
        left_boundaries.append(cursor)
    reflected_right_boundaries = sorted(len(right_tape) - boundary for boundary in right_boundaries)
    if "".join(left_tape_parts) != required_left or left_boundaries != reflected_right_boundaries:
        raise AssertionError("word-boundary shortcut witness no longer reconstructs")
    if not all(left_word == right_word[::-1] for left_word, right_word in zip(left_words, reversed(right_words))):
        raise AssertionError("the proposed continuation is not a direct word-reversal shortcut")

    return {
        "experiment_id": "audit-incumbent-568-quoted-endpoint-residual-20260923",
        "working_status": "rejected_endpoint_factorization_shortcut; no child emitted",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_sha,
        },
        "operator": {
            "name": "right-endpoint lexicalizer with live reversed-tape residual",
            "provenance": "bounded Luna endpoint attempt; reconstructed and boundary-audited independently",
            "normalized_spans": [list(LEFT_SPAN), list(RIGHT_SPAN)],
            "raw_spans": [list(LEFT_RAW), list(RIGHT_RAW)],
            "retained_left_prefix": LEFT_PREFIX,
            "retained_right_suffix": RIGHT_SUFFIX,
            "source_span_letters_each": len(left_source),
        },
        "endpoint_probe": {
            "right_phrase": RIGHT_ENDPOINT,
            "right_tape": right_tape,
            "required_left_tape": required_left,
            "left_prefix_phrase": LEFT_ENDPOINT_PREFIX,
            "matched_letters": len(left_prefix_tape),
            "residual_after_prefix": residual,
            "continuation_parse": "Draw a sward, rail a…",
            "left_words": left_words,
            "right_words": right_words,
            "right_word_boundaries": right_boundaries,
            "reflected_right_word_boundaries": reflected_right_boundaries,
            "left_word_boundaries": left_boundaries,
            "boundary_alignment_fraction": 1.0,
            "wordwise_reversal_pairs": [
                {"left": left_word, "right": right_word, "exact_reverse": True}
                for left_word, right_word in zip(left_words, reversed(right_words))
            ],
            "novelty_preflight": phrase_preflight(RIGHT_ENDPOINT),
        },
        "candidate": {
            "emitted": False,
            "rendered": None,
            "letters": None,
            "normalized_sha256": None,
            "exact_audit": None,
        },
        "admission": {
            "admitted": False,
            "reason": "The live residual after `Draw a` forces `sward rail a`; every token aligns to an exact reversed right token (`draws/a/sward/rail/a` versus `a/liar/draws/a/ward`), including the forbidden liar/rail pair and self-palindromic `a`. The bounded lane emitted no equal-length insertion longer than the 141-letter removed span, so no child exists to validate.",
            "human_readability_certified": False,
            "reader_evidence": False,
        },
        "next_construction": "Retire this endpoint. Select a disjoint unused 568 seam and construct a boundary-crossing pair whose live residual entails different token boundaries; preflight equal insertion length and positive growth before expanding the scene.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": payload["working_status"],
        "parent_letters": payload["parent"]["letters"],
        "prefix_match": payload["endpoint_probe"]["matched_letters"],
        "residual": payload["endpoint_probe"]["residual_after_prefix"],
        "boundary_alignment_fraction": payload["endpoint_probe"]["boundary_alignment_fraction"],
        "candidate_emitted": payload["candidate"]["emitted"],
    }, indent=2))
