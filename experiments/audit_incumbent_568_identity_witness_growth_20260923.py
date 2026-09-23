#!/usr/bin/env python3
"""Audit a novel short identity/witness insertion on the pinned 568 tape.

This preserves a real exact growth child as diagnostic evidence, while the
boundary audit decides whether it is admissible under the no-shortcut rule.
Exactness is not treated as readability or admission.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs" / "audit-incumbent-568-identity-witness-growth-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "290712c5"
LEFT_ANCHOR = "Nora saw Noel live."
RIGHT_ANCHOR = "“Evil Leon” was Aron."
LEFT_INSERT = "Noel was a lie."
RIGHT_INSERT = "Eila saw Leon."


def normalize(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def outside_in(tape: str) -> dict[str, Any]:
    i, j, count = 0, len(tape) - 1, 0
    while i < j:
        count += 1
        if tape[i] != tape[j]:
            return {"exact": False, "first_mismatch": [i, tape[i], j, tape[j]], "comparisons": count}
        i += 1
        j -= 1
    return {"exact": True, "first_mismatch": None, "comparisons": count}


def words(text: str) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    cursor = 0
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        token = normalize(match.group())
        rows.append((token, cursor, cursor + len(token)))
        cursor += len(token)
    return rows


def literal_preflight(phrases: tuple[str, ...]) -> dict[str, Any]:
    command = ["git", "grep", "-n", "-F", "-i"]
    for phrase in phrases:
        command.extend(("-e", phrase))
    command.extend((PREFLIGHT_REVISION, "--", "runs", "experiments", "docs", "data"))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"preflight failed: {result.stderr}")
    return {
        "revision": PREFLIGHT_REVISION,
        "scope": "tracked runs/, experiments/, docs/, and data/",
        "phrases": list(phrases),
        "hits": result.stdout.splitlines(),
        "status": "no_literal_phrase_hits" if not result.stdout else "literal_hits",
    }


def build_payload() -> dict[str, Any]:
    source = json.loads(PARENT_PATH.read_text())
    parent = str(next(r["rendered"] for r in source["rows"] if r.get("working_status") == "working_length_incumbent"))
    parent_tape = normalize(parent)
    parent_hash = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_hash != PARENT_SHA256:
        raise AssertionError("pinned parent identity changed")

    left_end = parent.index(LEFT_ANCHOR) + len(LEFT_ANCHOR)
    right_start = parent.index(RIGHT_ANCHOR)
    if left_end >= right_start:
        raise AssertionError("anchors are not ordered around the retained parent middle")
    left_cut = len(normalize(parent[:left_end]))
    right_cut = len(normalize(parent[:right_start]))
    if (left_cut, right_cut) != (178, 390):
        raise AssertionError(f"unexpected normalized cuts: {(left_cut, right_cut)}")
    if left_cut + right_cut != len(parent_tape):
        raise AssertionError("cuts are not a reflected parent seam")

    rendered = (
        parent[:left_end] + " " + LEFT_INSERT + parent[left_end:right_start]
        + RIGHT_INSERT + " " + parent[right_start:]
    )
    candidate_tape = normalize(rendered)
    left_tape, right_tape = normalize(LEFT_INSERT), normalize(RIGHT_INSERT)
    if left_tape != right_tape[::-1]:
        raise AssertionError("inserted identity/witness clauses do not discharge the local equation")

    scan = outside_in(candidate_tape)
    forward = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome as project_is_palindrome
    project_exact = bool(project_is_palindrome(rendered))
    if not (scan["exact"] and project_exact and forward == reverse):
        raise AssertionError("candidate failed an independent whole-tape exactness check")

    left_words, right_words = words(LEFT_INSERT), words(RIGHT_INSERT)
    left_boundaries = {end for _, _, end in left_words[:-1]}
    reflected_right_boundaries = {len(right_tape) - end for _, _, end in right_words[:-1]}
    shared_boundaries = sorted(left_boundaries & reflected_right_boundaries)
    union_cuts = sorted({0, len(left_tape), *left_boundaries, *reflected_right_boundaries})
    decomposition = []
    for start, end in zip(union_cuts, union_cuts[1:]):
        right_unit = right_tape[len(right_tape) - end : len(right_tape) - start]
        left_unit = left_tape[start:end]
        decomposition.append({
            "left_unit": left_unit,
            "right_authored_unit": right_unit,
            "equation": left_unit == right_unit[::-1],
            "left_whole_token": any(a == start and b == end for _, a, b in left_words),
            "right_whole_token": any(
                len(right_tape) - b == start and len(right_tape) - a == end
                for _, a, b in right_words
            ),
            "self_palindromic_unit": left_unit == left_unit[::-1],
        })

    preflight = literal_preflight((LEFT_INSERT, RIGHT_INSERT))
    return {
        "experiment_id": "audit-incumbent-568-identity-witness-growth-20260923",
        "method": "insert a copular identity allegation and a finite perception witness at the reflected sentence cuts",
        "parent": {"artifact": str(PARENT_PATH.relative_to(ROOT)), "letters": 568, "sha256_normalized": parent_hash},
        "edit": {
            "normalized_reflected_cuts": [left_cut, right_cut],
            "raw_insert_after": LEFT_ANCHOR,
            "raw_insert_before": RIGHT_ANCHOR,
            "left_insert": LEFT_INSERT,
            "right_insert": RIGHT_INSERT,
            "retained_parent_content_unchanged": True,
        },
        "candidate": {
            "rendered": rendered,
            "letters": len(candidate_tape),
            "growth_over_parent": len(candidate_tape) - len(parent_tape),
            "normalized_sha256": forward,
            "local_equation": {
                "left_tape": left_tape,
                "right_tape": right_tape,
                "exact": left_tape == right_tape[::-1],
                "letters_each": len(left_tape),
                "cursor_trace": [
                    {"cursor": i, "left": left_tape[i], "right_from_end": right_tape[-1-i]}
                    for i in range(len(left_tape))
                ],
            },
            "independent_validation": {
                "outside_in_exact": scan["exact"],
                "outside_in_comparisons": scan["comparisons"],
                "project_validator_exact": project_exact,
                "sha256_forward": forward,
                "sha256_reverse": reverse,
                "hashes_equal": forward == reverse,
            },
        },
        "novelty_preflight": {
            "literal_phrase_preflight": preflight,
            "seam_overlap": "The exact insertion cuts [178,390] touch the same broad region as the prior rejected 574-letter replacement [178,204)/[364,390); this is a new phrase pair, not a new seam family or general algorithm.",
            "method_novelty": "candidate-specific identity/witness wording only; no claim of a novel general generator",
        },
        "anti_shortcut_audit": {
            "left_tokens": [t for t, _, _ in left_words],
            "right_tokens": [t for t, _, _ in right_words],
            "left_internal_boundaries": sorted(left_boundaries),
            "reflected_right_internal_boundaries": sorted(reflected_right_boundaries),
            "shared_reflected_boundaries": shared_boundaries,
            "reflected_right_boundary_coverage": len(shared_boundaries) / max(1, len(reflected_right_boundaries)),
            "equation_decomposition": decomposition,
            "decision": "not admitted: Noel/Leon and was/saw are complete reversed-word pairs, and the article `a` is a one-letter self-palindromic unit; exactness and +22 growth do not override the no-shortcut criterion",
        },
        "readability": {
            "human_reader_evidence": False,
            "certified_readable": False,
            "scope": "The inserted clauses are intelligible in isolation; the 590-letter inherited prose remains rough and has not been reader-tested.",
        },
        "status": "exact_growth_diagnostic_rejected_by_shortcut_audit",
        "next_operator": "Retire the word-aligned identity/witness clause pair. Change from isolated reflected insertion to a two-seam discourse composition whose character debt crosses a retained sentence boundary; preregister the four word-boundary masks and require zero whole-token reversal pairs before lexicalization.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "artifact": str(OUT_PATH.relative_to(ROOT)),
        "status": payload["status"],
        "letters": payload["candidate"]["letters"],
        "sha256": payload["candidate"]["normalized_sha256"],
        "admitted": False,
    }, indent=2))
