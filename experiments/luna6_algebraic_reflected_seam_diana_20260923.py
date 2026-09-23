"""Exact X=reverse(Y)+L splice used to isolate a lexical seam obstruction."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-algebraic-reflected-seam-diana-20260923.json"
EXPECTED_PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_SEAM = 7
LEFT_END = 232
RIGHT_CUT = 561
Y = "Diana"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())


def raw_index_for_letter(text: str, offset: int) -> int:
    seen = 0
    for i, c in enumerate(text):
        if c.isalpha():
            if seen == offset:
                return i
            seen += 1
    if seen == offset:
        return len(text)
    raise IndexError(offset)


def outside_in(text: str) -> dict:
    tape = letters(text)
    mismatch = None
    for i in range(len(tape) // 2):
        j = len(tape) - i - 1
        if tape[i] != tape[j]:
            mismatch = {"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]}
            break
    return {"letters": len(tape), "exact": mismatch is None, "first_mismatch": mismatch}


def main() -> None:
    from llm_palindrome.validator import is_palindrome as project_is_palindrome

    source = json.loads(PARENT_PATH.read_text())
    parent = next(r["rendered"] for r in source["rows"] if r["id"] == "outer-causal-scene-568-working-incumbent")
    p = letters(parent)
    parent_sha = hashlib.sha256(p.encode("ascii")).hexdigest()
    assert len(p) == 568 and parent_sha == EXPECTED_PARENT_SHA256
    assert outside_in(parent)["exact"] and project_is_palindrome(parent)

    U = p[:LEFT_SEAM]
    L = p[LEFT_SEAM:LEFT_END]
    M = p[LEFT_END : len(p) - LEFT_END]
    mirrored_L = p[len(p) - LEFT_END : RIGHT_CUT]
    mirrored_U = p[RIGHT_CUT:]
    y = letters(Y)
    x = y[::-1] + L
    assert mirrored_L == L[::-1]
    assert mirrored_U == U[::-1]
    assert M == M[::-1]
    assert len(M) == 104
    assert x == y[::-1] + L

    left_raw_start = raw_index_for_letter(parent, LEFT_SEAM)
    left_raw_end = raw_index_for_letter(parent, LEFT_END)
    right_raw_cut = raw_index_for_letter(parent, RIGHT_CUT)
    rendered = (
        parent[:left_raw_start]
        + "An aid "
        + parent[left_raw_start:left_raw_end]
        + parent[left_raw_end:right_raw_cut]
        + Y
        + " "
        + parent[right_raw_cut:]
    )
    tape = letters(rendered)
    expected_tape = U + y[::-1] + L + M + L[::-1] + y + U[::-1]
    assert tape == expected_tape
    scan = outside_in(rendered)
    exact_project = project_is_palindrome(rendered)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()

    new_left_tokens = re.findall(r"[a-z]+", ("An aid").lower())
    new_right_tokens = re.findall(r"[a-z]+", Y.lower())
    token_reverse_pairs = sorted({(a, b) for a in new_left_tokens for b in new_right_tokens if a == b[::-1] and a != a[::-1]})
    self_pal = sorted({w for w in new_left_tokens + new_right_tokens if len(w) > 1 and w == w[::-1]})

    result = {
        "experiment_id": "luna6-algebraic-reflected-seam-diana-20260923",
        "status": "independently_exact_but_rejected_as_unreadable_seam_probe",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(p),
            "sha256_normalized": parent_sha,
        },
        "construction": {
            "geometry": {"replace_parent_span": [LEFT_SEAM, LEFT_END], "insert_at_parent_cut": RIGHT_CUT},
            "decomposition": {"U": U, "L": L, "M": M, "Y": y, "X": x, "reverse_Y": y[::-1]},
            "identity_used": "X = reverse(Y) + L; M is a 104-letter palindrome; mirrored retained shells remain unchanged.",
            "Y_letters": len(y),
            "predicted_candidate_letters": len(p) + 2 * len(y),
            "actual_candidate_letters": len(tape),
            "growth_over_parent": len(tape) - len(p),
        },
        "candidate": {
            "rendered": rendered,
            "normalized_sha256": forward,
            "reverse_sha256": reverse,
            "independent_outside_in": scan,
            "project_validator_exact": exact_project,
            "mechanically_exact": scan["exact"] and exact_project and forward == reverse,
        },
        "new_seam_readability_audit": {
            "left_surface_at_seam": "Leon won. An aid Wolf spots Nora...",
            "right_surface_at_seam": "...Aron stops flow Diana now, Noel.",
            "whole_token_reversal_pairs": [list(x) for x in token_reverse_pairs],
            "self_palindromic_words": self_pal,
            "repeated_new_units": [],
            "status": "unreadable_fragment/attachment debt; not a scene and not a reader-worthy result",
            "reader_evidence": False,
        },
        "admission": {
            "admitted_as_working_readable_candidate": False,
            "reason": "The identity guarantees exactness but this minimal Y does not lexicalize either boundary as intact prose. It is a seam diagnostic only; exactness does not certify English readability.",
        },
        "next_operator": {
            "geometry": {"replace_parent_span": [0, 232], "insert_at_parent_cut": 568},
            "exact_signature_found_in_registry_or_history": False,
            "change": "Release the 7-letter outer rails together with [7,232): replace [0,232) and insert Y at end cut 568, so reverse(Y) can be a complete sentence before retained Wolf and Y can be a complete terminal sentence without the forced `now, Noel` suffix. Do not reuse the `Diana` probe.",
            "reason": "The exact identity exposes two grammar debts at this seam: the left surface begins with fragment `An aid` before `Wolf`, and the right joins proper name `Diana` directly to `now, Noel`. The new seam frees both boundary contexts.",
        },
        "provenance": {
            "Y": Y,
            "lexical_source": "fresh hand-authored seam diagnostic; no catalogue import",
            "full_rendered_text_saved": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    assert scan["exact"] and exact_project and forward == reverse
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT_PATH), "letters": len(tape), "growth": len(tape) - len(p), "exact": scan["exact"] and exact_project, "sha256": forward}, indent=2))


if __name__ == "__main__":
    main()
