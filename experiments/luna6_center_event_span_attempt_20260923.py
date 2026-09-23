"""One fresh center-event sentence attempt against the pinned 568 parent."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT = ROOT / "runs/luna6-center-event-span-attempt-20260923.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def raw_offset(text: str, offset: int) -> int:
    n = 0
    for i, c in enumerate(text):
        if c.isascii() and c.isalpha():
            if n == offset:
                return i
            n += 1
    if n == offset:
        return len(text)
    raise ValueError(offset)


def outside_in(tape: str) -> dict:
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {"exact": i >= j, "letters": len(tape), "matched_outer_pairs": i,
            "first_mismatch": None if i >= j else {
                "offset_left": i, "offset_right": j, "left": tape[i], "right": tape[j]},
            "left_residual": tape[i:i + 40],
            "right_reverse_residual": tape[max(0, j - 39):j + 1][::-1]}


def main() -> dict:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    p = json.loads(PARENT.read_text())
    parent = next(r["rendered"] for r in p["rows"]
                  if r["id"] == "outer-causal-scene-568-working-incumbent")
    tape = letters(parent)
    if len(tape) != 568 or sha(tape) != PARENT_SHA:
        raise AssertionError("pinned parent identity changed")
    if not outside_in(tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("pinned parent failed independent exact audit")

    sentence = "Ned found ravens."
    center = 284
    literal = "Ned found ravens."
    match = subprocess.run(["git", "grep", "-F", literal, "HEAD", "--",
                            "experiments", "runs", "docs", "data"],
                           cwd=ROOT, capture_output=True, text=True)
    if match.returncode not in (0, 1):
        raise RuntimeError(match.stderr)
    if match.returncode == 0:
        raise AssertionError("authored literal already exists in tracked history")

    raw = raw_offset(parent, center)
    attempted = parent[:raw] + sentence + " " + parent[raw:]
    rendered_tape = letters(attempted)
    pointer = outside_in(rendered_tape)
    project_exact = bool(is_palindrome(attempted))
    forward, reverse = sha(rendered_tape), sha(rendered_tape[::-1])
    words = ["ned", "found", "ravens"]
    token_reversals = sorted({(a, b) for a in words for b in words if a != b and a[::-1] == b})
    self_pal = sorted(w for w in words if w == w[::-1])

    return {
        "experiment_id": "luna6-center-event-span-attempt-20260923",
        "status": "fresh_authored_center_sentence_but_not_palindromic",
        "novelty_preflight": {
            "parent": {"letters": 568, "sha256": PARENT_SHA, "exact": True},
            "center_cut": {"normalized_offset": center, "relation": "568 / 2 = 284",
                           "raw_context_before": parent[max(0, raw - 24):raw],
                           "raw_context_after": parent[raw:raw + 32]},
            "exact_literal_absent_from_tracked_HEAD": True,
            "source": "one freshly authored event sentence; no seed/catalogue line knowingly reused"},
        "authored_sentence": {
            "text": sentence,
            "parse": "Ned (agent) found (past transitive verb) ravens (plural theme).",
            "surface_quality": "complete, ordinary English event sentence; not human-rated",
            "shortcut_audit": {"whole_token_reversal_pairs": [list(x) for x in token_reversals],
                               "self_palindromic_tokens": self_pal,
                               "singleton_tokens": [],
                               "repeated_tokens": [],
                               "shortcut_free": not (token_reversals or self_pal)}},
        "center_span_equation": {
            "normalized_tape": letters(sentence),
            "reverse": letters(sentence)[::-1],
            "exact_palindromic_span": letters(sentence) == letters(sentence)[::-1],
            "matched_prefix": pointer["matched_outer_pairs"] - center,
            "first_span_mismatch": pointer["first_mismatch"]},
        "attempted_full_rendering": attempted,
        "candidate_audit": {
            "letters": len(rendered_tape), "growth": len(rendered_tape) - 568,
            "independent_outside_in": pointer, "project_validator_exact": project_exact,
            "forward_sha256": forward, "reverse_sha256": reverse,
            "sha_equal": forward == reverse,
            "exact": pointer["exact"] and project_exact and forward == reverse,
            "admitted": False,
            "reason": "The English event sentence does not form a palindromic center span; the exact outer parent pairs close to offset 284, then `n` conflicts with the final `s`."},
        "reader_status": "No human ratings; no readability certification."}


if __name__ == "__main__":
    result = main()
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"letters": result["candidate_audit"]["letters"],
                      "growth": result["candidate_audit"]["growth"],
                      "exact": result["candidate_audit"]["exact"],
                      "mismatch": result["candidate_audit"]["independent_outside_in"]["first_mismatch"]}, sort_keys=True))
