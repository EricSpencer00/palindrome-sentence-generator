"""One suffix-first cross-boundary pair at a fresh pinned-568 seam."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT = ROOT / "runs/luna6-suffix-first-new-seam-20260923.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isascii() and c.isalpha())


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def raw_offset(s: str, n: int) -> int:
    count = 0
    for i, c in enumerate(s):
        if c.isascii() and c.isalpha():
            if count == n:
                return i
            count += 1
    if count == n:
        return len(s)
    raise ValueError(n)


def outside_in(t: str) -> dict:
    i, j = 0, len(t) - 1
    while i < j and t[i] == t[j]:
        i += 1
        j -= 1
    return {"exact": i >= j, "letters": len(t), "matched_outer_pairs": i,
            "first_mismatch": None if i >= j else {
                "offset_left": i, "offset_right": j, "left": t[i], "right": t[j]},
            "left_residual": t[i:i + 40],
            "right_reverse_residual": t[max(0, j - 39):j + 1][::-1]}


def token_audit(left: str, right: str) -> dict:
    a, b = re.findall(r"[a-z]+", left.casefold()), re.findall(r"[a-z]+", right.casefold())
    reversals = sorted({(x, y) for x in a for y in b if x[::-1] == y})
    singleton = sorted({x for x in a + b if len(x) == 1})
    self_pal = sorted({x for x in a + b if len(x) > 1 and x == x[::-1]})
    shared = sorted(set(a) & set(b))
    repeated = sorted({x for x in a + b if (a + b).count(x) > 1 and x not in {"a", "an", "the"}})
    return {"left_tokens": a, "right_tokens": b,
            "whole_token_reversal_pairs": [list(x) for x in reversals],
            "singleton_tokens": singleton, "self_palindromic_tokens": self_pal,
            "shared_tokens": shared, "repeated_content_tokens": repeated,
            "shortcut_free": not (reversals or singleton or self_pal or shared or repeated)}


def git_grep_absent(needle: str) -> bool:
    r = subprocess.run(["git", "grep", "-F", needle, "HEAD", "--", "experiments", "runs"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(r.stderr)
    return r.returncode == 1


def main() -> dict:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    payload = json.loads(PARENT.read_text())
    parent = next(r["rendered"] for r in payload["rows"]
                  if r["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA:
        raise AssertionError("pinned parent identity changed")
    parent_exact = outside_in(parent_tape)["exact"] and bool(is_palindrome(parent))
    if not parent_exact:
        raise AssertionError("pinned parent validation failed")

    left_cut, right_cut = 163, 405
    signature = "pinned568|insert@163|mirror@405|suffix-first|cross-token-boundary"
    preflight = {
        "head": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(),
        "signature": signature,
        "signature_absent_from_tracked_experiment_history": git_grep_absent(signature),
        "left_clause_literal_absent": git_grep_absent('Ned owes rent.'),
        "right_clause_literal_absent": git_grep_absent('Tailors sew Oden.'),
        "excluded_geometries": [[81, 487], [20, 548], [178, 390], [16, 552], [120, 448]],
        "seam": {"left": left_cut, "right": right_cut, "reflected_relation": "568 - 163 = 405"},
        "flanks": {
            "left_before": "Now, Noel, did I live?",
            "left_after": "Nora saw Noel live.",
            "right_before": "“Evil Leon” was Aron.",
            "right_after": "Evil I did, Aras.",
            "raw_cut_check": "Both cuts are after sentence punctuation; insertion keeps parent separators and adds complete sentences."},
        "parent": {"letters": len(parent_tape), "sha256": PARENT_SHA, "independently_exact": parent_exact},
    }
    if not all(preflight[k] for k in ("signature_absent_from_tracked_experiment_history",
                                      "left_clause_literal_absent", "right_clause_literal_absent")):
        raise AssertionError("novelty collision")

    left_clause = "Ned owes rent."
    right_clause = "Tailors sew Oden."
    left_tape, right_tape = letters(left_clause), letters(right_clause)
    right_reversed = right_tape[::-1]
    cursor = 0
    while cursor < min(len(left_tape), len(right_reversed)) and left_tape[cursor] == right_reversed[cursor]:
        cursor += 1
    residual = {"cursor": cursor,
                "matched_prefix": left_tape[:cursor],
                "left_char": left_tape[cursor] if cursor < len(left_tape) else None,
                "required_char": right_reversed[cursor] if cursor < len(right_reversed) else None,
                "left_residual": left_tape[cursor:], "required_residual": right_reversed[cursor:]}
    if cursor < 7:
        raise AssertionError("expected suffix-first cross-boundary prefix did not match")

    # Raw offsets are the first letter after each punctuation boundary. The
    # parent already owns the preceding spaces, so additions end in one space.
    i, j = raw_offset(parent, left_cut), raw_offset(parent, right_cut)
    attempted = parent[:i] + left_clause + " " + parent[i:j] + right_clause + " " + parent[j:]
    attempted_tape = letters(attempted)
    pointer = outside_in(attempted_tape)
    project_exact = bool(is_palindrome(attempted))
    forward_sha, reverse_sha = sha(attempted_tape), sha(attempted_tape[::-1])
    audit = token_audit(left_clause, right_clause)

    return {
        "experiment_id": "luna6-suffix-first-new-seam-20260923",
        "operator": "one suffix-first multiword owner realization across a token boundary",
        "novelty_preflight": preflight,
        "authored_pair": {
            "left": left_clause,
            "left_parse": "Ned (subject) owes (present transitive verb) rent (object).",
            "right": right_clause,
            "right_parse": "Tailors (subject) sew (present transitive verb) Oden (object/name).",
            "event_relation": "A debt and the tailoring work are placed in one event frame; relation remains weak and is not a readability claim.",
            "provenance": "Freshly authored for this exact owner equation; exact phrase literals absent from tracked HEAD; no catalogue or seed text."},
        "suffix_first_owners": {
            "partner_final_sequence": [
                {"surface": "sew", "owner": "verb", "right_source_span": [7, 10], "reverse_consumed_span": [4, 7]},
                {"surface": "Oden", "owner": "object/name", "right_source_span": [10, 14], "reverse_consumed_span": [0, 4]}],
            "reverse_suffix": "nedowes",
            "left_opening_owners": [
                {"surface": "Ned", "owner": "subject", "left_span": [0, 3]},
                {"surface": "owes", "owner": "verb", "left_span": [3, 7]}],
            "crosses_word_boundary": True,
            "remaining_owner_equation": "left `rent` vs reverse of right-initial `Tailors` (`sroliat`)"},
        "live_residual": residual,
        "attempted_full_rendering": attempted,
        "candidate_audit": {
            "letters": len(attempted_tape), "growth": len(attempted_tape) - 568,
            "outside_in": pointer, "project_validator_exact": project_exact,
            "forward_sha256": forward_sha, "reverse_sha256": reverse_sha,
            "sha_equal": forward_sha == reverse_sha,
            "exact": pointer["exact"] and project_exact and forward_sha == reverse_sha,
            "shortcut_audit": audit,
            "admitted": False,
            "reason": "The suffix crosses the Ned|owes word boundary and matches exactly, but the next owner conflicts (`r` vs `s`); the full rendering is not exact."},
        "next_axis": "Stop suffix/prefix phrase pairing. Preflight a non-insertion internal owner-map edit (split one existing lexical owner across two grammatical clauses) before any new authored realization; do not move this suffix to another seam or search endings.",
        "reader_status": "No exact candidate; no human readability claim or ratings."}


if __name__ == "__main__":
    result = main()
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"seam": result["novelty_preflight"]["seam"],
                      "residual": result["live_residual"],
                      "letters": result["candidate_audit"]["letters"],
                      "exact": result["candidate_audit"]["exact"],
                      "shortcut_free": result["candidate_audit"]["shortcut_audit"]["shortcut_free"]}, sort_keys=True))
