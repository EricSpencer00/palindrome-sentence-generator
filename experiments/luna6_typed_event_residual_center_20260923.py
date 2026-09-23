"""One residual-first typed-event span test at the pinned 568 midpoint."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-typed-event-residual-center-20260923.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
# Freeze novelty against the committed tree immediately before this experiment
# so rerunning after this artifact has been committed cannot count itself.
PREFLIGHT_REVISION = "bb89ad91587fdc02651ebb0c2459c14b825107e4"
CANDIDATE = "No, Mar, I saw Eve; was I Ramon?"
LOCAL_TAPE = "nomarisaw" + "eve" + "wasiramon"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def raw_offset(text: str, target: int) -> int:
    count = 0
    for i, char in enumerate(text):
        if char.isascii() and char.isalpha():
            if count == target:
                return i
            count += 1
    if count == target:
        return len(text)
    raise ValueError(target)


def outside_in(tape: str) -> dict:
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {
        "exact": i >= j,
        "letters": len(tape),
        "matched_outer_pairs": i,
        "first_mismatch": None if i >= j else {
            "offset_left": i, "offset_right": j,
            "left": tape[i], "right": tape[j],
        },
        "left_residual": tape[i:i + 40],
        "right_reverse_residual": tape[max(0, j - 39):j + 1][::-1],
    }


def main() -> dict:
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome

    parent_json = json.loads(PARENT_PATH.read_text())
    parent = next(row["rendered"] for row in parent_json["rows"]
                  if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA:
        raise AssertionError("pinned 568 parent identity changed")
    if not outside_in(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("pinned 568 parent failed exact validation")

    # Check the authored literal, not just its topology, against tracked history.
    grep = subprocess.run(
        ["git", "grep", "-F", CANDIDATE, PREFLIGHT_REVISION,
         "--", "experiments", "runs", "docs", "data"],
        cwd=ROOT, capture_output=True, text=True)
    if grep.returncode not in (0, 1):
        raise RuntimeError(grep.stderr)
    novelty = grep.returncode == 1

    local_scan = outside_in(LOCAL_TAPE)
    local_validator = bool(is_palindrome(CANDIDATE))
    if letters(CANDIDATE) != LOCAL_TAPE:
        raise AssertionError("rendered phrase does not match explicitly tracked residual")

    center = 284
    raw = raw_offset(parent, center)
    full = parent[:raw] + CANDIDATE + " " + parent[raw:]
    tape = letters(full)
    full_scan = outside_in(tape)
    validator_exact = bool(is_palindrome(full))
    fwd, rev = sha(tape), sha(tape[::-1])

    tokens = [part.lower() for part in CANDIDATE.replace(",", "").replace(";", "").replace("?", "").split()]
    reversals = sorted({(a, b) for a in tokens for b in tokens
                        if a != b and a[::-1] == b})
    self_pal = sorted({token for token in tokens if token == token[::-1]})
    repeated = sorted({token for token in set(tokens) if tokens.count(token) > 1})
    singleton = sorted({token for token in tokens if len(token) == 1})

    return {
        "experiment_id": "luna6-typed-event-residual-center-20260923",
        "status": "exact_working_child_with_explicit_readability_and_shortcut_debt",
        "novelty_preflight": {
            "candidate_literal_absent_from_tracked_preflight_revision": novelty,
            "preflight_revision": PREFLIGHT_REVISION,
            "candidate_literal": CANDIDATE,
            "claim_scope": "incumbent-specific midpoint instantiation only; no claim of a novel general grammar",
            "parent": {"letters": 568, "sha256": PARENT_SHA, "exact": True},
            "center_geometry": {"offset": center, "parent_halves": [284, 284],
                                "topology_status": "reused incumbent-specific midpoint cut; not claimed novel"},
        },
        "typed_event_and_live_parse": {
            "fixed_frame": {"agent": "I", "action": "saw", "theme": "Eve"},
            "left_clause": {"text": "No, Mar, I saw Eve", "parse": "No (response); Mar (vocative); I (agent) saw (past transitive) Eve (theme)."},
            "right_clause": {"text": "was I Ramon?", "parse": "Was (copula, inverted); I (subject); Ramon (predicate nominative)."},
            "residual_accounting": {
                "left_outer_prefix": "nomarisaw",
                "reverse_character_residual": "wasiramon",
                "right_clause_tape": "wasiramon",
                "closure": "nomarisaw | eve | wasiramon",
                "local_normalized_tape": LOCAL_TAPE,
            },
            "semantic_reading": "A dialogue-like denial about seeing Eve followed by an identity question; grammatical pieces, but the discourse link is strained and not reader-validated.",
        },
        "local_audit": {
            "letters": len(LOCAL_TAPE),
            "outside_in": local_scan,
            "project_validator_exact": local_validator,
            "forward_sha256": sha(LOCAL_TAPE),
            "reverse_sha256": sha(LOCAL_TAPE[::-1]),
            "exact": local_scan["exact"] and local_validator and LOCAL_TAPE == LOCAL_TAPE[::-1],
        },
        "rendered_full_tape": full,
        "shortcut_and_repair_debt": {
            "whole_token_reversal_pairs": [list(pair) for pair in reversals],
            "self_palindromic_tokens": self_pal,
            "repeated_tokens": repeated,
            "singleton_tokens": singleton,
            "proper_palindromic_span": "Eve",
            "reader_status": "No reader study; not claimed readable as a whole. The local line is an exact working construction with rough dialogue/discourse and shortcut debt.",
        },
        "full_tape_audit": {
            "letters": len(tape), "growth": len(tape) - len(parent_tape),
            "outside_in": full_scan,
            "project_validator_exact": validator_exact,
            "forward_sha256": fwd, "reverse_sha256": rev,
            "hashes_equal": fwd == rev,
            "exact": full_scan["exact"] and validator_exact and fwd == rev,
            "admitted_working_child": full_scan["exact"] and validator_exact and fwd == rev,
            "display_policy": "Keep the original 568 parent alongside this 589-letter child; do not call the extension reader-ready.",
        },
        "next_repair": "Preserve this exact pair as the local residual witness; next change the event/theme representation so the central Eve span is replaced by a non-self-palindromic, nonrepeated lexical realization while retaining the same live outer residual, then independently retest before wrapping.",
    }


if __name__ == "__main__":
    result = main()
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "local_letters": result["local_audit"]["letters"],
        "local_exact": result["local_audit"]["exact"],
        "full_letters": result["full_tape_audit"]["letters"],
        "growth": result["full_tape_audit"]["growth"],
        "full_exact": result["full_tape_audit"]["exact"],
        "token_reversal_debt": result["shortcut_and_repair_debt"]["whole_token_reversal_pairs"],
    }, sort_keys=True))
