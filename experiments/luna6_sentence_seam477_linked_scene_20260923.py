"""One bounded linked-scene residual test on a sentence-bounded pinned-568 seam."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-sentence-seam477-linked-scene-20260923.json"
EXPECTED_PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_EVENT = (
    "Nora, a courier, hid the witness ledger under a flooded station and sent "
    "a sealed message to the captain before dawn."
)
RIGHT_EVENT = (
    "The keeper traced a charred note to the harbor, found its source, and "
    "recognized Aaron."
)


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())


def outside_in(text: str) -> dict:
    tape = letters(text)
    first_mismatch = None
    for i in range(len(tape) // 2):
        j = len(tape) - i - 1
        if tape[i] != tape[j]:
            first_mismatch = {"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]}
            break
    return {"letters": len(tape), "exact": first_mismatch is None, "first_mismatch": first_mismatch}


def repeated_ngrams(left_words: list[str], right_words: list[str], n: int = 2) -> list[list[str]]:
    left = {tuple(left_words[i : i + n]) for i in range(len(left_words) - n + 1)}
    right = {tuple(right_words[i : i + n]) for i in range(len(right_words) - n + 1)}
    return [list(x) for x in sorted(left & right)]


def main() -> None:
    from llm_palindrome.validator import is_palindrome as project_is_palindrome

    parent_json = json.loads(PARENT_PATH.read_text())
    parent_rendered = next(row["rendered"] for row in parent_json["rows"] if row["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent_rendered)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    assert len(parent_tape) == 568 and parent_sha == EXPECTED_PARENT_SHA256
    parent_scan = outside_in(parent_rendered)
    assert parent_scan["exact"] and project_is_palindrome(parent_rendered)

    left_tape = letters(LEFT_EVENT)
    right_tape = letters(RIGHT_EVENT)
    required_left = right_tape[::-1]
    cursor = 0
    while cursor < min(len(left_tape), len(required_left)) and left_tape[cursor] == required_left[cursor]:
        cursor += 1
    mismatch = None if cursor == len(required_left) else {
        "cursor": cursor,
        "left": left_tape[cursor],
        "required": required_left[cursor],
    }
    assert len(right_tape) == 70
    assert mismatch == {"cursor": 5, "left": "c", "required": "d"}

    left_words = re.findall(r"[a-z]+", LEFT_EVENT.lower())
    right_words = re.findall(r"[a-z]+", RIGHT_EVENT.lower())
    reversed_token_pairs = sorted({
        (a, b) for a in left_words for b in right_words
        if a == b[::-1] and a != a[::-1]
    })
    repeated_bigrams = repeated_ngrams(left_words, right_words)

    # This is the next materially distinct sentence-shell ownership, selected
    # after the cursor-5 failure. It is recorded, not executed in this run.
    next_geometry = {"replacement_parent_span": [7, 232], "opposing_insert_parent_cut": 561}
    assert next_geometry["opposing_insert_parent_cut"] == len(parent_tape) - next_geometry["replacement_parent_span"][0]
    assert parent_tape[7:10] == "wol"
    assert parent_tape[561:564] == "now"

    hypothetical_letters = len(parent_tape) - (232 - 91) + len(left_tape) + len(right_tape)
    result = {
        "experiment_id": "luna6-sentence-seam477-linked-scene-20260923",
        "status": "rejected_local_equation_mismatch_before_rendering",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "sha256_normalized": parent_sha,
            "independent_outside_in_exact": parent_scan["exact"],
            "project_validator_exact": project_is_palindrome(parent_rendered),
        },
        "novelty_preflight": {
            "geometry": {"replacement_parent_span": [91, 232], "opposing_insert_parent_cut": 477},
            "completed_registry_or_history_collision": False,
            "basis": "Exact signature absent from registry/history search; spans sentence-bounded source context, unlike prior partial phrase seams.",
        },
        "authored_linked_events": {"left": LEFT_EVENT, "right": RIGHT_EVENT},
        "live_equation": {
            "left_letters": len(left_tape),
            "right_letters": len(right_tape),
            "required_left_prefix": required_left,
            "left_prefix_tested": left_tape[: len(right_tape)],
            "matched_letters": cursor,
            "target_letters": len(right_tape),
            "first_mismatch": mismatch,
            "equation_closed": False,
        },
        "shortcut_and_reuse_audit": {
            "whole_token_reversal_pairs": [list(x) for x in reversed_token_pairs],
            "self_palindromic_words": sorted({w for w in left_words + right_words if len(w) > 1 and w == w[::-1]}),
            "cross_event_repeated_bigrams": repeated_bigrams,
            "shared_content_tokens": sorted(set(left_words) & set(right_words)),
            "candidate_admissible": False,
            "notes": "No phrase reused from the rejected 732 events. The local equation fails before any exact-child admission; no full tape was assembled.",
        },
        "candidate": None,
        "hypothetical_length_if_equation_closed": hypothetical_letters,
        "next_operator": {
            **next_geometry,
            "rationale": "Move the left seam to the sentence boundary before `Wolf spots Nora` and the reflected cut to the sentence-tail boundary before `now`; give the wider 225-letter scene shell one bounded live-equation realization.",
            "preflight_exact_signature_found": False,
        },
        "reader_status": "not_applicable_no_candidate",
        "provenance": {
            "source": "freshly authored linked event pair for this bounded local test",
            "whole_candidate_rendered": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT_PATH), "status": result["status"], "cursor": cursor, "target": len(right_tape), "next_geometry": next_geometry}, indent=2))


if __name__ == "__main__":
    main()
