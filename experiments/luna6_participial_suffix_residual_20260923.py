"""One authored participial-NP suffix equation at the pinned 568 outer seam.

The operator asks whether a reverse parse can start with a participial NP whose
first lexical item consumes characters across the final two source-token
owners. It records a single concrete failure rather than widening a lexicon.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs/luna6-participial-suffix-residual-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isascii() and ch.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def outside_in(tape: str) -> dict:
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {"exact": i >= j, "letters": len(tape), "matched_pairs": i,
            "first_mismatch": None if i >= j else {
                "offset_left": i, "offset_right": j,
                "left": tape[i], "right": tape[j]},
            "left_residual": tape[i:i + 24],
            "right_reverse_residual": tape[max(0, j - 23):j + 1][::-1]}


def main() -> dict:
    parent_payload = json.loads(PARENT_PATH.read_text())
    parent = next(r["rendered"] for r in parent_payload["rows"]
                  if r["id"] == "outer-causal-scene-568-working-incumbent")
    parent_tape = letters(parent)
    from llm_palindrome.validator import is_palindrome
    if len(parent_tape) != 568 or sha(parent_tape) != PARENT_SHA256:
        raise AssertionError("Pinned parent identity changed")
    if not outside_in(parent_tape)["exact"] or not is_palindrome(parent):
        raise AssertionError("Pinned parent failed independent exact validation")

    # Targeted preflight against committed work. Generic `draw` strings exist
    # in historical text data; this exact authored event and owner geometry do
    # not occur in the committed seam experiments examined here.
    preflight = {
        "checked_artifacts": [
            {"artifact": "experiments/luna6_three_token_cross_owner_20260923.py",
             "collision_check": "that lane closes `Mara saw rats` as `Star was a ram` with aligned reversed tokens; this probe instead uses a participial-NP chart and a different complete source event"},
            {"artifact": "experiments/luna6_grammatical_resegmentation_outer_20260923.py",
             "collision_check": "96-row SVO/adjective/PP bank, no participial-NP production"},
            {"artifact": "experiments/incumbent_568_outer_event_resegmentation_20260923.py",
             "collision_check": "multi-clause outer replacement with separate clause surfaces; no terminal participial residual chart"},
            {"artifact": "experiments/incumbent_568_bilateral_typed_grammar_chart_20261002.py",
             "collision_check": "typed chart at internal [194,374] seam; not this terminal participial attachment topology"},
        ],
        "exact_source_sentence_found_in_committed_experiments": False,
        "generic_draw_occurrences_in_historical_data": "yes; not used as source text",
        "signature": "pinned-568 outer suffix-first | participial adjective-NP initial production | final token `draw` plus preceding token reversal | live cursor",
        "distinct_for_one_bounded_obstruction_test": True,
    }

    # One freshly authored, intact source sentence, deliberately outside the
    # previous SVO word bank. A participial-NP parse is expected to begin
    # `Warded ...`; the source suffix emits `ward` from draw, then `n...` from
    # children, so the participial `-ed` residual is contradicted immediately.
    y = "Mara watched the children draw."
    y_tape = letters(y)
    reverse_y = y_tape[::-1]
    target_prefix = "warded"
    cursor = 0
    while cursor < min(len(reverse_y), len(target_prefix)) and reverse_y[cursor] == target_prefix[cursor]:
        cursor += 1
    source_token_owners = [
        {"source_token": "draw", "reverse_surface": "ward", "reverse_span": [0, 4]},
        {"source_token": "children", "reverse_surface": "nerdlihc", "reverse_span": [4, 12]},
        {"source_token": "the", "reverse_surface": "eht", "reverse_span": [12, 15]},
        {"source_token": "watched", "reverse_surface": "dehctaw", "reverse_span": [15, 22]},
        {"source_token": "Mara", "reverse_surface": "aram", "reverse_span": [22, 26]},
    ]
    actual = reverse_y[cursor:cursor + 16]
    expected = target_prefix[cursor:]
    obstruction = {
        "target_constituent": "Warded (past-participial adjective) + NP ...",
        "target_prefix": target_prefix,
        "matched_prefix": reverse_y[:cursor],
        "cursor": cursor,
        "expected_next_characters": expected,
        "actual_next_characters": actual,
        "first_contradiction": {"offset": cursor, "expected": expected[0], "actual": actual[0]},
        "owner_at_contradiction": "source token `children` reversed as `nerdlihc`",
        "remaining_reverse_tape": reverse_y[cursor:],
        "grammar_debt": "No finite participial-NP clause can continue `Warded` from this exact residual without changing the authored suffix; the first required `e` is forced to `n`.",
        "shortcut_note": "The provisional `ward` stem is fully owned by reverse(draw); treating it as a complete word would be a forbidden aligned reversal. It is retained only as a partial residual, not a clause parse.",
    }

    return {
        "experiment_id": "luna6-participial-suffix-residual-20260923",
        "method": "one authored suffix-first participial-NP residual attempt",
        "parent": {"artifact": str(PARENT_PATH.relative_to(ROOT)), "letters": len(parent_tape),
                   "sha256": PARENT_SHA256, "exact_outside_in": True,
                   "project_validator_exact": bool(is_palindrome(parent))},
        "novelty_preflight": preflight,
        "authored_realization": {"Y": y, "letters": len(y_tape), "reverse_Y": reverse_y,
                                 "provenance": "new hand-authored complete event sentence; no seed/catalogue text"},
        "source_token_owners": source_token_owners,
        "result": {"complete_reverse_parse": False, "rendered_child": None,
                   "candidate_admitted": False, "residual_obstruction": obstruction,
                   "live_cursor": cursor, "residual": reverse_y[cursor:]},
        "next_topology_preflight": {
            "initial_pivot_checked": "interior reduced-relative attachment",
            "collision": "committed relative-head, attachment-aware relative, normalized multiword-relative, recursive-clause, and typed NP-relative experiments already cover closely related owner grammars",
            "alternate_signature_checked": "verb-particle placement across an outer seam, with split-object/particle ownership and no reversed finished-word pair",
            "alternate_collision_found": False,
            "search_scope": "experiments and runs; targeted particle/phrasal-verb signature scan returned no matches",
        },
        "next_topology": "Pivot to a split phrasal-verb particle owner map, not another relative clause or suffix lexical bank: choose one transitive particle verb, place its object between verb and particle on one side, and have the reflected clause consume the particle/object boundary in a different grammatical role. Carry the particle as an independent typed slot and demand that content words span adjacent owners; preflight one exact parent seam before authoring a single frame.",
        "reader_status": "no candidate rendered; no readability claim",
    }


if __name__ == "__main__":
    result = main()
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"experiment": result["experiment_id"], "Y": result["authored_realization"]["Y"],
                      "reverse_prefix": result["authored_realization"]["reverse_Y"][:18],
                      "cursor": result["result"]["live_cursor"],
                      "residual": result["result"]["residual"][:24],
                      "expected": result["result"]["residual_obstruction"]["expected_next_characters"],
                      "actual": result["result"]["residual_obstruction"]["actual_next_characters"][:12]}, sort_keys=True))
