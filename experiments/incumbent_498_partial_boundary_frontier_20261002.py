"""Grow the verified 498-letter parent at two live partial-word seams.

The first stage opens parent cursors 46/452 and deliberately records the
awkward ``were he|r`` boundary it creates.  The repair changes the editable
seam, not the score: cursors 54/444 turn ``m|any`` and ``na|me`` into complete
words.  Both stages keep live owner/residual/cursor traces and independently
audit every saved child after rendering.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.validator import is_palindrome, normalize


PARENT = ROOT / "runs" / "overhang-growth-from-240-20261001.json"
OUT = ROOT / "runs" / "incumbent-498-partial-boundary-frontier-20261002.json"
PARENT_SHA256 = "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032"

# Pairs are intact event spans.  They are composed around the parent in a
# stack, so the right surface appears in reverse path order.  The ram rows add
# lexical and event content absent from the parent and prior frame bank.
PAIR_LIBRARY = {
    "P": ("A ram saw Leon.", "Noel was Mara."),
    "Q": ("A ram saw Nora.", "Aron was Mara."),
    "U": ("A ram saw Nadia.", "Aidan was Mara."),
    "F1": ("Nadia delivers maps.", "Spam's reviled, Aidan."),
    "F2": ("Nora delivers maps.", "Spam's reviled, Aron."),
    "G1": ("Nora stops rats.", "Star spots Aron."),
    "G2": ("Mara spots rats.", "Star stops Aram."),
    "H1": ("Mara maps Leon.", "Noel, spam Aram."),
    "H2": ("Leon maps Nora.", "Aron, spam Noel."),
    "K": ("Deliver no evil.", "Live on, reviled."),
    "L": ("Draw no maps.", "Spam onward."),
    "R": ("Aidan stops rats.", "Star spots Nadia."),
}

PATHS = {
    "amber": ("P", "F1", "G1", "H1", "K", "R"),
    "blue": ("Q", "F2", "G2", "H2", "L", "R"),
    "green": ("U", "F1", "G2", "H1", "L", "K"),
}

SEAMS = {
    "depth46-diagnostic": {
        "left_cursor": 46,
        "right_cursor": 452,
        "start_word_index": 13,
        "start_fragment": "e",
        "first_full_word_index": 14,
        "last_full_word_exclusive": 148,
        "end_fragment": "he",
        "left_boundary": "Fir",
        "right_boundary": "r, if ",
        "left_word_equation": "fir|e -> fire",
        "right_word_equation": "he|r -> her",
        "status": "diagnostic exact child; right seam reads 'were her'",
    },
    "depth54-repair": {
        "left_cursor": 54,
        "right_cursor": 444,
        "start_word_index": 16,
        "start_fragment": "Any. Me?",
        "first_full_word_index": 18,
        "last_full_word_exclusive": 146,
        "end_fragment": "na",
        "left_boundary": "A poem. ",
        "right_boundary": "me. Opa,",
        "left_word_equation": "m|any -> poem | any (new word boundary)",
        "right_word_equation": "na|me -> name",
        "status": "repaired exact child; both cut words close as ordinary words",
    },
}


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    value = tape(text)
    mismatch = next(
        (
            {"offset": i, "left": value[i], "right": value[-1 - i]}
            for i in range(len(value) // 2)
            if value[i] != value[-1 - i]
        ),
        None,
    )
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {
        "letters": len(value),
        "independent_normalizer_agrees": value == normalize(text),
        "two_pointer_exact": bool(value) and mismatch is None,
        "first_mismatch": mismatch,
        "project_validator_exact": bool(is_palindrome(text)),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def load_parent() -> tuple[dict[str, object], str, str, list[str]]:
    payload = json.loads(PARENT.read_text())
    row = max(payload["rows"], key=lambda item: item["audit"]["letters"])
    rendered = row["rendered"]
    value = tape(rendered)
    assert len(value) == 498
    assert value == value[::-1]
    assert hashlib.sha256(value.encode()).hexdigest() == PARENT_SHA256
    assert is_palindrome(rendered)
    return row, rendered, value, rendered.split()


def middle_surface(words: list[str], spec: dict[str, object]) -> str:
    full = words[
        int(spec["first_full_word_index"]):int(spec["last_full_word_exclusive"])
    ]
    return " ".join([str(spec["start_fragment"]), *full, str(spec["end_fragment"])])


def pair_audit() -> dict[str, dict[str, object]]:
    result = {}
    for key, (left, right) in PAIR_LIBRARY.items():
        left_tape = tape(left)
        right_tape = tape(right)
        assert left_tape == right_tape[::-1]
        result[key] = {
            "left": left,
            "right": right,
            "letters_per_side": len(left_tape),
            "exact_reverse_equation": True,
        }
    return result


def live_trace(path: tuple[str, ...], spec: dict[str, object]) -> list[dict[str, object]]:
    trace = []
    left_cursor = 0
    right_cursor = 0
    for key in path:
        left, right = PAIR_LIBRARY[key]
        residual = tape(left)[::-1]
        trace.append({
            "action": "emit_left_event",
            "pair": key,
            "owner": "L",
            "residual": residual,
            "left_shell_cursor": left_cursor + len(tape(left)),
            "right_shell_cursor": right_cursor,
        })
        left_cursor += len(tape(left))
        trace.append({
            "action": "consume_right_event",
            "pair": key,
            "owner": None,
            "residual": "",
            "left_shell_cursor": left_cursor,
            "right_shell_cursor": right_cursor + len(tape(right)),
        })
        right_cursor += len(tape(right))
    boundary_residual = tape(str(spec["left_boundary"]))[::-1]
    trace.append({
        "action": "open_partial_word_boundary",
        "owner": "L",
        "residual": boundary_residual,
        "left_parent_cursor": spec["left_cursor"],
        "right_parent_cursor": spec["right_cursor"],
        "left_shell_cursor": left_cursor + len(tape(str(spec["left_boundary"]))),
        "right_shell_cursor": right_cursor,
    })
    assert tape(str(spec["right_boundary"])) == boundary_residual
    trace.append({
        "action": "close_partial_word_boundary",
        "owner": None,
        "residual": "",
        "left_parent_cursor": spec["left_cursor"],
        "right_parent_cursor": spec["right_cursor"],
        "left_shell_cursor": left_cursor + len(tape(str(spec["left_boundary"]))),
        "right_shell_cursor": right_cursor + len(tape(str(spec["right_boundary"]))),
    })
    return trace


def render_child(
    words: list[str], parent_tape: str, seam_id: str, path_id: str,
) -> dict[str, object]:
    spec = SEAMS[seam_id]
    path = PATHS[path_id]
    middle = middle_surface(words, spec)
    left_spans = [PAIR_LIBRARY[key][0] for key in path]
    right_spans = [PAIR_LIBRARY[key][1] for key in reversed(path)]
    left = " ".join([*left_spans, str(spec["left_boundary"])])
    right = " ".join([str(spec["right_boundary"]), *right_spans])
    rendered = left + middle + right

    left_cursor = int(spec["left_cursor"])
    right_cursor = int(spec["right_cursor"])
    retained = parent_tape[left_cursor:right_cursor]
    assert tape(middle) == retained
    assert retained == retained[::-1]
    assert tape(left) == tape(right)[::-1]

    result_audit = audit(rendered)
    assert result_audit["letters"] > 530
    assert result_audit["two_pointer_exact"]
    assert result_audit["project_validator_exact"]
    assert result_audit["sha_equal"]

    parent_words = set(re.findall(r"[a-z]+", " ".join(words).casefold()))
    added_words = set(re.findall(r"[a-z]+", (left + " " + right).casefold()))
    new_words = sorted(added_words - parent_words)
    assert "ram" in new_words

    return {
        "id": f"{seam_id}-{path_id}",
        "seam_id": seam_id,
        "path_id": path_id,
        "pair_path": list(path),
        "rendered": rendered,
        "audit": result_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_sha256": PARENT_SHA256,
        "parent_letters": 498,
        "growth_over_parent": result_audit["letters"] - 498,
        "retained_parent_letters": len(retained),
        "new_lexical_content": new_words,
        "new_event_content": [PAIR_LIBRARY[key][0].removesuffix(".") for key in path],
        "live_state": {
            "left_cursor": left_cursor,
            "right_cursor": right_cursor,
            "left_partial_word": spec["left_word_equation"],
            "right_partial_word": spec["right_word_equation"],
            "trace": live_trace(path, spec),
            "final_owner": None,
            "final_residual": "",
        },
        "working_track_debt": [
            "the retained parent center remains rough and is not human-certified",
            "some reverse-facing event clauses use vocative or literary syntax",
            "proper palindromic event spans are allowed only on this construction track",
        ],
        "reader_status": "not promoted; exact working-track frontier only",
    }


def build_payload() -> dict[str, object]:
    _, parent_rendered, parent_tape, words = load_parent()
    pairs = pair_audit()
    rows = [
        render_child(words, parent_tape, seam_id, path_id)
        for seam_id in SEAMS
        for path_id in PATHS
    ]
    rows.sort(key=lambda row: (-int(row["audit"]["letters"]), str(row["id"])))
    repaired = [row for row in rows if row["seam_id"] == "depth54-repair"]
    diagnostic = [row for row in rows if row["seam_id"] == "depth46-diagnostic"]
    return {
        "experiment_id": "incumbent-498-partial-boundary-frontier-20261002",
        "method": (
            "load the independently exact 498-letter parent, open one live "
            "partial-word seam, then move the cursors eight letters inward "
            "to repair its worst boundary while preserving exact growth"
        ),
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "rendered": parent_rendered,
            "letters": 498,
            "sha256": PARENT_SHA256,
            "independent_exact": True,
            "role": "content incumbent",
        },
        "excluded_control": {
            "letters": 530,
            "role": "duplicated-wrapper repetition control only",
            "used_as_parent": False,
        },
        "pair_library": pairs,
        "seams": SEAMS,
        "stats": {
            "paths_per_seam": len(PATHS),
            "independently_exact_children": len(rows),
            "children_over_530": sum(int(row["audit"]["letters"]) > 530 for row in rows),
            "diagnostic_children": len(diagnostic),
            "repaired_children": len(repaired),
            "shortest_letters": min(int(row["audit"]["letters"]) for row in rows),
            "longest_letters": max(int(row["audit"]["letters"]) for row in rows),
        },
        "incumbent_specific_repair": {
            "diagnosed_seam": "depth46: retained 'were he' plus outer 'r' rendered 'were her'",
            "changed_editable_seam": "depth54: retained 'na' plus outer 'me' renders 'name'; the opposing m|any cut is resegmented after 'poem'",
            "cursor_delta": 8,
            "exact_children_after_repair": len(repaired),
            "worst_remaining_seam": "the inherited sequence after 'A poem. Any. Me?' remains discourse-poor",
            "next_operator": (
                "retain the depth54 amber/blue children and re-punctuate or "
                "replace the first inherited event span after 'A poem. Any. Me?' "
                "under the same live cursor equation"
            ),
        },
        "active_frontier": [row["id"] for row in repaired],
        "provenance": {
            "parent_loaded_and_verified_at_runtime": True,
            "parent_is_498_content_incumbent": True,
            "duplicate_530_control_excluded": True,
            "finished_parent_tape_reversal": False,
            "posthoc_character_repair": False,
            "per_candidate_model_scoring": False,
            "human_certified": False,
        },
        "rows": rows,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
