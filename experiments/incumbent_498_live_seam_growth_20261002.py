"""Grow the verified 498-letter incumbent through one live partial-word seam.

The parent is not treated as a bag of reusable phrases.  We reopen its outer
10 normalized letters on each side, retain the exact 478-letter middle, and
let the new right shell own the final ``e`` in the boundary word ``m|e``.
Every saved row is checked independently after rendering.
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
PRIOR_GROWTH = ROOT / "runs" / "syntax-residual-growth-from-498-20261001.json"
OUT = ROOT / "runs" / "incumbent-498-live-seam-growth-20261002.json"
PARENT_SHA256 = "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032"

SEAM_LEFT = "Nora saw an aide."
SEAM_RIGHT_TAIL = "Diana was Aron."

# Each pair is exact at the character level, but describes a distinct event.
# Pair order is reversed on the right when two pairs surround the open seam.
PAIR_LIBRARY = {
    "A": ("Nora saw Nadia.", "Aidan was Aron."),
    "B": ("Mara saw Leon.", "Noel was Aram."),
    "C": ("Nadia saw Leon.", "Noel was Aidan."),
    "D": ("Nadia saw Aron.", "Nora was Aidan."),
    "E": ("Aidan saw Noel.", "Leon was Nadia."),
}
COMBINATIONS = ("AB", "AC", "BD", "CD")

STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "did", "do", "for", "from",
    "her", "i", "in", "is", "it", "me", "my", "no", "not", "of", "on",
    "one", "or", "so", "that", "the", "to", "two", "was", "we",
}


def independent_tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = independent_tape(text)
    mismatch = next(
        (
            {"offset": i, "left": tape[i], "right": tape[-1 - i]}
            for i in range(len(tape) // 2)
            if tape[i] != tape[-1 - i]
        ),
        None,
    )
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "independent_normalizer_agrees": tape == normalize(text),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "project_validator_exact": bool(is_palindrome(text)),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def load_parent() -> tuple[dict[str, object], str, str]:
    payload = json.loads(PARENT.read_text())
    row = max(payload["rows"], key=lambda item: item["audit"]["letters"])
    rendered = row["rendered"]
    tape = independent_tape(rendered)
    assert len(tape) == 498
    assert tape == tape[::-1]
    assert hashlib.sha256(tape.encode()).hexdigest() == PARENT_SHA256
    assert is_palindrome(rendered)
    return row, rendered, tape


def retained_middle(parent_rendered: str, parent_tape: str) -> tuple[str, str]:
    words = parent_rendered.split()
    assert words[:4] == ["to", "get", "at", "see"]
    assert words[-3:] == ["me", "estate", "got"]

    # Keep all of ``me`` except its last character.  The new right shell owns
    # that ``e`` and renders it without whitespace after the retained ``m``.
    rendered = " ".join([*words[4:-3], "m"])
    tape = parent_tape[10:488]
    assert independent_tape(rendered) == tape
    assert len(tape) == 478
    assert tape == tape[::-1]
    return rendered, tape


def render_child(combo: str, middle_rendered: str) -> tuple[str, list[str], list[str]]:
    left = [PAIR_LIBRARY[key][0] for key in combo] + [SEAM_LEFT]
    right = [SEAM_RIGHT_TAIL] + [PAIR_LIBRARY[key][1] for key in reversed(combo)]
    # No space before the supplied e: it closes the retained partial word m|e.
    rendered = " ".join(left) + " " + middle_rendered + "e. " + " ".join(right)
    return rendered, left, right


def candidate_row(
    *,
    candidate_id: str,
    pair_path: str,
    left_spans: list[str],
    middle_rendered: str,
    supplied_boundary: str,
    right_spans: list[str],
    parent_tape: str,
    parent_words: set[str],
    seam_id: str,
    debt: list[str],
    inherited_residual: str = "",
) -> dict[str, object]:
    rendered = (
        " ".join(left_spans)
        + " "
        + middle_rendered
        + supplied_boundary
        + ". "
        + " ".join(right_spans)
    )
    row_audit = audit(rendered)
    added_words = re.findall(
        r"[a-z]+", " ".join([*left_spans, *right_spans]).casefold()
    )
    new_content_words = sorted(
        {word for word in added_words if word not in parent_words and word not in STOPWORDS}
    )
    left_tape = independent_tape(" ".join(left_spans))
    right_tape = supplied_boundary + independent_tape(" ".join(right_spans))
    assert right_tape == inherited_residual + left_tape[::-1]
    assert row_audit["letters"] > 530
    assert row_audit["two_pointer_exact"]
    assert row_audit["project_validator_exact"]
    assert row_audit["sha_equal"]
    assert new_content_words
    return {
        "id": candidate_id,
        "seam_id": seam_id,
        "pair_path": list(pair_path),
        "rendered": rendered,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_sha256": PARENT_SHA256,
        "parent_letters": 498,
        "growth_over_parent": row_audit["letters"] - 498,
        "left_spans": left_spans,
        "right_spans": right_spans,
        "new_content_words": new_content_words,
        "new_event_signatures": [
            span.removesuffix(".") for span in [*left_spans[:-1], *right_spans[1:]]
        ],
        "added_left_occurrences_in_parent": parent_tape.count(left_tape),
        "added_right_occurrences_in_parent": parent_tape.count(right_tape),
        "repetition_control": False,
        "audit": row_audit,
        "working_track_debt": debt,
    }


def build_payload() -> dict[str, object]:
    parent_row, parent_rendered, parent_tape = load_parent()
    middle_rendered, middle_tape = retained_middle(parent_rendered, parent_tape)
    parent_words = set(re.findall(r"[a-z]+", parent_rendered.casefold()))

    parent_words_rendered = parent_rendered.split()
    primary_middle = " ".join(parent_words_rendered[1:-1])
    assert independent_tape(primary_middle) == parent_tape[2:495]
    assert len(independent_tape(primary_middle)) == 493
    rows = []

    # The actual incumbent state immediately before its historical got -> to
    # closure.  A fresh Get-a-map/Pam-ate equation consumes that live g instead.
    for combo in COMBINATIONS:
        left_spans = [PAIR_LIBRARY[key][0] for key in combo] + ["Pam ate."]
        right_spans = ["Get a map."] + [
            PAIR_LIBRARY[key][1] for key in reversed(combo)
        ]
        rows.append(candidate_row(
            candidate_id=f"depth2-live-g-{combo.lower()}",
            pair_path=combo,
            left_spans=left_spans,
            middle_rendered=primary_middle,
            supplied_boundary="",
            right_spans=right_spans,
            parent_tape=parent_tape,
            parent_words=parent_words,
            seam_id="depth2-live-g",
            inherited_residual="g",
            debt=[
                "the inherited 493-letter middle remains rough generated text",
                "the outer name events retain literary inversion",
                "no human readability claim is made for this working-track child",
            ],
        ))

    seam_left_tape = independent_tape(SEAM_LEFT)
    supplied_boundary = "e"
    seam_tail_tape = independent_tape(SEAM_RIGHT_TAIL)
    residual_before = seam_left_tape[::-1]
    residual_after_boundary = residual_before[1:]
    assert supplied_boundary == residual_before[:1]
    assert seam_tail_tape == residual_after_boundary

    for combo in COMBINATIONS:
        _, left_spans, right_spans = render_child(combo, middle_rendered)
        rows.append(candidate_row(
            candidate_id=f"depth10-live-seam-{combo.lower()}",
            pair_path=combo,
            left_spans=left_spans,
            middle_rendered=middle_rendered,
            supplied_boundary=supplied_boundary,
            right_spans=right_spans,
            parent_tape=parent_tape,
            parent_words=parent_words,
            seam_id="depth10-m|e",
            debt=[
                "the inherited 478-letter middle remains rough generated text",
                "the outer event sequence uses literary inversion and name reuse",
                "no human readability claim is made for this working-track child",
            ],
        ))

    # A second, genuinely different cursor removes more of the inherited outer
    # filler while remaining above the 530-letter repetition control.
    deeper_specs = (
        {
            "depth": 35,
            "middle": " ".join([*parent_words_rendered[11:151], "op"]),
            "partial_word": "op|en",
            "supplied": "en",
            "seam_left": "Nora saw a nine.",
            "seam_right": "Ina was Aron.",
            "expected_sha": "78f8d4f17f2f204fee7619caefa70e28442ff6a0e09e3da263b01d0659bc17be",
        },
        {
            "depth": 39,
            "middle": " ".join([*parent_words_rendered[12:150], "i"]),
            "partial_word": "i|ts",
            "supplied": "ts",
            "seam_left": "Nora saw a post.",
            "seam_right": "Opa was Aron.",
            "expected_sha": "789121ad512b409e9d655cd8e62a56fb18ca8647061cc1027dbec4b401fabb80",
        },
    )
    deeper_pair_path = "ABCE"
    for spec in deeper_specs:
        left_spans = [PAIR_LIBRARY[key][0] for key in deeper_pair_path] + [spec["seam_left"]]
        right_spans = [spec["seam_right"]] + [
            PAIR_LIBRARY[key][1] for key in reversed(deeper_pair_path)
        ]
        row = candidate_row(
            candidate_id=f"depth{spec['depth']}-live-seam-abce",
            pair_path=deeper_pair_path,
            left_spans=left_spans,
            middle_rendered=spec["middle"],
            supplied_boundary=spec["supplied"],
            right_spans=right_spans,
            parent_tape=parent_tape,
            parent_words=parent_words,
            seam_id=f"depth{spec['depth']}-{spec['partial_word']}",
            debt=[
                f"the inherited {498 - 2 * spec['depth']}-letter middle remains rough generated text",
                "the added name events are formulaic working-track scaffolding",
                "no human readability claim is made for this working-track child",
            ],
        )
        assert row["audit"]["sha256_forward"] == spec["expected_sha"]
        rows.append(row)

    rows.sort(key=lambda row: (-row["audit"]["letters"], row["id"]))
    return {
        "experiment_id": "incumbent-498-live-seam-growth-20261002",
        "method": "incumbent-specific partial-word seam growth with live owner, residual, and cursor state",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "rendered": parent_rendered,
            "letters": 498,
            "sha256": PARENT_SHA256,
            "independent_exact": True,
            "role": "content-bearing raw-length incumbent",
        },
        "live_seam": {
            "left_cursor": {
                "normalized_offset": 2,
                "token_index": 1,
                "token": "get",
                "token_offset": 0,
            },
            "right_cursor": {
                "normalized_offset": 495,
                "token_index": 160,
                "token": "got",
                "token_offset": 0,
            },
            "removed_left_tape": parent_tape[:2],
            "removed_right_tape": parent_tape[495:],
            "retained_letters": 493,
            "retained_tape_plus_residual_exact": (
                independent_tape(primary_middle) + "g"
            ) == (independent_tape(primary_middle) + "g")[::-1],
            "partial_word": "(et)|g in get",
            "boundary_owner": "R",
            "trace": [
                {
                    "action": "reopen_parent_closure",
                    "surface": "get ... estate",
                    "owner_after": "R",
                    "residual_after": "g",
                    "ownership_proof": "left residual et + get -> right residual g",
                },
                {
                    "action": "emit_right",
                    "surface": "Get a map.",
                    "owner_after": "L",
                    "residual_after": "pamate",
                },
                {
                    "action": "emit_left",
                    "surface": "Pam ate.",
                    "owner_after": None,
                    "residual_after": "",
                },
            ],
        },
        "alternate_live_seams": [
            {
                "normalized_depth": 10,
                "left_cursor": 10,
                "right_cursor": 488,
                "retained_letters": len(middle_tape),
                "partial_word": "m|e",
                "boundary_owner": "R",
                "supplied_residual": supplied_boundary,
                "seam_left": SEAM_LEFT,
                "seam_right_tail": SEAM_RIGHT_TAIL,
                "residual_before_boundary": residual_before,
                "residual_after_boundary": residual_after_boundary,
            },
        ] + [
            {
                "normalized_depth": spec["depth"],
                "left_cursor": spec["depth"],
                "right_cursor": 498 - spec["depth"],
                "retained_letters": 498 - 2 * spec["depth"],
                "partial_word": spec["partial_word"],
                "boundary_owner": "R",
                "supplied_residual": spec["supplied"],
                "seam_left": spec["seam_left"],
                "seam_right_tail": spec["seam_right"],
            }
            for spec in deeper_specs
        ],
        "pair_library": {
            key: {"left": value[0], "right": value[1]}
            for key, value in PAIR_LIBRARY.items()
        },
        "stats": {
            "authored_paths": len(rows),
            "independently_exact_children": len(rows),
            "children_over_530": sum(row["audit"]["letters"] > 530 for row in rows),
            "shortest_letters": min(row["audit"]["letters"] for row in rows),
            "longest_letters": max(row["audit"]["letters"] for row in rows),
        },
        "frontier": {
            "retained_parent": {"letters": 498, "sha256": PARENT_SHA256},
            "prior_content_child": {
                "artifact": str(PRIOR_GROWTH.relative_to(ROOT)),
                "id": "syntax-window-0",
                "letters": 528,
                "sha256": "052505666bf5f660c56b8edf13e2b5282739ef6ea1b93a8be09685f356e7afe0",
            },
            "duplicate_control": {
                "artifact": str(PRIOR_GROWTH.relative_to(ROOT)),
                "id": "syntax-window-1",
                "letters": 530,
                "sha256": "3b44778dcb3f1542d71716f259d0190ebb47b675300f921422622bdf30b7034c",
                "role": "repetition control only",
            },
            "new_children": [
                {"id": row["id"], "letters": row["audit"]["letters"], "sha256": row["audit"]["sha256_forward"]}
                for row in rows
            ],
            "active_diverse_ids": [
                "depth2-live-g-ac",
                "depth2-live-g-cd",
                "depth10-live-seam-ac",
                "depth35-live-seam-abce",
            ],
        },
        "provenance": {
            "parent_loaded_and_verified_at_runtime": True,
            "finished_parent_tape_reversal": False,
            "posthoc_character_repair": False,
            "catalogue_text": False,
            "per_candidate_model_scoring": False,
            "new_event_content_required": True,
        },
        "next_repair": "retain both 554-letter depth-2 maxima and the depth-35 branch that removes 70 filler letters; repair the depth-35 branch's formulaic outer name sequence without changing its op|en cursor equation",
        "rows": rows,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
