"""Move the 498-parent frontier to the note/set partial-word seam.

At parent cursors 88/410, ``A note`` reverses to ``et on a``.  The retained
right boundary owns the initial ``s`` of ``still``; the right shell supplies
``et`` and therefore renders the complete word ``set``.  This removes the
previous depth-54 discourse-poor opening while keeping the 498 parent, exact
owner/residual state, and independently audited children.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_partial_boundary_frontier_20261002 import (
    PAIR_LIBRARY,
    PARENT,
    PARENT_SHA256,
    audit,
    load_parent,
    tape,
)


PRIOR = ROOT / "runs" / "incumbent-498-partial-boundary-frontier-20261002.json"
OUT = ROOT / "runs" / "incumbent-498-note-seam-growth-20261002.json"

LEFT_CURSOR = 88
RIGHT_CURSOR = 410
LEFT_BOUNDARY = "A note: "
RIGHT_BOUNDARY = "et on a "

# These paths are deliberately wider than the depth-54 paths because the
# deeper cut removes 68 additional parent letters.  The paths differ in event
# order and lexical choices; none is a repeated wrapper around the 530 control.
PATHS = {
    "amber": ("P", "F1", "G1", "H1", "K", "R", "Q", "L"),
    "blue": ("Q", "F2", "G2", "H2", "L", "R", "U", "K"),
    "green": ("U", "F1", "G2", "H1", "L", "K", "P", "R", "F2"),
}


def middle_surface(words: list[str]) -> str:
    # Offsets 88..410 are: state go two no do two ... get at + first s of
    # still.  Punctuation makes the first six words a quoted instruction and
    # makes the final s available to the right shell as Set.
    middle = " ".join([
        "State: 'Go two.' 'No, do two.'",
        *words[35:134],
        "S",
    ])
    return middle


def trace(path: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    left_shell_cursor = 0
    right_shell_cursor = 0
    for key in path:
        left, right = PAIR_LIBRARY[key]
        residual = tape(left)[::-1]
        left_shell_cursor += len(tape(left))
        rows.append({
            "action": "emit_left_event",
            "pair": key,
            "owner": "L",
            "residual": residual,
            "left_shell_cursor": left_shell_cursor,
            "right_shell_cursor": right_shell_cursor,
        })
        assert tape(right) == residual
        right_shell_cursor += len(tape(right))
        rows.append({
            "action": "consume_right_event",
            "pair": key,
            "owner": None,
            "residual": "",
            "left_shell_cursor": left_shell_cursor,
            "right_shell_cursor": right_shell_cursor,
        })

    boundary_residual = tape(LEFT_BOUNDARY)[::-1]
    assert boundary_residual == tape(RIGHT_BOUNDARY)
    rows.extend([
        {
            "action": "emit_left_note_boundary",
            "owner": "L",
            "residual": boundary_residual,
            "left_parent_cursor": LEFT_CURSOR,
            "right_parent_cursor": RIGHT_CURSOR,
        },
        {
            "action": "consume_right_set_boundary",
            "owner": None,
            "residual": "",
            "left_parent_cursor": LEFT_CURSOR,
            "right_parent_cursor": RIGHT_CURSOR,
            "surface_equation": "A note <-> et on a; retained s + et -> set",
        },
    ])
    return rows


def child(words: list[str], parent_tape: str, path_id: str) -> dict[str, object]:
    path = PATHS[path_id]
    middle = middle_surface(words)
    retained = parent_tape[LEFT_CURSOR:RIGHT_CURSOR]
    assert tape(middle) == retained
    assert retained == retained[::-1]

    left_spans = [PAIR_LIBRARY[key][0] for key in path]
    right_spans = [PAIR_LIBRARY[key][1] for key in reversed(path)]
    left = " ".join(left_spans) + " " + LEFT_BOUNDARY
    right = RIGHT_BOUNDARY + " ".join(right_spans)
    assert tape(left) == tape(right)[::-1]

    rendered = left + middle + right
    result_audit = audit(rendered)
    assert result_audit["letters"] > 530
    assert result_audit["two_pointer_exact"]
    assert result_audit["project_validator_exact"]
    assert result_audit["sha_equal"]
    assert "A note: State:" in rendered
    assert "at Set on a" in rendered

    parent_words = set(re.findall(r"[a-z]+", " ".join(words).casefold()))
    added = set(re.findall(r"[a-z]+", rendered.casefold()))
    new_words = sorted(added - parent_words)
    assert {"note", "ram", "set"} <= set(new_words)

    return {
        "id": f"depth88-note-set-{path_id}",
        "path_id": path_id,
        "pair_path": list(path),
        "rendered": rendered,
        "audit": result_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_sha256": PARENT_SHA256,
        "parent_letters": 498,
        "prior_frontier_artifact": str(PRIOR.relative_to(ROOT)),
        "growth_over_parent": int(result_audit["letters"]) - 498,
        "retained_parent_letters": len(retained),
        "removed_parent_letters_per_side": LEFT_CURSOR,
        "new_lexical_content": new_words,
        "new_event_content": [PAIR_LIBRARY[key][0].removesuffix(".") for key in path],
        "live_state": {
            "left_cursor": LEFT_CURSOR,
            "right_cursor": RIGHT_CURSOR,
            "left_boundary_equation": "A note | state",
            "right_partial_word": "s|et -> set",
            "trace": trace(path),
            "final_owner": None,
            "final_residual": "",
        },
        "working_track_debt": [
            "the retained 322-letter center is exact but still contains rough discourse",
            "some paired events use literary inversion or vocative syntax",
            "the note framing improves the seam but does not certify readability",
        ],
        "reader_status": "not promoted; exact working-track frontier only",
    }


def build_payload() -> dict[str, object]:
    _, parent_rendered, parent_tape, words = load_parent()
    prior = json.loads(PRIOR.read_text())
    prior_lengths = [
        int(row["audit"]["letters"])
        for row in prior["rows"]
        if row["seam_id"] == "depth54-repair"
    ]
    rows = [child(words, parent_tape, path_id) for path_id in PATHS]
    rows.sort(key=lambda row: (-int(row["audit"]["letters"]), str(row["id"])))
    return {
        "experiment_id": "incumbent-498-note-seam-growth-20261002",
        "method": (
            "move the editable incumbent seam from depth 54 to depth 88, "
            "carry A-note/Set partial-word ownership live, and widen only the "
            "outer event path enough to remain above the 530 control"
        ),
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "rendered": parent_rendered,
            "letters": 498,
            "sha256": PARENT_SHA256,
            "independent_exact": True,
            "role": "content incumbent",
        },
        "prior_repair": {
            "artifact": str(PRIOR.relative_to(ROOT)),
            "cursor": 54,
            "letters": sorted(prior_lengths),
            "worst_seam_removed": "A poem. Any. Me? rate yes up us if for even",
        },
        "excluded_control": {
            "letters": 530,
            "role": "duplicated-wrapper repetition control only",
            "used_as_parent": False,
        },
        "live_seam": {
            "left_cursor": LEFT_CURSOR,
            "right_cursor": RIGHT_CURSOR,
            "retained_letters": RIGHT_CURSOR - LEFT_CURSOR,
            "left_boundary": LEFT_BOUNDARY.strip(),
            "right_boundary": RIGHT_BOUNDARY.strip(),
            "boundary_tapes": [tape(LEFT_BOUNDARY), tape(RIGHT_BOUNDARY)],
            "reverse_exact": tape(LEFT_BOUNDARY) == tape(RIGHT_BOUNDARY)[::-1],
            "partial_word": "s|et",
        },
        "stats": {
            "authored_paths": len(rows),
            "independently_exact_children": len(rows),
            "children_over_530": sum(int(row["audit"]["letters"]) > 530 for row in rows),
            "shortest_letters": min(int(row["audit"]["letters"]) for row in rows),
            "longest_letters": max(int(row["audit"]["letters"]) for row in rows),
            "additional_parent_letters_removed_per_side": LEFT_CURSOR - 54,
        },
        "active_frontier": [row["id"] for row in rows],
        "worst_remaining_seam": {
            "text": "the retained function-word run after the quoted two/no instruction",
            "next_operator": (
                "keep the strongest depth88 child and reopen the next seam at "
                "the first complete Noel event, while requiring a new causal "
                "transition rather than another fixed outer wrapper"
            ),
        },
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
