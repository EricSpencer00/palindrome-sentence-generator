"""Solve the 550 parent's asymmetric ``smaps`` center equation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import (
    audit,
    independent_tape,
)
from experiments.incumbent_550_wide_center_story_20261002 import proper_spans
from llm_palindrome.admission import tokenize


PARENT = ROOT / "runs" / "incumbent-544-cross-boundary-seam-repair-20261002.json"
OUT = ROOT / "runs" / "incumbent-550-asymmetric-smaps-repair-20261002.json"
PARENT_SHA256 = "3040f0c4ac28002aa0edd7ce2fd920751b10e4a4f5430d82de3e46d09b3e7673"
EXPECTED_SHA256 = "69a586613709e0867c702b2c20401e5c7226165d5218d65a49eaaf69d9ae6002"
WINDOW_SHA256 = "d27bc2a511f089921a571aa69250fff542a4f3b4274ca1ad86dafdc6b5d81308"

OLD_WINDOW = "Leon, I saw diaper. Repaid was I, Noel; spam's"
NEW_WINDOW = "Pansy snaps. Pam's"
RESIDUAL = "smaps"


def spans_with_boundary(text: str, boundaries: set[int]) -> list[dict[str, object]]:
    units = tokenize(text)
    spans = proper_spans(text)
    return [
        span for span in spans
        if span["token_offsets"][0] in boundaries or span["token_offsets"][1] in boundaries
    ]


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = parent_payload["rows"][0]
    assert parent["audit"]["letters"] == 550
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    parent_rendered = str(parent["rendered"])
    assert parent_rendered.count(OLD_WINDOW) == 1

    window_tape = independent_tape(NEW_WINDOW)
    closure_tape = RESIDUAL + window_tape
    assert window_tape == "pansysnapspams"
    assert window_tape != window_tape[::-1]
    assert closure_tape == closure_tape[::-1]
    assert hashlib.sha256(window_tape.encode()).hexdigest() == WINDOW_SHA256

    rendered = parent_rendered.replace(OLD_WINDOW, NEW_WINDOW)
    result_audit = audit(rendered)
    assert result_audit["letters"] == 531
    assert result_audit["sha256_forward"] == EXPECTED_SHA256
    assert result_audit["two_pointer_exact"]
    assert result_audit["byte_pointer_exact"]
    assert result_audit["project_validator_exact"]

    parent_spans = proper_spans(parent_rendered)
    child_spans = proper_spans(rendered)
    assert len(parent_spans) == 67
    assert len(child_spans) == 63

    # Token offsets around maps | Pansy | snaps | Pam's | reviled in the child.
    child_units = tokenize(rendered)
    pansy_index = child_units.index("pansy")
    boundaries = {
        pansy_index,
        pansy_index + 1,
        pansy_index + 2,
        pansy_index + 3,
    }
    anchored = spans_with_boundary(rendered, boundaries)
    assert anchored == []

    return {
        "experiment_id": "incumbent-550-asymmetric-smaps-repair-20261002",
        "method": (
            "reopen an asymmetrically owned center window and require the "
            "residual-plus-window equation to close while the window itself is non-palindromic"
        ),
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 550,
            "sha256": PARENT_SHA256,
            "source_498_artifact": "runs/overhang-growth-from-240-20261001.json",
            "source_498_sha256": "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032",
        },
        "stats": {
            "independently_exact_children": 1,
            "longest_letters": 531,
            "parent_proper_span_count": len(parent_spans),
            "child_proper_span_count": len(child_spans),
            "proper_spans_removed": len(parent_spans) - len(child_spans),
            "proper_spans_created_at_replacement_boundaries": len(anchored),
            "globally_shortcut_clean_children": 0,
        },
        "rows": [{
            "id": "asymmetric-smaps-531",
            "rendered": rendered,
            "audit": result_audit,
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_sha256": PARENT_SHA256,
            "replacement": {
                "old_window": OLD_WINDOW,
                "old_raw_offsets": [357, 403],
                "old_normalized_offsets": [261, 294],
                "new_window": NEW_WINDOW,
                "new_raw_offsets": [357, 375],
                "new_normalized_offsets": [261, 275],
                "new_window_tape": window_tape,
                "new_window_sha256": WINDOW_SHA256,
                "new_window_is_palindromic": False,
                "new_event_content": ["Pansy snaps", "Pam is reviled"],
            },
            "live_state": {
                "owner": "L",
                "residual_before_window": RESIDUAL,
                "closure_tape": closure_tape,
                "closure_exact": True,
                "residual_after_window": "",
            },
            "boundary_audit": {
                "replacement_token_boundaries": sorted(boundaries),
                "proper_spans_anchored_at_replacement_boundaries": anchored,
                "parent_proper_span_count": len(parent_spans),
                "child_proper_span_count": len(child_spans),
                "old_window_spans_removed": 4,
                "new_window_spans_created": 0,
                "surviving_spans_have_frozen_endpoints": True,
                "globally_shortcut_clean": False,
            },
            "fixed_window_obstruction": {
                "statement": (
                    "with the frozen prefix ending in deliver|s maps and the "
                    "frozen suffix beginning reviled, every exact solution of "
                    "smaps + window creates an enclosing proper palindromic span"
                ),
                "frozen_left_tokens": ["delivers", "maps"],
                "frozen_right_token": "reviled",
                "proof_equation": (
                    "reverse(reviled)=deliver; exactness makes smaps+window "
                    "palindromic, so delivers maps + window + reviled is palindromic"
                ),
                "strict_accepts_at_fixed_window": 0,
                "next_editable_window": {
                    "raw_offsets": [357, 768],
                    "normalized_offsets": [261, 550],
                    "description": (
                        "one-sided replacement through EOF removes the frozen "
                        "right endpoint that forces the enclosing span"
                    ),
                    "minimum_new_window_letters_for_531_total": 270,
                },
            },
            "provenance": {
                "outer_wrapper_added": False,
                "fixed_finished_tape_reversal": False,
                "posthoc_character_repair": False,
                "human_certified": False,
                "working_track_only": True,
            },
            "next_operator": (
                "move to the recorded one-sided through-EOF window so the "
                "frozen reviled endpoint no longer forces an enclosing span"
            ),
        }],
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
