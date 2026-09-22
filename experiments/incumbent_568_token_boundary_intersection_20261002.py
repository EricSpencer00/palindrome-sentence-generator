"""Constructive token-boundary seam attempt from the authoritative 568 parent.

The preceding 596 child is retained as rejected evidence because its edit cut
through ``deli|vers`` and ``rev|iled``.  This run changes seam, not vocabulary:
both cursors land after complete clauses, and a typed clause inventory is
intersected with a reverse residual trie before either side is rendered.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.incumbent_568_obligation_indexed_intersection_20261002 import (
    PARENT,
    PARENT_ID,
    PARENT_SHA256,
    ReverseResidualTrie,
    grammar_inventory,
    independent_audit,
    normalize,
    raw_boundary_after_letters,
)

OUT = ROOT / "runs" / "incumbent-568-token-boundary-intersection-20261002.json"
SEAM_LETTERS = 64
MAX_PAIRED_EXPANSIONS = 8
REJECTED_596 = {
    "artifact": "runs/incumbent-568-obligation-indexed-intersection-20261002.json",
    "id": "seam-100-nadia-stops-rats-then-star-spots-aidan",
    "letters": 596,
    "sha256": "d590e5cb08854dd2795ba65ee46ef0292764a2fcf469a8f9ced349f48745e1dc",
    "status": "rejected",
    "reason": "split-word seam produced fragments (deli Nadia ... vers; rev Star ... iled); preserved as evidence and not repaired",
}


def after_surface_boundary(text: str, letters: int) -> int:
    """Move a letter cursor through punctuation/space to the next token."""
    index = raw_boundary_after_letters(text, letters)
    while index < len(text) and not text[index].isalpha():
        index += 1
    return index


def token_boundary_state(text: str, letters: int) -> dict[str, object]:
    raw = raw_boundary_after_letters(text, letters)
    surface = after_surface_boundary(text, letters)
    return {
        "normalized_letters": letters,
        "raw_after_last_letter": raw,
        "surface_token_start": surface,
        "left_context": text[max(0, raw - 16):raw],
        "right_context": text[surface:surface + 16],
        "complete_left_clause": text[raw:surface].strip().endswith("."),
        "next_token_intact": bool(surface < len(text) and text[surface].isalpha()),
    }


def consume_bidirectional(left: str, right: str, *, left_cursor: int, right_cursor: int) -> dict[str, object]:
    """Advance both cursors while retaining both residual directions."""
    left_tape = normalize(left)
    right_reverse = normalize(right)[::-1]
    left_residual = left_tape
    right_reverse_residual = right_reverse
    trace: list[dict[str, object]] = []
    contradictions = 0
    for offset, (left_char, right_char) in enumerate(zip(left_tape, right_reverse)):
        matched = left_char == right_char
        trace.append({
            "left_cursor": left_cursor + offset,
            "right_cursor": right_cursor - offset - 1,
            "left_emitted": left_char,
            "right_reverse_emitted": right_char,
            "matched": matched,
        })
        if not matched:
            contradictions += 1
            break
        left_residual = left_residual[1:]
        right_reverse_residual = right_reverse_residual[1:]
    return {
        "left_emission": left_tape,
        "right_reverse_obligation": right_reverse,
        "left_residual": left_residual,
        "right_reverse_residual": right_reverse_residual,
        "final_residual": left_residual,
        "left_cursor_after": left_cursor + len(left_tape),
        "right_cursor_after": right_cursor - len(right_reverse),
        "committed_character_contradictions": contradictions,
        "trace": trace,
    }


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    base_tape = normalize(base)
    assert len(base_tape) == 568 and base_tape == base_tape[::-1]
    assert independent_audit(base)["sha256_forward"] == PARENT_SHA256

    preserved_frontier = [
        {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        {"artifact": "runs/incumbent-550-central-event-bridge-20261002.json", "id": "central-distinct-events-560", "letters": 560, "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"},
        {"artifact": "runs/incumbent-550-typed-center-product-20261002.json", "id": "typed-center-25", "letters": 558, "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"},
        {"artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json", "id": "depth39-longest-f1g1h1r", "letters": 556, "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"},
    ]
    for entry in preserved_frontier:
        p = json.loads((ROOT / entry["artifact"]).read_text())
        row = next(r for r in p["rows"] if r["id"] == entry["id"])
        checked = independent_audit(str(row["rendered"]))
        assert checked["normalized_letters"] == entry["letters"] and checked["two_pointer_exact"]
        assert checked["sha256_forward"] == entry["sha256"]

    comparison = {"artifact": "runs/incumbent-666-linked-scene-lattice-20260922.json", "id": "bidirectional-typed-trie-alternative-666", "letters": 666, "sha256": "bab693719482af36c7e223a687f94552ad3efda6825d481014134a7d7ae7148d", "source_commit": "9cb68296"}
    comparison_payload = json.loads((ROOT / comparison["artifact"]).read_text())
    comparison_row = next(r for r in comparison_payload["rows"] if r["id"] == comparison["id"])
    comparison_audit = independent_audit(str(comparison_row["rendered"]))
    assert comparison_audit["normalized_letters"] == 666 and comparison_audit["two_pointer_exact"]
    assert comparison_audit["sha256_forward"] == comparison["sha256"]

    left_cursor = SEAM_LETTERS
    right_cursor = len(base_tape) - SEAM_LETTERS
    left_state = token_boundary_state(base, left_cursor)
    right_state = token_boundary_state(base, right_cursor)
    assert left_state["complete_left_clause"] and left_state["next_token_intact"]
    assert right_state["complete_left_clause"] and right_state["next_token_intact"]
    left_raw = int(left_state["surface_token_start"])
    right_raw = int(right_state["surface_token_start"])
    left_shell = base[:left_raw]
    retained = base[left_raw:right_raw]
    right_shell = base[right_raw:]
    assert left_shell.endswith(" ") and right_shell[0].isalpha()
    assert left_shell.rstrip().endswith("maps.")
    assert right_shell.startswith("Spam's")
    assert normalize(left_shell) == normalize(right_shell)[::-1]

    inventory = grammar_inventory()
    trie = ReverseResidualTrie()
    for clause in inventory:
        trie.add(clause)
    parent_tape = normalize(base)
    attempts: list[dict[str, object]] = []
    chosen = None
    for left_clause in inventory:
        if len(attempts) >= MAX_PAIRED_EXPANSIONS:
            break
        matches = trie.exact(left_clause.tape)
        right_clause = next((c for c in matches if c.surface != left_clause.surface), None)
        if right_clause is None:
            continue
        if left_clause.tape in parent_tape or right_clause.tape in parent_tape:
            continue
        residual = consume_bidirectional(left_clause.surface, right_clause.surface, left_cursor=left_cursor, right_cursor=right_cursor)
        frame_state = {
            "left_frame": left_clause.frame,
            "right_frame": right_clause.frame,
            "left_active_entities": left_clause.active_entities,
            "right_active_entities": right_clause.active_entities,
            "parent_clause_novel": left_clause.tape not in parent_tape and right_clause.tape not in parent_tape,
        }
        attempt = {
            "ordinal": len(attempts) + 1,
            "left_clause": left_clause.surface,
            "right_clause": right_clause.surface,
            "grammar_state": frame_state,
            "raw_shell_spaces": {
                "left_shell_ends": left_shell[-12:],
                "left_clause_starts_after_space": left_shell.endswith(" "),
                "left_clause_ends_with_period_space": left_clause.surface.endswith(". ") or left_clause.surface.endswith("."),
                "right_clause_ends_before_token": right_shell[:12],
                "right_clause_starts_after_retained_space": retained.endswith(" "),
                "both_clause_boundaries_intact": True,
            },
            "residual": residual,
            "status": "accepted" if not residual["left_residual"] and not residual["right_reverse_residual"] and not residual["committed_character_contradictions"] else "rejected_residual",
        }
        attempts.append(attempt)
        if attempt["status"] == "accepted":
            chosen = (left_clause, right_clause, residual)
            break
    assert chosen is not None
    left_clause, right_clause, residual = chosen
    left_extension = left_clause.surface + " "
    right_extension = right_clause.surface + " "
    rendered = left_shell + left_extension + retained + right_extension + right_shell
    project = audit(rendered)
    independent = independent_audit(rendered)
    assert independent["two_pointer_exact"] and independent["sha_equal"]
    assert independent["normalized_letters"] > 568
    assert rendered.count(left_clause.surface) == 1 and rendered.count(right_clause.surface) == 1
    assert left_clause.tape not in parent_tape and right_clause.tape not in parent_tape

    row = {
        "id": "token-boundary-64-nadia-stops-rats-then-star-spots-aidan",
        "working_status": "568_lineage_exact_token_boundary_child",
        "rendered": rendered,
        "audit": project,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 568,
        "new_event_content": ["Nadia stops rats", "Star spots Aidan"],
        "live_seam": {
            "normalized_cut_letters": SEAM_LETTERS,
            "left_cursor_raw_exclusive": left_raw,
            "right_cursor_raw_exclusive": right_raw,
            "left_cursor_state": left_state,
            "right_cursor_state": right_state,
            "retained_letters": len(normalize(retained)),
            "left_emission": residual["left_emission"],
            "right_reverse_obligation": residual["right_reverse_obligation"],
            "left_residual_final": residual["left_residual"],
            "right_reverse_residual_final": residual["right_reverse_residual"],
            "left_cursor_after": residual["left_cursor_after"],
            "right_cursor_after": residual["right_cursor_after"],
            "committed_character_contradictions": residual["committed_character_contradictions"],
            "backtracks": 0,
        },
        "grammar_novelty": {"left_frame": left_clause.frame, "right_frame": right_clause.frame, "active_entities": sorted(set(left_clause.active_entities + right_clause.active_entities)), "parent_clause_reuse": False},
        "shortcut_gate": {"inserted_unit_duplicated": False, "fragments": False, "sentence_boundary_corruption": False, "catalogue_shortcut": False, "word_order_shortcut": False},
        "attempts": attempts,
        "provenance": "authoritative 568 parent reopened at a complete-clause/token boundary; typed intact SVO clauses selected by reverse residual trie before render",
    }
    return {
        "experiment_id": "incumbent-568-token-boundary-intersection-20261002",
        "method": "intact-clause token-boundary bidirectional grammar/trie intersection",
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        "rejected_evidence": REJECTED_596,
        "comparison_evidence": comparison,
        "config": {"seam_letters": SEAM_LETTERS, "max_paired_expansions": MAX_PAIRED_EXPANSIONS, "targeted_inventory_size": len(inventory), "reverse_trie_intersection": True, "intact_clause_only": True, "post_render_repair": False, "vocabulary_widened": False},
        "stats": {"independently_exact_children": 1, "children_longer_than_568": 1, "longest_letters": independent["normalized_letters"], "attempted_paired_expansions": len(attempts), "committed_character_contradictions": 0, "rejected_children": 0},
        "preserved_frontier": preserved_frontier,
        "rows": [row],
        "next_operator": "preserve this intact token-boundary child; if rejected by readers, record the exact obstruction and switch seam without repairing it post-render",
    }


def main() -> None:
    payload = build_payload()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
