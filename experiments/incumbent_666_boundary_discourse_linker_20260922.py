"""Try one boundary-aware discourse linker, then switch seam once on conflict.

The first bounded operator is a hand-authored because/so/after/when/then
linker at [20,64)/[602,646).  It is neighbor-aware and carries the fixed
residual online.  Its obstruction is persisted before a single changed-seam
semantic-role repair at [127,197)/[469,539).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit


PARENT = ROOT / "runs" / "incumbent-666-comparison-alternative-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-boundary-discourse-linker-20260922.json"
PARENT_ID = "comparison-alternative-nora-sees-666"
PARENT_SHA256 = "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"
CHILD_SHA256 = "cb0ac54c46fcf48dcf518527996c915e55b947a198ca2acf257a65e6e23ca19d"
LINKER_LEFT = (20, 64)
LINKER_RIGHT = (602, 646)
LINKER_RAW_LEFT = (26, 86)
LINKER_RAW_RIGHT = (818, 877)
SWITCH_LEFT = (127, 197)
SWITCH_RIGHT = (469, 539)
SWITCH_RAW_LEFT = (170, 262)
SWITCH_RAW_RIGHT = (647, 739)

LINKER_CANDIDATES = (
    {
        "id": "because-nora-mara",
        "text": "Because Nora sees Nadia, Mara stops Aram.",
        "connector": "because",
        "frames": [("Nora", "sees"), ("Mara", "stops")],
    },
    {
        "id": "after-nadia-ari",
        "text": "After Nadia sees Ira, Ari spots God.",
        "connector": "after",
        "frames": [("Nadia", "sees"), ("Ari", "spots")],
    },
    {
        "id": "when-sara-aidan",
        "text": "When Sara saw God, Aidan sees Ira.",
        "connector": "when",
        "frames": [("Sara", "saw"), ("Aidan", "sees")],
    },
    {
        "id": "so-mara-nadia",
        "text": "Mara stops Aram, so Nadia sees Ira.",
        "connector": "so",
        "frames": [("Mara", "stops"), ("Nadia", "sees")],
    },
    {
        "id": "then-nora-ari",
        "text": "Nora sees Nadia; then Ari spots God.",
        "connector": "then",
        "frames": [("Nora", "sees"), ("Ari", "spots")],
    },
)

NEW_LEFT = (
    "Nadia sees Ira. Aidan stops Ira. Mara sees Ari. Ari stops Nadia. "
    "Ari sees God. Dog spots Ira."
)
NEW_RIGHT = (
    "Ari stops God. Dog sees Ira. Aidan spots Ira. Ira sees Aram. "
    "Ari spots Nadia. Ari sees Aidan."
)
OLD_SWITCH_LEFT = " Mara sees Nadia. Nadia saw Noel live. Mara stops Nadia. Nora sees Aram. Sara saw Noel live."
OLD_SWITCH_RIGHT = " Evil Leon was Aras. Mara sees Aron. Aidan spots Aram. Evil Leon was Aidan. Aidan sees Aram."

FRONTIER = (
    {
        "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "id": "outer-causal-scene-568-working-incumbent",
        "letters": 568,
        "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
    },
    {
        "artifact": "runs/incumbent-550-central-event-bridge-20261002.json",
        "id": "central-distinct-events-560",
        "letters": 560,
        "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc",
    },
    {
        "artifact": "runs/incumbent-550-typed-center-product-20261002.json",
        "id": "typed-center-25",
        "letters": 558,
        "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa",
    },
    {
        "artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json",
        "id": "depth39-longest-f1g1h1r",
        "letters": 556,
        "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14",
    },
)


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    reverse = tape[::-1]
    forward_sha = hashlib.sha256(tape.encode()).hexdigest()
    reverse_sha = hashlib.sha256(reverse.encode()).hexdigest()
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": all(
            tape[index] == tape[-1 - index] for index in range(len(tape) // 2)
        ),
        "sha256_forward": forward_sha,
        "sha256_reverse": reverse_sha,
        "sha_equal": forward_sha == reverse_sha,
    }


def validate_frontier_entry(entry: dict[str, object]) -> None:
    artifact = ROOT / str(entry["artifact"])
    assert artifact.exists(), artifact
    payload = json.loads(artifact.read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    recomputed = independent_audit(str(row["rendered"]))
    assert recomputed["normalized_letters"] == entry["letters"]
    assert recomputed["two_pointer_exact"]
    assert recomputed["sha256_forward"] == entry["sha256"]
    assert recomputed["sha_equal"]


def consume_online(emission: str, obligation: str, owner: str) -> dict[str, object]:
    residual = obligation
    trace = []
    for cursor, character in enumerate(normalize(emission)):
        if not residual:
            return {"exact": False, "cursor": cursor, "residual": residual, "trace": trace, "reason": "overrun"}
        expected = residual[0]
        if character != expected:
            return {
                "exact": False,
                "cursor": cursor,
                "residual": residual,
                "trace": trace,
                "reason": "character_contradiction",
                "expected": expected,
                "emitted": character,
            }
        residual = residual[1:]
        trace.append({"cursor": cursor, "owner": owner, "emitted": character, "expected": expected, "residual_after": residual})
    return {"exact": not residual, "cursor": len(trace), "residual": residual, "trace": trace, "reason": "closed" if not residual else "nonempty_residual"}


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    parent_independent = independent_audit(parent_rendered)
    assert parent_independent["normalized_letters"] == 666
    assert parent_independent["two_pointer_exact"]
    assert parent_independent["sha256_forward"] == PARENT_SHA256
    assert parent_independent["sha_equal"]
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    assert parent["promotion_status"]["promoted"] is True
    for frontier_entry in FRONTIER:
        validate_frontier_entry(frontier_entry)

    linker_left = parent_tape[LINKER_LEFT[0] : LINKER_LEFT[1]]
    linker_right = parent_tape[LINKER_RIGHT[0] : LINKER_RIGHT[1]]
    assert len(linker_left) == len(linker_right) == 44
    assert linker_left == linker_right[::-1]
    boundary_frames = {
        "left_before": ("Wolf", "spots"),
        "left_after": ("Mara", "stops"),
        "right_before": ("Aidan", "sees"),
        "right_after": ("Aron", "stops"),
    }
    all_boundary_frames = tuple(boundary_frames.values())
    linker_attempts = []
    for candidate in LINKER_CANDIDATES:
        neighbor_conflicts = [
            frame
            for frame in (candidate["frames"][0], candidate["frames"][-1])
            if frame in all_boundary_frames
        ]
        neighbor_checks = {
            side: all(frame != boundary_frame for frame in candidate["frames"])
            for side, boundary_frame in boundary_frames.items()
        }
        left_trace = consume_online(candidate["text"], linker_left, "linker_left")
        linker_attempts.append(
            {
                **candidate,
                "neighbor_conflicts": neighbor_conflicts,
                "neighbor_checks": neighbor_checks,
                "neighbor_gate_passed": not neighbor_conflicts,
                "left_trace": left_trace,
                "right_reverse_match": normalize(candidate["text"])[::-1] == linker_right,
                "closure": not neighbor_conflicts and left_trace["exact"] and normalize(candidate["text"])[::-1] == linker_right,
            }
        )
    linker_closures = [attempt for attempt in linker_attempts if attempt["closure"]]
    deepest = max(linker_attempts, key=lambda attempt: attempt["left_trace"]["cursor"])
    assert not linker_closures

    assert parent_tape[SWITCH_LEFT[0] : SWITCH_LEFT[1]] == normalize(OLD_SWITCH_LEFT)
    assert parent_tape[SWITCH_RIGHT[0] : SWITCH_RIGHT[1]] == normalize(OLD_SWITCH_RIGHT)
    assert parent_rendered[SWITCH_RAW_LEFT[0] : SWITCH_RAW_LEFT[1]] == OLD_SWITCH_LEFT
    assert parent_rendered[SWITCH_RAW_RIGHT[0] : SWITCH_RAW_RIGHT[1]] == OLD_SWITCH_RIGHT
    left_emission = normalize(NEW_LEFT)
    right_obligation = normalize(NEW_RIGHT)
    assert len(left_emission) == len(right_obligation) == 70
    assert left_emission == right_obligation[::-1]
    left_trace = consume_online(NEW_LEFT, left_emission, "switched_left_graft")
    right_trace = consume_online(NEW_RIGHT, right_obligation, "switched_right_graft")
    assert left_trace["exact"] and right_trace["exact"]
    left_clauses = [part.strip() for part in NEW_LEFT.split(".") if part.strip()]
    right_clauses = [part.strip() for part in NEW_RIGHT.split(".") if part.strip()]
    assert len(left_clauses) == len(set(left_clauses)) == 6
    assert len(right_clauses) == len(set(right_clauses)) == 6
    assert all(len(clause.split()) == 3 for clause in left_clauses + right_clauses)
    left_subject_verbs = [(clause.split()[0], clause.split()[1]) for clause in left_clauses]
    right_subject_verbs = [(clause.split()[0], clause.split()[1]) for clause in right_clauses]
    assert len(left_subject_verbs) == len(set(left_subject_verbs)) == 6
    assert len(right_subject_verbs) == len(set(right_subject_verbs)) == 6
    boundary_dedup = {
        "left_before": "Nora sees Aram.",
        "left_first_graft": left_clauses[0],
        "left_after": "Now, Noel, did I live?",
        "right_before": "Evil I did, Leon won.",
        "right_last_graft": right_clauses[-1],
        "right_after": "Mara sees Aron;",
        "checks": {
            "left_before": left_clauses[0] != "Nora sees Aram",
            "left_after": left_clauses[0] != "Now, Noel, did I live?",
            "right_before": right_clauses[-1] != "Evil I did, Leon won",
            "right_after": right_clauses[-1] != "Mara sees Aron",
        },
        "passed": left_clauses[0] != "Nora sees Aram"
        and right_clauses[-1] != "Mara sees Aron",
    }
    assert boundary_dedup["passed"]

    rendered = (
        parent_rendered[: SWITCH_RAW_LEFT[0]]
        + " "
        + NEW_LEFT
        + parent_rendered[SWITCH_RAW_LEFT[1] : SWITCH_RAW_RIGHT[0]]
        + " "
        + NEW_RIGHT
        + parent_rendered[SWITCH_RAW_RIGHT[1] :]
    )
    project_audit = audit(rendered)
    independent = independent_audit(rendered)
    assert project_audit["letters"] == 666
    assert independent["normalized_letters"] == 666
    assert independent["two_pointer_exact"]
    assert independent["sha256_forward"] == CHILD_SHA256
    assert independent["sha_equal"]
    assert project_audit["project_validator_exact"]

    row = {
        "id": "boundary-linker-switch-graft-666",
        "working_status": "boundary_discourse_linker_candidate",
        "promotion_status": {
            "promoted": False,
            "status": "selected_material_local_improvement_pending_review",
            "reason": "The linker contradicted at the promoted seam; the changed-seam graft removes a repeated pair and varies relations, pending full-text review.",
        },
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "new_event_content": left_clauses + right_clauses,
        "linker_attempt": {
            "normalized_windows": {"left": list(LINKER_LEFT), "right": list(LINKER_RIGHT)},
            "relation_inventory": ["because", "so", "after", "when", "then"],
            "neighbor_frames": boundary_frames,
            "attempts": linker_attempts,
            "closures": linker_closures,
            "deepest_obstruction": {
                "candidate": deepest["id"],
                "cursor": deepest["left_trace"]["cursor"],
                "residual": deepest["left_trace"]["residual"],
                "reason": deepest["left_trace"]["reason"],
            },
        },
        "live_seam": {
            "normalized_window_left": list(SWITCH_LEFT),
            "normalized_window_right": list(SWITCH_RIGHT),
            "raw_window_left": list(SWITCH_RAW_LEFT),
            "raw_window_right": list(SWITCH_RAW_RIGHT),
            "old_left": OLD_SWITCH_LEFT,
            "old_right": OLD_SWITCH_RIGHT,
            "new_left": NEW_LEFT,
            "new_right": NEW_RIGHT,
            "initial_owner": "switched_left_graft",
            "left_emission": left_emission,
            "right_obligation": right_obligation,
            "right_consumption": right_obligation,
            "final_owner": None,
            "final_residual": "",
            "left_trace": left_trace["trace"],
            "right_trace": right_trace["trace"],
            "boundary_dedup": boundary_dedup,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "semantic_roles": {
        "varied_relations": ["sees", "stops", "spots"],
        "complete_svo_clauses": True,
        "repeated_subject_verb_frames": [],
        "repeated_neighboring_clauses": False,
            "vocative_or_appositive_fragments": False,
        },
        "readability_delta": {
            "repeated_saw_noel_live_before": 2,
            "repeated_saw_noel_live_after": 0,
            "material_full_text_improvement": True,
        },
        "grammar_debt": {
            "inherited_proper_palindromic_spans": True,
            "remaining_delivers_maps_elsewhere": True,
            "remaining_star_spam_scaffolding": True,
            "remaining_mara_stops_rats_scaffolding": True,
            "rough_syntax_elsewhere": True,
            "human_reader_validation": False,
            "effect": "retain 568 incumbent and active 666 frontier until review",
        },
        "provenance": "boundary-aware linker zero closure persisted; one changed actual seam then closed with online residual trace",
    }

    return {
        "experiment_id": "incumbent-666-boundary-discourse-linker-20260922",
        "method": "bounded boundary-aware discourse linker with one changed-seam graft",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "linker_attempt": {"closures": linker_closures, "attempt_count": len(linker_attempts), "deepest_obstruction": {"candidate": deepest["id"], "cursor": deepest["left_trace"]["cursor"], "residual": deepest["left_trace"]["residual"], "reason": deepest["left_trace"]["reason"]}},
        "changed_seam_after_contradiction": {"normalized_left": list(SWITCH_LEFT), "normalized_right": list(SWITCH_RIGHT), "raw_left": list(SWITCH_RAW_LEFT), "raw_right": list(SWITCH_RAW_RIGHT), "exact_children": 1, "child_sha256": CHILD_SHA256},
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": "review the changed-seam boundary-aware graft; do not demote 568 or promote without full-text review",
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["changed_seam_after_contradiction"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
