"""Run a bounded typed discourse lattice, then change seam on zero closure.

The first operator is seam-local: it enumerates authored complete clauses and
two-clause causal/temporal connectors against the promoted 63-letter mirror
equation at [64,127)/[539,602), carrying the fixed character residual online.
If it finds no closure, the same run records the cursor obstruction and makes
one bounded direct repair at a different actual seam [127,197)/[469,539).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit


PARENT = ROOT / "runs" / "incumbent-666-comparison-alternative-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-typed-discourse-lattice-20260922.json"
PARENT_ID = "comparison-alternative-nora-sees-666"
PARENT_SHA256 = "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"
LATTICE_LEFT = (64, 127)
LATTICE_RIGHT = (539, 602)
LATTICE_RAW_LEFT = (86, 170)
LATTICE_RAW_RIGHT = (739, 822)
DIRECT_LEFT_WINDOW = (127, 197)
DIRECT_RIGHT_WINDOW = (469, 539)
DIRECT_RAW_LEFT = (170, 262)
DIRECT_RAW_RIGHT = (647, 739)
DIRECT_CHILD_SHA256 = "4f4aca22e8858379cb2f5a4931df1773c2a68b318ac9b8225ec447025f3d6230"

CLAUSES = (
    {"id": "mara_stops_rats", "text": "Mara stops rats.", "template": "mara_stops"},
    {"id": "nora_spots_ram", "text": "Nora spots a ram.", "template": "nora_spots"},
    {"id": "mara_sees_rats", "text": "Mara sees rats.", "template": "mara_sees"},
    {"id": "nora_sees_aram", "text": "Nora sees Aram.", "template": "nora_sees"},
    {"id": "nora_sees_ira", "text": "Nora sees Ira.", "template": "nora_sees"},
    {"id": "nadia_sees_ira", "text": "Nadia sees Ira.", "template": "nadia_sees"},
    {"id": "aidan_sees_aram", "text": "Aidan sees Aram.", "template": "aidan_sees"},
    {"id": "sara_saw_god", "text": "Sara saw God.", "template": "sara_saw"},
    {"id": "nora_saw_noel_live", "text": "Nora saw Noel live.", "template": "nora_saw"},
    {"id": "she_sees_aram", "text": "She sees Aram.", "template": "she_sees"},
    {"id": "he_sees_nadia", "text": "He sees Nadia.", "template": "he_sees"},
)
CONNECTORS = (
    "because",
    "so",
    "after",
    "when",
    "then",
)

DIRECT_LEFT = (
    "Nora saw Noel live. Mara saw God. Sara saw God. Pat notes. "
    "Nadia sees Mara. Nadia stops Aram."
)
DIRECT_RIGHT = (
    "Mara spots Aidan. Aram sees Aidan. Seton, tap. Dog was Aras. "
    "Dog was Aram. Evil Leon was Aron."
)

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


def consume_online(emission: str, obligation: str) -> dict[str, object]:
    """Consume one emitted character at a time against the live residual."""
    residual = obligation
    consumed = 0
    for cursor, character in enumerate(normalize(emission)):
        if not residual:
            return {
                "exact": False,
                "consumed": consumed,
                "cursor": cursor,
                "expected": None,
                "emitted": character,
                "residual": residual,
                "reason": "emission_overrun",
            }
        if character != residual[0]:
            return {
                "exact": False,
                "consumed": consumed,
                "cursor": cursor,
                "expected": residual[0],
                "emitted": character,
                "residual": residual,
                "reason": "character_contradiction",
            }
        residual = residual[1:]
        consumed += 1
    return {
        "exact": not residual,
        "consumed": consumed,
        "cursor": consumed,
        "expected": None,
        "emitted": None,
        "residual": residual,
        "reason": "closed" if not residual else "nonempty_residual",
    }


def compose_clause_pair(first: dict[str, str], connector: str, second: dict[str, str]) -> str:
    left = first["text"][:-1]
    right = second["text"]
    if connector == "because":
        return f"{left} because {right}"
    if connector == "so":
        return f"{left}, so {right}"
    if connector == "after":
        return f"After {left.lower()}, {right}"
    if connector == "when":
        return f"When {left.lower()}, {right}"
    return f"{left}; then {right}"


def run_lattice(target_left: str, target_right: str) -> dict[str, object]:
    candidates = []
    rejected_template_pairs = 0
    deepest = None
    for first, second, connector in itertools.product(CLAUSES, CLAUSES, CONNECTORS):
        if first["id"] == second["id"]:
            continue
        if first["template"] == second["template"]:
            rejected_template_pairs += 1
            continue
        emission = compose_clause_pair(first, connector, second)
        left_trace = consume_online(emission, target_left)
        candidates.append(
            {
                "first": first["id"],
                "second": second["id"],
                "connector": connector,
                "emission": emission,
                "left_trace": left_trace,
            }
        )
        if deepest is None or left_trace["consumed"] > deepest["left_trace"]["consumed"]:
            deepest = candidates[-1]

    closures = [
        candidate
        for candidate in candidates
        if candidate["left_trace"]["exact"]
        and normalize(candidate["emission"]) == target_left
        and normalize(candidate["emission"])[::-1] == target_right
    ]
    contradiction = deepest["left_trace"] if deepest else {}
    return {
        "target_equation": {
            "normalized_left": target_left,
            "normalized_right": target_right,
            "right_is_reverse": target_left[::-1] == target_right,
        },
        "authored_clause_count": len(CLAUSES),
        "connector_count": len(CONNECTORS),
        "relation_gate": "every candidate contains one of because/so/after/when/then",
        "template_gate": "reject identical subject-verb template pairs",
        "stats": {
            "ordered_pairs_considered": len(CLAUSES) * (len(CLAUSES) - 1) * len(CONNECTORS),
            "template_pair_rejections": rejected_template_pairs,
            "relation_candidates": len(candidates),
            "online_prefix_states": sum(c["left_trace"]["consumed"] for c in candidates),
            "character_contradictions": sum(
                c["left_trace"]["reason"] == "character_contradiction" for c in candidates
            ),
            "closures": len(closures),
        },
        "closures": closures,
        "deepest_prefix_obstruction": {
            "candidate": deepest,
            "cursor": contradiction.get("cursor"),
            "residual": contradiction.get("residual"),
            "reason": contradiction.get("reason"),
        },
    }


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

    lattice_left = parent_tape[LATTICE_LEFT[0] : LATTICE_LEFT[1]]
    lattice_right = parent_tape[LATTICE_RIGHT[0] : LATTICE_RIGHT[1]]
    lattice = run_lattice(lattice_left, lattice_right)
    assert lattice["target_equation"]["right_is_reverse"]
    assert lattice["stats"]["closures"] == 0

    direct_old_left = parent_rendered[DIRECT_RAW_LEFT[0] : DIRECT_RAW_LEFT[1]]
    direct_old_right = parent_rendered[DIRECT_RAW_RIGHT[0] : DIRECT_RAW_RIGHT[1]]
    assert parent_tape[DIRECT_LEFT_WINDOW[0] : DIRECT_LEFT_WINDOW[1]] == normalize(direct_old_left)
    assert parent_tape[DIRECT_RIGHT_WINDOW[0] : DIRECT_RIGHT_WINDOW[1]] == normalize(direct_old_right)
    direct_left = normalize(DIRECT_LEFT)
    direct_right = normalize(DIRECT_RIGHT)
    assert len(direct_left) == len(direct_right) == 70
    assert direct_left == direct_right[::-1]
    direct_rendered = (
        parent_rendered[: DIRECT_RAW_LEFT[0]]
        + " "
        + DIRECT_LEFT
        + parent_rendered[DIRECT_RAW_LEFT[1] : DIRECT_RAW_RIGHT[0]]
        + " "
        + DIRECT_RIGHT
        + parent_rendered[DIRECT_RAW_RIGHT[1] :]
    )
    direct_project_audit = audit(direct_rendered)
    direct_independent = independent_audit(direct_rendered)
    assert direct_project_audit["letters"] == 666
    assert direct_independent["normalized_letters"] == 666
    assert direct_independent["two_pointer_exact"]
    assert direct_independent["sha256_forward"] == DIRECT_CHILD_SHA256
    assert direct_independent["sha_equal"]
    assert direct_project_audit["project_validator_exact"]

    row = {
        "id": "typed-lattice-direct-seam-666",
        "working_status": "comparison_frontier_alternative",
        "promotion_status": {
            "promoted": False,
            "status": "pending_full_text_readability_review",
            "reason": (
                "The typed lattice had zero closure; the direct changed-seam child "
                "is exact but is not promoted without material full-text review."
            ),
        },
        "rendered": direct_rendered,
        "audit": direct_project_audit,
        "independent_audit": direct_independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "new_event_content": [
            "Nora saw Noel live",
            "Mara saw God",
            "Sara saw God",
            "Pat notes",
            "Nadia sees Mara",
            "Nadia stops Aram",
        ],
        "live_seam": {
            "normalized_window_left": list(DIRECT_LEFT_WINDOW),
            "normalized_window_right": list(DIRECT_RIGHT_WINDOW),
            "raw_window_left": list(DIRECT_RAW_LEFT),
            "raw_window_right": list(DIRECT_RAW_RIGHT),
            "old_left": direct_old_left,
            "old_right": direct_old_right,
            "new_left": DIRECT_LEFT,
            "new_right": DIRECT_RIGHT,
            "initial_owner": "left_direct_repair_window",
            "left_emission": direct_left,
            "right_obligation": direct_right,
            "right_consumption": direct_right,
            "final_owner": None,
            "final_residual": "",
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "typed_lattice_attempt": lattice,
        "readability_delta": {
            "material_full_text_improvement": False,
            "complete_clauses": True,
            "imperative_vocative_clause": {
                "text": "Seton, tap.",
                "complete": True,
                "vocative": "Seton",
                "predicate": "tap",
                "dangling_vocative_or_appositive": False,
            },
            "predicate_less_fragment": False,
            "dangling_vocative_or_appositive": False,
        },
        "grammar_debt": {
            "inherited_proper_palindromic_spans": True,
            "remaining_delivers_maps_elsewhere": True,
            "remaining_star_spam_scaffolding": True,
            "remaining_mara_stops_rats_scaffolding": True,
            "rough_syntax_elsewhere": True,
            "human_reader_validation": False,
            "effect": "retain active 666 and 568; comparison child awaits full-text review",
        },
        "provenance": (
            "zero-closure typed discourse lattice persisted with exact cursor/residual "
            "obstruction, followed immediately by a different actual seam direct repair"
        ),
    }

    return {
        "experiment_id": "incumbent-666-typed-discourse-lattice-20260922",
        "method": "bounded seam-local typed discourse lattice with immediate changed-seam repair",
        "active_frontier_parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 666,
            "sha256": PARENT_SHA256,
        },
        "working_incumbent": {
            "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
            "id": "outer-causal-scene-568-working-incumbent",
            "letters": 568,
            "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        },
        "lattice_attempt": lattice,
        "changed_seam_after_zero_closure": {
            "normalized_left": list(DIRECT_LEFT_WINDOW),
            "normalized_right": list(DIRECT_RIGHT_WINDOW),
            "raw_left": list(DIRECT_RAW_LEFT),
            "raw_right": list(DIRECT_RAW_RIGHT),
            "exact_children": 1,
            "child_sha256": DIRECT_CHILD_SHA256,
            "material_full_text_improvement": False,
        },
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": (
            "retain the zero-closure lattice obstruction and review the changed-seam "
            "comparison child before any promotion"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["lattice_attempt"]["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
