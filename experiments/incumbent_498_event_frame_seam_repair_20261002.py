"""Repair formulaic outer frames on deeper seams of the 498-letter parent."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_live_seam_growth_20261002 import (
    PARENT,
    PARENT_SHA256,
    audit,
    independent_tape,
    load_parent,
)


PRIOR = ROOT / "runs" / "incumbent-498-live-seam-growth-20261002.json"
OUT = ROOT / "runs" / "incumbent-498-event-frame-seam-repair-20261002.json"

# These are complete, varied event frames rather than the previous saw/was-only
# name shell.  Every pair is checked below instead of trusted as authored.
FRAME_PAIRS = {
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

SPECS = (
    {
        "id": "depth35-varied-f1g1h1",
        "depth": 35,
        "path": ("F1", "G1", "H1"),
        "middle_slice": (11, 151, "op"),
        "supplied": "en",
        "seam_left": "Nora saw a nine.",
        "seam_right": "Ina was Aron.",
        "partial_word": "op|en",
        "expected_sha": "93ffcba65f220c529e7e6ad1700d9cc543a0409991bcd2eea42dddb9cb8383b9",
    },
    {
        "id": "depth39-varied-f1g1h1k",
        "depth": 39,
        "path": ("F1", "G1", "H1", "K"),
        "middle_slice": (12, 150, "i"),
        "supplied": "ts",
        "seam_left": "Nora saw a post.",
        "seam_right": "Opa was Aron.",
        "partial_word": "i|ts",
        "expected_sha": "02e1ded5e201a2dac2b60a23c30eea7853527cab4ed99fcdb131ad2fd4aed08c",
    },
    {
        "id": "depth39-varied-f1g1h1l",
        "depth": 39,
        "path": ("F1", "G1", "H1", "L"),
        "middle_slice": (12, 150, "i"),
        "supplied": "ts",
        "seam_left": "Nora saw a post.",
        "seam_right": "Opa was Aron.",
        "partial_word": "i|ts",
        "expected_sha": "fdc696c8c4a01482ab0b88811cbc2b04b982c0b53ab90c151717bd54dc951377",
    },
    {
        "id": "depth39-varied-f2g2h2k",
        "depth": 39,
        "path": ("F2", "G2", "H2", "K"),
        "middle_slice": (12, 150, "i"),
        "supplied": "ts",
        "seam_left": "Nora saw a post.",
        "seam_right": "Opa was Aron.",
        "partial_word": "i|ts",
        "expected_sha": "b7520130866ba0819fe923ff78ba793a1c3eb9ddf3a31b4e348540902c56fda6",
    },
    {
        "id": "depth39-longest-f1g1h1r",
        "depth": 39,
        "path": ("F1", "G1", "H1", "R"),
        "middle_slice": (12, 150, "i"),
        "supplied": "ts",
        "seam_left": "Nora saw a post.",
        "seam_right": "Opa was Aron.",
        "partial_word": "i|ts",
        "expected_sha": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14",
    },
)

PREDICATES = {
    "ate", "deliver", "delivers", "draw", "live", "maps", "reviled",
    "saw", "spam", "spots", "stops", "was",
}


def predicates(text: str) -> list[str]:
    return sorted(set(re.findall(r"[a-z]+", text.casefold())) & PREDICATES)


def build_payload() -> dict[str, object]:
    _, parent_rendered, parent_tape = load_parent()
    parent_words = parent_rendered.split()
    prior = json.loads(PRIOR.read_text())
    prior_rows = {row["id"]: row for row in prior["rows"]}
    baseline = {
        35: prior_rows["depth35-live-seam-abce"],
        39: prior_rows["depth39-live-seam-abce"],
    }

    pair_audit = {}
    for key, (left, right) in FRAME_PAIRS.items():
        left_tape = independent_tape(left)
        right_tape = independent_tape(right)
        assert left_tape == right_tape[::-1]
        pair_audit[key] = {
            "left": left,
            "right": right,
            "letters_per_side": len(left_tape),
            "pair_exact": True,
            "left_predicates": predicates(left),
            "right_predicates": predicates(right),
        }

    rows = []
    for spec in SPECS:
        start, stop, partial = spec["middle_slice"]
        middle = " ".join([*parent_words[start:stop], partial])
        assert independent_tape(middle) == parent_tape[spec["depth"]:498 - spec["depth"]]

        left_spans = [FRAME_PAIRS[key][0] for key in spec["path"]] + [spec["seam_left"]]
        right_spans = [spec["seam_right"]] + [
            FRAME_PAIRS[key][1] for key in reversed(spec["path"])
        ]
        left_tape = independent_tape(" ".join(left_spans))
        right_tape = spec["supplied"] + independent_tape(" ".join(right_spans))
        assert right_tape == left_tape[::-1]

        rendered = (
            " ".join(left_spans)
            + " "
            + middle
            + spec["supplied"]
            + ". "
            + " ".join(right_spans)
        )
        row_audit = audit(rendered)
        assert row_audit["letters"] > 530
        assert row_audit["two_pointer_exact"]
        assert row_audit["project_validator_exact"]
        assert row_audit["sha_equal"]
        assert row_audit["sha256_forward"] == spec["expected_sha"]

        outer_text = " ".join([*left_spans, *right_spans])
        baseline_outer = " ".join([
            *baseline[spec["depth"]]["left_spans"],
            *baseline[spec["depth"]]["right_spans"],
        ])
        rows.append({
            "id": spec["id"],
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_sha256": PARENT_SHA256,
            "immediate_prior_artifact": str(PRIOR.relative_to(ROOT)),
            "depth": spec["depth"],
            "partial_word": spec["partial_word"],
            "left_cursor": spec["depth"],
            "right_cursor": 498 - spec["depth"],
            "owner": "R",
            "residual_before_boundary": independent_tape(spec["seam_left"])[::-1],
            "supplied_boundary": spec["supplied"],
            "residual_after_boundary": independent_tape(spec["seam_right"]),
            "pair_path": list(spec["path"]),
            "rendered": rendered,
            "left_spans": left_spans,
            "right_spans": right_spans,
            "audit": row_audit,
            "growth_over_parent": row_audit["letters"] - 498,
            "removed_inherited_outer_letters": spec["depth"] * 2,
            "frame_repair": {
                "baseline_id": baseline[spec["depth"]]["id"],
                "baseline_outer_predicates": predicates(baseline_outer),
                "repaired_outer_predicates": predicates(outer_text),
                "baseline_saw_was_count": sum(
                    word in {"saw", "was"}
                    for word in re.findall(r"[a-z]+", baseline_outer.casefold())
                ),
                "repaired_saw_was_count": sum(
                    word in {"saw", "was"}
                    for word in re.findall(r"[a-z]+", outer_text.casefold())
                ),
            },
            "working_track_debt": [
                f"the inherited {498 - 2 * spec['depth']}-letter middle remains rough",
                "some reverse-facing clauses use vocative or literary syntax",
                "exactness and frame diversity do not certify human readability",
            ],
        })

    rows.sort(key=lambda row: (-row["audit"]["letters"], row["id"]))
    return {
        "experiment_id": "incumbent-498-event-frame-seam-repair-20261002",
        "method": "replace formulaic name saw/was shells with varied exact event frames while preserving the incumbent's live partial-word equations",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 498,
            "sha256": PARENT_SHA256,
            "independent_exact": True,
        },
        "prior": str(PRIOR.relative_to(ROOT)),
        "pair_library": pair_audit,
        "stats": {
            "authored_paths": len(rows),
            "independently_exact_children": len(rows),
            "children_over_530": sum(row["audit"]["letters"] > 530 for row in rows),
            "shortest_letters": min(row["audit"]["letters"] for row in rows),
            "longest_letters": max(row["audit"]["letters"] for row in rows),
            "maximum_inherited_outer_letters_replaced": max(
                row["removed_inherited_outer_letters"] for row in rows
            ),
        },
        "active_frontier": [
            "depth39-longest-f1g1h1r",
            "depth39-varied-f1g1h1k",
            "depth39-varied-f2g2h2k",
            "depth35-varied-f1g1h1",
        ],
        "provenance": {
            "parent_loaded_and_verified_at_runtime": True,
            "pairs_authored_as_complete_event_frames": True,
            "pair_exactness_checked_before_composition": True,
            "finished_parent_tape_reversal": False,
            "posthoc_character_repair": False,
            "per_candidate_model_scoring": False,
            "human_certified": False,
        },
        "next_repair": "keep the 554 varied-frame child and reopen the next deeper inherited seam; require a finite main-clause continuation instead of another name pair",
        "rows": rows,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
