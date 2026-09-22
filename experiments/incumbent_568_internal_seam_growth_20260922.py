"""Grow the verified 568 incumbent through an internal live character seam.

The cut is deliberately not the previously explored outer ``w|on ... no|w``
cut.  At 73 normalized letters the incumbent exposes ``Mara stops|spots
Aram``.  A new authored event shell is admitted only when its left emission
and right obligation consume to an empty residual before the unchanged
center is rendered.
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


PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-internal-seam-growth-20260922.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
SEAM_LETTERS = 73
FRONTIER = (
    {
        "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "id": "outer-causal-scene-568-working-incumbent",
        "letters": 568,
        "sha256": PARENT_SHA256,
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


def raw_boundary_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if "a" <= char.lower() <= "z":
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(f"surface has fewer than {count} letters")


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    reverse_tape = tape[::-1]
    forward_sha = hashlib.sha256(tape.encode()).hexdigest()
    reverse_sha = hashlib.sha256(reverse_tape.encode()).hexdigest()
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "sha256_forward": forward_sha,
        "sha256_reverse": reverse_sha,
        "sha_equal": forward_sha == reverse_sha,
    }


def validate_frontier_entry(entry: dict[str, object]) -> None:
    artifact = ROOT / str(entry["artifact"])
    assert artifact.exists(), artifact
    payload = json.loads(artifact.read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    rendered = str(row["rendered"])
    recomputed = independent_audit(rendered)
    assert recomputed["normalized_letters"] == entry["letters"]
    assert recomputed["two_pointer_exact"]
    assert recomputed["sha256_forward"] == entry["sha256"]
    assert recomputed["sha_equal"]


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    parent_independent = independent_audit(parent_rendered)
    assert parent["audit"]["letters"] == 568
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    assert parent_independent["normalized_letters"] == 568
    assert parent_independent["two_pointer_exact"]
    assert parent_independent["sha256_forward"] == PARENT_SHA256
    assert parent_independent["sha_equal"]
    assert len(parent_tape) == 568 and parent_tape == parent_tape[::-1]
    for frontier_entry in FRONTIER:
        validate_frontier_entry(frontier_entry)

    left_raw = raw_boundary_after_letters(parent_rendered, SEAM_LETTERS)
    right_raw = raw_boundary_after_letters(parent_rendered, len(parent_tape) - SEAM_LETTERS)
    left_shell = parent_rendered[:left_raw]
    retained = parent_rendered[left_raw:right_raw]
    right_shell = parent_rendered[right_raw:]
    assert left_shell.endswith("Mara stops")
    assert right_shell.startswith(" spots Aram")
    assert normalize(left_shell) == normalize(right_shell)[::-1]

    left_extension = " rats. Nora spots a ram."
    right_extension = " Mara stops Aron. Star"
    left_emission = normalize(left_extension)
    right_obligation = normalize(right_extension)
    assert left_emission == right_obligation[::-1]

    rendered = left_shell + left_extension + retained + right_extension + right_shell
    project_audit = audit(rendered)
    independent = independent_audit(rendered)
    assert project_audit["letters"] == 602
    assert independent["normalized_letters"] == project_audit["letters"]
    assert independent["two_pointer_exact"]
    assert independent["sha_equal"]
    assert project_audit["two_pointer_exact"]
    assert project_audit["byte_pointer_exact"]
    assert project_audit["project_validator_exact"]

    row = {
        "id": "internal-mara-stops-602",
        "working_status": "568_lineage_growth_frontier",
        "rendered": rendered,
        "audit": project_audit,
        "independent_audit": independent,
        "lineage_root_artifact": str(PARENT.relative_to(ROOT)),
        "lineage_root_id": PARENT_ID,
        "lineage_root_sha256": PARENT_SHA256,
        "growth_over_root": project_audit["letters"] - 568,
        "new_event_content": [
            "Nora spots a ram",
            "Mara stops Aron",
            "Star spots Aram",
        ],
        "live_seam": {
            "normalized_cut_letters": SEAM_LETTERS,
            "left_cursor_raw_exclusive": left_raw,
            "right_cursor_raw_exclusive": right_raw,
            "left_partial_join": "Mara stops|spots",
            "right_partial_join": "spots|Aram",
            "retained_letters": len(normalize(retained)),
            "initial_owner": "left_extension",
            "left_emission": left_emission,
            "right_obligation": right_obligation,
            "right_consumption": right_obligation,
            "final_owner": None,
            "final_residual": "",
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "grammar_debt": {
            "inherited_proper_spans": True,
            "inherited_repeated_scaffolding": True,
            "rough_syntax": True,
            "human_reader_validation": False,
            "status": "exact growth candidate retained; debt does not demote 568",
        },
        "provenance": (
            "new authored internal seam shell over the loaded 568 artifact; "
            "center retained byte-for-byte and residual closed before admission"
        ),
    }

    return {
        "experiment_id": "incumbent-568-internal-seam-growth-20260922",
        "method": "internal 73-letter residual-owned seam growth",
        "working_incumbent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 568,
            "sha256": PARENT_SHA256,
        },
        "stats": {
            "independently_exact_children": 1,
            "children_longer_than_568": 1,
            "longest_letters": project_audit["letters"],
            "seam_letters": SEAM_LETTERS,
            "retained_center_letters": len(normalize(retained)),
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "preserved_frontier": [
            {
            **FRONTIER[0],
            },
            {
                **FRONTIER[1],
            },
            {
                **FRONTIER[2],
            },
            {
                **FRONTIER[3],
            },
        ],
        "rows": [row],
        "next_operator": (
            "retain the 568 incumbent and this 602 child; if the internal seam "
            "is changed, record its residual obstruction before moving to a new seam"
        ),
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
