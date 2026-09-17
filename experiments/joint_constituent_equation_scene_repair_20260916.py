"""One held-out paired-constituent repair for the 2026-09-16 scene frontier.

The repair replaces only the complete outer constituents at the first mismatch
of the selected baseline scene.  All events stay in ordinary prose order; no
character, word, or mirrored-unit editing is used.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.joint_constituent_equation_scene_solver_20260916 import (
    LEFT,
    RIGHT,
    Constituent,
    exact_sha,
    exact_two_pointer,
    equation_frontier,
    independent_admission,
    novelty_preflight as baseline_novelty_preflight,
    tape,
)
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT = "joint-constituent-equation-scene-repair-20260916"
SIGNATURE = (
    "bilateral-complete-constituent-scene|paired-outer-frontier-repair|"
    "heldout-svo-adjunct|ordinary-right-scene-order|independent-pointer-sha-audit"
)
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

# The baseline rank-1 scene is retained in place except for its first and last
# complete constituents, which are replaced as one semantic pair.
REPAIR_LEFT = Constituent(
    "archivist-journals",
    "The patient archivist",
    "catalogs",
    "sealed journals",
    "beneath winter rafters",
    "archivist catalogs sealed journals beneath winter rafters",
)
REPAIR_RIGHT = Constituent(
    "signaler-boats",
    "The coastal signaler",
    "guides",
    "lantern boats",
    "at twilight",
    "signaler guides lantern boats at twilight",
)


def novelty_preflight() -> dict:
    """Check this repair family before constructing/rendering any prose."""
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [
        e["id"] for e in entries
        if e.get("id") != EXPERIMENT and e.get("signature") == SIGNATURE
    ]
    return {
        "entries_inspected": len(entries),
        "exact_signature_collisions_before_render": collisions,
        "passed": not collisions,
        "repair_scope": "replace exactly the paired complete constituents at the first frontier mismatch",
        "state_space_distinction": "one held-out SVO-adjunct pair replaces the outer baseline events; two interior events and normal scene order are fixed",
    }


def render() -> str:
    # Rank-1 baseline: left (0, 2), right (1, 0). Replace left[0] and right[1].
    return " ".join(
        [REPAIR_LEFT.text, LEFT[2].text, RIGHT[1].text, REPAIR_RIGHT.text]
    )


def audit() -> dict:
    text = render()
    normalized = tape(text)
    pointer = exact_two_pointer(text)
    sha = exact_sha(text)
    central = mechanical_admission_checks(text, min_letters=90, max_letters=220)
    independent = independent_admission(text)
    return {
        "rendered": text,
        "normalized_tape": normalized,
        "letters": len(normalized),
        "replaced_pair": {
            "left": {"id": REPAIR_LEFT.id, "meaning": REPAIR_LEFT.meaning, "position": "first"},
            "right": {"id": REPAIR_RIGHT.id, "meaning": REPAIR_RIGHT.meaning, "position": "last"},
        },
        "preserved_constituents": [LEFT[2].id, RIGHT[1].id],
        "equation_frontier": equation_frontier(
            [REPAIR_LEFT.text, LEFT[2].text], [RIGHT[1].text, REPAIR_RIGHT.text]
        ),
        "exact_check_two_pointer": pointer,
        "exact_check_sha256": sha,
        "independent_exact_agreement": pointer["exact"] == sha["exact"],
        "central_admission": central,
        "independent_admission": independent,
        "anti_shortcut_flags": {
            "fixed_tape": False,
            "reverse_decoder": False,
            "mirrored_word_units": False,
            "repeated_palindromic_unit": False,
            "catalogue_text_used": False,
            "isolated_character_edit": False,
            "complete_constituents_only": True,
            "normal_order_events": True,
        },
        "mechanically_admitted": bool(normalized)
        and normalized == normalized[::-1]
        and pointer["exact"]
        and sha["exact"]
        and all(central.values())
        and all(independent.values()),
        "provenance": {
            "baseline_experiment": "joint-constituent-equation-scene-solver-20260916",
            "baseline_selection": {"left_indices": [0, 2], "right_indices": [1, 0], "rank": 1},
            "repair_operator": "paired complete constituent replacement at first mirrored mismatch",
            "authoring": "fresh held-out role-compatible SVO-adjunct events",
        },
        "next_repair": "If the repaired outer pair remains open, replace only the next complete constituent pair at the first reported frontier mismatch and rerun novelty preflight before rendering.",
    }


def run() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_render']}")
    # Baseline preflight is evidence that this is a targeted continuation, not
    # a duplicate sweep; it does not enumerate or render the baseline space.
    baseline = baseline_novelty_preflight()
    row = audit()
    return {
        "experiment": EXPERIMENT,
        "signature": SIGNATURE,
        "status": "complete; one targeted frontier repair, no exact closure",
        "novelty_preflight": preflight,
        "baseline_novelty_preflight": baseline,
        "candidate_count": 1,
        "candidate": row,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
            "ordinary_order_rendering": True,
        },
        "anti_shortcut_policy": "No fixed tape, reverse decoding, mirrored units, catalogue import, repetition, or isolated-character edits; only one paired complete constituent replacement was attempted.",
    }


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "letters": audit()["letters"]}, indent=2))
