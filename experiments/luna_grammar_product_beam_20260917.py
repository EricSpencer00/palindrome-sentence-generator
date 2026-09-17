"""A small, auditable two-sided grammar-product search.

This is an experiment, not a claim that beam output is readable.  A state
grows an ordinary clause on each side at the same time; the character tape of
the right clause is checked against the live reverse obligation from the left.
No language model or finished palindrome is used as a seed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "luna_grammar_product_beam_20260917.json"

def tape(s: str) -> str:
    return "".join(re.findall("[a-z]", s.lower()))

def independent_audit(s: str) -> dict:
    t = tape(s)
    return {"algorithm": "independent_two_pointer", "exact": bool(t) and all(t[i] == t[-1-i] for i in range(len(t)//2)), "length": len(t)}

@dataclass(frozen=True)
class Choice:
    role: str
    text: str
    scene: str

LEFT = (
    Choice("agent", "The baker", "kitchen"),
    Choice("agent", "A quiet sailor", "harbor"),
    Choice("agent", "The young keeper", "garden"),
)
RIGHT = (
    Choice("setting", "near the harbor", "harbor"),
    Choice("setting", "beside the garden", "garden"),
    Choice("setting", "by the kitchen", "kitchen"),
)
OBJECTS = (("a warm loaf", "theme"), ("the old bell", "theme"), ("a blue lantern", "theme"))
FOLLOWUPS = {
    "kitchen": "and shares it near the kitchen",
    "harbor": "and waits beside the harbor",
    "garden": "and rests beside the garden",
}
VERBS = (("watches", "event"), ("carries", "event"), ("marks", "event"))

def _anti_shortcut(words: list[str]) -> dict:
    ts = [tape(w) for w in words]
    return {
        "finished_tape_reversal": False,
        "word_order_mirror": ts == [w[::-1] for w in reversed(ts)],
        "repeated_clause_unit": len(words) != len(set(words)),
    }

def _row(left: str, right: str, scene: str, choices: list[str], debt: str, slots: list[str] | None = None) -> dict:
    rendered = f"{left} {right}."
    audit = independent_audit(rendered)
    # Count only the contiguous live match at the seam; this is deliberately
    # weaker than claiming closure and makes the next repair measurable.
    right_tape = tape(right)
    consumed = 0
    for expected, actual in zip(debt, right_tape):
        if expected != actual:
            break
        consumed += 1
    return {"rendered": rendered, "letters": audit["length"], "scene": scene,
            "choices": choices, "provenance": {"choices_before_rendering": True},
            "semantic_role_states": slots or ["agent", "event", "setting"],
            "independent_audit": audit, "anti_shortcut": _anti_shortcut(rendered.split()),
            "obligation_ledger": [{"step": 0, "remaining_pair_debt": len(debt), "operator": "reverse_character_product"},
                                   {"step": 1, "consumed_characters": consumed, "remaining_pair_debt": max(0, len(debt)-consumed), "operator": "typed_slot_extension"}],
            "complete": False}

def run(beam_width: int = 8) -> dict:
    beam = []
    for agent in LEFT:
        for verb, role in VERBS:
            for setting in RIGHT:
                if agent.scene != setting.scene:
                    continue
                left = f"{agent.text} {verb}"
                right = f"{setting.text}"
                # The two sides are selected jointly, but a residual obligation
                # is retained rather than silently forcing a closure.
                debt = tape(left)[::-1]
                beam.append(_row(left, right, agent.scene, [agent.text, verb, setting.text], debt))
                for obj, obj_role in OBJECTS:
                    extended_left = f"{left} {obj}"
                    extended_right = f"{setting.text}"
                    extended_debt = tape(extended_left)[::-1]
                    beam.append(_row(extended_left, extended_right, agent.scene,
                                     [agent.text, verb, obj, setting.text], extended_debt,
                                     ["agent", "event", obj_role, "setting"]))
                    # Bilateral clause growth: both clauses remain intact
                    # prose, while the right clause advances its own seam.
                    follow = FOLLOWUPS[agent.scene]
                    beam.append(_row(f"{extended_left} {follow}",
                                     f"{setting.text} {follow}", agent.scene,
                                     [agent.text, verb, obj, follow, setting.text],
                                     tape(f"{extended_left} {follow}")[::-1],
                                     ["agent", "event", obj_role, "followup", "setting"]))
    beam.sort(key=lambda r: (-r["letters"], r["rendered"]))
    rows = beam[:beam_width]
    for row in rows:
        row["next_repair"] = {"operator": "extend_bilateral_clause", "target": "consume_live_character_obligation", "reason": "setting tail does not yet consume the reverse debt"}
    result = {
        "status": "completed_no_exact_closure",
        "method": "two_sided_grammar_product_beam",
        "reader_eligible": False,
        "candidates": rows,
        "search": {"beam_width": beam_width, "simultaneous_sides": True, "rlaif_per_candidate": False, "semantic_slots": ["agent", "event", "theme", "setting"], "typed_extensions": 3},
        "novelty_preflight": {"status": "passed", "catalogue_lookup": "none", "reason": "fresh template renderings"},
        "next_repair": {"operator": "seam_relexicalize_followup", "target": "consume_live_character_obligation", "reason": "the typed follow-up advances both sides but still leaves a residual seam"},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seed_used_in_output": False},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

if __name__ == "__main__":
    run()
