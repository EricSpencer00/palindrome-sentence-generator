"""Replace a blocked embedded question with a complete declarative message.

Tell licenses both an indirect question and a finite declarative complement.
This operator adds the latter with an ordinary plural craft-agent subject,
bounded identifying repair relatives, and a modal tool-ownership predicate.
It is not a fragment deletion: the complement retains its subject and finite
modal. Every added lexical item remains an exposed W leaf in one tree.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).with_name("modal_tool_explanation_repair_20260913.py")
SPEC = importlib.util.spec_from_file_location("declarative_modal_parent", SOURCE)
MODAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODAL
SPEC.loader.exec_module(MODAL)
BASE, CAPTURE, PARENT = MODAL.BASE, MODAL.CAPTURE, MODAL.PARENT
W = PARENT.word


class Grammar(MODAL.Grammar):
    def productions(self, lhs):
        if lhs.name == "W" and lhs.feature("category") in ("craft_plural", "frequency"):
            category = lhs.feature("category")
            forms = (("workers", "woodworkers", "workmen", "carpenters", "artisans", "cabinetmakers", "builders")
                     if category == "craft_plural" else ("never", "ever", "also", "often", "rarely", "sometimes", "seldom"))
            return tuple(BASE.Production(f"word:{category}:{form}", lhs,
                         (BASE.sym("T", form=form, label=category),)) for form in forms)
        if lhs.name == "MESSAGE":
            rows = []
            for adverbial in (False, True):
                rhs = (BASE.sym("CRAFT_GROUP"), W("modal")) + ((W("frequency"),) if adverbial else ()) + (W("own"), BASE.sym("ARTIFACT", role="tool"))
                rows.append(BASE.Production(f"message:modal_ownership:{adverbial}", lhs, rhs))
            return tuple(rows)
        if lhs.name == "CRAFT_GROUP":
            rows = []
            for modified in (False, True):
                prefix = ((W("adj_person"),) if modified else ()) + (W("craft_plural"),)
                rows.append(BASE.Production(f"craft_group:{modified}:plain", lhs, prefix))
                rows.append(BASE.Production(f"craft_group:{modified}:repair_relative", lhs,
                            prefix + (W("who"), W("repair"), BASE.sym("ARTIFACT", role="wooden_object"))))
            return tuple(rows)
        original = super().productions(lhs)
        if lhs.name == "S":
            return (BASE.Production("instruction:declarative_message", lhs,
                    (W("instruction"), BASE.sym("HUMAN", role="recipient"), BASE.sym("MESSAGE"))),) + original
        return original


CONTROL = ("tell a careful manager who repaired the broken cabinet skilled woodworkers "
           "who restored the old furniture might also own a wooden mallet")


def run(max_states=100000):
    grammar = Grammar()
    prior_path = MODAL.INSTRUCTION.ROOT / "runs/modal-tool-explanation-repair-2026-09-13/result-01.json"
    prior = json.loads(prior_path.read_text())
    prior_state = CAPTURE.replay(MODAL.Grammar(), prior["deepest_actual_search_witness"]["ledger"])
    assert prior_state.length == 19
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    control["semantic_relation"] = "Tell a manager that skilled woodworkers might also own a woodworking tool; the two repair relatives identify the people by their craft work."
    search = CAPTURE.solve(grammar, max_states)
    return {"method": "declarative_tool_ownership_message_connected_tree",
            "config": {"max_states": max_states, "control_used_as_search_seed": False},
            "prior_failure": {"artifact": str(prior_path), "replay_verified": True,
                              "conflict": prior["deepest_actual_search_witness"]["next_character_conflict"]},
            "repair": "Replace the incompatible wh leaf with a complete finite declarative complement and ordinary plural craft agents; retain a finite modal and a physical tool object.",
            "endpoint_preflight": grammar.endpoint_rows, "control": control,
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                           "source": "task-authored typed productions and single lexemes; no catalogue, frozen half, or candidate seed"},
            "immediate_successor_repair": "Jointly replace the plural-agent/adverbial boundary through an ordinary craft-agent phrase and a complete modal event; do not insert nonwords or treat the frontier as a candidate.",
            **search}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists(): parser.error("refusing to overwrite evidence")
    result = run(args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "stats": result["stats"], "conflict": result["deepest_actual_search_witness"]["next_character_conflict"],
                      "exact_candidates": len(result["exact_closures"])}))


if __name__ == "__main__": main()
