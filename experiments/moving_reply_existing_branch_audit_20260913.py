"""Isolate a PRE-EXISTING moving reply branch; this is not a new generator."""
from __future__ import annotations
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).with_name("causal_shutter_reply_tree_20260913.py")
SPEC = importlib.util.spec_from_file_location("moving_reply_existing_parent", SOURCE)
SHUTTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = SHUTTER
SPEC.loader.exec_module(SHUTTER)
BASE, CAPTURE, PARENT = SHUTTER.BASE, SHUTTER.CAPTURE, SHUTTER.PARENT


class Grammar(SHUTTER.Grammar):
    def productions(self, lhs):
        rows = super().productions(lhs)
        return tuple(row for row in rows if row.identifier == "reply:moving") if lhs.name == "S" else rows


CONTROL = ("no it is moving because the experienced technician who repaired that damaged handle "
           "set that wooden shutter in motion")


def run(max_states=100000):
    grammar = Grammar()
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    control["semantic_witness"] = SHUTTER.semantic_witness(grammar, CONTROL)
    assert control["independent_parse"]
    return {"method": "existing_moving_reply_branch_isolation", "new_construction": False,
            "accounting": "This exact moving branch was already searched by causal_shutter_reply_tree_20260913; this artifact isolates its witness without claiming a new repair.",
            "config": {"max_states": max_states, "control_used_as_seed": False},
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest()},
            "endpoint_preflight": grammar.endpoint_rows, "control": control,
            "structural_failure_prediction": {"opening_letters": "noitis", "reversed_terminal_letters": "noitom",
                                              "first_conflict_zero_based": 4, "characters": ["i", "o"]},
            **CAPTURE.solve(grammar, max_states)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error("refusing to overwrite evidence")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "conflict": result["deepest_actual_search_witness"]["next_character_conflict"]}))


if __name__ == "__main__": main()
