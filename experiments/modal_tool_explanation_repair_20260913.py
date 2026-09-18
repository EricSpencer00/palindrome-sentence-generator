"""Repair the n-ending demand with a typed modal ownership explanation.

This changes finite-clause structure rather than inventing past morphology.
The infinitive own has a human subject and physical tool object, licensed by
might/could/would. The normal past event remains as an alternative. All words
are selected at exposed leaves and every character uses the same exact zipper.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).with_name("unequal_boundary_instruction_tree_20260913.py")
SPEC = importlib.util.spec_from_file_location("modal_instruction_parent", SOURCE)
INSTRUCTION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = INSTRUCTION
SPEC.loader.exec_module(INSTRUCTION)
BASE, CAPTURE, PARENT = INSTRUCTION.BASE, INSTRUCTION.CAPTURE, INSTRUCTION.PARENT
W = PARENT.word


class Grammar(INSTRUCTION.Grammar):
    def productions(self, lhs):
        if lhs.name == "W" and lhs.feature("category") in ("modal", "own"):
            category = lhs.feature("category")
            forms = ("might", "could", "would") if category == "modal" else ("own",)
            return tuple(BASE.Production(f"word:{category}:{form}", lhs,
                         (BASE.sym("T", form=form, label=category),)) for form in forms)
        original = super().productions(lhs)
        if lhs.name == "CONTENT":
            return (BASE.Production("content:modal_tool_ownership", lhs,
                    (BASE.sym("HUMAN", role="craftsperson"), W("modal"), W("own"), BASE.sym("ARTIFACT", role="tool"))),) + original
        return original


CONTROL = ("tell a careful worker who repaired the old furniture why the experienced carpenter "
           "who restored the broken cabinet might own a wooden mallet")


def run(max_states=100000):
    grammar = Grammar()
    prior_path = INSTRUCTION.ROOT / "runs/unequal-boundary-instruction-tree-2026-09-13/result-02.json"
    prior = json.loads(prior_path.read_text())
    original_state = CAPTURE.replay(INSTRUCTION.Grammar(), prior["deepest_actual_search_witness"]["ledger"])
    assert original_state.length == 15 and original_state.residual == "n"
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    control["semantic_witness"] = INSTRUCTION.semantic_witness(grammar, CONTROL)
    control["semantic_witness"]["relation"] = "The recipient is to learn why a carpenter might own a woodworking tool; repair relatives identify the people by their craft work."
    search = CAPTURE.solve(grammar, max_states)
    witness = search["deepest_actual_search_witness"]
    return {"method": "modal_tool_ownership_explanation_structural_repair",
            "config": {"max_states": max_states, "control_used_as_search_seed": False},
            "prior_failure": {"artifact": str(prior_path), "replay_verified": True,
                              "certificate": prior["lexical_rejection_certificate"]},
            "repair": "A complete modal ownership predicate licenses an n-final infinitive while preserving a human agent and a physical woodworking tool.",
            "endpoint_preflight": grammar.endpoint_rows, "control": control,
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                           "source": "task-authored typed productions and individual words; no phrase seeds or catalogue material"},
            "lexical_rejection_certificate": INSTRUCTION.lexical_rejection_certificate(grammar, witness),
            "immediate_successor_repair": "If the wh/own boundary fails, change the communication valency to a complete declarative content clause; do not alter word spellings or substitute fragments.",
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
