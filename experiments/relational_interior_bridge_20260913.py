"""Residual-safe insertion of complete relational modifier trees.

The shallow imperative run exhausted its finite grammar without a closure.
Its deepest path had left ``teacher`` and right ``met`` disagreeing at the
next character. This repair changes the interior tree topology: object NPs
can open one or two right-branching relational modifiers, each with a complete
typed complement. A bridge is expanded only at the currently exposed edge;
its words remain unassigned until their own edge is reached. The same exact
character zipper processes the newly opened subtree. No rendered half or
partial rejected text is retained as a construction seed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/shallow_imperative_lockstep_20260913.py"
SPEC = importlib.util.spec_from_file_location("relational_bridge_parent", SOURCE)
PARENT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = PARENT
SPEC.loader.exec_module(PARENT)
BASE = PARENT.BASE

# Relational words carry a complement type. This is a grammar extension,
# not an inventory of text fragments or reversed phrase pairs.
RELATIONS = {"of": "person", "beside": "object", "near": "place"}


class BridgeGrammar(PARENT.Grammar):
    def productions(self, lhs):
        if lhs.name == "NP" and lhs.feature("type") == "object":
            rows = list(super().productions(lhs))
            for modified in (False, True):
                prefix = (PARENT.word("det"),) + ((PARENT.word("adj_object"),) if modified else ()) + (PARENT.word("object"),)
                rows.append(BASE.Production(f"np:object:{int(modified)}:bridge", lhs,
                                            prefix + (BASE.sym("BRIDGE", remaining="2"),)))
            return tuple(rows)
        if lhs.name == "BRIDGE":
            depth = int(lhs.feature("remaining"))
            rows = []
            for relation, kind in RELATIONS.items():
                prefix = (BASE.sym("RELWORD", relation=relation), BASE.sym("BASE_NP", type=kind))
                rows.append(BASE.Production(f"bridge:{relation}:stop", lhs, prefix))
                if depth > 1:
                    rows.append(BASE.Production(f"bridge:{relation}:continue", lhs,
                                                prefix + (BASE.sym("BRIDGE", remaining=str(depth-1)),)))
            return tuple(rows)
        if lhs.name == "BASE_NP":
            # Bridge complements are full NPs but cannot recursively reopen
            # bridges. This yields a finite shallow control/search grammar.
            kind = lhs.feature("type")
            return tuple(BASE.Production(f"base-np:{kind}:{int(modified)}", lhs,
                          (PARENT.word("det"),) + ((PARENT.word("adj_"+kind),) if modified else ()) + (PARENT.word(kind),))
                         for modified in (False, True))
        if lhs.name == "RELWORD":
            # Keep the actual word unselected until this exposed slot expands.
            return (BASE.Production("relation-slot", lhs,
                      (BASE.sym("W", category="relation:" + lhs.feature("relation")),)),)
        if lhs.name == "W" and lhs.feature("category").startswith("relation:"):
            relation = lhs.feature("category").split(":", 1)[1]
            return (BASE.Production("relation-word:"+relation, lhs,
                      (BASE.sym("T", form=relation, label="relation"),)),)
        return super().productions(lhs)


CONTROL = ("draw the detailed portrait of the patient teacher beside the old drawing in the quiet studio "
           "and reward the careful artist with a rare medal")


def bridge_witness(grammar, text):
    tree = BASE.parse_tree(grammar, text)
    rows = []
    def walk(node):
        if node.symbol.name == "BRIDGE":
            relation = node.children[0].symbol.feature("relation")
            rows.append({"relation": relation, "complement_type": node.children[1].symbol.feature("type"),
                         "typed_complement_ok": node.children[1].symbol.feature("type") == RELATIONS[relation]})
        for child in node.children:
            walk(child)
    if tree:
        walk(tree)
    return {"independent_parse": tree is not None, "bridges": rows,
            "all_complements_complete_and_typed": tree is not None and all(row["typed_complement_ok"] for row in rows)}


def run(max_states=100000):
    grammar = BridgeGrammar()
    control = PARENT.audit(grammar, CONTROL, "intact_prose_grammar_control")
    control["bridge_witness"] = bridge_witness(grammar, CONTROL)
    control["diagnostic_only"] = True
    control["reader_status"] = "grammar control only; not a palindrome candidate or human evidence"
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    assert len(control["bridge_witness"]["bridges"]) == 2
    result = PARENT.solve(grammar, max_states)
    for collection in (result["exact_closures"], result["mechanically_admitted_closures"]):
        for row in collection:
            # Independently parse the exact rendered letters again with the
            # punctuation removed solely for this grammar parser.
            row["bridge_witness"] = bridge_witness(grammar, row["rendered"][:-1].lower())
    return {"method": "exposed_leaf_relational_interior_bridge",
            "construction_change": "insert complete typed relational modifier subtrees inside a shared object NP; expose their words through the exact character zipper",
            "prior_failure": {"artifact": "runs/shallow-imperative-lockstep-2026-09-13/result-01.json",
                              "deepest_emitted_letters": 19, "live_left_debt": "a", "conflicting_right_character": "m",
                              "interpretation": "existing lexical slots could not bridge the live interior; no prior text is reused as a seed"},
            "config": {"max_states": max_states, "min_letters": PARENT.MIN_LETTERS, "max_letters": PARENT.MAX_LETTERS,
                       "maximum_bridge_relations": 2, "lexicalize_only_exposed_leaf": True,
                       "single_shared_tree": True, "canonical_character_schedule": True},
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "parent_sha256": sha256(SOURCE.read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                           "catalogue_generation_material": False, "relations": RELATIONS},
            "complete_grammar_control": control, "outer_seam_oracle": PARENT.seam_oracle(grammar), **result,
            "next_reader_facing_test": "Any new mechanically admitted long closure must enter randomized blinded human reading with intact and shuffled controls; require coherent paraphrases and ordinary-English ratings.",
            "next_construction_if_no_candidate": "Introduce whole-clause causal and temporal adjunct bridges with finite typed subjects and predicates, because NP-only relations still constrain the opposite lexical endpoint."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-states", default=100000, type=int)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing artifact")
    result = run(args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "stats": result["stats"],
                      "exact": len(result["exact_closures"]), "admitted": len(result["mechanically_admitted_closures"])}))


if __name__ == "__main__":
    main()
