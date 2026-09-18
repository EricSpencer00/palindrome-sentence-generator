"""New instruction topology with non-word-aligned outer boundaries.

The Reward/drawer family is excluded before expansion: reverse outer words
would force a proper palindromic interior in every complete output. Here a
typed communication command requests an explanation of a woodworking event.
The tell/mallet overlap crosses a word boundary. The preflight is only a
compatibility filter; it supplies neither a phrase nor a lexicalized tree.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/causal_subject_witness_capture_20260913.py"
SPEC = importlib.util.spec_from_file_location("instruction_capture", SOURCE)
CAPTURE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = CAPTURE
SPEC.loader.exec_module(CAPTURE)
BASE, PAIR, PARENT = CAPTURE.BASE, CAPTURE.PAIR, CAPTURE.PARENT
W = PARENT.word

LEXICON = {
    "instruction": ("tell", "show", "teach"),
    "det": ("a", "the", "my", "your", "our"),
    "recipient": ("man", "manager", "mechanic", "mason", "maker", "woman", "worker", "waiter", "carpenter", "apprentice"),
    "craftsperson": ("carpenter", "artisan", "craftsman", "woodworker", "builder", "cabinetmaker", "engineer"),
    "adj_person": ("careful", "experienced", "young", "skilled", "patient", "tired"),
    "wh": ("why", "how", "where", "when"),
    "who": ("who",),
    "repair": ("repaired", "restored", "rebuilt", "mended"),
    "wooden_object": ("cabinet", "chair", "table", "bench", "desk", "furniture", "stool", "shelf"),
    "adj_object": ("broken", "old", "damaged", "new", "wooden"),
    "test": ("tested", "inspected", "checked", "examined", "used", "held", "bought", "found", "made"),
    "tool": ("mallet", "hammer", "chisel", "plane", "saw", "drill", "clamp", "file", "sander"),
}


def endpoint_preflight(openings=None, endings=None):
    rows = []
    for opening in openings or LEXICON["instruction"]:
        for ending in endings or LEXICON["tool"]:
            forced_interior = opening == ending[::-1]
            compatible = PAIR.overlap_compatible(opening, ending)
            rows.append({"opening": opening, "ending": ending,
                         "forces_proper_palindromic_interior": forced_interior,
                         "outer_character_overlap": compatible,
                         "retained": compatible and not forced_interior,
                         "lexicalization": "deferred_until_exposed_leaf"})
    return rows


class Grammar(BASE.FeatureGrammar):
    def __init__(self):
        self.endpoint_rows = endpoint_preflight()
        self.retained = [row for row in self.endpoint_rows if row["retained"]]
        assert self.retained, "no admissible unequal-boundary endpoint relation"
        # One domain is kept for the entire tree; there is no frozen phrase.
        self.openings = tuple(dict.fromkeys(row["opening"] for row in self.retained))
        self.endings = tuple(dict.fromkeys(row["ending"] for row in self.retained))

    def start(self):
        return BASE.sym("S")

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        rows = []
        def add(label, *rhs):
            rows.append(BASE.Production(label, lhs, tuple(rhs)))
        if lhs.name == "S":
            add("instruction:request_explanation", W("instruction"), BASE.sym("HUMAN", role="recipient"), W("wh"), BASE.sym("CONTENT"))
        elif lhs.name == "CONTENT":
            add("content:woodworking_event", BASE.sym("HUMAN", role="craftsperson"), W("test"), BASE.sym("ARTIFACT", role="tool"))
        elif lhs.name == "HUMAN":
            role = lhs.feature("role")
            for modified in (False, True):
                prefix = (W("det"),) + ((W("adj_person"),) if modified else ()) + (W(role),)
                add(f"human:{role}:{modified}:plain", *prefix)
                add(f"human:{role}:{modified}:repair_relative", *prefix, W("who"), W("repair"), BASE.sym("ARTIFACT", role="wooden_object"))
        elif lhs.name == "ARTIFACT":
            role = lhs.feature("role")
            for count in (0, 1, 2):
                add(f"artifact:{role}:{count}", W("det"), *((W("adj_object"),) * count), W(role))
        elif lhs.name == "W":
            category = lhs.feature("category")
            forms = self.openings if category == "instruction" else self.endings if category == "tool" else LEXICON[category]
            for form in forms:
                add(f"word:{category}:{form}", BASE.sym("T", form=form, label=category))
        return tuple(rows)


CONTROL = ("tell a careful worker who repaired the old furniture why the experienced carpenter "
           "who restored the broken cabinet tested the damaged wooden mallet")


def semantic_witness(grammar, text):
    tree = BASE.parse_tree(grammar, text)
    roles, events = [], []
    def walk(node):
        if node.symbol.name in ("HUMAN", "ARTIFACT"):
            roles.append({"kind": node.symbol.name, "role": node.symbol.feature("role")})
        if node.production in ("instruction:request_explanation", "content:woodworking_event"):
            events.append(node.production)
        for child in node.children:
            walk(child)
    if tree is not None:
        walk(tree)
    return {"independent_complete_reparse": tree is not None, "typed_roles": roles, "events": events,
            "relation": "An addressee is to receive an explanation of a craftsperson's tool-use or tool-inspection event; optional relatives identify humans by furniture repairs.",
            "human_readability_certified": False}


def lexical_rejection_certificate(grammar, witness):
    state = CAPTURE.replay(grammar, witness["ledger"])
    side, edge = PARENT.active_edge(state)
    node = BASE.node_map(state)[state.frontier[edge]]
    alternatives = []
    for production in grammar.productions(node.symbol):
        if len(production.rhs) == 1 and production.rhs[0].name == "T":
            form = production.rhs[0].feature("form")
            char = form[0] if side == 1 else form[-1]
            alternatives.append({"word": form, "required_character": state.residual,
                                 "offered_character": char, "matches": char == state.residual})
    return {"exposed_symbol": {"name": node.symbol.name, "features": list(node.symbol.features)},
            "live_debt": state.residual, "side": side, "alternatives": alternatives,
            "all_lexical_alternatives_reject": bool(alternatives) and not any(row["matches"] for row in alternatives)}


def run(max_states=100000):
    grammar = Grammar()
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    control["semantic_witness"] = semantic_witness(grammar, CONTROL)
    assert control["independent_parse"]
    assert control["independent_exact_audit"]["letters"] > 100
    search = CAPTURE.solve(grammar, max_states)
    return {"method": "unequal_boundary_explanatory_instruction_connected_tree",
            "replacement_reason": "Exact reverse outer words force an inadmissible proper interior palindrome; their topology is rejected before search.",
            "endpoint_preflight": grammar.endpoint_rows,
            "config": {"max_states": max_states, "control_used_as_search_seed": False},
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                           "source": "task-authored individual lexemes and typed productions; no catalogue, half, phrase seed, or borrowed output"},
            "control": control,
            "lexical_rejection_certificate": lexical_rejection_certificate(grammar, search["deepest_actual_search_witness"]),
            "immediate_successor_repair": "Replace the forced past-tense test verb with a complete modal-plus-infinitive ownership event; retain the explanatory instruction and human/tool roles.",
            **search}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=100000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite evidence")
    result = run(args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "stats": result["stats"], "conflict": result["deepest_actual_search_witness"]["next_character_conflict"],
                      "exact_candidates": len(result["exact_closures"])}))


if __name__ == "__main__":
    main()
