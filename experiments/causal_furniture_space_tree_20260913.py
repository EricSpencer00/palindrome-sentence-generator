"""New unequal A/sofa construction: a complete causal furniture-moving event.

A household or craft agent moves one furniture item because another human
needs room for a sofa. Moved furniture and incoming sofa have disjoint lexical
heads by construction. No previous reply, proper name, word sequence, or
mirrored segment is used as a generation seed. Only exposed W slots choose
words; existing exact search also rejects repeated non-function content.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("moving_reply_existing_branch_audit_20260913.py")
SPEC = importlib.util.spec_from_file_location("furniture_moving_evidence", SOURCE)
PRIOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = PRIOR
SPEC.loader.exec_module(PRIOR)
BASE, CAPTURE, PARENT = PRIOR.BASE, PRIOR.CAPTURE, PRIOR.PARENT
W = PARENT.word
LEXICON = {
    "opening": ("a",), "det": ("a", "the", "that", "this", "one"),
    "person": ("father", "mother", "worker", "carpenter", "foreman", "farmer", "friend", "relative", "helper", "guardian"),
    "family_modifier": ("foster",), "family_head": ("parent", "carer"),
    "adj_person": ("careful", "young", "patient", "tired", "skilled"),
    "who": ("who",), "repair": ("repaired", "restored", "fixed", "rebuilt"),
    "repair_object": ("cabinet", "cupboard", "door", "frame"),
    "adj_repair": ("broken", "damaged", "wooden", "old"),
    "move": ("moved", "shifted", "dragged", "pushed", "repositioned"),
    "moved_furniture": ("table", "desk", "chair", "bench"),
    "adj_furniture": ("heavy", "wooden", "metal", "old", "new"),
    "because": ("because", "since"), "need": ("needed",), "space": ("space", "room"), "for": ("for",),
    "incoming_furniture": ("sofa",),
    "adj_incoming": ("soft", "old", "new", "large", "wide", "comfortable", "red", "blue", "velvet"),
}


def endpoint_preflight():
    rows = []
    for left in LEXICON["opening"]:
        for right in LEXICON["incoming_furniture"]:
            count = min(len(left), len(right))
            rows.append({"opening": left, "ending": right, "outer_character_overlap": left[:count] == right[::-1][:count],
                         "forces_proper_palindromic_interior": left == right[::-1],
                         "retained": left[:count] == right[::-1][:count] and left != right[::-1],
                         "lexicalization": "deferred_until_exposed_leaf"})
    return rows


class Grammar(BASE.FeatureGrammar):
    def __init__(self):
        self.endpoint_rows = endpoint_preflight()
        assert all(row["retained"] for row in self.endpoint_rows)
        assert not set(LEXICON["moved_furniture"]) & set(LEXICON["incoming_furniture"])

    def start(self): return BASE.sym("S")

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        rows = []
        def add(name, *rhs): rows.append(BASE.Production(name, lhs, tuple(rhs)))
        if lhs.name == "S":
            add("causal:make_space_for_distinct_furniture", W("opening"), BASE.sym("AGENT_BODY"), W("move"),
                BASE.sym("MOVED_ITEM"), W("because"), BASE.sym("BENEFICIARY"), W("need"), W("space"), W("for"), BASE.sym("INCOMING_ITEM"))
        elif lhs.name == "AGENT_BODY":
            add("agent:ordinary", W("person"))
            add("agent:modified", W("adj_person"), W("person"))
            add("agent:foster_family_role", W("family_modifier"), W("family_head"))
        elif lhs.name == "BENEFICIARY":
            add("beneficiary:plain", W("det"), W("person"))
            add("beneficiary:identified_by_repair", W("det"), W("adj_person"), W("person"), W("who"), W("repair"),
                W("det"), W("adj_repair"), W("repair_object"))
        elif lhs.name == "MOVED_ITEM":
            add("moved:plain", W("det"), W("moved_furniture"))
            add("moved:modified", W("det"), W("adj_furniture"), W("moved_furniture"))
        elif lhs.name == "INCOMING_ITEM":
            add("incoming:plain", W("det"), W("incoming_furniture"))
            add("incoming:modified", W("det"), W("adj_incoming"), W("incoming_furniture"))
        elif lhs.name == "W":
            category = lhs.feature("category")
            for form in LEXICON[category]: add(f"word:{category}:{form}", BASE.sym("T", form=form, label=category))
        return tuple(rows)


CONTROL = ("a foster parent moved the heavy table because the careful carpenter who repaired "
           "the damaged cabinet needed room for a comfortable sofa")


def semantic_witness(grammar, text):
    tree = BASE.parse_tree(grammar, text)
    if tree is None: return {"independent_complete_reparse": False}
    moved, incoming = [], []
    def visit(node):
        if node.symbol.name == "T":
            if node.symbol.feature("label") == "moved_furniture": moved.append(node.symbol.feature("form"))
            if node.symbol.feature("label") == "incoming_furniture": incoming.append(node.symbol.feature("form"))
        for child in node.children: visit(child)
    visit(tree)
    return {"independent_complete_reparse": True, "moved_items": moved, "incoming_items": incoming,
            "distinct_furniture_heads": not set(moved) & set(incoming),
            "relation": "Moving the existing furniture makes space for a different incoming item requested by a human beneficiary.",
            "human_readability_certified": False}


def run(max_states=100000):
    grammar = Grammar()
    prior_path = ROOT / "runs/moving-reply-existing-branch-audit-2026-09-13/result-01.json"
    prior = json.loads(prior_path.read_text())
    prior_state = CAPTURE.replay(PRIOR.Grammar(), prior["deepest_actual_search_witness"]["ledger"])
    assert prior_state.length == 9 and prior_state.residual == "i"
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    control["semantic_witness"] = semantic_witness(grammar, CONTROL)
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    return {"method": "causal_furniture_clearance_unequal_endpoint_full_tree",
            "config": {"max_states": max_states, "control_used_as_seed": False},
            "prior_failure": {"artifact": str(prior_path), "replay_verified": True, "new_construction": False,
                              "conflict": prior["deepest_actual_search_witness"]["next_character_conflict"]},
            "replacement": "Replace the reply/coplanar motion seam with a complete furniture-moving event and its explicit space-making cause; use A/sofa instead of No/motion.",
            "endpoint_preflight": grammar.endpoint_rows, "control": control,
            "construction_invariants": {"moved_and_incoming_heads_disjoint": True,
                "repeated_non_function_content_pruned_at_exposed_lexical_selection": True,
                "central_admission_applied_to_every_closure": True},
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                "source": "task-authored individual lexemes and typed causal productions; no proper names, catalogue material, frozen half, or mirrored phrase"},
            "immediate_successor_repair": "Use the actual household-agent/sofa-modifier boundary to replace that paired realization through another ordinary complete causal event, without lexical padding.",
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
