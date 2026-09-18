"""Replace the blocked work*/adverb/own seam with a reported craft event.

An ordinary first-person plural agent reports a completed woodworking action.
The clause is finite (we have ... hewn/carved/shaped ... tool), not a fragment
or an invented suffix. The joint repair changes subject realization, aspect,
predicate argument semantics, and the optional adverbial position together.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).with_name("declarative_tool_message_repair_20260913.py")
SPEC = importlib.util.spec_from_file_location("perfect_craft_parent", SOURCE)
DECLARATIVE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = DECLARATIVE
SPEC.loader.exec_module(DECLARATIVE)
BASE, CAPTURE, PARENT = DECLARATIVE.BASE, DECLARATIVE.CAPTURE, DECLARATIVE.PARENT
W = PARENT.word
LEXICON = {
    "first_person_plural": ("we",),
    "perfect_auxiliary": ("have", "had"),
    "craft_participle": ("hewn", "carved", "shaped", "sanded", "made", "cut"),
    "craft_adverb": ("carefully", "recently", "already", "just"),
    "and": ("and",),
}


class Grammar(DECLARATIVE.Grammar):
    def productions(self, lhs):
        if lhs.name == "W" and lhs.feature("category") in LEXICON:
            category = lhs.feature("category")
            return tuple(BASE.Production(f"word:{category}:{form}", lhs,
                         (BASE.sym("T", form=form, label=category),)) for form in LEXICON[category])
        if lhs.name == "CRAFT_REPORT":
            return tuple(BASE.Production(f"craft_report:perfect:{adverbial}", lhs,
                         (W("first_person_plural"), W("perfect_auxiliary"))
                         + ((W("craft_adverb"),) if adverbial else ())
                         + (W("craft_participle"), BASE.sym("ARTIFACT", role="tool")))
                         for adverbial in (False, True))
        original = super().productions(lhs)
        if lhs.name == "S":
            return (BASE.Production("instruction:reported_craft_event", lhs,
                    (W("instruction"), BASE.sym("HUMAN", role="recipient"), BASE.sym("CRAFT_REPORT"))),) + original
        if lhs.name == "HUMAN" and lhs.feature("role") == "recipient":
            # One human head owns both identifying predicates. This expands
            # grammatical control length without repeating a phrase or person.
            return original + (BASE.Production("human:recipient:coordinated_repairs", lhs,
                   (W("det"), W("adj_person"), W("recipient"), W("who"),
                    W("repair"), BASE.sym("ARTIFACT", role="wooden_object"), W("and"),
                    W("repair"), BASE.sym("ARTIFACT", role="wooden_object"))),)
        return original


CONTROL = ("tell a careful manager who restored the broken cabinet and repaired the old furniture "
           "we have recently hewn a new wooden mallet")


def semantic_witness(grammar, text):
    tree = BASE.parse_tree(grammar, text)
    reports = []
    def visit(node):
        if node.symbol.name == "CRAFT_REPORT":
            reports.append({"subject": node.children[0].symbol.feature("category"),
                            "finite_auxiliary": node.children[1].symbol.feature("category"),
                            "predicate": node.children[-2].symbol.feature("category"),
                            "patient": node.children[-1].symbol.feature("role")})
        for child in node.children:
            visit(child)
    if tree is not None: visit(tree)
    return {"independent_complete_reparse": tree is not None,
            "typed_report": reports,
            "relation": "The addressee is to learn that the speakers have made a woodworking tool by carving it. The repair relative identifies the manager.",
            "human_readability_certified": False}


def run(max_states=100000):
    grammar = Grammar()
    prior_path = DECLARATIVE.MODAL.INSTRUCTION.ROOT / "runs/declarative-tool-message-repair-2026-09-13/result-01.json"
    prior = json.loads(prior_path.read_text())
    state = CAPTURE.replay(DECLARATIVE.Grammar(), prior["deepest_actual_search_witness"]["ledger"])
    assert state.length == 23 and state.residual == "k"
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    control["semantic_witness"] = semantic_witness(grammar, CONTROL)
    search = CAPTURE.solve(grammar, max_states)
    return {"method": "perfect_aspect_reported_craft_event_joint_repair",
            "config": {"max_states": max_states, "control_used_as_search_seed": False},
            "prior_failure": {"artifact": str(prior_path), "replay_verified": True,
                              "conflict": prior["deepest_actual_search_witness"]["next_character_conflict"],
                              "joint_boundary_requirement": "Keeping workers and own forces the preceding adverb to end kr; no adverb in the typed inventory has that suffix."},
            "repair": "Replace plural occupational subject plus modal ownership with a first-person-plural completed craft event, a perfect auxiliary, and optional aspect/manner adverb.",
            "endpoint_preflight": grammar.endpoint_rows, "control": control,
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                           "source": "task-authored individual lexemes and typed clauses; no catalogue, frozen half, phrase seed, or preassigned interior words"},
            "immediate_successor_repair": "If the perfect-auxiliary seam fails, replace the recipient/terminal-event pair jointly, retaining a complete communication act and the unequal outer word boundaries.",
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
