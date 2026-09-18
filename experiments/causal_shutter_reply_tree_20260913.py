"""A new communication act with the unequal No/position endpoint relation.

The full root owns a negative reply and its causal explanation. Semantic mode
links the shutter's stated state to the technician's action and resulting
position/motion. There is no Tell/mallet grammar inheritance or phrase seed.
Lexical words are assigned only by exposed W productions. The context question
is diagnostic metadata and never enters the generation search.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/perfect_craft_report_repair_20260913.py"
SPEC = importlib.util.spec_from_file_location("shutter_reply_evidence", SOURCE)
PRIOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = PRIOR
SPEC.loader.exec_module(PRIOR)
BASE, CAPTURE, PARENT = PRIOR.BASE, PRIOR.CAPTURE, PRIOR.PARENT
W = PARENT.word

LEXICON = {
    "reply": ("no",), "referent": ("it",), "copula": ("is",), "because": ("because", "since"),
    "det": ("the", "a", "that", "this", "one"),
    "technician": ("technician", "carpenter", "worker", "repairer", "engineer"),
    "adj_person": ("careful", "experienced", "skilled", "young"),
    "who": ("who",), "repair": ("repaired", "replaced", "fixed", "restored"),
    "component": ("hinge", "handle", "frame", "latch"),
    "adj_component": ("damaged", "broken", "old", "rusty"),
    "shutter": ("shutter", "door", "panel"), "adj_shutter": ("wooden", "metal", "heavy", "new"),
    "in": ("in",), "position": ("position",), "motion": ("motion",),
}
MODES = {
    "accessible": {"state": ("open", "ajar"), "action": ("left", "kept", "held"),
                   "position": ("raised", "high", "open"), "question": "Is the shutter closed?"},
    "closed": {"state": ("closed", "shut"), "action": ("left", "kept", "held"),
               "position": ("lowered", "low", "closed"), "question": "Is the shutter open?"},
    "moving": {"state": ("moving",), "action": ("set", "put"),
               "position": (), "question": "Is the shutter still?"},
}


def endpoint_preflight(openings=("no",), endings=("position", "motion")):
    result = []
    for left in openings:
        for right in endings:
            overlap = min(len(left), len(right))
            forced_interior = left == right[::-1]
            compatible = left[:overlap] == right[::-1][:overlap]
            result.append({"opening": left, "ending": right, "outer_character_overlap": compatible,
                           "forces_proper_palindromic_interior": forced_interior,
                           "retained": compatible and not forced_interior,
                           "lexicalization": "deferred_until_exposed_leaf"})
    return result


class Grammar(BASE.FeatureGrammar):
    def __init__(self):
        self.endpoint_rows = endpoint_preflight()
        assert all(row["retained"] for row in self.endpoint_rows)

    def start(self): return BASE.sym("S")

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        result = []
        def add(name, *rhs): result.append(BASE.Production(name, lhs, tuple(rhs)))
        if lhs.name == "S":
            for mode in MODES:
                add("reply:" + mode, W("reply"), W("referent"), W("copula"), W("state_" + mode),
                    W("because"), BASE.sym("CAUSE", mode=mode))
        elif lhs.name == "CAUSE":
            mode = lhs.feature("mode")
            add("cause:" + mode, BASE.sym("AGENT"), W("action_" + mode), BASE.sym("SHUTTER"), W("in"), BASE.sym("RESULT", mode=mode))
        elif lhs.name == "RESULT":
            mode = lhs.feature("mode")
            if mode == "moving":
                add("result:motion", W("motion"))
            else:
                add("result:position:" + mode, W("det"), W("position_" + mode), W("position"))
        elif lhs.name == "AGENT":
            for modified in (False, True):
                prefix = (W("det"),) + ((W("adj_person"),) if modified else ()) + (W("technician"),)
                add(f"agent:{modified}:plain", *prefix)
                add(f"agent:{modified}:repair_relative", *prefix, W("who"), W("repair"), BASE.sym("COMPONENT"))
        elif lhs.name == "COMPONENT":
            add("component:modified", W("det"), W("adj_component"), W("component"))
            add("component:plain", W("det"), W("component"))
        elif lhs.name == "SHUTTER":
            add("shutter:modified", W("det"), W("adj_shutter"), W("shutter"))
            add("shutter:plain", W("det"), W("shutter"))
        elif lhs.name == "W":
            category = lhs.feature("category")
            if category in LEXICON:
                forms = LEXICON[category]
            else:
                role, mode = category.split("_", 1)
                forms = MODES[mode][role]
            for form in forms:
                add(f"word:{category}:{form}", BASE.sym("T", form=form, label=category))
        return tuple(result)


CONTROL = ("no it is open because the experienced technician who repaired the damaged hinge "
           "left the wooden shutter in the raised position")


def semantic_witness(grammar, text):
    tree = BASE.parse_tree(grammar, text)
    if tree is None: return {"independent_complete_reparse": False}
    mode = tree.production.split(":")[1]
    cause = tree.children[-1]
    return {"independent_complete_reparse": True, "context_question": MODES[mode]["question"],
            "diagnostic_context_only": True, "mode": mode,
            "state_and_result_share_semantic_mode": cause.symbol.feature("mode") == mode == cause.children[-1].symbol.feature("mode"),
            "causal_roles": {"agent": cause.children[0].symbol.name, "patient": cause.children[2].symbol.name,
                             "result": cause.children[-1].symbol.name},
            "relation": "A technician's action leaves the shutter raised/open, lowered/closed, or moving; the reply denies the corresponding contrasting state.",
            "human_readability_certified": False}


def rejection_certificate(grammar, witness):
    state = CAPTURE.replay(grammar, witness["ledger"])
    side, edge = PARENT.active_edge(state)
    node = BASE.node_map(state)[state.frontier[edge]]
    choices = []
    for production in grammar.productions(node.symbol):
        if len(production.rhs) == 1 and production.rhs[0].name == "T":
            form = production.rhs[0].feature("form")
            char = form[0] if side == 1 else form[-1]
            choices.append({"word": form, "offered_character": char, "matches": char == state.residual})
    return {"required_character": state.residual, "side": side,
            "exposed_role": node.symbol.feature("category"), "lexical_alternatives": choices,
            "all_alternatives_reject": bool(choices) and not any(row["matches"] for row in choices)}


def run(max_states=100000):
    grammar = Grammar()
    prior_path = ROOT / "runs/perfect-craft-report-repair-2026-09-13/result-01.json"
    prior = json.loads(prior_path.read_text())
    prior_state = CAPTURE.replay(PRIOR.Grammar(), prior["deepest_actual_search_witness"]["ledger"])
    assert prior_state.length == 23 and prior_state.residual == "a"
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    control["rendered"] = "No, " + CONTROL[3:] + "."
    control["semantic_witness"] = semantic_witness(grammar, CONTROL)
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    search = CAPTURE.solve(grammar, max_states)
    return {"method": "causally_typed_shutter_state_reply_unequal_endpoint_tree",
            "config": {"max_states": max_states, "control_or_context_used_as_search_seed": False},
            "prior_failure": {"artifact": str(prior_path), "replay_verified": True,
                              "conflict": prior["deepest_actual_search_witness"]["next_character_conflict"]},
            "replacement": "Replace the entire tool-report communication act and its lexical endpoints with a negative state reply causally explained by a technician's complete action.",
            "endpoint_preflight": grammar.endpoint_rows, "control": control,
            "rejection_certificate": rejection_certificate(grammar, search["deepest_actual_search_witness"]),
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                           "source": "task-authored semantic modes, typed trees, and individual lexemes; no Tell/mallet productions, frozen text, catalogue, or candidate seed"},
            "immediate_successor_repair": "Use the persisted reply-state/result-position mismatch to replace that entire causal pair with a different ordinary state explanation; preserve exact exposed-leaf construction.",
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
