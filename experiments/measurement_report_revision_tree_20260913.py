"""New causal event: a quantitative finding motivates a report revision.

This replaces the furniture/space topology rather than editing its vocabulary.
An analyst revises a document because a distinct measurement agent measures a
comparative nutrient quantity in fruit. The comparison is the measure verb's
object; its fruit source is an in-phrase. One root owns all syntax, with all
individual forms chosen only at exposed W leaves. No phrase is a search seed.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("causal_furniture_space_tree_20260913.py")
SPEC = importlib.util.spec_from_file_location("measurement_revision_evidence", SOURCE)
PRIOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = PRIOR
SPEC.loader.exec_module(PRIOR)
BASE, CAPTURE, PARENT = PRIOR.BASE, PRIOR.CAPTURE, PRIOR.PARENT
W = PARENT.word
LEXICON = {
    "opening": ("an",), "analyst": ("analyst", "editor", "author", "auditor", "observer"),
    "adj_analyst": ("experienced", "alert"), "revision": ("revised", "updated", "corrected"),
    "det": ("the", "a", "an", "one", "that", "this"),
    "document": ("report", "summary", "chart", "draft"),
    "adj_document": ("preliminary", "detailed", "initial", "earlier"),
    "because": ("because", "since"),
    "measurement_agent": ("nutritionist", "researcher", "scientist", "technician"),
    "adj_measurement_agent": ("careful", "skilled", "patient"),
    "measure": ("measured",), "degree": ("more", "less"),
    "nutrient": ("sugar", "starch", "fiber"), "than": ("than",), "expected": ("expected",), "in": ("in",),
}
FRUITS = {
    "banana": {"head": ("banana",), "adjective": ("ripe", "fresh", "large", "small", "sweet")},
    "sultana": {"head": ("sultana",), "adjective": ("large", "small", "sweet")},
}


def endpoint_preflight():
    rows = []
    for left in LEXICON["opening"]:
        for kind, record in FRUITS.items():
            for right in record["head"]:
                count = min(len(left), len(right))
                compatible = left[:count] == right[::-1][:count]
                forced_interior = left == right[::-1]
                rows.append({"opening": left, "ending": right, "fruit_type": kind,
                             "outer_character_overlap": compatible, "forces_proper_palindromic_interior": forced_interior,
                             "retained": compatible and not forced_interior, "lexicalization": "deferred_until_exposed_leaf"})
    return rows


class Grammar(BASE.FeatureGrammar):
    def __init__(self):
        self.endpoint_rows = endpoint_preflight()
        self.fruit_types = tuple(row["fruit_type"] for row in self.endpoint_rows if row["retained"])
        roles = [set(LEXICON[key]) for key in ("analyst", "measurement_agent", "document", "nutrient")]
        roles.append({word for record in FRUITS.values() for word in record["head"]})
        assert all(not left & right for index, left in enumerate(roles) for right in roles[index + 1:])

    def start(self): return BASE.sym("S")

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        rows = []
        def add(name, *rhs): rows.append(BASE.Production(name, lhs, tuple(rhs)))
        if lhs.name == "S":
            for fruit in self.fruit_types:
                add("revision:measurement_caused:" + fruit, W("opening"), BASE.sym("ANALYST"), W("revision"),
                    BASE.sym("DOCUMENT"), W("because"), BASE.sym("FINDING", fruit=fruit))
        elif lhs.name == "ANALYST":
            add("analyst:plain", W("analyst"))
            add("analyst:modified", W("adj_analyst"), W("analyst"))
        elif lhs.name == "DOCUMENT":
            add("document:plain", W("det"), W("document"))
            add("document:modified", W("det"), W("adj_document"), W("document"))
        elif lhs.name == "FINDING":
            add("finding:comparative_measurement", BASE.sym("MEASUREMENT_AGENT"), W("measure"), BASE.sym("QUANTITY"),
                W("in"), BASE.sym("FRUIT", kind=lhs.feature("fruit")))
        elif lhs.name == "MEASUREMENT_AGENT":
            add("measurement_agent:plain", W("det"), W("measurement_agent"))
            add("measurement_agent:modified", W("det"), W("adj_measurement_agent"), W("measurement_agent"))
        elif lhs.name == "QUANTITY":
            add("quantity:compared_with_expectation", W("degree"), W("nutrient"), W("than"), W("expected"))
        elif lhs.name == "FRUIT":
            kind = lhs.feature("kind")
            add("fruit:plain:" + kind, W("det"), W("fruit_head_" + kind))
            add("fruit:modified:" + kind, W("det"), W("fruit_adjective_" + kind), W("fruit_head_" + kind))
        elif lhs.name == "W":
            category = lhs.feature("category")
            if category in LEXICON:
                forms = LEXICON[category]
            else:
                _, role, kind = category.split("_", 2)
                forms = FRUITS[kind][role]
            for form in forms: add(f"word:{category}:{form}", BASE.sym("T", form=form, label=category))
        return tuple(rows)


CONTROL = ("an experienced analyst revised the preliminary report because the careful nutritionist "
           "measured more sugar than expected in a ripe banana")


def semantic_witness(grammar, text):
    tree = BASE.parse_tree(grammar, text)
    if tree is None: return {"independent_complete_reparse": False}
    finding = tree.children[-1]
    quantity = finding.children[2]
    words = []
    def visit(node):
        if node.symbol.name == "T": words.append({"word": node.symbol.feature("form"), "role": node.symbol.feature("label")})
        for child in node.children: visit(child)
    visit(tree)
    return {"independent_complete_reparse": True,
            "comparison_is_measurement_object": quantity.symbol.name == "QUANTITY" and finding.children[1].symbol.feature("category") == "measure",
            "quantity_children": [child.symbol.feature("category") for child in quantity.children],
            "fruit_is_measured_source": finding.children[-1].symbol.name == "FRUIT",
            "role_words": words,
            "relation": "An analyst revises a document because a different measurement agent measured a nutrient quantity above or below expectation in fruit.",
            "human_readability_certified": False}


def run(max_states=100000):
    grammar = Grammar()
    prior_path = ROOT / "runs/causal-furniture-space-tree-2026-09-13/result-01.json"
    prior = json.loads(prior_path.read_text())
    prior_state = CAPTURE.replay(PRIOR.Grammar(), prior["deepest_actual_search_witness"]["ledger"])
    assert prior_state.length == 13 and prior_state.residual == "r"
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    control["semantic_witness"] = semantic_witness(grammar, CONTROL)
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    return {"method": "measurement_driven_report_revision_full_tree",
            "config": {"max_states": max_states, "control_used_as_seed": False},
            "prior_failure": {"artifact": str(prior_path), "replay_verified": True,
                              "conflict": prior["deepest_actual_search_witness"]["next_character_conflict"]},
            "replacement": "Replace household movement and space-making with a quantitative finding that causes document revision; replace A/sofa with An/banana or An/sultana.",
            "endpoint_preflight": grammar.endpoint_rows, "control": control,
            "construction_invariants": {"agent_document_nutrient_fruit_heads_pairwise_disjoint": True,
                "content_repetition_pruned_at_exposed_lexical_selection": True,
                "central_admission_applied_to_every_closure": True},
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                "source": "task-authored typed causal and comparative productions with individual lexemes; no names, catalogue, frozen half, or phrase seed"},
            "immediate_successor_repair": "Use the persisted analyst/fruit boundary to select a genuinely different initial clause realization with a compatible live frontier, rather than swapping an adjective.",
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
