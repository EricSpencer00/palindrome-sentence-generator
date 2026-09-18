"""A new causal frame: sustained demonstration use overheats equipment.

Some devices overheated because operators used THEM without breaks for long
demonstrations. The typed pronoun refers back to the plural device subject,
so the causal relation does not depend on an unrelated activity. Duration and
lack of breaks explain overheating; no repair-relative is added for length.
Every word remains an exposed W choice. Central endpoint provenance runs
before any expansion; every closure also receives full central admission.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("measurement_report_revision_tree_20260913.py")
SPEC = importlib.util.spec_from_file_location("device_overheating_evidence", SOURCE)
PRIOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = PRIOR
SPEC.loader.exec_module(PRIOR)
BASE, CAPTURE, PARENT = PRIOR.BASE, PRIOR.CAPTURE, PRIOR.PARENT
from llm_palindrome.admission import has_forbidden_catalogue_endpoint_scaffold
W = PARENT.word
LEXICON = {
    "opening": ("some",), "device": ("devices", "projectors", "screens", "monitors", "computers"),
    "adj_device": ("electronic", "small", "new", "old"), "overheat": ("overheated",),
    "because": ("because", "since"), "det": ("the", "some", "those"),
    "operator": ("technicians", "engineers", "presenters", "designers"),
    "adj_operator": ("experienced", "skilled", "careful", "young"),
    "use": ("used", "operated"), "device_anaphor": ("them",),
    "without": ("without",), "break": ("breaks", "pauses", "rests"), "for": ("for",),
    "duration": ("prolonged", "long", "extended", "continuous"), "live": ("live",),
    "demonstration": ("demos", "demonstrations", "presentations"),
}


def endpoint_preflight():
    rows = []
    for left in LEXICON["opening"]:
        for right in LEXICON["demonstration"]:
            count = min(len(left), len(right))
            compatible = left[:count] == right[::-1][:count]
            forced = left == right[::-1]
            provenance_allowed = not has_forbidden_catalogue_endpoint_scaffold((left, right))
            rows.append({"opening": left, "ending": right, "outer_character_overlap": compatible,
                         "forces_proper_palindromic_interior": forced,
                         "central_endpoint_provenance_allowed": provenance_allowed,
                         "retained": compatible and not forced and provenance_allowed,
                         "lexicalization": "deferred_until_exposed_leaf"})
    return rows


class Grammar(BASE.FeatureGrammar):
    def __init__(self):
        self.endpoint_rows = endpoint_preflight()
        self.endings = tuple(row["ending"] for row in self.endpoint_rows if row["retained"])
        assert self.endings, "no endpoint survives overlap and central provenance checks"
        groups = [set(LEXICON[role]) for role in ("device", "operator", "demonstration", "break")]
        assert all(not left & right for index, left in enumerate(groups) for right in groups[index + 1:])

    def start(self): return BASE.sym("S")

    def terminal(self, symbol):
        return symbol.feature("form") if symbol.name == "T" else None

    def productions(self, lhs):
        rows = []
        def add(label, *rhs): rows.append(BASE.Production(label, lhs, tuple(rhs)))
        if lhs.name == "S":
            add("causal:overheating_after_sustained_use", W("opening"), BASE.sym("DEVICES", number="plural", referent="equipment"),
                W("overheat"), W("because"), BASE.sym("USE_EVENT", patient="equipment"))
        elif lhs.name == "DEVICES":
            add("devices:plain", W("device"))
            add("devices:modified", W("adj_device"), W("device"))
        elif lhs.name == "USE_EVENT":
            add("use:without_rest_for_demonstrations", BASE.sym("OPERATORS", number="plural"), W("use"),
                BASE.sym("ANAPHOR", number="plural", referent=lhs.feature("patient")), W("without"), W("break"), W("for"), BASE.sym("DEMONSTRATION"))
        elif lhs.name == "OPERATORS":
            add("operators:plain", W("det"), W("operator"))
            add("operators:modified", W("det"), W("adj_operator"), W("operator"))
        elif lhs.name == "ANAPHOR":
            if lhs.feature("number") == "plural" and lhs.feature("referent") == "equipment":
                add("anaphor:previous_plural_equipment", W("device_anaphor"))
        elif lhs.name == "DEMONSTRATION":
            add("demonstration:prolonged", W("duration"), W("demonstration"))
            add("demonstration:prolonged_live", W("duration"), W("live"), W("demonstration"))
        elif lhs.name == "W":
            category = lhs.feature("category")
            forms = self.endings if category == "demonstration" else LEXICON[category]
            for form in forms: add(f"word:{category}:{form}", BASE.sym("T", form=form, label=category))
        return tuple(rows)


CONTROL = ("some electronic devices overheated because the experienced technicians operated them "
           "without breaks for prolonged live demos")


def semantic_witness(grammar, text):
    tree = BASE.parse_tree(grammar, text)
    if tree is None: return {"independent_complete_reparse": False}
    devices, event = tree.children[1], tree.children[-1]
    anaphor = event.children[2]
    return {"independent_complete_reparse": True,
            "equipment_coreference": devices.symbol.feature("referent") == event.symbol.feature("patient") == anaphor.symbol.feature("referent"),
            "plural_anaphor": anaphor.symbol.feature("number") == devices.symbol.feature("number") == "plural",
            "lack_of_rest_is_part_of_use_event": event.children[3].symbol.feature("category") == "without" and event.children[4].symbol.feature("category") == "break",
            "demonstration_is_use_purpose": event.children[-1].symbol.name == "DEMONSTRATION",
            "relation": "The same electronic equipment overheats because operators use it for prolonged demonstrations without rest periods.",
            "human_readability_certified": False}


def run(max_states=100000):
    grammar = Grammar()
    prior_path = ROOT / "runs/measurement-report-revision-tree-2026-09-13/result-01.json"
    prior = json.loads(prior_path.read_text())
    prior_state = CAPTURE.replay(PRIOR.Grammar(), prior["deepest_actual_search_witness"]["ledger"])
    assert prior_state.length == 11 and prior_state.residual == "l"
    control = PARENT.audit(grammar, CONTROL, "intact_prose_diagnostic_only")
    control["semantic_witness"] = semantic_witness(grammar, CONTROL)
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"] > 100
    assert control["mechanical_checks"]["not_forbidden_catalogue_endpoint_scaffold"]
    return {"method": "device_overheating_caused_by_sustained_demo_use",
            "config": {"max_states": max_states, "control_used_as_seed": False},
            "prior_failure": {"artifact": str(prior_path), "replay_verified": True,
                              "conflict": prior["deepest_actual_search_witness"]["next_character_conflict"]},
            "replacement": "Retire analyst/fruit revision; a plural equipment subject and an explicitly coreferential prolonged-use event explain overheating with Some/demos endpoints.",
            "endpoint_preflight": grammar.endpoint_rows, "control": control,
            "construction_invariants": {"device_operator_demo_rest_heads_pairwise_disjoint": True,
                "content_repetition_pruned_at_exposed_lexical_selection": True,
                "central_endpoint_provenance_checked_before_expansion": True,
                "central_admission_applied_to_every_closure": True},
            "provenance": {"method_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "grammar_sha256": grammar.digest(),
                "source": "task-authored individual lexemes and coreferential causal productions; no catalogue, names, frozen half, or phrase seed"},
            "immediate_successor_repair": "Replace any persisted device/adverbial seam through a different whole causal construction; do not introduce a token selected solely to spell a missing suffix.",
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
