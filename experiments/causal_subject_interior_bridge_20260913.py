"""Replace a relative-pronoun collision by a complete causal-clause subject.

The preceding run got through temp/met, then failed on temp's p against who's
o. This topology introduces a complete finite causal clause after the main
imperative(s), so an ordinary subject such as bishop or help occupies the
formerly blocking boundary before met. Compositional work-site person NPs
also permit the interior contact to span two independently selected words.
All lexical choices still occur at exposed W leaves of one shared tree.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/typed_occupation_pair_consistency_20260913.py"
SPEC = importlib.util.spec_from_file_location("causal_subject_pair", SOURCE)
PAIR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = PAIR
SPEC.loader.exec_module(PAIR)
BASE, PARENT = PAIR.BASE, PAIR.PARENT
ADDITIONAL_WORDS = {
    "causal": ("because", "since", "after"),
    "workplace_modifier": ("temple", "school", "harbor", "market", "garden"),
    "person": ("bishop", "chap", "cop", "group", "help", "helper", "keeper", "worker", "clerk"),
}


class Grammar(PAIR.Grammar):
    def productions(self, lhs):
        if lhs.name == "S":
            return (BASE.Production("sentence:causal-bridge", lhs, (BASE.sym("MAIN"), BASE.sym("CAUSE"))),
                    BASE.Production("sentence:main-only", lhs, (BASE.sym("MAIN"),)))
        if lhs.name == "MAIN":
            return tuple(BASE.Production("main:"+p.identifier, lhs, p.rhs)
                         for p in super().productions(BASE.sym("S")))
        if lhs.name == "CAUSE":
            return (BASE.Production("causal:complete-past-event", lhs,
                      (PARENT.word("causal"), PARENT.np("person",False), PARENT.word("past_person"),
                       PARENT.np("person",False))),)
        if lhs.name == "NP" and lhs.feature("type") == "person":
            # Both the site and occupation remain unassigned W slots. Their
            # compositional reading is a person working at the named site.
            return super().productions(lhs) + (BASE.Production("np:workplace-person", lhs,
                      (PARENT.word("det"), PARENT.word("workplace_modifier"), PARENT.word("person"))),)
        if lhs.name == "W" and lhs.feature("category") in ADDITIONAL_WORDS:
            category = lhs.feature("category")
            original = super().productions(lhs) if category == "person" else ()
            return original + tuple(BASE.Production("causal-word:"+category+":"+form, lhs,
                      (BASE.sym("T", form=form,label=category),)) for form in ADDITIONAL_WORDS[category])
        return super().productions(lhs)


CONTROL = ("reward a temple guard with a rare medal and carry the detailed portrait of the patient teacher "
           "beside the old drawing to the careful artist because the help met a drawer")


def causal_witness(grammar, text):
    tree = BASE.parse_tree(grammar,text)
    causes = []
    def walk(node):
        if node.symbol.name == "CAUSE":
            causes.append({"complete_finite_clause": len(node.children)==4,
                "subject_type": node.children[1].symbol.feature("type"),
                "verb_category": node.children[2].symbol.feature("category"),
                "object_type": node.children[3].symbol.feature("type")})
        for child in node.children:
            walk(child)
    if tree:
        walk(tree)
    return {"independent_parse": tree is not None,"causal_clauses":causes,
            "typed_valency_ok": tree is not None and all(row["complete_finite_clause"]
                and row["subject_type"]==row["object_type"]=="person"
                and row["verb_category"]=="past_person" for row in causes)}


def run(max_states=100000):
    grammar=Grammar()
    control=PARENT.audit(grammar,CONTROL,"intact_prose_grammar_control")
    control["diagnostic_only"]=True
    control["reader_status"]="grammar control only; not a palindrome candidate or human evidence"
    control["causal_witness"]=causal_witness(grammar,CONTROL)
    trace=PAIR.emitted_control_trace(grammar,CONTROL)
    assert trace["emitted_letters"]>21
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"]>100
    result=PAIR.solve(grammar,max_states)
    for collection in (result["exact_closures"],result["mechanically_admitted_closures"]):
        for row in collection:
            row["causal_witness"]=causal_witness(grammar,row["rendered"][:-1].lower())
    return {"method":"complete_causal_subject_and_workplace_np_interior_bridge",
        "construction_change":"replace the restrictive relative pronoun seam with a complete typed causal subject; permit compositional workplace person NPs",
        "config":{"max_states":max_states,"single_shared_tree":True,"words_only_at_exposed_leaves":True,
                  "character_equality_during_emission":True,"maximum_causal_clauses":1},
        "provenance":{"generator_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
                      "pair_operator_sha256":sha256(SOURCE.read_bytes()).hexdigest(),
                      "grammar_sha256":grammar.digest(),"additional_individual_words":ADDITIONAL_WORDS,
                      "catalogue_generation_material":False,"control_used_as_search_seed":False},
        "complete_grammar_control":control,"cross_21_emitted_control_trace":trace,**result,
        "next_reader_facing_test":"Any admitted new long closure must enter randomized blinded human reading with intact and shuffled controls, grammaticality judgments and coherent paraphrases.",
        "next_construction_if_no_candidate":"Use the new deepest emitted trace to replace the specific compound-head/determiner boundary with a complete grammatical nominal alternative."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--max-states",default=100000,type=int)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing artifact")
    result=run(args.max_states)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"out":str(args.out),"stats":result["stats"],
                      "exact":len(result["exact_closures"]),"admitted":len(result["mechanically_admitted_closures"])}))


if __name__=="__main__":
    main()
