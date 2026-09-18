"""Repair the recorded helper(p)/gentle(t) boundary with a typed property phrase.

The independent witness in causal-subject-witness-capture/result-01.json is
the source of the failure diagnosis. This grammar adds coordinated person
properties, with independent adjective leaves and a real conjunction. The
ordinary property word 'simple' supplies elp at the exposed reverse edge,
where 'gentle' supplied elt. No rejected text is frozen into generation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"experiments/causal_subject_witness_capture_20260913.py"
SPEC=importlib.util.spec_from_file_location("property_phrase_capture",SOURCE)
CAPTURE=importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name]=CAPTURE
SPEC.loader.exec_module(CAPTURE)
CAUSAL,BASE,PAIR,PARENT=CAPTURE.CAUSAL,CAPTURE.BASE,CAPTURE.PAIR,CAPTURE.PARENT
PRIOR=ROOT/"runs/causal-subject-witness-capture-2026-09-13/result-01.json"
PROPERTY_WORDS=("simple","supple","strong","modest","polite","cheerful")


class Grammar(CAUSAL.Grammar):
    def productions(self,lhs):
        if lhs.name=="NP" and lhs.feature("type")=="person":
            phrase=BASE.Production("np:coordinated-person-properties",lhs,
                (PARENT.word("det"),BASE.sym("PROPERTIES",type="person"),PARENT.word("person")))
            return super().productions(lhs)+(phrase,)
        if lhs.name=="PROPERTIES":
            return (BASE.Production("properties:person-coordination",lhs,
                (PARENT.word("adj_person"),PARENT.word("property_conjunction"),PARENT.word("adj_person"))),)
        if lhs.name=="W" and lhs.feature("category")=="adj_person":
            return super().productions(lhs)+tuple(BASE.Production("property:"+form,lhs,
                (BASE.sym("T",form=form,label="adj_person"),)) for form in PROPERTY_WORDS)
        if lhs.name=="W" and lhs.feature("category")=="property_conjunction":
            return tuple(BASE.Production("property-conjunction:"+form,lhs,
                (BASE.sym("T",form=form,label="conjunction"),)) for form in ("and","but"))
        return super().productions(lhs)


CONTROL=("reward a temple helper with a rare medal and carry the detailed portrait of the patient teacher "
         "beside the old drawing to the careful artist because the gentle but simple help met a drawer")


def property_witness(grammar,text):
    tree=BASE.parse_tree(grammar,text)
    phrases=[]
    def walk(node):
        if node.symbol.name=="PROPERTIES":
            categories=[child.symbol.feature("category") for child in node.children]
            phrases.append({"type":node.symbol.feature("type"),"categories":categories,
                "complete_typed_phrase":categories==["adj_person","property_conjunction","adj_person"]})
        for child in node.children:walk(child)
    if tree:walk(tree)
    return {"independent_parse":tree is not None,"property_phrases":phrases}


def run(max_states=100000):
    prior=json.loads(PRIOR.read_text())
    old=prior["deepest_actual_search_witness"]
    restored=CAPTURE.replay(CAUSAL.Grammar(),old["ledger"])
    assert CAPTURE.digest_state(restored)==old["state_sha256"]
    assert old["next_character_conflict"]["expected_character"]=="p"
    assert old["next_character_conflict"]["opposing_character"]=="t"
    grammar=Grammar()
    control=PARENT.audit(grammar,CONTROL,"intact_prose_grammar_control")
    control["diagnostic_only"]=True
    control["reader_status"]="grammar control only; not a palindrome candidate or human evidence"
    control["property_witness"]=property_witness(grammar,CONTROL)
    trace=PAIR.emitted_control_trace(grammar,CONTROL)
    assert trace["emitted_letters"]>33
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"]>100
    result=CAPTURE.solve(grammar,max_states)
    return {"method":"coordinated_typed_person_property_phrase_repair",
        "construction_change":"add a complete coordinated adjective phrase at a person NP; choose its two properties independently at exposed leaves",
        "prior_recorded_failure":{"artifact":str(PRIOR.relative_to(ROOT)),"artifact_sha256":sha256(PRIOR.read_bytes()).hexdigest(),
            "state_sha256":old["state_sha256"],"replay_verified_now":True,"conflict":old["next_character_conflict"]},
        "config":{"max_states":max_states,"single_shared_tree":True,"words_only_at_exposed_leaves":True,
                  "character_equality_during_emission":True},
        "provenance":{"generator_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
            "capture_sha256":sha256(SOURCE.read_bytes()).hexdigest(),"grammar_sha256":grammar.digest(),
            "additional_properties":PROPERTY_WORDS,"control_used_as_search_seed":False,"catalogue_generation_material":False},
        "complete_grammar_control":control,"cross_33_emitted_control_trace":trace,**result,
        "next_reader_facing_test":"Any new long admitted closure requires randomized blinded human reading with intact and shuffled controls, plus coherent paraphrases.",
        "next_construction_if_no_candidate":"Use the persisted replay of the new deepest state to repair its exact next lexical or structural boundary."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--max-states",default=100000,type=int)
    args=parser.parse_args()
    if args.out.exists():parser.error("refusing to overwrite an existing artifact")
    result=run(args.max_states)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"out":str(args.out),"stats":result["stats"],"exact":len(result["exact_closures"]),
        "admitted":len(result["mechanically_admitted_closures"]),
        "conflict":result["deepest_actual_search_witness"]["next_character_conflict"]}))


if __name__=="__main__":main()
