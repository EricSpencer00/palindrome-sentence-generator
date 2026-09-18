"""Ordinary trade-role NPs paired with complete artifact-testing events.

Repair the failed person-role construction by replacing its roles entirely:
the right clause has a human expert testing a physical drawer, while the left
NP can describe a vendor by commodity and venue. Bare 'help' and 'drawer' are
removed from the person inventory. Three independently exposed word slots
compose the trade role; there is no stored phrase or frozen partial sentence.
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
SPEC=importlib.util.spec_from_file_location("trade_artifact_capture",SOURCE)
CAPTURE=importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name]=CAPTURE
SPEC.loader.exec_module(CAPTURE)
CAUSAL,BASE,PAIR,PARENT=CAPTURE.CAUSAL,CAPTURE.BASE,CAPTURE.PAIR,CAPTURE.PARENT
LEXICAL_ROLES={
    "commodity":("dessert","bread","fruit","flower"),
    "sales_venue":("street","market"),
    "trade_head":("vendor","seller"),
    "artifact_test_past":("stressed","tested","inspected","checked"),
    "temporal":("after","while"),
    "expertise_domain":("safety","design","materials"),
    "specialist_head":("expert",),
    "reward_for":("reward","thank","honor"),
    "repair_gerund":("repairing","restoring","rebuilding"),
    "for":("for",),
}
FORBIDDEN_PERSON_FORMS={"help","drawer"}


class Grammar(CAUSAL.Grammar):
    def productions(self,lhs):
        if lhs.name=="IMP":
            purpose=BASE.Production("imperative:reward-for-artifact-repair",lhs,
                (PARENT.word("reward_for"),PARENT.np("person",False),BASE.sym("PURPOSE")))
            return (purpose,)+super().productions(lhs)
        if lhs.name=="PURPOSE":
            return (BASE.Production("purpose:single-repair",lhs,
                (PARENT.word("for"),BASE.sym("REPAIR"))),
                BASE.Production("purpose:coordinated-repairs",lhs,
                (PARENT.word("for"),BASE.sym("REPAIR"),PARENT.word("and"),BASE.sym("REPAIR"))))
        if lhs.name=="REPAIR":
            return (BASE.Production("repair:gerund-artifact-object",lhs,
                (PARENT.word("repair_gerund"),PARENT.np("object",False))),)
        if lhs.name=="NP" and lhs.feature("type")=="person":
            original=super().productions(lhs)
            trade=BASE.Production("np:commodity-venue-trader",lhs,
                (PARENT.word("det"),BASE.sym("TRADE",agent_type="person",goods_type="commodity")))
            specialist=BASE.Production("np:domain-specialist",lhs,
                (PARENT.word("det"),PARENT.word("expertise_domain"),PARENT.word("specialist_head")))
            return original+(trade,specialist)
        if lhs.name=="TRADE":
            return (BASE.Production("trade:commodity-venue-head",lhs,
                (PARENT.word("commodity"),PARENT.word("sales_venue"),PARENT.word("trade_head"))),)
        if lhs.name=="CAUSE":
            test=BASE.Production("temporal:human-tests-artifact",lhs,
                (PARENT.word("temporal"),PARENT.np("person",False),PARENT.word("artifact_test_past"),
                 PARENT.np("object",False)))
            return (test,)+super().productions(lhs)
        if lhs.name=="W":
            category=lhs.feature("category")
            if category in LEXICAL_ROLES:
                return tuple(BASE.Production("trade-role:"+category+":"+form,lhs,
                    (BASE.sym("T",form=form,label=category),)) for form in LEXICAL_ROLES[category])
            original=super().productions(lhs)
            if category=="person":
                original=tuple(prod for prod in original if prod.rhs[0].feature("form") not in FORBIDDEN_PERSON_FORMS)
                original+=(BASE.Production("human:expert",lhs,(BASE.sym("T",form="expert",label="person"),)),)
            if category=="object":
                original+=tuple(BASE.Production("artifact:"+form,lhs,(BASE.sym("T",form=form,label="object"),))
                                for form in ("cart","cabinet"))
            if category=="adj_object":
                original+=(BASE.Production("artifact-property:damaged",lhs,(BASE.sym("T",form="damaged",label="adj_object"),)),)
            return original
        return super().productions(lhs)


CONTROL=("reward a dessert street vendor for repairing the damaged cabinet and restoring the old cart "
         "after the safety expert stressed a drawer")


def semantic_witness(grammar,text):
    tree=BASE.parse_tree(grammar,text)
    roles,events,repairs=[],[],[]
    def leaves(node):
        if node.terminal:return [node.terminal]
        return [word for child in node.children for word in leaves(child)]
    def walk(node):
        if node.symbol.name=="TRADE":
            roles.append({"words":leaves(node),"agent_type":node.symbol.feature("agent_type"),
                "slot_categories":[child.symbol.feature("category") for child in node.children]})
        if node.production=="temporal:human-tests-artifact":
            events.append({"agent_type":node.children[1].symbol.feature("type"),
                "agent":leaves(node.children[1]),"predicate":leaves(node.children[2]),
                "patient_type":node.children[3].symbol.feature("type"),"patient":leaves(node.children[3])})
        if node.symbol.name=="REPAIR":
            repairs.append({"predicate":leaves(node.children[0]),"patient_type":node.children[1].symbol.feature("type"),
                            "patient":leaves(node.children[1])})
        for child in node.children:walk(child)
    if tree:walk(tree)
    return {"independent_parse":tree is not None,"trade_roles":roles,"artifact_events":events,"rewarded_repairs":repairs,
        "correct_artifact_event_types":bool(events) and all(e["agent_type"]=="person" and e["patient_type"]=="object" for e in events),
        "bare_help_and_drawer_excluded_from_person_lexicon":not any(p.rhs[0].feature("form") in FORBIDDEN_PERSON_FORMS
            for p in grammar.productions(PARENT.word("person")))}


def run(max_states=100000):
    grammar=Grammar()
    control=PARENT.audit(grammar,CONTROL,"intact_prose_grammar_control")
    control["diagnostic_only"]=True
    control["reader_status"]="grammar control only; not a palindrome candidate or human evidence"
    control["semantic_witness"]=semantic_witness(grammar,CONTROL)
    trace=PAIR.emitted_control_trace(grammar,CONTROL)
    assert trace["emitted_letters"]>35
    assert control["independent_parse"] and control["independent_exact_audit"]["letters"]>100
    assert control["semantic_witness"]["correct_artifact_event_types"]
    result=CAPTURE.solve(grammar,max_states)
    for collection in (result["exact_closures"],result["mechanically_admitted_closures"]):
        for row in collection:row["semantic_witness"]=semantic_witness(grammar,row["rendered"][:-1].lower())
    return {"method":"compositional_trade_role_and_typed_artifact_event_bridge",
        "construction_change":"replace the prior helper/simple role pair with an independently composed trade-role NP, an explicit reward-for-repair complement, and a human-agent/artifact-patient temporal clause",
        "config":{"max_states":max_states,"single_shared_tree":True,"words_only_at_exposed_leaves":True,
            "character_equality_during_emission":True,"forbidden_person_forms":sorted(FORBIDDEN_PERSON_FORMS)},
        "provenance":{"generator_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
            "capture_sha256":sha256(SOURCE.read_bytes()).hexdigest(),"grammar_sha256":grammar.digest(),
            "individual_lexical_roles":LEXICAL_ROLES,"control_used_as_search_seed":False,"catalogue_generation_material":False},
        "complete_grammar_control":control,"cross_35_emitted_control_trace":trace,**result,
        "next_reader_facing_test":"Any admitted new long closure requires randomized blinded human reading with intact and shuffled controls, grammaticality judgments and coherent paraphrases.",
        "next_construction_if_no_candidate":"Use the persisted actual-search ledger to repair its next typed boundary while retaining ordinary person and artifact senses."}


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
