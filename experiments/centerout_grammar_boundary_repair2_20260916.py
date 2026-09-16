"""Second single center-out repair: author an adjunct for the outer letter.

The prior candidate begins with ``t`` and ends with ``child`` (``d``).  This
held-out adjunct preserves the porter/parcel event and changes only the right
adjunct so its final word ends in ``t``.  It is not a substitution sweep.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID="centerout-grammar-boundary-repair2-20260916"
SIGNATURE="center-out-grammar-state|outer-letter-directed-adjunct-authorship|heldout-semantic-boundary-repair|event-preserving-valency|independent-exact-hash-audit"
OUT=ROOT/"runs"/f"{EXPERIMENT_ID}.json"
PARENT=ROOT/"runs/centerout-grammar-boundary-repair-20260916.json"

def audit(text):
    s=normalize_letters(text); rev=s[::-1]; mism=[]; i=0; j=len(s)-1
    while i<j:
        if s[i]!=s[j]: mism.append({"left":i,"right":j,"a":s[i],"b":s[j]})
        i+=1; j-=1
    hf=hashlib.sha256(s.encode()).hexdigest(); hr=hashlib.sha256(rev.encode()).hexdigest()
    return {"algorithm":"independent_two_pointer_plus_sha256","letters":len(s),"two_pointer_exact":bool(s) and not mism,
            "mismatch_count":len(mism),"first_mismatch":mism[0] if mism else None,
            "sha256_forward":hf,"sha256_reverse":hr,"sha_equal":hf==hr}

def run():
    parent=json.loads(PARENT.read_text()); base=parent["candidate"]["rendered"]
    old="near the lamplit school for a waiting child"
    new="near the lamplit school for a waiting guest"
    assert old in base
    rendered=base.replace(old,new,1); a=audit(rendered); checks=mechanical_admission_checks(rendered,min_letters=39,max_letters=220)
    row={"label":"heldout-outer-letter-adjunct-repair","rendered":rendered,"letters":a["letters"],"exact_audit":a,"checks":checks,
         "mechanically_admitted":bool(a["two_pointer_exact"] and a["sha_equal"] and all(checks.values())),
         "repair":{"parent_run":str(PARENT.relative_to(ROOT)),"changed_span":{"from":old,"to":new},"target":"outer t/d mismatch","new_final_letter":"t"},
         "provenance":{"authored_event_preserved":True,"catalogue_text_copied":False,"finished_sentence_reversed":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False}}
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed","method":"single authored adjunct whose final letter targets the observed outer mismatch; all other slots fixed","candidate":row,
            "stats":{"candidates":1,"exact":int(a["two_pointer_exact"]),"mechanically_admitted":int(row["mechanically_admitted"])},"next_repair":"author one complete adjunct against the next two mirrored boundary obligations, preserving the same event and grammar state","reader_status":"not eligible: no exact mechanically admitted candidate","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"generated_not_catalogue":True}}

if __name__=="__main__":
    p=run(); OUT.write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p,indent=2))
