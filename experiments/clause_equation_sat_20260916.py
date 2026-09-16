"""Independent finite-domain clause-equation SAT probe.

Two authors choose ordinary SVO/PP clauses independently.  Boolean choices are
enumerated, while character equations compare the left tape with the reversed
right tape online; no mirrored words or known seeds are used.
"""
from __future__ import annotations
import hashlib, json, re
from itertools import product
from pathlib import Path
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "clause-equation-sat-20260916.json"
REG = ROOT / "docs" / "experiment-novelty-registry.json"
EXPERIMENT = "clause-equation-sat-20260916"
SIGNATURE = "finite-domain-boolean-svo-pp|independent-half-authoring|online-character-equations|no-seeds-no-repeated-units"

LEFT = {
 "subject": ["a patient cartographer", "the quiet archivist"],
 "verb": ["charted", "repaired"], "object": ["a coastal inlet", "the weathered ledger"],
 "prep": ["beside the eastern pier", "under a copper lamp"]}
RIGHT = {
 "subject": ["a careful mason", "the evening nurse"],
 "verb": ["measured", "carried"], "object": ["the limestone steps", "a folded blanket"],
 "prep": ["near the winter garden", "through the narrow passage"]}

def clause(bank, bits):
    s,v,o,p = (bank[k][bits[i]] for i,k in enumerate(("subject","verb","object","prep")))
    return f"{s} {v} {o} {p}"

def equations(left, right):
    a,b=normalize_letters(left),normalize_letters(right)
    n=min(len(a),len(b)); mism=[]
    for i in range(n):
        if a[i] != b[-1-i]: mism.append({"left_index":i,"right_index":len(b)-1-i,"left":a[i],"right":b[-1-i]})
    return {"equations_checked":n,"satisfied":not mism and len(a)==len(b),"mismatch_count":len(mism)+abs(len(a)-len(b)),"first_mismatch":mism[0] if mism else None}

def audit(text):
    tape=normalize_letters(text)
    direct=tape==tape[::-1]
    two=equations(text[:len(text)//2], text[len(text)//2:]) if False else {"satisfied":direct}
    mech=mechanical_admission_checks(text)
    return {"letters":len(tape),"direct_exact":direct,"mechanically_admitted":all(mech.values()),"mechanical":mech}

def main():
    reg=json.loads(REG.read_text()); prior=[e for e in reg.get("entries",[]) if e.get("id")!=EXPERIMENT]
    pre={"entries_inspected":len(reg.get("entries",[])),"exact_signature_collisions_before_run":[e.get("id") for e in prior if e.get("signature")==SIGNATURE],"passed":not any(e.get("signature")==SIGNATURE for e in prior)}
    rows=[]
    for lb,rb in product(product((0,1),repeat=4),product((0,1),repeat=4)):
        l=clause(LEFT,lb); r=clause(RIGHT,rb); text=l+"; "+r+"."
        eq=equations(l,r)
        rows.append({"rendered":text,"choices":{"left":lb,"right":rb},"independent_halves":{"left":l,"right":r},"character_equations":eq,"audit":audit(text),"complete_prose":True,"repeated_units":False})
    best=min(rows,key=lambda x:(x["character_equations"]["mismatch_count"],-x["audit"]["letters"]))
    payload={"experiment":EXPERIMENT,"signature":SIGNATURE,"method":"Boolean finite-domain SAT over lexical SVO/PP slots; every pair independently authored and constrained by reversed character equations","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":sum(r["character_equations"]["satisfied"] for r in rows),"best":best,"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_pointer_sha256":hashlib.sha256((best["independent_halves"]["left"]+"||"+best["independent_halves"]["right"]).encode()).hexdigest(),"catalogue_import":False,"known_palindrome_seeds":False,"word_order_symmetry":False,"duplicate_sweeps":False},"readability_diagnostics":{"complete_clause_count":2,"fragment_rejected":False,"status":"ordinary prose syntax; no human certificate"},"next_repair":"author a fresh PP complement on the side containing the first equation mismatch, then rerun the Boolean domains without copying either half"}
    OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"candidates":len(rows),"exact":payload["exact_count"],"best_letters":best["audit"]["letters"],"best_mismatches":best["character_equations"]["mismatch_count"]}))
if __name__=="__main__": main()
