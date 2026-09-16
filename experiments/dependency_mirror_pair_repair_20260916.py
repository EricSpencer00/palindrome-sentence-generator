"""Held-out dependency repair for the joint mirror-pair constructor.

This is a bounded second pass, not a new inventory sweep: each base pair gets
one held-out subject/adjunct substitution chosen by residual reverse-tape debt.
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT=Path(__file__).parents[1]
ID="dependency-mirror-pair-repair-20260916"
SIG="dependency-mirror-pair-repair|heldout-subject-adjunct-substitution|global-character-debt-selection|typed-argument-order|independent-two-pointer-hash-audit"
BASE=ROOT/"runs/dependency-mirror-pair-constructor-20260916.json"
OUT=ROOT/f"runs/{ID}.json"
REG=ROOT/"docs/experiment-novelty-registry.json"

def t(s): return normalize_letters(s)
def audit(s):
    x=t(s); m=next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=x[-1-i]),None)
    h=lambda z:hashlib.sha256(z.encode()).hexdigest()
    return {"letters":len(x),"two_pointer_exact":bool(x) and m is None,"first_mismatch":m,"sha256_forward":h(x),"sha256_reverse":h(x[::-1]),"sha256_exact":bool(x) and h(x)==h(x[::-1])}
def split(sentence):
    words=sentence.rstrip(".").split()
    # authored constructor schema is subject(3), verb(1), object(3), adjunct(rest)
    return words[:3],words[3],words[4:7],words[7:]
def render(s,v,o,a):
    subject = [s] if isinstance(s, str) else s
    adjunct = [a] if isinstance(a, str) else a
    return " ".join((*subject,v,*o,*adjunct))+"."
def debt(left,right):
    a,b=t(left),t(right)
    return Counter(a)-Counter(b[::-1]),Counter(b[::-1])-Counter(a)
def residual_score(left,right):
    dl,dr=debt(left,right)
    n=min(len(t(left)),len(t(right))); m=next(((i,t(left)[i],t(right)[-1-i]) for i in range(n) if t(left)[i]!=t(right)[-1-i]),None)
    return (len(dl)+len(dr),abs(len(t(left))-len(t(right))),m[0] if m else 999)
def preflight():
    rows=json.loads(REG.read_text()).get("entries",[])
    collision=any(r.get("id")!=ID and r.get("signature")==SIG for r in rows)
    return {"entries_inspected":len(rows),"exact_signature_collision":collision,"passed":not collision,"reason":"bounded debt-directed held-out repair is distinct from base joint construction" if not collision else "signature collision"}
def run():
    p=preflight(); base=json.loads(BASE.read_text()); heldout_subjects=["the gentle messenger","the silent gardener","the careful scholar"]; heldout_adjs=["for the waiting child before dusk","near the stone bridge after rain","for the patient nurse before noon"]
    rows=[]
    for b in base["candidates"]:
        ls,lv,lo,la=split(b["rendered"]); rs,rv,ro,ra=split(b["paired_rendered"]); left=b["rendered"]; right=b["paired_rendered"]
        trials=[]
        for side in ("left","right"):
            for subj in heldout_subjects:
                for adj in heldout_adjs:
                    if side=="left": nl,nr=render(subj,lv,lo,adj),right
                    else: nl,nr=left,render(subj,rv,ro,adj)
                    trials.append((residual_score(nl,nr),nl,nr,side,subj,adj))
        score,nl,nr,side,subj,adj=min(trials,key=lambda z:z[0]); al,ar=audit(nl),audit(nr)
        rows.append({"base_rendered":left,"base_paired_rendered":right,"rendered":nl,"paired_rendered":nr,"letters":al["letters"],"paired_letters":ar["letters"],"selection":{"side":side,"subject":subj,"adjunct":adj,"residual_score":score,"trials":len(trials)},"audit":{"left":al,"right":ar},"admission":{"left":mechanical_admission_checks(nl,min_letters=39,max_letters=300),"right":mechanical_admission_checks(nr,min_letters=39,max_letters=300),"complete_prose":True},"provenance":{"source":"held-out substitutions from joint dependency constructor","argument_order":["subject","verb","object","adjunct"],"catalogue_text_used":False,"word_order_mirrored":False,"repeated_unit_shortcut":False},"repair":{"operator":"global-reverse-tape-character-debt-directed subject/adjunct substitution","next":"author a held-out verb-object agreement-preserving alternative targeted at the remaining debt, then re-run the blinded reader gate"}})
    result={"experiment_id":ID,"signature":SIG,"novelty_preflight":p,"operator":"one bounded held-out substitution per base pair, ranked by global reverse-tape residual","base_count":len(base["candidates"]),"repair_count":len(rows),"exact_count":sum(r["audit"]["left"]["two_pointer_exact"] and r["audit"]["right"]["two_pointer_exact"] for r in rows),"candidates":rows,"reader_eligible":False,"status":"completed_no_exact_closure","next_repair":"held-out agreement-preserving verb-object substitution against each remaining residual"}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); return result
if __name__=="__main__":
    r=run(); print(json.dumps({"experiment_id":ID,"novelty":r["novelty_preflight"],"repairs":r["repair_count"],"exact":r["exact_count"],"best":max(r["candidates"],key=lambda x:x["letters"])["rendered"]},indent=2))
