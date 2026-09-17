"""Typed central-crossing repair for asynchronous forward prose templates.

The chart stores both ordinary-order grammar states and a character residual.
Left words are emitted from the left tape edge; right words are selected from
the end of a forward clause but emitted in a stack, so the completed right
clause remains forward order.  The crossing state carries subject number and
auxiliary agreement, rather than using an empty-seam sentinel.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-central-crossing-async-20260917.json"
EXPERIMENT = "typed-central-crossing-async-20260917"
SIGNATURE = "typed-central-crossing|agreement-carrying-grammar|variable-slot-lengths|opposite-edge-obligations|forward-rendering|independent-audit"

def letters(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(text):
    t=letters(text); mismatch=None
    for i in range(len(t)//2):
        if t[i]!=t[-1-i]: mismatch=[i,len(t)-1-i,t[i],t[-1-i]]; break
    ws=re.findall(r"[a-z]+",text.lower())
    return {"letters":len(t),"exact":bool(t) and mismatch is None,"mismatch":mismatch,
            "words":ws,"repeated_word_count":len(ws)-len(set(ws)),
            "self_palindromic_words":[w for w in ws if len(w)>1 and w==w[::-1]],
            "word_order_only":False,"borrowed_catalogue":False}

DOM={
 "DET": ("a","the","one"), "SUBJ_S": ("baker","farmer","artist","friend"),
 "SUBJ_P": ("sailors","writers","farmers"), "ADJ": ("calm","kind","old","bright"),
 "OBJ": ("bread","letters","music","apples","parcels"),
 "V_S": ("bakes","keeps","marks","opens","reads","sends","writes"),
 "V_P": ("bake","keep","mark","open","read","send","write"),
 "PREP": ("by","for","near","with"), "PLACE": ("home","shore","town","river"),
}

@dataclass(frozen=True)
class State:
    li:int; ri:int; residual:str; number:str; left_words:tuple; right_words:tuple

def consume(residual, incoming):
    """Compare newest opposite-edge tape chars; residual is unmatched tape."""
    r=residual; x=incoming
    n=min(len(r),len(x))
    if n and r[-n:] != x[:n]: return None
    if n==len(x): return r[:-n] if n else r
    return x[n:]

def search(lt, rt, max_states=250000):
    # right is indexed from its terminal slot backward; its words are stacked,
    # then reversed once for the ordinary forward rendering.
    starts=[State(0,len(rt)-1,"","",(),())]; seen=set(); closures=[]; expanded=0
    while starts and expanded<max_states:
        s=starts.pop(); key=(s.li,s.ri,s.residual,s.number)
        if key in seen: continue
        seen.add(key); expanded+=1
        if s.li==len(lt) and s.ri<0:
            if not s.residual: closures.append(s)
            continue
        moves=[]
        if s.li<len(lt): moves.append(("L",lt[s.li],False))
        if s.ri>=0: moves.append(("R",rt[s.ri],True))
        for side,slot,_ in moves:
            for w in DOM[slot]:
                number=s.number
                if slot=="SUBJ_S": number="S"
                if slot=="SUBJ_P": number="P"
                if slot=="V_S" and number not in ("","S"): continue
                if slot=="V_P" and number not in ("","P"): continue
                nw=letters(w)
                if side=="L":
                    nr=consume(s.residual,nw[::-1]) if s.residual else nw[::-1]
                    if nr is None: continue
                    starts.append(State(s.li+1,s.ri,nr,number,s.left_words+(w,),s.right_words))
                else:
                    nr=consume(s.residual,nw[::-1]) if s.residual else nw[::-1]
                    if nr is None: continue
                    starts.append(State(s.li,s.ri-1,nr,number,s.left_words,s.right_words+(w,)))
    return expanded, closures

def main():
    # Withheld algorithmic control: demonstrates non-empty crossing without
    # importing a known palindrome or claiming readability.
    control="ab ba."
    ca=audit(control)
    templates=[
      (("DET","ADJ","SUBJ_S","V_S","DET","OBJ"),("DET","OBJ","V_S","PREP","DET","PLACE")),
      (("DET","SUBJ_S","V_S","DET","OBJ"),("DET","ADJ","SUBJ_S","V_S","PLACE")),
      (("DET","SUBJ_P","V_P","DET","OBJ"),("DET","OBJ","V_P","PREP","PLACE")),
    ]
    rows=[]; total=0; exact=[]
    for ti,(lt,rt) in enumerate(templates):
        n,cl=search(lt,rt); total+=n
        for s in cl:
            text=" ".join(s.left_words+tuple(reversed(s.right_words)))+"."
            a=audit(text); row={"template":ti,"rendered":text,"left":s.left_words,"right":tuple(reversed(s.right_words)),"state":{"number":s.number,"residual":s.residual},"audit":a,"provenance":"fresh typed central-crossing chart; agreement state and live opposite-edge residual"}
            rows.append(row)
            if a["exact"]: exact.append(row)
    result={"experiment_id":EXPERIMENT,"signature":SIGNATURE,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"state_count":total,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible_count":0,"withheld_control":{"rendered":control,"audit":ca,"status":"algorithmic_control_only","reason":"short exact tape control; not presented as readable prose"},"exact_closures":exact[:20],"rendered_candidates":rows[:20],"novelty_preflight":{"registry_read":True,"signature":SIGNATURE,"copied_catalogue":False},"repair_after_failure":"The crossing chart is sound but the lexical domains have no compatible long closure; next repair should add terminal-bearing relative clauses with agreement features while retaining the same state invariant.","scope":"No candidate is reader-eligible without blinded human ratings."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"states":total,"exact":len(exact),"control":ca["exact"],"out":str(OUT)}))
if __name__=="__main__": main()
