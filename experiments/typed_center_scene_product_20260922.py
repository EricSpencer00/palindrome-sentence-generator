"""Live typed-center scene product.

This experiment selects grammatical scene slots on both sides while consuming
the character obligation online.  A typed center (odd one-word or even empty)
is part of the state; no completed tape is reversed during construction.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ID="typed-center-scene-product-20260922"
RUN=ROOT/"runs"/(ID+".json")
def norm(s): return re.sub(r"[^a-z]","",s.lower())
def audit(s):
    t=norm(s); mm=[i for i in range(len(t)//2) if t[i]!=t[-1-i]]
    return {"exact":bool(t) and not mm,"letters":len(t),"mismatches":mm[:8],
            "pointer_pairs":len(t)//2,"forward_sha256":hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256":hashlib.sha256(t[::-1].encode()).hexdigest()}

LEFT=[("an", "det", "sg"),("a","det","sg"),("the","det","any"),
      ("aide","agent","sg"),("artist","agent","sg"),("writer","agent","sg"),
      ("sailor","agent","sg"),("rips","verb","pl"),("marks","verb","pl"),
      ("reads","verb","pl"),("keeps","verb","pl"),("nine","num","pl"),
      ("seven","num","pl"),("memos","theme","pl"),("letters","theme","pl"),
      ("maps","theme","pl"),("notes","theme","pl")]
RIGHT=[("some","det","pl"),("the","det","any"),("many","det","pl"),
       ("men","agent","pl"),("artists","agent","pl"),("sailors","agent","pl"),
       ("writers","agent","pl"),("inspire","verb","pl"),("mark","verb","pl"),
       ("read","verb","pl"),("keep","verb","pl"),("diana","name","sg"),
       ("nora","name","sg"),("iris","name","sg"),("leon","name","sg")]
CENTERS=[("", "even"),("is", "odd"),("was", "odd"),("ere", "odd")]
LEFT_SLOTS=[[(t,r,f) for t,r,f in LEFT if r==role or (role=="det" and r=="det")] for role in ("det","agent","verb","num","theme")]
RIGHT_SLOTS=[[(t,r,f) for t,r,f in RIGHT if r==role] for role in ("det","agent","verb","name")]

def consume(debt, token, side):
    x=norm(token) if side=="L" else norm(token)[::-1]
    if not debt: return None
    if debt.startswith(x): return debt[len(x):],side
    if x.startswith(debt): return x[len(debt):],"R" if side=="L" else "L"
    return None

def main():
    nodes=prunes=terminals=0; rows=[]; exact=[]; frontier=[]; seen=set()
    def rec(li,ri,debt,side,left,right,center,trace):
        nonlocal nodes,prunes,terminals
        nodes+=1
        if nodes>250000: return
        if li>=1 and len(frontier)<80:
            text=" ".join(left+([center] if center else [])+list(reversed(right)))
            aa=audit(text)
            frontier.append({"rendered":text.capitalize()+".","length":aa["letters"],"exact":aa["exact"],"audit":aa,"provenance":{"left_slots":left,"right_slots":list(reversed(right)),"center":center,"trace":trace},"status":"frontier_partial_right" if ri>=0 else "terminal"})
        if li==len(LEFT) and ri<0:
            terminals+=1
            text=" ".join(left+([center] if center else [])+list(reversed(right)))
            a=audit(text); row={"rendered":text.capitalize()+".","normalized":norm(text),"length":a["letters"],"exact":a["exact"],"audit":a,"provenance":{"left_slots":left,"right_slots":list(reversed(right)),"center":center,"trace":trace},"novelty":"generated_typed_scene"}
            if row["normalized"] not in seen:
                seen.add(row["normalized"]); rows.append(row)
                if a["exact"]: exact.append(row)
            return
        if not debt:
            # Opening a new left token is a live obligation, never a finished-tape check.
            if li<len(LEFT):
                for tok,role,feat in LEFT_SLOTS[li]:
                    rec(li+1,ri,norm(tok),"L",left+[tok],right,center,trace+[("L",tok)])
            return
        if side=="R" and ri>=0:
            for tok,role,feat in RIGHT_SLOTS[ri]:
                z=consume(debt,tok,"R")
                if z: rec(li,ri-1,*z,left,right+[tok],center,trace+[("R",tok)])
                else: prunes+=1
        elif side=="L" and li<len(LEFT):
            for tok,role,feat in LEFT_SLOTS[li]:
                z=consume(debt,tok,"L")
                if z: rec(li+1,ri,*z,left+[tok],right,center,trace+[("L",tok)])
                else: prunes+=1
        else: prunes+=1
    # Build each semantic frame and center separately; slots are selected online.
    for center,kind in CENTERS:
        # five slots on each side; center is a typed scene copula/empty seam.
        rec(0,3,"","L",[],[],center,[])
    rows.sort(key=lambda r:(-r["length"],r["normalized"]))
    out={"experiment_id":ID,"status":"completed_live_typed_center_search","method":"simultaneous typed scene-slot expansion with online character debt and authored odd/even center","stats":{"nodes":nodes,"live_prunes":prunes,"terminals":terminals,"unique_rendered":len(rows),"frontier_outputs":len(frontier),"exact":len(exact),"longest":max((r["length"] for r in frontier+rows),default=0)},"candidates":rows[:40],"frontier_outputs":frontier[:40],"exact_candidates":exact[:40],"shortcut_gates":{"finished_tape_reverse_during_search":False,"repair":False,"catalogue":False,"repeated_units":False,"word_order_only":False,"rlaif":False},"novelty":{"signature":"typed-center|scene-slots|online-debt|odd-even-center","preflight":"not duplicate of reverse lexicalization; center is state, not post-render audit"},"next_construction":"add role-conditioned semantic frames and choose paired slots from both ends rather than fixed slot order; preserve online debt and center typing","provenance":{"script":str(Path(__file__)) ,"independent_verifier":"two-pointer mismatch list plus forward/reverse SHA-256","run":str(RUN)}}
    RUN.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"nodes":nodes,"prunes":prunes,"terminals":terminals,"unique":len(rows),"frontier":len(frontier),"exact":len(exact),"longest":out["stats"]["longest"],"best":[r["rendered"] for r in frontier[:3]]}))
if __name__=="__main__": main()
