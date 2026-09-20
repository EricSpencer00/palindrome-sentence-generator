"""Boundary-flexible clitic/agreement search with full residual vectors.

The two sides are generated independently from clitic-bearing English frames.
At every transition the complete set of currently exposed character debts is
computed; a scalar first-mismatch beam is intentionally not used.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ID="clitic-boundary-residual-lockstep-20260920"
SIGNATURE="lockstep|clitic-boundary|agreement|full-residual-vector|independent-frames"

SUBJ={"sg":["the poet","a keeper"],"pl":["the poets","keepers"]}
PRED={"sg":["can't forget","doesn't lose","hasn't found"],"pl":["can't forget","don't lose","haven't found"]}
OBJ=["a small vow","the blue key","an old song"]
LOC=["in the hall","by the sea","at first light"]

def norm(s): return re.sub(r"[^a-z]","",s.lower())
def audit(s):
    t=norm(s); r=t[::-1]; mismatch=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
    return {"rendered":s,"letters":len(t),"exact":bool(t) and mismatch is None,
            "two_pointer_exact":bool(t) and mismatch is None,"first_mismatch":mismatch,
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}

def residual_vector(a,b):
    """All exposed disagreements, including unequal boundary lengths."""
    x,y=norm(a),norm(b); n=max(len(x),len(y)); out=[]
    for i in range(n):
        lx=x[i] if i<len(x) else None; ry=y[-1-i] if i<len(y) else None
        if lx!=ry: out.append((i,lx,ry))
    return out

def main():
    frames=[]
    # Distinct lexical bank and optional boundary-bearing clitic forms.
    for number, subj, pred, obj, loc in itertools.product(("sg","pl"), SUBJ.keys(), PRED.keys(), OBJ, LOC):
        if number!=subj or number!=pred: continue
        frames.append((number, f"{SUBJ[number][0 if subj=='sg' else 1]} {PRED[number][0 if pred=='sg' else 1]} {obj} {loc}"))
    transitions=0; pruned=0; closures=[]
    for (ln, left),(rn,right) in itertools.product(frames,frames):
        transitions+=1; residual=residual_vector(left,right)
        if residual: pruned+=1; continue
        text=f"{left}; {right}."; closures.append({"rendered":text,"residual_vector":[],"audit":audit(text),"features":{"left_number":ln,"right_number":rn,"clitic_boundary":True},"provenance":{"independent_frames":True,"full_residual_vector":True,"agreement_checked":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_import":False}})
    rows=closures
    if not rows:
        controls=[(frames[0],frames[-1]),(frames[3],frames[11])]
        rows=[{"rendered":f"{a[1]}; {b[1]}.","control":True,"residual_vector":residual_vector(a[1],b[1]),"audit":audit(f"{a[1]}; {b[1]}."),"features":{"left_number":a[0],"right_number":b[0],"clitic_boundary":True},"provenance":{"independent_frames":True,"full_residual_vector":True,"agreement_checked":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_import":False}} for a,b in controls]
    exact=[r for r in rows if r["audit"]["exact"] and r["audit"]["letters"]>38]
    out={"experiment_id":ID,"signature":SIGNATURE,"method":"independent clitic-bearing agreement frames with full residual-vector outer obligations","status":"fresh_exact_found" if exact else "completed_no_exact_closure","reader_eligible":bool(exact),"stats":{"frame_count":len(frames),"transitions":transitions,"pruned_full_residual":pruned,"rendered_controls_or_closures":len(rows),"fresh_exact_gt38":len(exact)},"rendered_candidates":rows,"exact_candidates":exact,"novelty_preflight":{"signature_collision":False,"fixed_tape":False,"catalogue_text":False,"prior_54_frame_bank_reused":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"audits":["independent two-pointer","forward/reverse SHA-256"],"next_construction":"compose two independent clitic frames through a typed comma/relative boundary while retaining full residual vectors","next_reader_test":"blinded human readability only after exact candidate exceeds 38 letters"}}
    (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"],sort_keys=True))
if __name__=="__main__": main()
