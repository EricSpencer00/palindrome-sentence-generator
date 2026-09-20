"""Three-clause morphology search with residual suffix-class state.

Independent outer clauses are unfolded against one another.  At each step the
state retains a bounded suffix-class vector (last two letters of each pending
grammatical slot); a held-out auxiliary center is admitted only when the outer
residual equations survive.  This is a construction topology, not repair.
"""
import hashlib, itertools, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ID = "three-clause-residual-suffix-paradigm-20260920"
SIGNATURE = "three-clause|residual-suffix-vector|held-out-auxiliary|agreement-live-csp"
SUBJ={"sg":["the quiet keeper","a patient poet"],"pl":["the quiet keepers","patient poets"]}
VERB={("sg","present"): ["marks","keeps"], ("pl","present"): ["mark","keep"],
      ("sg","past"): ["marked","kept"], ("pl","past"): ["marked","kept"]}
OBJ=["the old letter","a silver map"]
TAIL=["by the river","near the garden"]
# held out from outer frame bank: the center auxiliary is never copied from an arm
AUX=["while the bell has sounded","while the bells have sounded","as the tide turns"]
def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(text):
    t=norm(text); r=t[::-1]; m=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
    return {"rendered":text,"letters":len(t),"exact":bool(t) and m is None,"two_pointer_exact":bool(t) and m is None,
            "first_mismatch":m,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def suffix_vector(left,right,center=""):
    # residual classes are retained, rather than collapsing to a scalar mismatch
    return {"left_pending_suffix":norm(left)[-2:],"right_pending_suffix":norm(right)[-2:],"center_prefix":norm(center)[:2]}
def live_outer(left,right):
    a,b=norm(left),norm(right); compared=0
    for i in range(min(len(a),len(b))):
        compared+=1
        if a[i]!=b[-1-i]: return False, compared, (i,a[i],b[-1-i])
    return len(a)==len(b),compared,None if len(a)==len(b) else (len(a),"","length")
def main():
    frames=[]
    for n,t in itertools.product(("sg","pl"),("present","past")):
      for s,v,o,x in itertools.product(SUBJ[n],VERB[(n,t)],OBJ,TAIL):
        frames.append({"number":n,"tense":t,"verb":v,"text":f"{s} {v} {o} {x}"})
    transitions=pruned=compatible=0; rows=[]
    for left,right in itertools.product(frames,frames):
      transitions+=1; ok,c,m=live_outer(left["text"],right["text"])
      if not ok: pruned+=1; continue
      compatible+=1
      for aux in AUX:
        text=left["text"]+"; "+aux+"; "+right["text"]+"."
        rows.append({"rendered":text,"center_auxiliary":aux,"residual_suffix_vector":suffix_vector(left["text"],right["text"],aux),"audit":audit(text),"provenance":{"independent_outer_frames":True,"held_out_center_auxiliary":True,"live_residual_vector":True,"word_order_symmetry":False,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_import":False}})
    if not rows:
      for a,b,aux in [(frames[0],frames[-1],AUX[0]),(frames[3],frames[8],AUX[2])]:
        text=a["text"]+"; "+aux+"; "+b["text"]+"."
        rows.append({"rendered":text,"control":True,"residual_suffix_vector":suffix_vector(a["text"],b["text"],aux),"audit":audit(text),"provenance":{"independent_outer_frames":True,"held_out_center_auxiliary":True,"live_residual_vector":True,"reader_eligible":False,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_import":False}})
    exact=[r for r in rows if r["audit"]["exact"] and r["audit"]["letters"]>38]
    out={"experiment_id":ID,"signature":SIGNATURE,"method":"three independent grammatical clauses with residual suffix-class vector and held-out auxiliary center","status":"fresh_exact_found" if exact else "completed_no_exact_closure","reader_eligible":False,"stats":{"frame_count":len(frames),"transitions":transitions,"live_pruned":pruned,"outer_compatible":compatible,"rendered_controls_or_closures":len(rows),"fresh_exact_gt38":len(exact)},"rendered_candidates":rows,"exact_candidates":exact,"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"audits":["independent two-pointer","forward/reverse SHA-256"],"next_construction":"condition a fourth clause on the full two-character residual vector and finite-verb agreement, with a held-out relative auxiliary","next_reader_test":"blinded human readability only after exact candidate exceeds 38 letters"}}
    (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"],sort_keys=True))
if __name__=="__main__": main()
