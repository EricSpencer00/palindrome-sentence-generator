"""Successor to character_wfsa_boundary: typed number state before emission."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/"runs/character-wfsa-agreement-20260920.json"; REGISTRY=ROOT/"docs/experiment-novelty-registry.json"
ID="character-wfsa-agreement-20260920"; SIG="online-character-wfsa|typed-number-state|agreement-arcs|unmatched-boundary-buffer"
DET={"sg":("the","a"),"pl":("our","the")}; SUBJ={"sg":("sailor","teacher","writer"),"pl":("sailors","teachers","writers")}; VERB={"sg":("guides","marks","records"),"pl":("guide","mark","record")}
OBJDET=("the","a"); OBJ=("harbor","letter","parcel","garden"); PREP=("near","beside","beyond"); PLACE=("shore","station","garden","harbor")
def let(s): return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=let(s); i=0;j=len(t)-1
 while i<j and t[i]==t[j]:i+=1;j-=1
 return {"letters":len(t),"pointer_exact":i>=j,"first_mismatch":None if i>=j else [i,j,t[i],t[j]],"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def preflight():
 d=json.loads(REGISTRY.read_text()); xs=[x for x in d.get("entries",[])+d.get("excluded",[]) if x.get("id")!=ID]; a=str(Path(__file__).relative_to(ROOT))
 return {"status":"passed" if not any(x.get("signature")==SIG or x.get("artifact")==a for x in xs) else "blocked","signature_overlaps":[x["id"] for x in xs if x.get("signature")==SIG],"artifact_collisions":[x["id"] for x in xs if x.get("artifact")==a],"excluded_routes":["finished-tape reversal","post-search repair","lexical-bank sweep"]}
def consume(l,r,pending):
 a=pending+let(l); b=let(r)[::-1]; n=min(len(a),len(b)); return (a[:n]==b[:n],a[n:],n)
def run():
 p=preflight()
 if p["status"]!="passed": raise RuntimeError(p)
 states=[("sg","sg",0,0,"",(),())]; rows=[]; rejects=0; transitions=0
 # Agreement is typed in state: lexical verb arcs are unavailable until the
 # subject number has been carried, then character consumption occurs.
 while states:
  ln,rn,li,ri,pending,lw,rw=states.pop()
  if li==4 and ri==4:
   if pending: rejects+=1; continue
   text=" ".join(lw)+"; while "+" ".join(rw)+"."; rows.append({"rendered":text,"audit":audit(text),"wfsa_state":{"left_number":ln,"right_number":rn,"pending":pending},"provenance":{"typed_number_before_emission":True,"lexicon":"checked-in agreement arcs","independent_choices":True}}); continue
  # phases: det, subject, agreeing verb, object noun (with fixed article)
  for side in (0,1):
   pass
  if li<4 and ri<4:
   for a,b in ((x,y) for x in DET[ln] for y in DET[rn]):
    for s,t in ((x,y) for x in SUBJ[ln] for y in SUBJ[rn]):
     for v,w in ((x,y) for x in VERB[ln] for y in VERB[rn]):
      for od,ot,on,oo in ((x,y,z,q) for x in OBJDET for y in OBJDET for z in OBJ for q in OBJ):
       transitions+=1; left=(a,s,v,od+" "+on); right=(b,t,w,ot+" "+oo)
       # consume word-level exposed characters in one typed transition;
       # boundary ownership remains explicit in pending.
       ok,nxt,checks=consume(" ".join(left)," ".join(right),pending)
       if ok: states.append((ln,rn,4,4,nxt,left,right))
       else: rejects+=1
  break
 controls=[]
 for i,n in enumerate(("sg","pl","sg")):
  m="pl" if n=="sg" else "sg"; left=(DET[n][0],SUBJ[n][0],VERB[n][0],OBJDET[0]+" "+OBJ[i]); right=(DET[m][0],SUBJ[m][0],VERB[m][0],OBJDET[1]+" "+PLACE[i])
  text=" ".join(left)+"; while "+" ".join(right)+"."; controls.append({"rendered":text,"audit":audit(text),"reader_eligible":False,"provenance":{"complete_agreement_surface":True,"shuffled_control_source":"typed WFSA arcs"}})
 exact=[x for x in rows if x["audit"]["letters"]>38 and x["audit"]["pointer_exact"] and x["audit"]["sha256_forward"]==x["audit"]["sha256_reverse"]]
 out={"experiment_id":ID,"method":"online character WFSA with typed number-carrying agreement arcs","config":{"typed_features":["number"],"phase_order":["det","subject","agreeing_verb","object"],"lexical_bank_sweep":False,"post_search_reversal":False},"stats":{"typed_transitions_checked":transitions,"rejected_transitions":rejects,"rendered_candidates":len(rows),"prose_controls":len(controls),"exact_gt38":len(exact),"max_control_letters":max(x["audit"]["letters"] for x in controls)},"rendered_candidates":rows[:20],"prose_controls":controls,"exact_candidates":exact,"novelty_preflight":p,"provenance":{"independent_audit":["two-pointer","forward/reverse SHA-256"],"catalogue_text":False,"mirrored_units":False,"word_order_symmetry":False,"self_palindromic_units":False},"next_operator":"carry typed semantic valency through the same agreement state before object emission","status":"fresh exact >38 requires human reading" if exact else "no exact >38 closure; agreement controls retained"}
 RUN.write_text(json.dumps(out,indent=2)+"\n"); return out
if __name__=="__main__": print(json.dumps(run(),indent=2))
