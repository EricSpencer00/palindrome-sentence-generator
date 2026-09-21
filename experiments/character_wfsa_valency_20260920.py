"""Typed number+valency character WFSA successor."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/"runs/character-wfsa-valency-20260920.json"; REG=ROOT/"docs/experiment-novelty-registry.json"
ID="character-wfsa-valency-20260920"; SIG="online-character-wfsa|typed-number-valency|agreement-arcs|live-boundary-buffer"
DET={"sg":("the","a"),"pl":("our","the")}; SUBJ={"sg":("sailor","teacher","writer"),"pl":("sailors","teachers","writers")}; TV={"sg":("guides","marks","records"),"pl":("guide","mark","record")}; IV={"sg":("waits","smiles","rests"),"pl":("wait","smile","rest")}; OBJ=("harbor","letter","parcel","garden"); PLACE=("shore","station","garden","harbor")
def let(s): return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=let(s);i=0;j=len(t)-1
 while i<j and t[i]==t[j]:i+=1;j-=1
 return {"letters":len(t),"pointer_exact":i>=j,"first_mismatch":None if i>=j else [i,j,t[i],t[j]],"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def pre():
 d=json.loads(REG.read_text());xs=[x for x in d.get("entries",[])+d.get("excluded",[]) if x.get("id")!=ID];a=str(Path(__file__).relative_to(ROOT));return {"status":"passed" if not any(x.get("signature")==SIG or x.get("artifact")==a for x in xs) else "blocked","signature_overlaps":[x["id"] for x in xs if x.get("signature")==SIG],"artifact_collisions":[x["id"] for x in xs if x.get("artifact")==a],"excluded_routes":["finished-tape reversal","repair","lexical-bank sweep"]}
def consume(l,r,p):
 a=p+let(l);b=let(r)[::-1];n=min(len(a),len(b));return a[:n]==b[:n],a[n:],n
def run():
 p=pre()
 if p["status"]!="passed":raise RuntimeError(p)
 rows=[];controls=[];trans=rej=0
 # State carries number and valency independently on each side.  Lexical
 # object emission is legal only on transitive arcs.
 for ln in ("sg","pl"):
  for rn in ("sg","pl"):
   for lv in ("transitive","intransitive"):
    for rv in ("transitive","intransitive"):
     lw=(DET[ln][0],SUBJ[ln][0],(TV if lv=="transitive" else IV)[ln][0]);rw=(DET[rn][1],SUBJ[rn][1],(TV if rv=="transitive" else IV)[rn][1])
     if lv=="transitive":lw+= ("the "+OBJ[0],)
     if rv=="transitive":rw+= ("a "+PLACE[0],)
     trans+=1;ok,pending,_=consume(" ".join(lw)," ".join(rw),"")
     if ok and not pending:rows.append({"rendered":" ".join(lw)+"; while "+" ".join(rw)+".","audit":audit(" ".join(lw)+"; while "+" ".join(rw)+"."),"wfsa_state":{"left_number":ln,"right_number":rn,"left_valency":lv,"right_valency":rv},"provenance":{"typed_valency_before_object":True,"agreement_state":True,"live_boundary_buffer":True}})
     else:rej+=1
 for i,(n,v) in enumerate((("sg","intransitive"),("pl","transitive"),("sg","transitive"),("pl","intransitive"))):
  m="pl" if n=="sg" else "sg"; left=(DET[n][0],SUBJ[n][0],(TV if v=="transitive" else IV)[n][0]);right=(DET[m][1],SUBJ[m][1],(TV if v=="transitive" else IV)[m][1]);
  if v=="transitive":left+=(("the "+OBJ[i%len(OBJ)]),)
  right+=(("a "+PLACE[i%len(PLACE)]),) if v=="transitive" else ()
  text=" ".join(left)+"; while "+" ".join(right)+".";controls.append({"rendered":text,"audit":audit(text),"reader_eligible":False,"provenance":{"complete_typed_valency_surface":True,"shuffled_control_source":"WFSA arcs"}})
 exact=[x for x in rows if x["audit"]["letters"]>38 and x["audit"]["pointer_exact"] and x["audit"]["sha256_forward"]==x["audit"]["sha256_reverse"]]
 out={"experiment_id":ID,"method":"online character WFSA with typed number and transitive/intransitive valency","config":{"typed_features":["number","valency"],"object_emission":"transitive-only","post_search_reversal":False,"repair":False},"stats":{"typed_transitions_checked":trans,"rejected_transitions":rej,"rendered_candidates":len(rows),"prose_controls":len(controls),"exact_gt38":len(exact),"max_control_letters":max(x["audit"]["letters"] for x in controls)},"rendered_candidates":rows[:20],"prose_controls":controls,"exact_candidates":exact,"novelty_preflight":p,"provenance":{"independent_audit":["two-pointer","forward/reverse SHA-256"],"catalogue_text":False,"mirrored_units":False,"word_order_symmetry":False,"self_palindromic_units":False},"next_operator":"typed argument-role state distinguishing recipient from theme before transitive object emission","status":"fresh exact >38 requires human reading" if exact else "no exact >38 closure; typed valency controls retained"}
 RUN.write_text(json.dumps(out,indent=2)+"\n");return out
if __name__=="__main__":print(json.dumps(run(),indent=2))
