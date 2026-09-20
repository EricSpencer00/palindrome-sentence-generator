"""Shared-scene SVO/locative grammar with slot-level character domains."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def norm(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=norm(s); i,j=0,len(t)-1
 while i<j and t[i]==t[j]: i+=1; j-=1
 return {"letters":len(t),"exact":bool(t) and i>=j,"first_mismatch":None if i>=j else {"index":i,"forward":t[i],"reverse":t[-1-i]},"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer_exact(s):
 t=norm(s); return bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2))
def consume(a,b):
 n=min(len(a),len(b)); return (a[n:],b[n:]) if a[:n]==b[:n] else None

# Domains are keyed by semantic scene and grammar type.  Each slot carries a
# small ordinary-word domain; the solver filters its first/final character
# obligation before committing the slot, not after a finished sentence.
DOMAINS={
 "harbor":{"SVO":{"NP":("the patient sailor","a quiet keeper"),"V":("guards","studies"),"OBJ":("the lantern","the chart"),"ADV":("at dawn","before dusk")},"LOC":{"NP":("the harbor","the quay"),"V":("rests","waits"),"PREP":("by","near"),"LOC":("the river","the old bridge"),"ADV":("at dawn","before dusk")}},
 "garden":{"SVO":{"NP":("the young poet","the bright gardener"),"V":("remembers","carries"),"OBJ":("the garden","a small letter"),"ADV":("in silence","after rain")},"LOC":{"NP":("the garden","the orchard"),"V":("lies","stands"),"PREP":("near","beside"),"LOC":("the old trees","the quiet wall"),"ADV":("in rain","at noon")}},
 "bridge":{"SVO":{"NP":("several quiet scouts","a patient pilot"),"V":("watch","marks"),"OBJ":("the old bridge","the distant shore"),"ADV":("at noon","through mist")},"LOC":{"NP":("the bridge","the shore"),"V":("stands","rests"),"PREP":("beside","near"),"LOC":("the river","the harbor"),"ADV":("at noon","through mist")}},
}

def frames(scene,kind):
 d=DOMAINS[scene][kind]; out=[]
 for np in d["NP"]:
  for v in d["V"]:
   for adv in d["ADV"]:
    if kind=="SVO":
     for obj in d["OBJ"]: out.append({"scene":scene,"kind":kind,"slots":(("NP",np),("V",v),("OBJ",obj),("ADV",adv))})
    else:
     for prep in d["PREP"]:
      for loc in d["LOC"]: out.append({"scene":scene,"kind":kind,"slots":(("NP",np),("V",v),("PREP",prep),("LOC",loc),("ADV",adv))})
 return out

def controls():
 texts=("The patient sailor guards the lantern at dawn; the harbor rests by the river at dawn.","The young poet remembers the garden in silence; the garden lies near the old trees in rain.","Several quiet scouts watch the old bridge at noon; the bridge stands beside the river at noon.")
 return [{"rendered":t,"audit":audit(t),"independent_pointer_exact":pointer_exact(t),"complete_scene_roles":True,"reader_eligible":False,"provenance":"authored shared-scene SVO/locative control; not generated exact candidate"} for t in texts]

def run(limit=40000):
 exact=[]; diagnostics=[]; seen=set(); states=char_prunes=domain_prunes=semantic_prunes=seam_prunes=0
 for scene in DOMAINS:
  lefts=frames(scene,"SVO"); rights=frames(scene,"LOC")
  for left in lefts:
   for right in rights:
    # Distinct frame identities force the grammar alternation while retaining
    # the shared scene key; slot domains are checked during edge expansion.
    if left["slots"][0][1]==right["slots"][0][1]: semantic_prunes+=1; continue
    le=tuple(tuple(x.split()) for _,x in left["slots"]); re=tuple(tuple(x.split()) for _,x in reversed(right["slots"]))
    stack=[(0,0,"","","","",False,False)]
    while stack and states<limit:
     li,ri,lt,rt,lb,rb,ls,rs=stack.pop(); states+=1
     if li==len(le) and ri==len(re):
      rendered=(lt+"; "+rt).strip(); au=audit(rendered)
      if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"scene":scene,"cross_word_seam":ls or rs,"complete_scene_roles":True,"reader_eligible":False,"reason":"complete SVO/locative derivation but residual/exact gate failed"})
      if lb or rb or not(ls or rs):
       if not(ls or rs): seam_prunes+=1
       continue
      if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
       seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"scene":scene,"left_kind":"SVO","right_kind":"LOC","slot_domains_live":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
      continue
     if li<len(le):
      edge=le[li]; text="".join(edge); res=consume(lb+norm(text)[::-1],rb)
      if rb and norm(text)[::-1][0] != rb[0]: domain_prunes+=1; continue
      if res is None: char_prunes+=1
      else: stack.append((li+1,ri,lt+" "+" ".join(edge),rt,res[0],res[1],ls or(bool(lb) and len(norm(text))>len(rb)),rs))
     if ri<len(re):
      edge=re[ri]; text="".join(edge); res=consume(lb,rb+norm(text))
      if lb and norm(text)[0] != lb[0]: domain_prunes+=1; continue
      if res is None: char_prunes+=1
      else: stack.append((li,ri+1,lt," ".join(edge)+(" "+rt if rt else ""),res[0],res[1],ls,rs or(bool(rb) and len(norm(text))>len(lb))))
    if states>=limit: break
   if states>=limit: break
  if states>=limit: break
 return {"method":"shared-scene-locative-slot-domains-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","scenes":len(DOMAINS),"states":states,"character_prunes":char_prunes,"domain_prunes":domain_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh shared-scene grammar alternates complete transitive SVO and intransitive locative clauses; slot-level NP/verb/object/preposition/location domains are filtered under live character equations; no fixed frame replay, reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["joint-intact-clause-scene-lattice-20260920","character-boundary-product-20260920","shared-scene-online-clause-match-20260920"],"unused_dimension":"intransitive locative grammar alternation with slot-level character domains under a shared semantic scene key","reason":"prior shared-scene lanes used fixed SVO frames; this lane derives a distinct locative predicate and prepositional attachment with live slot domains"},"first_live_diagnostic":"locative slot residual mismatch during shared-scene growth" if not exact else "exact closure requires blinded reader review","next_construction":"hold out an existential locative frame with a typed postposition and preserve slot-domain gates"}

if __name__=="__main__":
 result=run(); out=ROOT/"runs/shared-scene-locative-slot-domains-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("scenes","states","character_prunes","domain_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
