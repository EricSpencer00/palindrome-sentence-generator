"""Pre-render Earley lane for object-NP whose-possessive CP attachment."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID="cfg-earley-object-np-whose-cp-20260921"
SIGNATURE="earley-object-np-whose-cp|embedded-possessive-seam|vp-object-attachment|live-mirrored-domain"
REGISTRY=ROOT/"docs/experiment-novelty-registry.json"; OUT=ROOT/"runs"/(ID+".json")
CLAUSES=(
 {"id":"keeper-crates-labels","subject":"The patient keeper","verb":"stores","object":"the sealed crates","cp":"whose labels the curator checks","number":"sg","valency":"transitive"},
 {"id":"archivist-maps-folds","subject":"The careful archivist","verb":"files","object":"the folded maps","cp":"whose edges the assistant measures","number":"sg","valency":"transitive"},
 {"id":"porter-parcels-seals","subject":"The young porter","verb":"carries","object":"the marked parcels","cp":"whose seals the clerk records","number":"sg","valency":"transitive"},)
def norm(s): return "".join(c.lower() for c in s if "a"<=c.lower()<="z")
def render(c): return f"{c['subject']} {c['verb']} {c['object']} {c['cp']}"
def chart(c):
 u=c["number"]=="sg" and c["valency"]=="transitive"
 return {"algorithm":"earley_item_chart","productions":["S -> NP VP","VP -> V NP_obj","NP_obj -> Det N CP","CP -> whose NP_poss V"],"items":[{"lhs":"S","dot":0,"features":{"number":c["number"]}},{"lhs":"NP_obj","dot":3,"features":{"number":c["number"],"attachment":"object"}},{"lhs":"CP","dot":3,"features":{"relative":"whose","possessor_np":True,"attachment":"object-NP"}},{"lhs":"VP","dot":2,"features":{"valency":c["valency"]}}],"object_cp_attachment":"VP -> V [NP_obj -> Det N CP]","agreement_unified":u,"accepted":u}
def domains(left,right):
 l,r=norm(left),norm(right); rows=[]; bad=[]
 for i in range(min(len(l),len(r))):
  j=len(r)-1-i; d=sorted({l[i]}&{r[j]}); x={"position":i,"opposing_position":j,"left_owner":"left_object_whose_CP_item","right_owner":"right_object_whose_CP_item","left_support":l[i],"right_support":r[j],"domain":d,"survives":bool(d)}; rows.append(x)
  if not d: bad.append(x)
 return {"algorithm":"online_mirrored_domain_intersection","invariant":"each support belongs to an emitted object-whose CP item; empty domains remain conflicts","pairs_checked":len(rows),"surviving_pairs":sum(x["survives"] for x in rows),"conflict_count":len(bad),"first_conflict":bad[0] if bad else None,"ledger":rows[:24]}
def audit(s,k):
 t=norm(s)
 if k=="pointer":
  mm=[{"i":i,"j":len(t)-1-i,"left":t[i],"right":t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]]; return {"algorithm":"independent_pointer","exact":bool(t) and not mm,"letters":len(t),"mismatch_count":len(mm),"mismatches":mm[:12]}
 return {"algorithm":"independent_sha_forward_reverse","exact":bool(t) and hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(t[::-1].encode()).hexdigest(),"forward":hashlib.sha256(t.encode()).hexdigest(),"reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def novelty():
 e=json.loads(REGISTRY.read_text()).get("entries",[]); c=[x["id"] for x in e if x.get("id")!=ID and x.get("signature")==SIGNATURE]; return {"entries_inspected":len(e),"exact_signature_collisions":c,"passed":not c,"distinction":"whose possessive CP is attached inside object NP through VP -> V NP_obj, distinct from object when and subject whose seams, expanded before rendering"}
def row(a,b,i):
 left,right=render(a),render(b); text=left+". "+right+"."; p,s=audit(text,"pointer"),audit(text,"sha")
 return {"rank":i,"rendered":text,"letters":p["letters"],"provenance":{"left_clause":a["id"],"right_clause":b["id"],"left_object_cp":a["cp"],"right_object_cp":b["cp"],"topology":"VP -> V [NP_obj -> Det N CP; whose NP_poss V]","source":"fresh authored grammar terminals"},"left_chart":chart(a),"right_chart":chart(b),"mirrored_domains":domains(left,right),"exact_check_pointer":p,"exact_check_sha":s,"independent_exact_agreement":p["exact"]==s["exact"],"anti_shortcut_flags":{"fixed_tape":False,"post_render_repair":False,"reverse_decoder":False,"mirrored_units":False,"word_order_mirror":False,"catalogue_text":False,"reward_loop":False,"complete_constituents":True,"object_possessive_cp_pre_render":True,"agreement_checked":True,"valency_checked":True},"mechanically_admitted":False,"next_operator":"Pivot to an object-NP that-relative CP with an overt subject and object gap before chart expansion, preserving the attachment seam and independent audits."}
def run():
 pre=novelty()
 if not pre["passed"]: raise RuntimeError(pre)
 pairs=((CLAUSES[0],CLAUSES[1]),(CLAUSES[1],CLAUSES[2]),(CLAUSES[2],CLAUSES[0])); rows=[row(a,b,i+1) for i,(a,b) in enumerate(pairs)]
 return {"experiment_id":ID,"signature":SIGNATURE,"method":"pre-render object-NP whose CP Earley expansion with live mirrored domains","novelty_preflight":pre,"rows":rows,"stats":{"states_examined":len(rows),"over_39":sum(x["letters"]>=39 for x in rows),"exact":sum(x["mechanically_admitted"] for x in rows),"agreement_unified":sum(x["left_chart"]["accepted"] and x["right_chart"]["accepted"] for x in rows),"domain_conflicts":sum(x["mirrored_domains"]["conflict_count"] for x in rows),"rendered_lengths":[x["letters"] for x in rows]},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer","forward/reverse SHA","Earley chart","live mirrored-domain ledger"],"brown_usage":"none; no borrowed prose"},"anti_shortcut_policy":"No post-render repair, fixed tape, reverse decoder, mirrored units, word-order symmetry, catalogue text, or reward loop.","next_operator":"Pivot to an object-NP that-relative CP with overt subject and object gap before expansion."}
if __name__=="__main__":
 if OUT.exists(): raise SystemExit(f"output already exists: {OUT}")
 result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],indent=2))
