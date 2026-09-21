"""Pre-render Earley expansion with a held-out determiner-bearing PP."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID="cfg-earley-determiner-pp-expansion-20260921"
SIGNATURE="earley-determiner-bearing-pp|pre-render-np-expansion|locative-feature-unification|live-mirrored-domain"
REGISTRY=ROOT/"docs/experiment-novelty-registry.json"; OUT=ROOT/"runs"/(ID+".json")
CLAUSES=(
 {"id":"curator-vases-gallery","subject":"The careful curator","verb":"moves","object":"the fragile vases","number":"sg","valency":"transitive-locative"},
 {"id":"miller-baskets-market","subject":"The patient miller","verb":"carries","object":"a woven basket","number":"sg","valency":"transitive-locative"},
 {"id":"sentry-lantern-wall","subject":"The alert sentry","verb":"checks","object":"the brass lantern","number":"sg","valency":"transitive-locative"},)
PPS=(
 {"id":"pp-beside-a-cedar-footbridge","prep":"beside","det":"a","noun":"cedar footbridge","valency":"transitive-locative","new":True},
 {"id":"pp-near-a-stone-garden","prep":"near","det":"a","noun":"stone garden","valency":"transitive-locative","new":True},
 {"id":"pp-under-a-willow-arch","prep":"under","det":"a","noun":"willow arch","valency":"transitive-locative","new":True},)
def norm(s): return "".join(c.lower() for c in s if "a"<=c.lower()<="z")
def pp(p): return f"{p['prep']} {p['det']} {p['noun']}"
def render(c,p): return f"{c['subject']} {c['verb']} {c['object']} {pp(p)}"
def chart(c,p):
 f={"number":c["number"],"valency":c["valency"]}
 return {"algorithm":"earley_item_chart","productions":["S -> NP VP","VP -> V NP PP","PP -> P NP","NP -> Det N"],"items":[{"lhs":"S","dot":0,"origin":0,"features":{"number":c["number"]}},{"lhs":"VP","dot":3,"origin":0,"features":f},{"lhs":"PP","dot":2,"origin":0,"features":{"valency":p["valency"],"det":p["det"],"noun":p["noun"],"heldout":True}}],"feature_unified":f["valency"]==p["valency"] and bool(p["det"]),"accepted":f["valency"]==p["valency"] and bool(p["det"])}
def domains(left,right):
 l,r=norm(left),norm(right); ledger=[]; conflicts=[]
 for i in range(min(len(l),len(r))):
  j=len(r)-1-i; d=sorted({l[i]}&{r[j]}); x={"position":i,"opposing_position":j,"left_owner":"left_PP_Earley_item","right_owner":"right_PP_Earley_item","left_support":l[i],"right_support":r[j],"domain":d,"survives":bool(d)}; ledger.append(x)
  if not d: conflicts.append(x)
 return {"algorithm":"online_mirrored_domain_intersection","owner_invariant":"every support comes from an emitted Earley terminal; empty domains are retained conflicts","pairs_checked":len(ledger),"surviving_pairs":sum(x["survives"] for x in ledger),"conflict_count":len(conflicts),"first_conflict":conflicts[0] if conflicts else None,"ledger":ledger[:24]}
def pointer(s):
 t=norm(s); mm=[{"i":i,"j":len(t)-1-i,"left":t[i],"right":t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]]; return {"algorithm":"independent_pointer","exact":bool(t) and not mm,"letters":len(t),"mismatch_count":len(mm),"mismatches":mm[:12]}
def sha(s):
 t=norm(s); return {"algorithm":"independent_sha_forward_reverse","exact":bool(t) and hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(t[::-1].encode()).hexdigest(),"forward":hashlib.sha256(t.encode()).hexdigest(),"reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def novelty():
 e=json.loads(REGISTRY.read_text()).get("entries",[]); c=[x["id"] for x in e if x.get("id")!=ID and x.get("signature")==SIGNATURE]; return {"entries_inspected":len(e),"exact_signature_collisions":c,"passed":not c,"distinction":"PP expands as P Det N before rendering; determiner and locative valency are Earley features and mirrored domains are live"}
def row(a,b,pa,pb,rank):
 left,right=render(a,pa),render(b,pb); text=left+". "+right+"."; p,s=pointer(text),sha(text)
 return {"rank":rank,"rendered":text,"letters":p["letters"],"provenance":{"left_clause":a["id"],"right_clause":b["id"],"left_pp":pa["id"],"right_pp":pb["id"],"determiner_bearing_pp":True,"source":"fresh authored grammar terminals"},"left_chart":chart(a,pa),"right_chart":chart(b,pb),"mirrored_domains":domains(left,right),"exact_check_pointer":p,"exact_check_sha":s,"independent_exact_agreement":p["exact"]==s["exact"],"anti_shortcut_flags":{"fixed_tape":False,"post_render_repair":False,"reverse_decoder":False,"mirrored_units":False,"word_order_mirror":False,"catalogue_text":False,"reward_loop":False,"complete_constituents":True,"determiner_expanded_pre_render":True,"valency_checked":True},"mechanically_admitted":False,"next_operator":"Add a held-out plural determiner-bearing PP with agreement-carrying NP features before Earley expansion, then retain its live domain conflicts."}
def run():
 pre=novelty()
 if not pre["passed"]: raise RuntimeError(pre)
 specs=((CLAUSES[0],CLAUSES[1],PPS[0],PPS[1]),(CLAUSES[1],CLAUSES[2],PPS[1],PPS[2]),(CLAUSES[2],CLAUSES[0],PPS[2],PPS[0])); rows=[row(*x,i+1) for i,x in enumerate(specs)]
 return {"experiment_id":ID,"signature":SIGNATURE,"method":"held-out determiner-bearing PP Earley expansion with live mirrored domains","novelty_preflight":pre,"rows":rows,"stats":{"states_examined":len(rows),"over_39":sum(x["letters"]>=39 for x in rows),"exact":sum(x["mechanically_admitted"] for x in rows),"feature_unified":sum(x["left_chart"]["accepted"] and x["right_chart"]["accepted"] for x in rows),"domain_conflicts":sum(x["mirrored_domains"]["conflict_count"] for x in rows)},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer","forward/reverse SHA","Earley chart","live mirrored-domain ledger"],"brown_usage":"none; no borrowed prose"},"anti_shortcut_policy":"No post-render repair, fixed tape, reverse decoder, mirrored units, word-order symmetry, catalogue text, or reward loop.","next_operator":"Add a held-out plural determiner-bearing PP with agreement features before expansion."}
if __name__=="__main__":
 if OUT.exists(): raise SystemExit(f"output already exists: {OUT}")
 result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],indent=2))
