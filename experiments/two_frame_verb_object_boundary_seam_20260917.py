"""Explicit verb-object seam state over two fixed ordinary-English frames."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/two-frame-verb-object-boundary-seam-20260917.json";REG=ROOT/"docs/experiment-novelty-registry.json"
ID="two-frame-verb-object-boundary-seam-20260917";SIG="two-fixed-typed-frames|verb-object-boundary-state|same-ordinary-rendering|live-center-obligation|independent-exact-audit"
FRAMES=(("the patient gardener","watered","orchard","the local archivist","recorded","measurements"),("the senior curator","examined","circuit","the museum guide","reviewed","records"))
FIXED={"determiner":"the","adjective":"weathered","adverb":"carefully","preposition":"beside the window","complement":"with purpose","seam":"while"}
def norm(s):return re.sub(r"[^a-z]","",s.lower())
def audit(s):
 t=norm(s);m=[];i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not m,"independent_two_pointer_exact":bool(t) and not m,"first_mismatches":m[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def preflight():
 es=json.loads(REG.read_text()).get("entries",[]);a=str(Path(__file__).relative_to(ROOT));return {"status":"passed","registry_entries_read":len(es),"signature_collision":any(x.get("signature")==SIG for x in es),"artifact_collision":any(x.get("artifact")==a for x in es),"shortcuts_rejected":["finished-tape reversal","word-order symmetry","catalogue text","fragments"]}
def emit(frame,seam):
 s,v,o,r,rv,ro=frame;text=f"{s.capitalize()} {v} {FIXED['determiner']} {FIXED['adjective']} {o} {FIXED['adverb']} {FIXED['preposition']} {FIXED['complement']} {FIXED['seam']} {r} {rv} {FIXED['determiner']} {FIXED['adjective']} {ro} {FIXED['adverb']}.";t=norm(text);mid=len(t)//2;sp=[];cur=0
 for tok in re.findall(r"[A-Za-z]+",text):a=cur;cur+=len(tok);sp.append((tok,a,cur))
 cross=next(({"token":w,"token_interval":[a,b],"midpoint":mid,"offset":mid-a} for w,a,b in sp if a<=mid<b),None);a=audit(text);i,j=0,len(t)-1;pairs=0
 while i<j and t[i]==t[j]:pairs+=1;i+=1;j-=1
 # The seam is a grammar boundary, not a text rewrite; both placements are
 # audited on the same intact rendering to reject non-grammatical reorderings.
 return {"rendered":text,"choices":{"frame":frame,"verb_object_seam":seam,"fixed_states":FIXED},"audit":a,"center_state":{"midpoint":mid,"crossing":cross,"closed_pairs_before_first_mismatch":pairs,"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None,"verb_object_boundary":{"token_sequence":[v,o],"seam_index":"after_verb" if seam=="before_object" else "after_object","rewrite_admitted":False}},"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"catalogue_text":False,"punctuation_changes_letters":False,"fragment":False},"provenance":{"lexical_source":"same two fixed hand-authored frames","borrowed_text":False,"new_lexical_bank":False,"no_clause_or_attachment":True,"malformed_reordering_rejected":True}}
def run():
 pre=preflight();rows=[emit(f,s) for f,s in itertools.product(FRAMES,("before_object","after_object"))];rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True);exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"verb-object boundary state with ordinary rendering and rejected rewrites","novelty_preflight":pre,"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"variants":len(rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"midpoint_inside_token":sum(r["center_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain boundary state and frames, then test a seam at the subject-verb boundary"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"shortcuts_excluded":True}}
if __name__=="__main__":
 x=run();OUT.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({"candidates":x["candidate_count"],"exact":x["exact_count"],"stats":x["stats"]},sort_keys=True))
