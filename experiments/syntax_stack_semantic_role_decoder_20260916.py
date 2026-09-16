"""Syntax-stack semantic-role decoder with live mirrored character debt."""
from __future__ import annotations
import hashlib,itertools,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters,tokenize
EXPERIMENT="syntax-stack-semantic-role-decoder-20260916"
SIGNATURE="semantic-role-stack-pushdown|ordinary-order-complete-scene-realization|live-mirrored-character-debt|role-frame-return-continuation|heldout-role-repair|independent-four-audit"
OUT=ROOT/"runs"/(EXPERIMENT+".json");REGISTRY=ROOT/"docs/experiment-novelty-registry.json"
ROLE_FRAMES={"agent":["the harbor clerk","the patient harbor clerk","the station porter"],"action":["records the arrival","checks the sealed ledger","carries a wet parcel"],"object":["for the evening watch","beside the quiet bench","to the waiting office"],"response":["then the night guard locks the gate","while the tired traveler waits by the door","and the late train crosses the bridge"]}
SCENE={"id":"harbor-arrival-stack","meaning":"A harbor clerk records an arrival, checks a ledger, carries a parcel, and a guard or traveler responds in the same scene."}
def tape(s):return normalize_letters(s)
def two_pointer(s):
 t=tape(s);bad=[(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {"algorithm":"independent_two_pointer","exact":bool(t) and not bad,"letters":len(t),"mismatch_count":len(bad),"first_mismatch":bad[0] if bad else None}
def hash_audit(s):
 t=tape(s);return {"algorithm":"forward_reverse_sha256","exact":bool(t) and hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest(),"forward":hashlib.sha256(t.encode()).hexdigest(),"reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def stack_render(c):return "; ".join(ROLE_FRAMES[r][c[i]] for i,r in enumerate(("agent","action","object","response")))+"."
def debt(s):
 t=tape(s);n=len(t);pairs=[(i,n-1-i) for i in range(n//2)];good=[p for p in pairs if t[p[0]]==t[p[1]]]
 return {"equation":"new emitted character agrees with far-end obligation","positions_checked":len(pairs),"matching_pairs":len(good),"mismatch_pairs":len(pairs)-len(good),"match_rate":len(good)/len(pairs),"first_mismatch_offset":next((i for i,j in pairs if t[i]!=t[j]),None)}
def shape(s):
 w=[normalize_letters(x) for x in tokenize(s)];return {"algorithm":"independent_prose_shape","word_count":len(w),"sentence_count":len(re.findall(r"[.!?]",s)),"content_words_unique":len(set(w))>=len(w)*.7,"not_word_order_mirror":w!=list(reversed(w))}
def audit(s,c,rank=0):
 a=two_pointer(s);h=hash_audit(s);m=mechanical_admission_checks(s,min_letters=45,max_letters=180);q=shape(s)
 return {"rank":rank,"rendered":s,"letters":len(tape(s)),"role_stack":["agent","action","object","response"],"choice":c,"debt_audit":debt(s),"exact_check_1":a,"exact_check_2":h,"independent_exact_agreement":a["exact"]==h["exact"],"central_admission":m,"independent_shape":q,"mechanically_admitted":a["exact"] and h["exact"] and all(m.values()) and all(q.values()),"provenance":{"scene":SCENE["id"],"human_authored_frames":True,"catalogue_text_used":False,"pre_existing_palindrome_wrapped":False,"word_order_symmetry_used":False,"repeated_palindromic_unit_used":False},"next_repair_operator":"replace the complete role frame containing the first debt mismatch, then replay the held-out continuation stack"}
def preflight():
 e=json.loads(REGISTRY.read_text()).get("entries",[]);same=[x["id"] for x in e if x.get("signature")==SIGNATURE and x.get("id")!=EXPERIMENT]
 return {"entries_inspected":len(e),"exact_signature_collisions_before_run":same,"passed":not same,"distinction":"pushdown role stack with continuation returns and live end-character obligations; no tape segmentation, grammar intersection, or bilateral decoding"}
def main():
 p=preflight()
 if not p["passed"]:raise SystemExit(p)
 rows=[audit(stack_render(c),c) for c in itertools.product(range(3),repeat=4)];rows.sort(key=lambda r:(-r["debt_audit"]["match_rate"],r["exact_check_1"]["mismatch_count"],-r["letters"]))
 for i,r in enumerate(rows[:12],1):r["rank"]=i
 exact=[r for r in rows if r["exact_check_1"]["exact"] and r["exact_check_2"]["exact"]]
 out={"experiment":EXPERIMENT,"signature":SIGNATURE,"status":"complete_syntax_stack_semantic_role_search","novelty_preflight":p,"states_examined":len(rows),"exact_count":len(exact),"mechanically_admitted_count":sum(r["mechanically_admitted"] for r in rows),"best_rendered_candidates":rows[:12],"failed_attempts":rows,"readability_evidence":{"status":"diagnostic_only_human_reading_not_run","reader_eligible_count":0},"failure_evidence":{"all_nonexact_assignments_retained":True,"next_repair_operator":"held-out complete role-frame replacement at first debt mismatch","exact_closure_found":bool(exact)},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"catalogue_or_corpus_import":False}}
 OUT.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"states":len(rows),"exact":len(exact),"best":rows[0]["rendered"]}))
if __name__=="__main__":main()
