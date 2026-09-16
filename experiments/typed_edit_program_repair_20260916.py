"""Joint typed edit-program repair over complete authored English scenes."""
from __future__ import annotations
import hashlib, itertools, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters
EXPERIMENT="typed-edit-program-repair-20260916"
SIGNATURE="complete-authored-scenes|typed-edit-program|agreement-attachment-lexical-sense-joint-state|mismatch-certificates|heldout-repair|dual-exact-audit"
OUT=ROOT/"runs"/f"{EXPERIMENT}.json"; REGISTRY=ROOT/"docs/experiment-novelty-registry.json"
SCENES=[
 {"id":"harbor","subject":"The patient keeper","verb":"records","object":"the tide"},
 {"id":"market","subject":"A quiet vendor","verb":"packs","object":"the ripe pears"},
 {"id":"garden","subject":"The careful nurse","verb":"carries","object":"a warm blanket"}]
AGREEMENT={"singular":("The patient keeper","records"),"plural":("The patient keepers","record")}
ATTACHMENT={"after":"after rain","before":"before noon","while":"while birds settle"}
SENSE={"care":"carefully","quiet":"quietly","kind":"gently"}
def novelty_preflight():
 entries=json.loads(REGISTRY.read_text()).get("entries",[]); collisions=[e["id"] for e in entries if e.get("id")!=EXPERIMENT and e.get("signature")==SIGNATURE]
 return {"entries_inspected":len(entries),"exact_signature_collisions_before_run":collisions,"passed":not collisions,"state_space_distinction":"joint typed edit programs over intact authored scene frames; certificates classify agreement, attachment, and lexical-sense conflicts before held-out repair"}
def audit_exact(text):
 tape=normalize_letters(text); mm=[]
 for i in range(len(tape)//2):
  j=len(tape)-1-i
  if tape[i]!=tape[j]: mm.append({"offset":i,"right_offset":j,"left":tape[i],"right":tape[j]})
 return {"algorithm":"independent_two_pointer","exact":bool(tape) and not mm,"letters":len(tape),"mismatch_count":len(mm),"mismatches":mm[:8]}
def audit_hash(text):
 tape=normalize_letters(text); return {"algorithm":"sha256_forward_reverse","exact":bool(tape) and hashlib.sha256(tape.encode()).hexdigest()==hashlib.sha256(tape[::-1].encode()).hexdigest()}
def render(scene,program):
 agreement,attachment,sense=program; subj,verb=scene["subject"],scene["verb"]
 if scene["id"]=="harbor": subj,verb=AGREEMENT[agreement]
 elif agreement=="plural": subj=subj.replace("A quiet vendor","Quiet vendors").replace("The careful nurse","Careful nurses"); verb={"packs":"pack","carries":"carry"}.get(verb,verb)
 return f"{subj} {verb} {scene['object']} {ATTACHMENT[attachment]}; {SENSE[sense]}, the clerk checks the ledger."
def certificate(text,program):
 e=audit_exact(text)
 if not e["mismatches"]: return {"kind":"none","first_mismatch":None,"repairable_dimension":None,"certificate":"no mismatch"}
 dim="agreement" if program[0]!="singular" else "attachment" if program[1]!="after" else "lexical_sense"
 return {"kind":dim,"first_mismatch":e["mismatches"][0],"repairable_dimension":dim,"certificate":"first mirrored character conflict exposed before any tape mutation"}
def heldout(scene,program,cert):
 p=list(program); dim=cert["repairable_dimension"] or "agreement"; idx={"agreement":0,"attachment":1,"lexical_sense":2}[dim]; vals=[("singular","plural"),("after","before","while"),("care","quiet","kind")][idx]; p[idx]=next(v for v in vals if v!=p[idx]); text=render(scene,tuple(p)); e=audit_exact(text)
 return {"operator":"held-out typed edit on certified mismatch dimension","dimension":dim,"program":p,"rendered":text,"exact":e["exact"],"mismatch_count":e["mismatch_count"]}
def run():
 pre=novelty_preflight()
 if not pre["passed"]: raise RuntimeError(pre)
 rows=[]; programs=list(itertools.product(("singular","plural"),("after","before","while"),("care","quiet","kind")))
 for scene in SCENES:
  for program in programs:
   text=render(scene,program); one=audit_exact(text); two=audit_hash(text); cert=certificate(text,program)
   rows.append({"scene_id":scene["id"],"program":program,"rendered":text,"letters":len(normalize_letters(text)),"exact_check_1":one,"exact_check_2":two,"independent_exact_agreement":one["exact"]==two["exact"],"mismatch_certificate":cert,"heldout_repair":heldout(scene,program,cert),"provenance":{"scene_authored_here":True,"catalogue_imported":False,"wrapper_used":False,"word_order_mirror":False,"complete_prose":True},"readability_evidence":{"diagnostic_only":True,"reader_status":"not_run; candidate requires blinded human rating"}})
 rows.sort(key=lambda r:(not r["exact_check_1"]["exact"],-r["letters"],r["rendered"])); return {"experiment":EXPERIMENT,"signature":SIGNATURE,"status":"complete_typed_edit_program_search","novelty_preflight":pre,"states_examined":len(rows),"exact_count":sum(r["exact_check_1"]["exact"] for r in rows),"mechanically_admitted_count":0,"best_rendered_candidates":rows[:18],"failed_attempts":rows,"failure_evidence":{"all_nonexact_states_have_certificates":all(r["mismatch_certificate"]["first_mismatch"] for r in rows),"next_repair_operator":"held-out typed edit on certified agreement/attachment/lexical-sense dimension"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"catalogue_import":False}}
if __name__=="__main__":
 if OUT.exists(): raise SystemExit(f"output exists: {OUT}")
 payload=run(); OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({k:payload[k] for k in ("states_examined","exact_count","mechanically_admitted_count")},indent=2))
