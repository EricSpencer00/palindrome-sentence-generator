"""Third held-out repair: match the repaired outer endpoint with a fresh subject."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters
EXPERIMENT_ID="authored-clause-template-sat-repair3-20260916"
SIGNATURE="heldout-third-subject-repair|fresh-t-initial-usher-realization|usher-guide-visitor-valency|outer-endpoint-character-match|independent-pointer-sha-audit"
OUT=ROOT/"runs"/(EXPERIMENT_ID+".json")
PARENT="the observant usher guides a young visitor toward the quiet gallery at sunset. The quiet guide checks the gallery at sunset."
FRAME={"sense":"a thoughtful usher guides a young visitor toward the quiet gallery at sunset","subject":"the thoughtful usher","verb":"guides","object":"a young visitor","adjunct":"toward the quiet gallery at sunset"}
def audit(text):
 l=normalize_letters(text); r=l[::-1]; hf=hashlib.sha256(l.encode()).hexdigest(); hr=hashlib.sha256(r.encode()).hexdigest()
 return {"letters":len(l),"forward":l,"reverse":r,"exact":l==r,"hash_forward":hf,"hash_reverse":hr,"hash_equal":hf==hr,"independent_pointer_audit":all(l[i]==l[-i-1] for i in range(len(l)))}
def run():
 p=normalize_letters(PARENT); mismatch=next((i for i,(a,b) in enumerate(zip(p,p[::-1])) if a!=b),len(p))
 left=f"{FRAME['subject']} {FRAME['verb']} {FRAME['object']} {FRAME['adjunct']}"; text=left+". The quiet guide checks the gallery at sunset."
 a=audit(text); checks=mechanical_admission_checks(text,min_letters=39,max_letters=240); letters=normalize_letters(text)
 c={"rendered":text,"letters":a["letters"],"source_first_mismatch_index":mismatch,"targeted_outer_pair":{"left":letters[0],"right":letters[-1],"matched":letters[0]==letters[-1]},"fresh_valency_frame":FRAME,"exact_audit":a,"checks":checks,"admitted":bool(a["exact"] and all(checks.values())),"provenance":{"fresh_subject_authored":True,"fresh_frame_authored":True,"preserved_roles":["usher","guide","visitor"],"source_sentences_copied":False,"catalogue_imported":False,"reversed_finished_sentence":False,"word_order_symmetry":False,"repair_parent":PARENT}}
 return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed","method":"replace the subject with one fresh t-initial realization while preserving usher-guide-visitor valency and the repaired adjunct, then replay the full character equation","parent":{"rendered":PARENT,"first_mismatch_index":mismatch},"candidates":[c],"stats":{"candidates":1,"exact":int(a["exact"]),"admitted":int(c["admitted"])},"next_repair":"Author one fresh guide-clause complement ending in the same outer character, preserving the complete scene roles; test as one held-out frame.","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"copied_text":False}}
if __name__=="__main__":
 if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
 q=run(); OUT.write_text(json.dumps(q,indent=2)+"\n"); print(json.dumps(q,indent=2))
