"""Second held-out repair: target the first outer character debt by adjunct choice."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters
EXPERIMENT_ID="authored-clause-template-sat-repair2-20260916"
SIGNATURE="heldout-second-outer-character-repair|usher-guide-visitor-valency|adjunct-final-character-target|full-character-equation-replay|independent-pointer-sha-audit"
OUT=ROOT/"runs"/(EXPERIMENT_ID+".json")
PARENT="the observant usher guides a young visitor toward the quiet gallery before noon. The quiet guide checks the gallery before noon."
FRAME={"sense":"an observant usher guides a young visitor toward the quiet gallery at sunset","subject":"the observant usher","verb":"guides","object":"a young visitor","adjunct":"toward the quiet gallery at sunset"}
def audit(text):
 l=normalize_letters(text); r=l[::-1]; hf=hashlib.sha256(l.encode()).hexdigest(); hr=hashlib.sha256(r.encode()).hexdigest()
 return {"letters":len(l),"forward":l,"reverse":r,"exact":l==r,"hash_forward":hf,"hash_reverse":hr,"hash_equal":hf==hr,"independent_pointer_audit":all(l[i]==l[-i-1] for i in range(len(l)))}
def run():
 parent=normalize_letters(PARENT); mismatch=next((i for i,(a,b) in enumerate(zip(parent,parent[::-1])) if a!=b),len(parent))
 left=f"{FRAME['subject']} {FRAME['verb']} {FRAME['object']} {FRAME['adjunct']}"
 text=left+". The quiet guide checks the gallery at sunset."
 a=audit(text); checks=mechanical_admission_checks(text,min_letters=39,max_letters=240)
 c={"rendered":text,"letters":a["letters"],"source_first_mismatch_index":mismatch,"targeted_outer_pair":{"left":normalize_letters(text)[0],"right":normalize_letters(text)[-1],"matched":normalize_letters(text)[0]==normalize_letters(text)[-1]},"fresh_valency_frame":FRAME,"exact_audit":a,"checks":checks,"admitted":bool(a["exact"] and all(checks.values())),"provenance":{"fresh_frame_authored":True,"adjunct_only_repair":True,"source_sentences_copied":False,"catalogue_imported":False,"reversed_finished_sentence":False,"word_order_symmetry":False,"repair_parent":PARENT}}
 return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed","method":"retain usher-guide-visitor valency and replace only the complete adjunct realization so its final character targets the first outer debt before replaying the full equation","parent":{"rendered":PARENT,"first_mismatch_index":mismatch},"candidates":[c],"stats":{"candidates":1,"exact":int(a["exact"]),"admitted":int(c["admitted"])},"next_repair":"Use a fresh subject whose initial character matches the repaired adjunct endpoint, preserving usher-guide-visitor roles; test as one complete frame.","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"copied_text":False}}
if __name__=="__main__":
 if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
 p=run(); OUT.write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p,indent=2))
