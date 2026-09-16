"""Held-out repair for the authored clause-template SAT lane.

The first parent probe's outer mismatch is recorded, then one fresh complete
valency frame is authored and substituted. This is a repair, not a sweep.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters
EXPERIMENT_ID="authored-clause-template-sat-repair-20260916"
SIGNATURE="heldout-first-debt-valency-frame|complete-clause-substitution|agreement-valency-preservation|full-character-equation-replay|independent-pointer-sha-audit"
OUT=ROOT/"runs"/(EXPERIMENT_ID+".json")
PARENT="the patient courier delivers a sealed letter before dusk. the patient courier delivers a sealed letter before dusk."
FRAME={"sense":"an observant usher guides a visitor toward the quiet gallery before noon","subject":"the observant usher","verb":"guides","object":"a young visitor","adjunct":"toward the quiet gallery before noon"}
def audit(text):
 letters=normalize_letters(text); rev=letters[::-1]; hf=hashlib.sha256(letters.encode()).hexdigest(); hr=hashlib.sha256(rev.encode()).hexdigest()
 return {"letters":len(letters),"forward":letters,"reverse":rev,"exact":letters==rev,"hash_forward":hf,"hash_reverse":hr,"hash_equal":hf==hr,"independent_pointer_audit":all(letters[i]==letters[-i-1] for i in range(len(letters)))}
def run():
 first=normalize_letters(PARENT); reverse=first[::-1]; mismatch=next((i for i,(a,b) in enumerate(zip(first,reverse)) if a!=b),min(len(first),len(reverse)))
 left=f"{FRAME['subject']} {FRAME['verb']} {FRAME['object']} {FRAME['adjunct']}"
 text=left+". The quiet guide checks the gallery before noon."
 a=audit(text); checks=mechanical_admission_checks(text,min_letters=39,max_letters=240)
 candidate={"rendered":text,"letters":a["letters"],"source_first_mismatch_index":mismatch,"fresh_valency_frame":FRAME,"exact_audit":a,"checks":checks,"admitted":bool(a["exact"] and all(checks.values())),"provenance":{"fresh_frame_authored":True,"source_sentences_copied":False,"catalogue_imported":False,"reversed_finished_sentence":False,"word_order_symmetry":False,"repair_parent":PARENT}}
 return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed","method":"replace the complete parent clause at its first mirrored character debt with one fresh typed valency frame, then replay full rendering and exact gate","parent":{"rendered":PARENT,"first_mismatch_index":mismatch},"candidates":[candidate],"stats":{"candidates":1,"exact":int(a["exact"]),"admitted":int(candidate["admitted"])},"next_repair":"Author a second fresh frame whose final adjunct supplies the unmatched outer character while preserving usher-guide-visitor valency; test it as a single held-out repair.","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"copied_text":False}}
if __name__=="__main__":
 if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
 p=run(); OUT.write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p,indent=2))
