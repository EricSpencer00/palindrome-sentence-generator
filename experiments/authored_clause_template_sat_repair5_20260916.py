"""Fifth held-out repair: fresh guide verb, fixed valency and endpoint."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters
EXPERIMENT_ID="authored-clause-template-sat-repair5-20260916"
SIGNATURE="heldout-fifth-guide-verb-repair|fresh-guide-verb-realization|usher-guide-visitor-valency|outer-endpoint-character-match|independent-pointer-sha-audit"
OUT=ROOT/"runs"/(EXPERIMENT_ID+".json")
PARENT="the thoughtful usher guides a young visitor toward the quiet gallery at sunset. the quiet guide checks the lantern at sunset."
FRAME={"subject":"the thoughtful usher","verb":"guides","object":"a young visitor","adjunct":"toward the quiet gallery at sunset","guide_clause":"the quiet guide leads the lantern at sunset"}
def audit(text):
 l=normalize_letters(text); r=l[::-1]; hf=hashlib.sha256(l.encode()).hexdigest(); hr=hashlib.sha256(r.encode()).hexdigest()
 return {"letters":len(l),"forward":l,"reverse":r,"exact":l==r,"hash_forward":hf,"hash_reverse":hr,"hash_equal":hf==hr,"independent_pointer_audit":all(l[i]==l[-i-1] for i in range(len(l)))}
def run():
 p=normalize_letters(PARENT); mismatch=next((i for i,(a,b) in enumerate(zip(p,p[::-1])) if a!=b),len(p)); text=f"{FRAME['subject']} {FRAME['verb']} {FRAME['object']} {FRAME['adjunct']}. {FRAME['guide_clause']}."; a=audit(text); checks=mechanical_admission_checks(text,min_letters=39,max_letters=240); l=normalize_letters(text)
 c={"rendered":text,"letters":a["letters"],"source_first_mismatch_index":mismatch,"targeted_outer_pair":{"left":l[0],"right":l[-1],"matched":l[0]==l[-1]},"fresh_guide_verb":"leads","exact_audit":a,"checks":checks,"admitted":bool(a["exact"] and all(checks.values())),"provenance":{"fresh_verb_authored":True,"fresh_frame_authored":True,"preserved_roles":["usher","guide","visitor"],"source_sentences_copied":False,"catalogue_imported":False,"reversed_finished_sentence":False,"word_order_symmetry":False,"repair_parent":PARENT}}
 return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed","method":"replace only the guide verb with one fresh valency-compatible realization, preserve scene roles and endpoint adjunct, then replay the full equation","parent":{"rendered":PARENT,"first_mismatch_index":mismatch},"candidates":[c],"stats":{"candidates":1,"exact":int(a["exact"]),"admitted":int(c["admitted"])},"next_repair":"Author one fresh guide subject/object pair with the same endpoint adjunct, preserving the complete scene roles; test as one held-out frame.","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"copied_text":False}}
if __name__=="__main__":
 if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
 q=run(); OUT.write_text(json.dumps(q,indent=2)+"\n"); print(json.dumps(q,indent=2))
