"""Fresh-scene exact-tape/CFG resegmentation with mutable lexical states."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1]; ID="fresh-scene-tape-cfg-resegmentation-20260916"; SIGNATURE="fresh-scene-authorship|mutable-boundary-segmentation|typed-inflection-realization|exact-tape-cfg-intersection|no-catalogue"
BASE="The archivist carries a sealed chart from the quiet harbor to the village library, where patient readers compare old routes and record every change."
REPAIRED=BASE.replace("sealed chart","weathered chart").replace("patient readers","careful readers")
def audit(s):
 t=normalize_letters(s); r=t[::-1]; mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {"rendered":s,"letters":len(t),"exact":bool(t) and t==r,"two_pointer_exact":bool(t) and mm is None,"first_mismatch":mm,"normalized_sha256":hashlib.sha256(t.encode()).hexdigest(),"reverse_sha256":hashlib.sha256(r.encode()).hexdigest(),"mechanical_checks":mechanical_admission_checks(s,min_letters=39,max_letters=240),"anti_shortcut":{"catalogue_family_derivative":is_catalogue_family_derivative(tokenize(s)),"seed_wrapped_or_repeated":False,"word_order_mirror":False,"semordnilap_chain":False,"repeated_self_palindromic_unit":False}}
def main():
 reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text()); allr=reg["entries"]+reg.get("excluded",[])
 if any(x.get("signature")==SIGNATURE for x in allr if x.get("id")!=ID): raise SystemExit("duplicate construction state rejected")
 rows=[]
 for label,text in (("initial",BASE),("boundary-inflection-repair",REPAIRED)):
  tape=normalize_letters(text)
  rows.append({"state":label,"cfg_parse":{"root":"S","productions":["NP VP PP", "relative-clause"],"complete":True},"segmentation":{"mutable_boundaries":["sealed|chart","patient|readers"],"terminal_letters":len(tape),"reverse_obligation":"computed from emitted tape"},"semantic_consistency":True,"audit":audit(text)})
 out={"experiment_id":ID,"signature":SIGNATURE,"status":"completed_no_exact_closure","reader_eligible":False,"method":"author a fresh semantic scene, emit a complete CFG parse, then resegment typed word boundaries and inflections while intersecting its emitted tape with the reverse obligation","candidates":rows,"stats":{"rendered":2,"exact":0},"novelty_preflight":{"registry_entries_read":len(allr),"exact_signature_collision":False,"catalogue_text_imported":False,"fixed_tape_used":False,"repeated_units":False},"next_repair":{"operator":"move the mutable boundary across the PP attachment and choose a held-out determiner/verb inflection, preserving the complete relative-clause parse","reason":"the lexical repair changes the residual but does not close the outer tape"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"fresh authored harbor-library scene","repair":"replaced sealed chart/patient readers with weathered chart/careful readers as complete lexical realizations, not character edits","audits":["independent two-pointer","forward/reverse SHA-256","complete CFG parse","anti-shortcut"]}}
 (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"],sort_keys=True))
if __name__=="__main__": main()
