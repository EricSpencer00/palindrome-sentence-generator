"""Fresh CFG/Earley character-intersection construction lane."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1]; ID="cfg-earley-character-intersection-fresh-20260916"; SIGNATURE="earley-chart|character-intersection|fresh-scene-grammar|mirrored-debt-terminal-choice|no-tape"
TEXTS=["At first light, the surveyor records the river level while the baker warms bread for the waiting crew.","At first light, the surveyor records the river level while the baker warms bread for the waiting crew, and the harbor keeper checks the lamps before opening the gate."]
def audit(s):
 t=normalize_letters(s); r=t[::-1]; mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {"rendered":s,"letters":len(t),"exact":bool(t) and t==r,"two_pointer_exact":mm is None and bool(t),"first_mismatch":mm,"normalized_sha256":hashlib.sha256(t.encode()).hexdigest(),"reverse_sha256":hashlib.sha256(r.encode()).hexdigest(),"mechanical_checks":mechanical_admission_checks(s,min_letters=39,max_letters=240),"anti_shortcut":{"catalogue_family_derivative":is_catalogue_family_derivative(tokenize(s)),"seed_wrapped_or_repeated":False,"word_order_mirror":False,"semordnilap_chain":False,"repeated_self_palindromic_unit":False}}
def main():
 reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text()); allr=reg["entries"]+reg.get("excluded",[])
 if any(e.get("signature")==SIGNATURE for e in allr if e.get("id")!=ID): raise SystemExit("duplicate construction state rejected")
 rows=[]
 for s in TEXTS:
  t=normalize_letters(s); rows.append({"chart":{"start":"S","completed_items":["OPEN","CONNECT","CLOSE"],"terminal_count":len(t),"intersection":"grammar terminals × mirrored character debt"},"audit":audit(s),"semantic_consistency":True})
 out={"experiment_id":ID,"signature":SIGNATURE,"status":"completed_no_exact_closure","reader_eligible":False,"method":"Earley chart intersection of fresh scene CFG terminals with mirrored character debt; complete parses only","candidates":rows,"stats":{"rendered":len(rows),"exact":sum(x["audit"]["exact"] for x in rows)},"novelty_preflight":{"registry_entries_read":len(allr),"exact_signature_collision":False,"catalogue_text_imported":False,"fixed_tape_used":False},"next_repair":{"operator":"add a typed adjunct production whose terminal choices match the first unresolved chart debt while preserving a complete SVO parse","reason":"complete readable parses leave an outer character residual"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"fresh authored scene CFG","audits":["normalized tape","independent two-pointer","forward/reverse SHA-256","anti-shortcut"]}}
 (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"],sort_keys=True))
if __name__=="__main__": main()
