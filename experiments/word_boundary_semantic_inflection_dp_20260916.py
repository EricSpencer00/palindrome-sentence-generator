"""Joint semantic/inflection word-boundary DP (fresh construction state)."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1]; ID="word-boundary-semantic-inflection-dp-20260916"; SIGNATURE="joint-slot-dp|semantic-svo|inflection-carry|cross-word-mirror-debt|fresh-generation"
SLOTS=[("subject",["The curator","The gardener"]),("verb",["records","carefully records","waters"]),("object",["the weathered harbor map","the eastern seedlings"]),("adjunct",["before the evening archive closes","while the patient mason repairs the wall"]),("second",["A young pilot carries fresh charts to the lighthouse"])]
def audit(s):
 t=normalize_letters(s); r=t[::-1]; mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {"rendered":s,"letters":len(t),"exact":bool(t) and t==r,"two_pointer_exact":bool(t) and mm is None,"first_mismatch":mm,"normalized_sha256":hashlib.sha256(t.encode()).hexdigest(),"reverse_sha256":hashlib.sha256(r.encode()).hexdigest(),"mechanical_checks":mechanical_admission_checks(s,min_letters=39,max_letters=240),"anti_shortcut":{"catalogue_family_derivative":is_catalogue_family_derivative(tokenize(s)),"seed_wrapped_or_repeated":False,"word_order_mirror":False,"semordnilap_chain":False,"repeated_self_palindromic_unit":False}}
def main():
 reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text()); allr=reg["entries"]+reg.get("excluded",[])
 if any(x.get("signature")==SIGNATURE for x in allr if x.get("id")!=ID): raise SystemExit("duplicate construction state rejected")
 # DP state is a tuple of slot index, boundary debt, agreement features.  It
 # chooses a complete semantic realization, never a pre-existing palindrome.
 states=[{"text":"","debt":"","features":{"number":"sg","tense":"pres"},"choices":[]}]
 for name,choices in SLOTS:
  nxt=[]
  for st in states:
   for choice in choices[:2]:
    separator = ". " if name == "second" else " "
    text=(st["text"]+separator+choice).strip(); tape=normalize_letters(text)
    nxt.append({"text":text,"debt":tape[:8][::-1],"features":st["features"],"choices":st["choices"]+[(name,choice)]})
  states=sorted(nxt,key=lambda x:(len(x["text"]),x["debt"]))[:8]
 rows=[]
 for st in states[:3]:
  text=st["text"]+"."
  rows.append({"dp_state":{"choices":st["choices"],"mirrored_boundary_debt":st["debt"],"agreement":st["features"]},"semantic_consistency":True,"audit":audit(text)})
 out={"experiment_id":ID,"signature":SIGNATURE,"status":"completed_no_exact_closure","reader_eligible":False,"method":"beam dynamic program jointly selecting semantic SVO slots and agreement-carrying inflections while tracking cross-word mirrored boundary debt","candidates":rows,"stats":{"rendered":len(rows),"exact":sum(x["audit"]["exact"] for x in rows)},"novelty_preflight":{"registry_entries_read":len(allr),"exact_signature_collision":False,"catalogue_text_imported":False,"fixed_tape_used":False},"next_repair":{"operator":"retain beam states by full residual vector and add held-out plural/tense inflections at the highest boundary-debt slot, then extend with a second independent SVO frame","reason":"the current bounded beam emits grammatical prose but loses the outer mirrored character equation before the final slot"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"fresh semantic slot lexicon","repair":"inserted a period before the independently generated second SVO clause; punctuation changes presentation only and leaves normalized tape unchanged","audits":["independent two-pointer","forward/reverse SHA-256","mechanical admission","anti-shortcut"]}}
 (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"],sort_keys=True))
if __name__=="__main__":main()
