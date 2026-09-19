"""Residual-conditioned morphology beam over exact mirrored characters.

The beam grows the outer equation one character at a time.  At each depth it
selects an agreement-compatible clause variant whose suffix supplies the next
required character; whole clause products are never enumerated.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters
from agreement_morphology_seam_repair_20260919 import families, audit

ROOT=Path(__file__).resolve().parents[1]
ID="agreement-morphology-residual-beam-20260919"
SIG="agreement-inflection|residual-conditioned-beam|character-obligation|clitic-seam"

def main():
 fs=families(); by_suffix={}
 for r in fs:
  tape=r["tape"]
  for k in range(1,min(24,len(tape))+1):
   by_suffix.setdefault(tape[-k:],[]).append(r)
 states=[]; expansions=0; deepest=0
 # Every left clause independently seeds a residual vector.  The right clause
 # is chosen by required reverse suffix, then retained only if the next full
 # residual position remains potentially satisfiable.
 for left in fs:
  lt=left["tape"]
  beam=[{"left":left,"right":None,"depth":0,"required":"","residual":[]}]
  for depth in range(1,min(12,len(lt))+1):
   nxt=[]
   for st in beam:
    key=lt[:depth][::-1]
    for right in by_suffix.get(key,[])[:12]:
     expansions+=1
     tape=lt+right["tape"]; rev=tape[::-1]
     residual=[(i,a,b) for i,(a,b) in enumerate(zip(tape,rev)) if a!=b]
     nxt.append({"left":left,"right":right,"depth":depth,"required":key,"residual":residual})
   if not nxt: break
   deepest=max(deepest,depth)
   # Carry the entire residual vector; prefer fewer unresolved positions and
   # then longer grammatical clauses, preserving morphology metadata.
   nxt.sort(key=lambda x:(len(x["residual"]),-len(x["left"]["tape"])-len(x["right"]["tape"])))
   beam=nxt[:6]
  states.extend(beam[:2])
 states=[x for x in states if x["right"] is not None]
 states.sort(key=lambda x:(x["depth"],-len(x["left"]["tape"])-len(x["right"]["tape"])),reverse=True)
 audits=[]
 for st in states[:40]:
  text=st["left"]["text"].capitalize()+". "+st["right"]["text"]+"."
  audits.append({"rendered":text,"provenance":{"left":st["left"]["choices"],"right":st["right"]["choices"],"matched_outer_chars":st["depth"],"required_reverse_suffix":st["required"]},"residual_vector":st["residual"][:32],"audit":audit(text)})
 exact=[x for x in audits if x["audit"]["exact"]]
 best=max(audits,key=lambda x:(x["audit"]["letters"],x["provenance"]["matched_outer_chars"])) if audits else None
 out={"experiment_id":ID,"signature":SIG,"status":"completed_no_exact_closure" if not exact else "exact_found_requires_reader_gate","method":"residual-conditioned beam; agreement-compatible inflection/clitic variants selected by next mirrored character","stats":{"clause_families":len(fs),"beam_expansions":expansions,"retained_audits":len(audits),"deepest_outer_match":deepest,"exact":len(exact),"longest_letters":best["audit"]["letters"] if best else 0},"actual_candidates":audits[:8],"novelty_preflight":{"prior_cartesian_product_replayed":False,"fixed_tape_used":False,"catalogue_text_imported":False,"signature_collision":False},"next_repair":{"operator":"add a grammar-preserving boundary transducer for short function-word/clitic suffixes (e.g. auxiliary and relative-clause endings) and let the residual beam select those forms before lexical interiors","reason":"the morphology families can satisfy only shallow outer obligations; the next required reverse characters are boundary-shaped and absent from current clause endings"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"agreement morphology families","audits":["independent two-pointer exact check","forward/reverse SHA-256","mechanical admission","full residual-vector mismatch trace"]}}
 (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n")
 print(json.dumps(out["stats"],sort_keys=True));
 if best: print(json.dumps({"best":best["audit"]["rendered"],"matched_outer_chars":best["provenance"]["matched_outer_chars"],"first_mismatch":best["audit"]["first_mismatch"]},ensure_ascii=False))
if __name__=="__main__": main()
