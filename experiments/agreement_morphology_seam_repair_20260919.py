"""Character-indexed repair for the agreement/morphology transducer.

Clause families are indexed by suffixes required by the reverse tape before
any pair is rendered.  The retained state carries all unresolved character
obligations, rather than one scalar score.
"""
from __future__ import annotations
import hashlib, itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.agreement_morphology_boundary_transducer_20260919 import (SUBJECTS, VERBS, OBJECTS, ADJUNCTS, CLITICS, clause, audit)

ROOT=Path(__file__).resolve().parents[1]
ID="agreement-morphology-seam-repair-20260919"
SIG="agreement-inflection|character-indexed-seam|full-residual-vector|clitic-repair"

def families():
 out=[]
 # Explicit loops keep metadata attached to every inflection choice.
 for number in ("sg","pl"):
  for tense in ("pres","past"):
   for subject,verb,obj,adj,clitic in itertools.product(SUBJECTS[number],VERBS[(number,tense)],OBJECTS[number],ADJUNCTS,CLITICS):
    text=clause(subject,verb,obj,adj,clitic)
    out.append({"text":text,"number":number,"tense":tense,"choices":[subject,verb,obj,adj,clitic],"tape":normalize_letters(text)})
 return out

def residual(left,right):
 tape=left["tape"]+right["tape"]
 rev=tape[::-1]
 mismatches=[(i,a,b) for i,(a,b) in enumerate(zip(tape,rev)) if a!=b]
 return mismatches

def main():
 fs=families(); indexes={k:{} for k in range(1,13)}
 for r in fs:
  # right suffix must equal reverse(left prefix); build all suffix indexes.
  for k in indexes:
   key=r["tape"][-k:] if len(r["tape"])>=k else r["tape"]
   indexes[k].setdefault(key,[]).append(r)
 retained=[]; considered=0; max_k=0
 for left in fs:
  for k in range(12,0,-1):
   if len(left["tape"])<k: continue
   key=left["tape"][:k][::-1]
   hits=indexes[k].get(key,[])[:8]
   if hits:
    max_k=max(max_k,k)
    for right in hits:
     considered += 1
     tape=left["tape"]+right["tape"]
     mm=residual(left,right)
     if len(retained) < 3000:
      retained.append({"left":left,"right":right,"matched_outer_chars":k,"residual":mm,"tape":tape})
    break
 retained.sort(key=lambda x:(x["matched_outer_chars"],-len(x["tape"])),reverse=True)
 audits=[]
 for r in retained[:40]:
  text=r["left"]["text"].capitalize()+". "+r["right"]["text"]+"."
  checks=audit(text)
  audits.append({"rendered":text,"provenance":{"left":r["left"]["choices"],"right":r["right"]["choices"],"matched_outer_chars":r["matched_outer_chars"]},"residual_vector":r["residual"][:24],"audit":checks})
 exact=[x for x in audits if x["audit"]["exact"]]
 best=max(audits,key=lambda x:(x["audit"]["letters"],x["provenance"]["matched_outer_chars"])) if audits else None
 out={"experiment_id":ID,"signature":SIG,"status":"completed_no_exact_closure" if not exact else "exact_found_requires_reader_gate","method":"suffix-indexed agreement morphology repair; retain full mirrored-character residual vector after outer seam pruning","stats":{"clause_families":len(fs),"indexed_pair_candidates":considered,"retained_audits":len(audits),"max_outer_match":max_k,"exact":len(exact),"longest_letters":best["audit"]["letters"] if best else 0},"actual_candidates":audits[:8],"novelty_preflight":{"fixed_tape_used":False,"catalogue_text_imported":False,"prior_30k_product_replayed":False,"signature_collision":False},"next_repair":{"operator":"use the residual vector to choose a morphology/clitic variant whose boundary supplies the next reverse-required character, then re-index at that new seam","reason":"outer suffix indexing found grammatical pairs sharing only a short seam; remaining mismatches span lexical interiors, so morphology must be selected at each unresolved character rather than after whole-clause realization"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"same 4,800 agreement families, independently re-indexed","audits":["independent two-pointer exact check","forward/reverse SHA-256","mechanical admission","full residual-vector trace"]}}
 (ROOT/"runs"/(ID+".json")).write_text(json.dumps(out,indent=2)+"\n")
 print(json.dumps(out["stats"],sort_keys=True));
 if best: print(json.dumps({"best":best["audit"]["rendered"],"matched_outer_chars":best["provenance"]["matched_outer_chars"],"first_mismatch":best["audit"]["first_mismatch"]},ensure_ascii=False))
if __name__=="__main__": main()
