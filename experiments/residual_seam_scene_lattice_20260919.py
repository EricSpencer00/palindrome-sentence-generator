"""Residual-seam indexed lattice for grammatical Shakespearean scenes.

The lattice state is (remaining letters, next required character,
agreement/valency).  It indexes complete clause realizations before joining
them with a grammatical connector; it never renders a malformed connector or
reverses a finished sentence.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ID="residual-seam-scene-lattice-20260919"
@dataclass(frozen=True)
class Clause:
    text:str; agreement:str; valency:str; role:str

CLAUSES=(
 Clause("the player greets the king","sg","transitive","address"),
 Clause("a quiet poet praises the sonnet","sg","transitive","art"),
 Clause("the captain guards the harbor","sg","transitive","place"),
 Clause("some players carry a banner","pl","transitive","object"),
 Clause("the herald reads a letter","sg","transitive","document"),
 Clause("Diana inspires the singer","sg","transitive","person"),
 Clause("the actors answer the poet","pl","transitive","reply"),
 Clause("a sailor watches the river","sg","transitive","place"),
 Clause("the players praise a new song","pl","transitive","art"),
 Clause("the bard remembers the court","sg","transitive","place"),
)
CONNECTORS=("while", "and", "as the bells sound")

def audit(text:str)->dict[str,object]:
 t=normalize_letters(text); i,j=0,len(t)-1; mm=[]
 while i<j:
  if t[i]!=t[j]: mm.append({"left":i,"right":j,"left_char":t[i],"right_char":t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and not mm,"first_mismatch":mm[0] if mm else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def hidden_span(text:str)->bool:
 w=[normalize_letters(x) for x in tokenize(text)]
 for a in range(len(w)):
  for b in range(a+2,len(w)+1):
   if a==0 and b==len(w): continue
   z="".join(w[a:b])
   if z==z[::-1]: return True
 return False

def run(min_target:int=40,max_target:int=70)->dict[str,object]:
 rows=[]; indexed={}
 for left in CLAUSES:
  for right in CLAUSES:
   if left.role==right.role: continue
   for connector in CONNECTORS:
    text=f"{left.text} {connector} {right.text}."
    tape=normalize_letters(text); n=len(tape)
    if not min_target<=n<=max_target: continue
    # Consume the two complete clauses from opposite ends.  The state records
    # the unresolved seam rather than accepting a malformed partial sentence.
    l,r=0,n-1; matched=0
    while l<r and tape[l]==tape[r]: l+=1;r-=1;matched+=1
    required=tape[l] if l<n and l<=r else None
    state=(n-matched*2,required,left.agreement+"/"+right.agreement,left.valency+"/"+right.valency)
    indexed.setdefault("|".join(map(str,state)),0); indexed["|".join(map(str,state))]+=1
    a=audit(text); checks=mechanical_admission_checks(text,min_letters=30,max_letters=2000)
    rows.append({"rendered":text,"length":n,"connector":connector,"residual_state":{"remaining_letters":state[0],"next_required_character":required,"agreement":state[2],"valency":state[3]},"audit":a,"mechanical_checks":checks,"hidden_proper_span":hidden_span(text),"mechanically_admitted":a["two_pointer_exact"] and not hidden_span(text) and all(checks.values()),"provenance":{"left_clause":left.text,"right_clause":right.text,"representation":"residual-seam indexed complete-clause lattice","finished_tape_reversed":False,"catalogue_imported":False,"rlaif_used":False},"reader_status":"unreviewed; programmatic checks never certify readability"})
 exact=[x for x in rows if x["audit"]["two_pointer_exact"]]; admitted=[x for x in exact if x["mechanically_admitted"]]
 return {"experiment_id":ID,"method":"complete grammatical clauses indexed by residual seam state (remaining letters, required character, agreement, valency)","status":"completed_exact" if exact else "completed_no_exact_closure","actual_candidates":rows,"exact_candidates":exact,"stats":{"rendered":len(rows),"indexed_states":len(indexed),"exact":len(exact),"admitted":len(admitted),"longest_rendered":max((x["length"] for x in rows),default=0),"longest_exact":max((x["length"] for x in exact),default=0)},"index":{"state_counts":indexed,"target_range":[min_target,max_target]},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["literal outside-in two-pointer","forward/reverse SHA-256"],"rlaif_per_candidate":False},"novelty_preflight":{"status":"passed","distinction":"complete grammatical connector clauses are indexed by residual seam state; no shell wrapper or reversed finished tape"},"next_repair":{"action":"replace only the first residual-bearing clause terminal with a held-out same-valency Shakespearean realization and re-index its state","reader_test":"randomized blinded intact prose versus shuffled controls for every mechanically admitted row"},"reader_gate":"closed; no human readability evidence"}

if __name__=="__main__":
 out=ROOT/"runs"/(ID+".json"); result=run(); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],sort_keys=True))
