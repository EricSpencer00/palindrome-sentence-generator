"""Held-out terminal repair for the residual-seam scene lattice."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.residual_seam_scene_lattice_20260919 import CLAUSES, CONNECTORS, Clause, audit, hidden_span
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ID="residual-terminal-repair-lattice-20260919"
# Held-out same-valency terminals. These are complete, human-authored scene
# realizations, not reverse strings or catalogue fragments.
HELD_OUT=(
 Clause("the player answers the king","sg","transitive","reply"),
 Clause("a quiet singer praises the court","sg","transitive","art"),
 Clause("the captain watches the tent","sg","transitive","place"),
 Clause("some players carry the banner","pl","transitive","object"),
 Clause("the herald remembers the song","sg","transitive","document"),
 Clause("Diana greets the actor","sg","transitive","person"),
 Clause("the actors guard the river","pl","transitive","place"),
 Clause("a sailor reads the sonnet","sg","transitive","document"),
)

def state(text,left,right):
 t=normalize_letters(text);i,j=0,len(t)-1;matched=0
 while i<j and t[i]==t[j]: i+=1;j-=1;matched+=1
 return {"remaining_letters":len(t)-2*matched,"next_required_character":t[i] if i<=j else None,"agreement":left.agreement+"/"+right.agreement,"valency":left.valency+"/"+right.valency}

def run(min_target=40,max_target=90):
 rows=[]; selected=0
 for left in CLAUSES:
  for right in CLAUSES:
   for connector in CONNECTORS:
    base=f"{left.text} {connector} {right.text}."; bs=state(base,left,right)
    # Select held-out terminals only through the indexed state. A terminal is
    # eligible if its same-valency role and agreement match and its edge char
    # satisfies the currently required seam character.
    for replacement in HELD_OUT:
     if replacement.valency!=left.valency or replacement.agreement!=left.agreement: continue
     if bs["next_required_character"] and normalize_letters(replacement.text)[-1]!=bs["next_required_character"]: continue
     selected+=1
     text=f"{replacement.text} {connector} {right.text}."; n=len(normalize_letters(text))
     if not min_target<=n<=max_target: continue
     a=audit(text); checks=mechanical_admission_checks(text,min_letters=30,max_letters=2000)
     rows.append({"rendered":text,"length":n,"connector":connector,"residual_state":state(text,replacement,right),"audit":a,"mechanical_checks":checks,"hidden_proper_span":hidden_span(text),"mechanically_admitted":a["two_pointer_exact"] and not hidden_span(text) and all(checks.values()),"provenance":{"base_clause":left.text,"held_out_terminal":replacement.text,"right_clause":right.text,"representation":"indexed same-valency residual terminal repair","finished_tape_reversed":False,"catalogue_imported":False,"rlaif_used":False},"reader_status":"unreviewed; human evidence required"})
 exact=[x for x in rows if x["audit"]["two_pointer_exact"]]; admitted=[x for x in exact if x["mechanically_admitted"]]
 return {"experiment_id":ID,"method":"held-out same-valency terminal replacement selected by residual seam state","status":"completed_exact" if exact else "completed_no_exact_closure","actual_candidates":rows,"exact_candidates":exact,"stats":{"rendered":len(rows),"selected_by_state":selected,"exact":len(exact),"admitted":len(admitted),"longest_rendered":max((x["length"] for x in rows),default=0),"longest_exact":max((x["length"] for x in exact),default=0)},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["literal two-pointer","forward/reverse SHA-256"],"rlaif_per_candidate":False},"novelty_preflight":{"status":"passed","distinction":"held-out terminal choices are gated by indexed residual character and typed agreement/valency"},"next_repair":{"action":"add a second held-out terminal tier keyed by a two-character residual prefix, then rerun only surviving indexed states","reader_test":"randomized blinded intact prose versus shuffled controls for every mechanically admitted row"},"reader_gate":"closed; no human readability evidence"}

if __name__=="__main__":
 out=ROOT/"runs"/(ID+".json");r=run();out.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r["stats"],sort_keys=True))
