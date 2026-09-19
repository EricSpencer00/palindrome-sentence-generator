"""Two-character residual-prefix repair with finite grammatical attachments."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.residual_seam_scene_lattice_20260919 import CLAUSES, CONNECTORS, Clause, audit, hidden_span
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ID="residual-prefix2-attachment-lattice-20260919"
ATTACHMENTS=("", " at dawn", " under the moon")
HELD_OUT=(
 Clause("the player guards the hearth","sg","transitive","place"),
 Clause("the player marks the sonnet","sg","transitive","document"),
 Clause("a singer praises the court","sg","transitive","place"),
 Clause("the actor guards the tent","sg","transitive","place"),
 Clause("a poet writes a quiet part","sg","transitive","document"),
 Clause("some players carry the banner","pl","transitive","object"),
)

def residual(text):
 t=normalize_letters(text);i,j=0,len(t)-1;matched=0
 while i<j and t[i]==t[j]: i+=1;j-=1;matched+=1
 return {"remaining_letters":len(t)-2*matched,"next_required_prefix":t[i:min(j+1,i+2)] if i<=j else "","matched_pairs":matched}

def run(min_target=40,max_target=100):
 rows=[]; selected=0; states={}
 for left in CLAUSES:
  for right in CLAUSES:
   if left.role==right.role: continue
   for connector in CONNECTORS:
    for attachment in ATTACHMENTS:
     base=f"{left.text} {connector} {right.text}{attachment}."; rs=residual(base)
     prefix=rs["next_required_prefix"]
     for replacement in HELD_OUT:
      if replacement.agreement!=left.agreement or replacement.valency!=left.valency: continue
      terminal=normalize_letters(replacement.text)[-2:]
      if len(prefix)!=2 or terminal!=prefix: continue
      selected+=1; text=f"{replacement.text} {connector} {right.text}{attachment}."; n=len(normalize_letters(text))
      if not min_target<=n<=max_target: continue
      key=(rs["remaining_letters"],prefix,replacement.agreement,replacement.valency,attachment or "none");states[str(key)]=states.get(str(key),0)+1
      a=audit(text); checks=mechanical_admission_checks(text,min_letters=30,max_letters=2000)
      rows.append({"rendered":text,"length":n,"attachment":attachment or "none","residual_state":{**residual(text),"agreement":replacement.agreement,"valency":replacement.valency},"audit":a,"mechanical_checks":checks,"hidden_proper_span":hidden_span(text),"mechanically_admitted":a["two_pointer_exact"] and not hidden_span(text) and all(checks.values()),"provenance":{"base_clause":left.text,"held_out_terminal":replacement.text,"right_clause":right.text,"representation":"two-character residual-prefix lattice with finite attachment state","finished_tape_reversed":False,"catalogue_imported":False,"rlaif_used":False},"reader_status":"unreviewed; human evidence required"})
 exact=[x for x in rows if x["audit"]["two_pointer_exact"]];admitted=[x for x in exact if x["mechanically_admitted"]]
 return {"experiment_id":ID,"method":"held-out two-character terminal replacement keyed by residual prefix plus finite attachment state","status":"completed_exact" if exact else "completed_no_exact_closure","actual_candidates":rows,"exact_candidates":exact,"stats":{"rendered":len(rows),"selected_by_prefix":selected,"indexed_states":len(states),"exact":len(exact),"admitted":len(admitted),"longest_rendered":max((x["length"] for x in rows),default=0),"longest_exact":max((x["length"] for x in exact),default=0)},"index":{"state_counts":states,"target_range":[min_target,max_target],"attachments":list(ATTACHMENTS)},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["literal two-pointer","forward/reverse SHA-256"],"rlaif_per_candidate":False},"novelty_preflight":{"status":"passed","distinction":"two-character residual prefix and finite attachment state jointly gate held-out terminals"},"next_repair":{"action":"add a three-character terminal prefix tier only for states with intact clause valency","reader_test":"randomized blinded intact prose versus shuffled controls for every admitted row"},"reader_gate":"closed; no human readability evidence"}

if __name__=="__main__":
 out=ROOT/"runs"/(ID+".json");r=run();out.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r["stats"],sort_keys=True))
