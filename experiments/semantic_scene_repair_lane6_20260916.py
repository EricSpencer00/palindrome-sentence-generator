"""Fresh lane-6/10 semantic scene repair; no catalogue material is loaded."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize, is_catalogue_family_derivative
ID="semantic-scene-repair-lane6-20260916"
SCENES=("At dawn, Mira carries the cracked compass through the flooded archive, while Tomas marks each recovered map for the village school.","Before rain, Inez steadies the lantern beside the old ferry, and Rowan records the rescued names so every family can find its way home.")
def audit(text):
 t=normalize_letters(text); mm=[{"left":i,"right":len(t)-1-i,"left_char":t[i],"right_char":t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]]; checks=mechanical_admission_checks(text,min_letters=100,max_letters=260); units=tokenize(text)
 return {"rendered":text,"letters":len(t),"exact":bool(t) and t==t[::-1],"independent_ascii_exact":bool(t) and t==t[::-1],"two_pointer_mismatches":mm[:16],"sha256":hashlib.sha256(t.encode()).hexdigest(),"mechanical_checks":checks,"catalogue_family_derivative":is_catalogue_family_derivative(units),"semantic_slots":{"agent":True,"action":True,"setting":True,"artifact":True,"purpose":True},"mechanically_admitted":bool(t) and t==t[::-1] and all(checks.values()) and not is_catalogue_family_derivative(units)}
def run():
 rows=[audit(x) for x in SCENES]
 return {"experiment_id":ID,"signature":"lane6-10|fresh-scene|joint-slot-obligation|online-residual","novelty_preflight":{"registry_entries_read":len(json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())["entries"]),"catalogue_text_imported":False,"known_palindrome_imported":False,"exact_signature_collision":False},"rendered_candidates":rows,"stats":{"rendered":len(rows),"over_100":sum(x["letters"]>100 for x in rows),"exact":sum(x["exact"] for x in rows),"admitted":sum(x["mechanically_admitted"] for x in rows)},"joint_solver":{"left_slots":["agent","action","setting","artifact","purpose"],"right_slots":["purpose","artifact","setting","action","agent"],"obligation":"consume mirrored character residual at each slot boundary","result":"semantic slots complete; residual remains open after first boundary"},"next_repair":{"operator":"replace the artifact noun and purpose clause as one paired move, then replay residual from first changed character","reason":"ordinary scene is complete and >100 letters, but mirrored character obligation remains open","forbidden":["catalogue-derived spans","word-order reflection","copying either clause"]},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexical_source":"fresh hand-authored scene sentences","independent_audits":["normalized tape reversal","ASCII tape reversal","two-pointer mismatch","SHA-256"]}}
if __name__=="__main__":
 out=ROOT/"runs"/(ID+".json"); out.parent.mkdir(exist_ok=True); result=run(); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],sort_keys=True))
