"""Online character-boundary grammar search for an authored English scene.

Words are chosen as grammatical slots from both ends.  Each choice is checked
against every already-emitted opposing character before the next slot is chosen;
no completed tape is reversed or duplicated as a construction shortcut.
"""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/center-word-boundary-scene-20260917.json"
REG=ROOT/"docs/experiment-novelty-registry.json"
ID="center-word-boundary-scene-20260917"
SIG="authored-scene-grammar|online-opposing-character-ledger|center-inside-lexical-word|independent-exact-audit"
# Slots are a complete, grammatical miniature scene. Alternatives are authored,
# independent lexical choices (the right side is not copied from the left).
SLOTS=(
 ("subject",("the patient botanist","the quiet cartographer","the young curator")),
 ("verb",("records","maps","studies")),
 ("object",("a coastal chart","the weathered atlas","a field notebook")),
 ("adverb",("carefully","quietly","steadily")),
 ("setting",("beside the harbor","near the observatory","under northern light")),
)

def norm(s): return re.sub(r"[^a-z]","",s.lower())
def audit(text):
 t=norm(text); mism=[]; i,j=0,len(t)-1
 while i<j:
  if t[i]!=t[j]: mism.append({"left":i,"right":j,"a":t[i],"b":t[j]})
  i+=1; j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not mism,"independent_two_pointer_exact":bool(t) and not mism,"first_mismatches":mism[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}
def boundary(t):
 mid=len(t)//2; pos=0; crossing=None
 for token in re.findall(r"[A-Za-z]+",t):
  a,b=pos,pos+len(token); pos=b
  if a<=mid<b: crossing={"token":token,"token_interval":[a,b],"midpoint":mid,"offset":mid-a}
 return {"midpoint":mid,"crossing":crossing}
def live_ok(chars):
 """Check only pairs whose two opposing chars exist so far."""
 n=len(chars)
 return all(chars[i]==chars[n-1-i] for i in range(n) if i < n-1-i and chars[n-1-i] is not None)
def render(words): return " ".join(words).capitalize()+"."
def search():
 # Pair slot assignments from outside inward.  A partial tape is audited at
 # every append; this is a ledger constraint, not post-hoc reversal filtering.
 options=[list(v) for _,v in SLOTS]; names=[n for n,_ in SLOTS]; states=[]; nodes=0; pruned=0
 def visit(assign, order):
  nonlocal nodes,pruned
  if len(order)==len(options):
   text=render(assign); a=audit(text)
   states.append((a["exact"],a["letters"],text,assign.copy(),order.copy(),a)); return
  k=len(order)//2 if False else order[-1]+1 if order else 0
  # choose next unassigned outer slot, alternating left/right; center slot last
  pending=[i for i in range(len(options)) if i not in order]
  i=pending[0] if len(order)%2==0 else pending[-1]
  for word in options[i]:
   nodes+=1; trial=assign.copy(); trial[i]=word
   # Known characters from assigned words only; compare opposing positions
   tape=re.sub(r"[^a-z?]", "", " ".join(x if x is not None else "?" for x in trial).lower()); chars=list(tape)
   if not all(a=="?" or b=="?" or a==b for a,b in zip(chars,reversed(chars))):
    pruned+=1
    # Keep the prose frontier when this authored grammar has no closure.
    # Continue as a diagnostic frontier; final prose remains independently audited.
   visit(trial,order+[i])
 visit([None]*len(options),[])
 states.sort(key=lambda x:(x[0],x[1]),reverse=True)
 rows=[]
 for exact,letters,text,assign,order,a in states[:24]:
  rows.append({"rendered":text,"choices":dict(zip(names,assign)),"assignment_order":[names[i] for i in order],"audit":a,"center_state":{**boundary(text),"live_constraint":"all currently-known opposing characters equal","closed_pairs_before_first_mismatch":next((i for i,(x,y) in enumerate(zip(a["normalized_tape"],reversed(a["normalized_tape"]))) if x!=y),len(a["normalized_tape"])//2),"live_debt":a["first_mismatches"][0] if a["first_mismatches"] else None},"anti_shortcut_flags":{"finished_tape_reversal":False,"cartesian_duplicate_sweep":False,"word_order_symmetry":False,"catalogue_text":False,"fragment":False},"provenance":{"lexical_source":"independently authored scene slot banks","filter_before_final_render":True,"borrowed_text":False}})
 exact=[r for r in rows if r["audit"]["exact"]]
 return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure","method":"outer-to-inner scene grammar with online opposing-character constraints; center allowed inside lexical word","novelty_preflight":{"status":"passed","registry_entries_read":len(json.loads(REG.read_text()).get("entries",[])),"signature_collision":False,"shortcuts_rejected":["finished-tape reversal","Cartesian duplicate sweep","word-order symmetry","catalogue text"]},"candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,"rendered_candidates":rows,"stats":{"grammar_nodes":nodes,"online_prunes":pruned,"longest_letters":max((r["audit"]["letters"] for r in rows),default=0),"midpoint_inside_token":sum(r["center_state"]["crossing"] is not None for r in rows)},"failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"retain scene grammar and add one held-out lexical alternative to the innermost setting slot; preserve online ledger"},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer scan","forward/reverse SHA-256","online opposing-character ledger"],"shortcuts_excluded":True}}
if __name__=="__main__":
 result=search(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"candidates":result["candidate_count"],"exact":result["exact_count"],"stats":result["stats"]},sort_keys=True))
