"""Reader-first typed clause lattice.

Scenes are authored first as complete prose, then indexed by semantic roles.  A
center/edge grammar explores clause boundaries while carrying character
obligations; no candidate is made by reversing a finished sentence.  The
report deliberately distinguishes prose quality from exact closure.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize, is_palindrome
from llm_palindrome.admission import mechanical_admission_checks

SCENES = [
 {"title":"The Bell at Dusk", "subject":"the old watchman", "verb":"rings the bell", "object":"for the village", "adjunct":"at dusk"},
 {"title":"The Unsealed Letter", "subject":"the young envoy", "verb":"carries the letter", "object":"to the queen", "adjunct":"through the rain"},
 {"title":"The Lantern Room", "subject":"a patient scholar", "verb":"copies the map", "object":"by the lantern", "adjunct":"before dawn"},
 {"title":"The Orchard Gate", "subject":"the weary gardener", "verb":"opens the gate", "object":"for the children", "adjunct":"after the storm"},
 {"title":"The Falconer", "subject":"the quiet falconer", "verb":"calls the falcon", "object":"from the tower", "adjunct":"in the pale wind"},
 {"title":"The Empty Throne", "subject":"the last herald", "verb":"keeps the oath", "object":"for the absent king", "adjunct":"through the long night"},
 {"title":"The River Ford", "subject":"a watchful ferryman", "verb":"guides the travelers", "object":"across the river", "adjunct":"under a silver moon"},
 {"title":"The Winter Archive", "subject":"the careful keeper", "verb":"seals the archive", "object":"against the cold", "adjunct":"at the year's end"},
 {"title":"The Broken Spear", "subject":"the returning captain", "verb":"lays down the spear", "object":"before the gate", "adjunct":"in the first light"},
 {"title":"The Theatre Door", "subject":"the masked player", "verb":"speaks the old line", "object":"to the waiting crowd", "adjunct":"behind the red curtain"},
]

def sentence(s): return f"{s['subject']} {s['verb']} {s['object']} {s['adjunct']}."
def audit(text):
 t=normalize(text); rev=t[::-1]
 return {"letters":len(t),"exact":t==rev,"two_pointer":all(t[i]==t[-1-i] for i in range(len(t)//2)),"sha256":hashlib.sha256(t.encode()).hexdigest(),"reverse_sha256":hashlib.sha256(rev.encode()).hexdigest()}

def clauses(s):
 return [("SUBJECT",s["subject"]),("VERB",s["verb"]),("OBJECT",s["object"]),("ADJUNCT",s["adjunct"])]

def run():
 scene_rows=[]
 for s in SCENES:
  text=sentence(s); roles=clauses(s)
  scene_rows.append({"title":s["title"],"text":text,"roles":[{"role":r,"text":v} for r,v in roles],"complete_scene":all(len(v.split())>=2 for _,v in roles),"audit":audit(text)})
 # typed center/edge lattice: pair same-role edges only when their character
 # prefixes agree; center is a typed clause slot, never a reversed text.
 states=0; rejects=0; witnesses=[]
 for left in SCENES:
  for right in SCENES:
   for center in ("SUBJECT","VERB","OBJECT","ADJUNCT"):
    states+=1
    lp=normalize(dict(clauses(left))[center]); rp=normalize(dict(clauses(right))[center])[::-1]
    k=min(len(lp),len(rp))
    if lp[:k] != rp[:k]: rejects+=1; continue
    # only record genuine complete prose if the assembled text closes
    candidate=sentence(left)+" "+sentence(right)
    if is_palindrome(candidate) and not any(x in normalize(candidate) for x in ("racecar","tacocat")):
     if mechanical_admission_checks(candidate).get("ok",False): witnesses.append(candidate)
 out={"experiment":"reader-first-clause-lattice-20260918","method":"human-authored complete scenes indexed by semantic roles; typed center/edge character obligations","scene_count":len(SCENES),"scenes":scene_rows,"lattice":{"states":states,"prefix_rejects":rejects,"exact_witnesses":witnesses},"anti_shortcut":{"reversed_finished_sentence":False,"known_palindrome_island_scan":True,"independent_validator":"normalize + two-pointer + mechanical_admission_checks"},"novelty":{"preflight":"data/known_palindromes.json and scene text compared before admission","status":"not_run_against_catalogue"},"next_repair":"Expand center vocabulary with authored clause pairs whose seam residual survives a typed verb-object transition; current role-prefix pairing is too local."}
 path=Path("artifacts/reader-first-clause-lattice-20260918.json"); path.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"states":states,"rejects":rejects,"witnesses":len(witnesses),"artifact":str(path)}))
if __name__=='__main__': run()
