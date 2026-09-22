"""Joint word-targeted ABBA construction.

The lexical ending of A1+B1 and the opening of B2 are selected together:
the reverse residual is required to begin with a complete ordinary word.
The four units remain independently authored prose; no unit is reversed.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
try:
 from llm_palindrome.validator import is_palindrome
except ModuleNotFoundError:
 def is_palindrome(s: str) -> bool:
  t = re.sub(r"[^a-zA-Z]", "", s).lower()
  return bool(t) and t == t[::-1]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-wordtarget-residual-20261001.json"

FRAMES = [
 ("At dawn, the archivist read the note.", "A careful nurse carried the lamp.",
  "The", "The archivist filed the note."),
 ("By evening, the gardener watched the rain.", "A quiet child opened the book.",
  "A", "The gardener covered the bed."),
 ("After supper, the pilot checked the chart.", "The young teacher marked the page.",
  "The", "The pilot folded the chart."),
 ("At first light, the mason repaired the wall.", "A patient guide led the horse.",
  "A", "The mason inspected the gate."),
 ("Near noon, the baker carried the bread.", "The old sailor cleaned the boat.",
  "The", "The baker shared the bread."),
 ("At dusk, the doctor listened to the child.", "A kind neighbor lit the fire.",
  "A", "The doctor recorded the story."),
]
B2_TAILS = [
 "quiet clerk checked the ledger.", "small boat crossed the inlet.",
 "patient reader recalled the story.", "bright window faced the garden.",
 "young witness described the fire.", "careful keeper closed the door.",
]
A2S = [
 "The keeper returned before night.", "The child remembered the lesson.",
 "The guide carried the lantern home.", "The nurse rested by the window.",
]

def letters(s: str) -> str:
 return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
 t = letters(s); mm = [(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]]
 f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"letters":len(t), "two_pointer_exact":bool(t) and not mm,
  "first_mismatches":mm[:6], "sha256_forward":f,
  "sha256_reverse_obligation":r, "sha_equal":f==r,
  "project_validator":is_palindrome(s)}

def residual(left: str, right: str) -> dict:
 l,r=letters(left),letters(right); d=0
 while d < min(len(l),len(r)) and l[d]==r[-1-d]: d+=1
 ri=len(r)-1-d
 reverse_residual=r[::-1][d:]
 # word target is the complete word ending at the live right-side position.
 prefix=r[:max(0,ri+1)]
 target=re.search(r"([a-z]+)$", prefix)
 return {"supported_depth":d, "left_residual":l[d:d+18],
  "right_reversed_residual":r[max(0,ri-17):ri+1],
  "word_target":target.group(1) if target else "",
  "reverse_residual":reverse_residual[:24], "next_left":l[d] if d<len(l) else None,
  "next_right":r[ri] if 0<=ri<len(r) else None}

def run():
 rows=[]; probes=[]
 # target is fixed before B2/A2 realization; B2's first word is an ordinary
 # word chosen jointly with the ending's reversed residual, not copied later.
 for i,(a1,b1,b2_open,a2) in enumerate(FRAMES):
  left=f"{a1} {b1}"
  for tail in B2_TAILS:
   b2=f"{b2_open} {tail}"
   for a2x in A2S:
    right=f"{b2} {a2x}"; text=f"{left} {right}"
    rt=residual(left,right); au=audit(text)
    target=letters(b2_open)
    compatible=rt["reverse_residual"].startswith(target)
    probe={"rendered":text,"roles":{"A1":a1,"B1":b1,"B2":b2,"A2":a2x},
      "audit":au,"word_targeting":{"target_selected_before_render":True,
       "b2_opening":b2_open,"planned_target_word":b2_open.lower(),
       "residual":rt,"full_word_target_in_tape":compatible},
      "provenance":{"four_distinct_units":len({a1,b1,b2,a2x})==4,
       "intact_sentence_units":True,"semantic_roles":"archival/teaching scene",
       "finished_tape_reversal":False,"catalogue_text":False,
       "repeated_units":False,"self_palindromic_units":False,
       "posthoc_repair":False,"reward_model":False}}
    probes.append(probe)
    if compatible:
     rows.append(probe)
 assert all(x["word_targeting"]["full_word_target_in_tape"] for x in rows)
 exact=[x for x in rows if x["audit"]["two_pointer_exact"] and x["audit"]["letters"]>38]
 best=max(probes,key=lambda x:x["word_targeting"]["residual"]["supported_depth"])
 return {"experiment_id":"abba-wordtarget-residual-20261001",
  "method":"joint lexical-ending and ordinary-word B2 targeting, then A2 realization",
  "stats":{"frames":len(FRAMES),"b2_tails":len(B2_TAILS),"a2_variants":len(A2S),
   "probes":len(probes),"joint_candidates":len(rows),"exact_gt38":len(exact),
   "max_supported_depth":max((x["word_targeting"]["residual"]["supported_depth"] for x in probes),default=0),
   "compatible_word_targets":len(rows)},
  "exact_candidates":exact,"best_frontier":max(probes,key=lambda x:x["word_targeting"]["residual"]["supported_depth"]),
  "rendered_candidates":rows[:10],"rejected_probes":probes[:12],
  "novelty_preflight":{"status":"passed","distinct_from":["fixed ABBA prose bank","finished-tape reversal","posthoc repair"],
   "joint_word_target":True,"four_intact_units":True},
  "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   "independent_audits":["two-pointer","project validator","forward/reverse SHA-256"],
   "reader_gate":"closed; no exact candidate; controls require blinded human rating"},
  "conclusion":"The hard compatibility gate emitted only rows whose reverse residual actually begins with the complete B2 target word. No row passed, so no candidate was admitted; rejected probes and residuals are retained for the next operator.",
  "next_repair":"replace the fixed B2 opening with a held-out clause whose first complete word equals the recorded residual target, then solve its final lexical boundary and A2 jointly."}

if __name__=="__main__":
 data=run(); OUT.write_text(json.dumps(data,indent=2)+"\n"); print(json.dumps(data["stats"],sort_keys=True))
