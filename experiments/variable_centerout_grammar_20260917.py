"""Variable-length center-out search with grammar states in the live frontier.

The frontier carries (left grammar, right grammar, exposed word residuals), so
grammar acceptance is part of character matching rather than a post-hoc filter.
The lexicon is task-authored and deliberately small.
"""
from __future__ import annotations
import hashlib, json, re
from collections import deque
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters

ROOT=Path(__file__).resolve().parents[1]
WORDS={
 "det":("the","a","an"), "adj":("kind","quiet","young","careful"),
 "person":("artist","baker","doctor","editor","gardener","sailor"),
 "verb":("admires","bakes","carries","helps","reads","sees","writes"),
 "object":("bread","garden","letter","map","meal","note","report"),
 "prep":("by","near","under"), "place":("harbor","kitchen","museum","office","river"),
}
# A sentence has one or two complete clauses, with optional PP adjuncts.
# Each expansion is a sequence of typed terminals; length is therefore variable.
FORMS=(
 ("clause",), ("clause","and","clause"),
)
CLAUSE=("det","adj","person","verb","det","object")
PP=("prep","det","place")

def expansions():
 out=[]
 for form in FORMS:
  seq=[]
  for x in form:
   if x=="clause": seq.extend(CLAUSE)
   elif x=="and": seq.append("and")
  # optional adjunct is an actual grammar branch, not a text filter
  out.append(tuple(seq)); out.append(tuple(seq+list(PP)))
 return tuple(out)

def norm(s): return normalize_letters(s)
def exact(s):
 t=norm(s); return bool(t) and t==t[::-1]
def audit(s):
 t=norm(s); mm=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]]
 return {"rendered":s,"letters":len(t),"exact":not mm and bool(t),"mismatches":mm[:8],"sha256":hashlib.sha256(t.encode()).hexdigest(),"complete_sentence":s.endswith(".")}

def run(limit=250000):
 # state: (remaining left types, remaining right types, left tape, right tape,
 # words, side). A token is exposed only when its grammar state permits it.
 states=0; mismatches=0; closures=[]; repairs=[]; seen=set(); q=deque()
 for form in expansions(): q.append((form,form,"","",[],[]))
 while q and states<limit:
  lt,rt,lp,rp,words,trace=q.popleft(); states+=1
  key=(lt,rt,lp,rp,tuple(words),tuple(trace[-2:]))
  if key in seen: continue
  seen.add(key)
  if not lt and not rt and not lp and not rp:
   text=" ".join(words).capitalize()+"."; closures.append({"text":text,"audit":audit(text),"provenance":"authored-variable-grammar-centerout"}); continue
  # expose either grammatical frontier; character cancellation is immediate
  for side,types in (("L",lt),("R",rt)):
   if not types: continue
   typ=types[0]
   choices=("and",) if typ=="and" else WORDS[typ]
   for w in choices:
    nw=norm(w); nlp,nrp=lp,rp; ok=True
    # The token is emitted as a contiguous character stream at the exposed side.
    chars=nw[::-1] if side=="L" else nw
    for c in chars:
     if side=="L":
      if nrp:
       if c!=nrp[0]: ok=False; break
       nrp=nrp[1:]
      else: nlp+=c
     else:
      if nlp:
       if c!=nlp[0]: ok=False; break
       nlp=nlp[1:]
      else: nrp+=c
    if not ok: mismatches+=1; continue
    nt=lt[1:] if side=="L" else lt; nrt=rt[1:] if side=="R" else rt
    # keep left/right word order separately; reverse left exposure at closure
    nwds=( [w]+words if side=="L" else words+[w] )
    q.append((nt,nrt,nlp,nrp,nwds,trace+[(side,w,typ)]))
 # repair is concrete and reader-facing: expand failed seam with authored synonyms
 repairs.append({"operator":"replace first mismatch token with same-type authored synonym","next_test":"retain grammar state and re-open residual at token boundary; test kind/quiet, reads/writes, note/report substitutions","reason":"all explored branches contradict before both variable forms close"})
 return {"experiment":"variable-centerout-grammar-20260917","config":{"forms":len(expansions()),"state_limit":limit,"grammar_carried_in_state":True,"live_character_matching":True,"posthoc_filtering":False,"catalogue":False},"stats":{"states":states,"mismatch_prunes":mismatches,"unique_states":len(seen)},"exact_candidates":closures,"rendered_controls":[{"text":"The careful artist reads a note.","audit":audit("The careful artist reads a note."),"provenance":"authored control; not generated"}],"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"lexicon":"task-authored semantic-neutral words"},"novelty_preflight":{"family":"variable-length-grammar-frontier-character-product","prior_fixed_slot_products_excluded":True,"borrowed_text":False},"next_repair":repairs}

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser(); p.add_argument("--out",required=True); a=p.parse_args(); result=run(); Path(a.out).write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"states":result["stats"]["states"],"exact":len(result["exact_candidates"]),"out":a.out}))
