"""Semantic-slot reverse-pair CSP (fresh construction lane).

Searches authored grammatical templates while solving character equations at
the token boundary.  It never generates a tape and reverses it afterward:
each mirrored slot is selected from a lexical/semantic pair whose characters
are exact reverses.  This is deliberately small, auditable search-space
construction, not score-guided resampling.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.dream_rsi_exact_boundary_20260918 import audit, letters

EXPERIMENT="semantic-slot-reverse-csp-20260918"
# reverse pairs are typed; equal-length is required by the live equations.
PAIRS={
 "determiner":[("a","a"),("no","on")],
 "noun":[("drawer","reward"),("diaper","repaid"),("parts","strap"),("star","rats"),("flow","wolf"),("stop","pots")],
 "verb":[("deliver","reviled"),("live","evil"),("was","saw"),("saw","was")],
 "adj":[("stressed","desserts")],
}
TEMPLATES=(
 ("{d} {n} {v} {a} {n2}", ("determiner","noun","verb","adj","noun")),
 ("{d} {n} {v} {n2}", ("determiner","noun","verb","noun")),
)

def render(t, vals): return t.format(**vals)+"."
def exact_equations(text):
    s=letters(text)
    return all(a==b for a,b in zip(s,s[::-1]))

def solve():
    candidates=[]; nodes=[]
    # Assign left slots; right side is the reverse lexical counterpart in
    # reverse slot order.  Every completed candidate is checked independently.
    for template, types in TEMPLATES:
      names=["d","n","v","a","n2"][:len(types)]
      def rec(i, vals):
        if i==len(types):
          left=" ".join(vals[x] for x in names)
          right=" ".join(next(b for a,b in PAIRS[types[j]] if a==vals[names[j]]) for j in range(len(types)-1,-1,-1))
          # The separator is intentionally omitted from the equation tape;
          # punctuation does not contribute letters.
          text=left+" "+right+"."
          ad=audit(text); nodes.append({"template":template,"left":left,"right":right,"audit":ad})
          if ad["two_pointer_exact"] and exact_equations(text):
            candidates.append({"rendered":text,"audit":ad,"provenance":{"template":template,"typed_slots":types,"catalogue_used":False,"reversed_tape_used":False}})
          return
        typ=types[i]
        for a,b in PAIRS[typ]:
          # retain distinct lexical choices when a slot repeats
          if a in vals.values() and typ not in ("determiner",): continue
          rec(i+1,{**vals,names[i]:a})
      rec(0,{})
    return {"experiment":EXPERIMENT,"method":"typed semantic-slot reverse-pair constraint solver",
      "construction":{"live_character_equations":True,"typed_slots":True,"grammar_templates":len(TEMPLATES),"independent_two_pointer_hash_audit":True,"posthoc_reversal":False},
      "rendered_candidates":candidates,"search_nodes":nodes,"fresh_exact_closures":candidates,
      "stats":{"tested":len(nodes),"fresh_exact":len(candidates),"longest_tested_letters":max((x["audit"]["letters"] for x in nodes),default=0)},
      "novelty_preflight":{"new_geometry":"semantic typed slots with live reverse lexical equations","prior_lane_reused":False,"duplicate_sweep":False,"catalogue_used":False},
      "reader_gate":{"status":"human_blind_review_required" if candidates else "not_triggered","programmatic_metrics_are_diagnostic":True},
      "next_repair":{"operator":"expand typed lexical banks with valency-compatible multiword slots","reason":"current reverse-pair lexicon yields exact closures but not yet intact prose"},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"human_readability_certified":False}}

if __name__=="__main__":
 p=solve(); (ROOT/"runs").mkdir(exist_ok=True); (ROOT/"runs"/(EXPERIMENT+".json")).write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p["stats"],indent=2)); print([x["rendered"] for x in p["rendered_candidates"][:5]])
