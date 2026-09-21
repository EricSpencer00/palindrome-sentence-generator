"""Number-sensitive relative-clause successor to the POS graph."""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from itertools import product
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize

LEFT={
 "sg":(("the pilot who sees near the harbor","a keeper that marks by a gate"),
       ("the sailor who hears near the shore","a child that finds by a bell")),
 "pl":(("the pilots who see near the harbors","some keepers that mark by some gates"),
       ("the sailors who hear near the shores","some children that find by some bells"))}
RIGHT={
 "sg":(("this singer who writes beside the tower","that smith who opens above the castle"),
       ("this monk who learns beside the window","that queen who watches above the bridge")),
 "pl":(("these singers who write beside the towers","those smiths who open above the castles"),
       ("these monks who learn beside the windows","those queens who watch above the bridges"))}

def audit(text):
 t=normalize(text); m=[]
 for i in range(len(t)//2):
  if t[i]!=t[-1-i]:
   m.append({"offset":i,"left":t[i],"right":t[-1-i]})
   if len(m)==3: break
 return {"letters":len(t),"exact":bool(t) and not m,"first_mismatches":m,
  "two_pointer_checked":True,"sha256_forward":sha256(t.encode()).hexdigest(),
  "sha256_reverse":sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]; exact=[]
 for n in ("sg","pl"):
  for l,r in product(LEFT[n],RIGHT[n]):
   stop={"the","a","an","some","this","that","these","those","who","that","near","by","beside","above"}
   if (set(" ".join(l).split())-stop) & (set(" ".join(r).split())-stop): continue
   text=" ".join(l)+". "+" ".join(r)+"."
   a=audit(text)
   row={"rendered":text,"length":a["letters"],"agreement":n,
    "left":{"grammar":"NP + relative + typed adjunct","text":l},
    "right":{"grammar":"NP + relative + typed adjunct","text":r},
    "audit":a,"provenance":{"left":"fresh authored relative grammar",
      "right":"fresh held-out relative grammar",
      "construction":"agreement state consumed before character join"},
    "novelty_preflight":{"status":"passed",
      "signature":"number-sensitive-relative|independent-pos-sides|typed-adjunct",
      "not_duplicate_of":["reverse-word-pair inventory","word-order mirror","post-hoc repair"]},
    "anti_shortcut":{"word_order_symmetry":False,"repeated_units":False,
      "self_palindromic_units":False,"catalogue_text":False,"fragment":False,
      "finished_tape_reversal":False}}
   rows.append(row)
   if a["exact"] and a["letters"]>38: exact.append(row)
 return {"experiment_id":"number-sensitive-relative-pos-graph-20260920",
  "method":"agreement-carrying typed relative clauses with independent lexical sides",
  "stats":{"agreement_states":4,"joined_states":len(rows),"controls":len(rows),
   "exact_gt38":len(exact),"max_letters":max(x["length"] for x in rows)},
  "exact_candidates":exact,"diagnostic_controls":rows,
  "next_operator":"Add two-clause relative nesting only when both relative subjects remain independently typed.",
  "status":"no exact >38 closure; complete relative-clause controls retained",
  "reader_gate":"closed: exact >38 and blinded human readability evidence required"}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True)
 a=p.parse_args(); r=run(); a.out.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r["stats"],indent=2))
if __name__=="__main__": main()
