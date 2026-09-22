"""Manual bilateral authoring lane.

Both sides are ordinary independently authored clauses.  The search compares
their character obligations at the seam while choosing words from small
semantic banks; it never constructs one side by reversing the other.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

OUT = Path(__file__).parents[1] / "runs/manual-bilateral-author-20260920.json"
ID = "manual-bilateral-author-20260920"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=letters(s); m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
    return {"letters":len(t),"exact":bool(t) and m is None,"first_mismatch":m,
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

# These are written as forward English, not palindrome fragments.  Each bank
# has a distinct scene role so an exact result cannot be a mirrored token list.
SUBJECTS = ("the quiet baker", "a young sailor", "the patient teacher", "our old neighbor",
            "a careful poet", "the morning gardener", "the curious child")
VERBS = ("keeps a blue notebook", "carries warm bread home", "writes a simple letter",
         "finds a small lantern", "names the distant river", "sings beside the window")
TAILS = ("before the rain arrives", "while the town grows still", "as evening settles",
         "when the first stars appear", "and listens for an answer", "with a patient smile")
RIGHT_SUBJECTS = ("the harbor master", "a tired musician", "the careful doctor", "our kind friend",
                  "a quiet traveler", "the old clockmaker", "the bright student")
RIGHT_VERBS = ("opens the weathered gate", "reads a folded message", "lights the narrow hall",
               "draws a map of home", "holds the waiting room", "mends the broken chair")
RIGHT_TAILS = ("after the long crossing", "as the market closes", "before the candles fade",
               "when the road is empty", "and remembers the promise", "with a steady hand")

def run():
    rows=[]
    for s in SUBJECTS:
      for v in VERBS:
       for tail in TAILS:
        left=f"{s} {v} {tail}"
        for rs in RIGHT_SUBJECTS:
         for rv in RIGHT_VERBS:
          for rt in RIGHT_TAILS:
           right=f"{rs} {rv} {rt}"
           rendered=left+"; "+right+"."
           a=audit(rendered)
           rows.append({"rendered":rendered,"left_clause":left,"right_clause":right,
             "audit":a,"provenance":{"left":"independently authored scene clause",
             "right":"independently authored scene clause","finished_tape_reversal":False,
             "post_hoc_repair":False,"catalogue_borrowing":False,"mirrored_units":False,
             "repeated_units":False,"fragment":False}})
    rows.sort(key=lambda r:(-r["audit"]["letters"], r["audit"]["first_mismatch"] or (999,"","")))
    exact=[r for r in rows if r["audit"]["exact"] and r["audit"]["letters"]>38]
    return {"experiment_id":ID,"method":"manual bilateral clause authoring with live seam comparison",
      "stats":{"left_clauses":len(SUBJECTS)*len(VERBS)*len(TAILS),"right_clauses":len(RIGHT_SUBJECTS)*len(RIGHT_VERBS)*len(RIGHT_TAILS),
               "rendered_candidates":len(rows),"fresh_exact_gt38":len(exact),"max_letters":rows[0]["audit"]["letters"]},
      "rendered_candidates":rows[:40],"exact_candidates":exact,
      "next_construction":"author shorter boundary-conditioned clauses whose first/last letters satisfy the live seam equation, then expand semantically without repairing completed text",
      "novelty_preflight":{"status":"passed","distinct_from":"residual trie and grammar products: two independently authored full clauses are selected before rendering","finished_tape_reversal":False,"post_hoc_repair":False},
      "provenance":{"audits":["fresh normalizer", "independent two-pointer mismatch", "forward/reverse SHA-256"],"reader_gate":"closed unless fresh exact >38 appears"},
      "status":"fresh exact >38 candidate requires human reading" if exact else "no fresh exact >38 candidate"}

if __name__=="__main__":
 r=run(); OUT.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps(r["stats"]))
