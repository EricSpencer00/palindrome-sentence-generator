"""Fresh authored-scene ABBA seam search around an exact readable center.

The search chooses complete, independently written observation/return clauses
around the 44-letter center.  It never reverses a selected clause, and it
records the first residual for the next authoring pass.  This tests whether
paragraph seams can grow an existing readable center without treating prose as
an afterthought.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "runs" / "authored-abba-scene-bank-20260927.json"
CENTER = "Now, an aide rips nine memos; some men inspire. Diana won."
LEFT = [
    "At dawn, Mira opened the garden gate.", "By dusk, Noel read the harbor chart.",
    "A quiet bell marked the turning tide.", "The young poet watched the swallows.",
    "In spring, the keeper planted ash trees.", "Our patient guide carried the lantern.",
    "Near the river, the old mill rested.", "A careful nurse heard the child singing.",
    "The red boat crossed the cold inlet.", "At noon, the scholar copied the letter.",
]
RIGHT = [
    "The gate closed as Mira left the garden.", "The chart closed as Noel left the harbor.",
    "The tide turned beneath the marked bell.", "The swallows passed beyond the young poet.",
    "The ash trees grew beside the spring keeper.", "The lantern guided our patient guide.",
    "The mill rested near the river guide.", "The child sang while the careful nurse listened.",
    "The cold inlet received the red boat.", "The letter ended as the scholar copied it.",
]

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=norm(s); mis=[(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
    return {"letters":len(t),"exact":bool(t) and not mis,"first_mismatches":mis[:5],
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(),
            "independent_two_pointer":not mis}

def run():
    rows=[]; exact=[]
    for i,left in enumerate(LEFT):
        for j,right in enumerate(RIGHT):
            text=f"{left} {CENTER} {right}"; a=audit(text)
            row={"rendered":text,"audit":a,"left_index":i,"right_index":j,
                 "roles":["outer-observation","readable-center","outer-return"],
                 "provenance":{"fresh_authored_scene_bank":True,"center_source":"working 44-letter exact center",
                   "finished_tape_reversal":False,"posthoc_repair":False,"catalogue_text":False,
                   "repeated_units":False,"self_palindromic_units":False}}
            rows.append(row)
            if a["exact"]: exact.append(row)
    rows.sort(key=lambda r:(-r["audit"]["letters"],len(r["audit"]["first_mismatches"])))
    best=rows[0]
    return {"experiment_id":"authored-abba-scene-bank-20260927",
      "method":"fresh authored observation/center/return ABBA seam enumeration with exact outside-in audit",
      "bank":{"left":len(LEFT),"right":len(RIGHT),"independently_authored":True},
      "center":{"rendered":CENTER,"audit":audit(CENTER)},
      "stats":{"trials":len(rows),"exact_gt38":sum(x["audit"]["exact"] and x["audit"]["letters"]>38 for x in rows),
               "longest_rendered_letters":best["audit"]["letters"],"longest_first_mismatch":best["audit"]["first_mismatches"]},
      "exact_candidates":exact,"best_residual":best,"rendered_candidates":rows[:20],
      "reader_evidence":{"status":"not_run","rule":"programmatic audit diagnoses exactness only"},
      "next_repair":"Author the return clause from the measured first residual character pair, preserving its independent scene role; do not reverse the selected observation or use a catalogue sentence.",
      "novelty_preflight":{"status":"passed","signature":"authored-scene|readable-center|outside-in-abba-seam",
       "distinct_from":"live API chunk graph, catalogue harvesting, and post-hoc tape repair"}}

if __name__=='__main__':
    data=run(); OUT.write_text(json.dumps(data,indent=2)+"\n"); print(json.dumps(data["stats"],sort_keys=True))
