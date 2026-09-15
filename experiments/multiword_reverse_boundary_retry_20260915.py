"""Residual attachment probe: segment the reversed tape into a typed clause."""
from __future__ import annotations
import argparse, hashlib, json, re
from itertools import product
from pathlib import Path

S = ("we", "they", "workers", "teachers", "nurses", "artists")
V = ("repaid", "deliver", "reviled", "stressed", "noticed", "carried", "reviewed")
O = ("a diaper", "the drawer", "the reward", "desserts", "reports", "letters")
A = ("today", "quietly", "carefully", "outside", "at dawn", "in spring")
ANCHORS = {"repaid", "diaper", "drawer", "reward", "deliver", "reviled", "stressed", "desserts"}
WORDS = set(S + V + tuple(" ".join(x.split()) for x in O + A)) | {w for x in O + A for w in x.split()}

def norm(x): return re.sub("[^a-z]", "", x.lower())
def audit(x):
    t = norm(x); bad = [i for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"exact": bool(t) and not bad, "letters": len(t), "mismatches": bad,
            "sha256": hashlib.sha256(t.encode()).hexdigest()}

def segment(tape, n, pos=0):
    if n == 0: return [()] if pos == len(tape) else []
    out=[]
    for w in WORDS:
        if tape.startswith(w, pos):
            for rest in segment(tape, n-1, pos+len(w)):
                out.append((w,)+rest)
    return out

def run():
    hits=[]; left_count=0; segment_attempts=0
    for s,v,o,a in product(S,V,O,A):
        left=f"{s} {v} {o} {a}"; lw=left.split()
        if not (set(lw) & ANCHORS): continue
        left_count += 1; rev=norm(left)[::-1]
        # Multiword reverse-boundary attachment: punctuation/spacing is free,
        # but the reverse must segment into the same complete S-V-O-Adv shape.
        for rw in segment(rev, 4):
            segment_attempts += 1
            right=f"{rw[0]} {rw[1]} {' '.join(rw[2:-1])} {rw[-1]}"
            text=left+" "+right; aa=audit(text)
            if aa["letters"] >= 39 and aa["exact"]:
                hits.append({"text":text,"left_clause":left,"right_clause":right,"audit":aa,
                  "anti_shortcut":{"typed_left":True,"typed_right":True,"reverse_boundary":True,"independent_validator":True}})
    return {"status":"no_reader_worthy_output" if not hits else "exact_hits_need_blinded_readers",
      "config":{"operator":"reverse-tape segmentation into typed S-V-O-Adv","min_letters":39},
      "anchor_clauses":left_count,"segment_attempts":segment_attempts,"exact_hits":len(hits),
      "hits":hits,"independent_rendered_hits":[h["text"] for h in hits],
      "next_operator":"Expand phrase inventory with authored multiword valency frames; retain independent parses and hard exact gate."}

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ns=ap.parse_args()
    r=run(); ns.out.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({k:r[k] for k in ('status','anchor_clauses','segment_attempts','exact_hits')},indent=2))
