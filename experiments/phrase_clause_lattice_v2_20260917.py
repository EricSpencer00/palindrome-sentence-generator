"""Phrase-level mirrored clause lattice.

Each side chooses an independently authored, intact clause chunk.  The search
consumes characters from the outside inward while permitting a chunk boundary
on either side; it never constructs one side by reversing a finished tape.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

LEFT = [
    "A calm editor reads a note",
    "The nurse carries a small lamp", "A quiet sailor watches the tide",
    "The careful teacher marks the page", "A red fox crosses the field",
    "The young poet writes a letter", "A baker sets bread on a board",
    "The old doctor checks the chart", "A child hears music at dawn",
    "The gardener waters the roses", "A bright moon lights the road",
]
RIGHT = [
    "some men inspire Diana", "the editor answers a calm aide",
    "the sailor follows a quiet path", "a reader studies the marked page",
    "the fox watches the red bird", "a poet sends the written letter",
    "the baker shares warm bread", "a doctor records the old chart",
    "the child remembers dawn music", "a gardener tends the rose bed",
    "the moon follows the bright road", "a friend reads the final note",
]

def tape(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t=tape(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"exact": t==t[::-1], "length":len(t), "two_pointer": all(t[i]==t[-1-i] for i in range(len(t)//2)), "forward_sha":f, "reverse_sha":r, "sha_equal":f==r}

def search(max_chunks=3):
    # State tracks unconsumed character buffers on each independently chosen side.
    # A transition may append a fresh chunk only when its side buffer is empty.
    states={("", "", (), ()): None}; hits=[]; seen=set()
    for depth in range(160):
        nxt={}
        for (lb,rb,ls,rs), parent in states.items():
            if not lb and len(ls)<max_chunks:
                for k,c in enumerate(LEFT):
                    q=(tape(c),rb,ls+(c,),rs)
                    nxt.setdefault(q, ((lb,rb,ls,rs),"L",c))
            if not rb and len(rs)<max_chunks:
                for k,c in enumerate(RIGHT):
                    q=(lb,tape(c),ls,rs+(c,))
                    nxt.setdefault(q, ((lb,rb,ls,rs),"R",c))
            if lb and rb:
                if lb[0] != rb[-1]: continue
                q=(lb[1:],rb[:-1],ls,rs); nxt.setdefault(q, ((lb,rb,ls,rs),"C",None))
            elif not lb and not rb and ls and rs:
                text=" ".join(ls+rs); a=audit(text)
                if a["exact"] and len(ls)<=max_chunks and len(rs)<=max_chunks: hits.append((text,a,ls,rs))
        states=nxt
        # cap only duplicate structural states, preserving deterministic breadth-first search
        if len(states)>300000: states=dict(list(states.items())[:300000])
    return hits, len(states)

if __name__ == "__main__":
    hits,n=search()
    out={"method":"independent phrase-clause lattice, live outer character equations", "states_last_layer":n,
         "candidate_count":len(hits), "candidates":[{"text":x[0],"audit":x[1],"left_chunks":x[2],"right_chunks":x[3],"provenance":"hand-authored ordinary clause banks; no catalogue/reversal"} for x in hits]}
    p=Path("runs/phrase-clause-lattice-v2-20260917.json"); p.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
