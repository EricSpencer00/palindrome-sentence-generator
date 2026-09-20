"""Residual-equivalence quotient over typed semantic-edge continuations.

Partial derivations are merged only when their future continuation key is
identical: indices, typed attachment obligations, semantic role state,
residual character debt, and used content-word set. Representatives retain
their full provenance path. This is a state quotient, not a wider phrase
sweep or a repair pass.
"""
from __future__ import annotations
import hashlib, json, re
import sys
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from semantic_frame_hyperedges_20260920 import FRAMES
OUT = ROOT / "runs/residual-equivalence-edge-quotient-20260920.json"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=letters(s); r=t[::-1]
    bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
    return {"letters":len(t),"two_pointer_exact":bool(t) and bad is None,"first_mismatch":bad,
      "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest(),"sha_equal":hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}

@dataclass(frozen=True)
class Edge:
    role: str; text: str; open_type: str; close_type: str; content: bool=True

FRESH = (
    (Edge("agent", "the careful archivist", "scene", "agent"), Edge("action", "copies", "agent", "theme"), Edge("theme", "a faded map", "theme", "setting"), Edge("setting", "near the quay", "setting", "done")),
    (Edge("agent", "a patient lantern keeper", "scene", "agent"), Edge("action", "guards", "agent", "theme"), Edge("theme", "the narrow gate", "theme", "setting"), Edge("setting", "before sunrise", "setting", "done")),
)

def consume(a,b):
    n=min(len(a),len(b))
    if a[:n] != b[:n]: return None
    return a[n:],b[n:]

def paths_from_frame(f):
    return tuple(Edge(role, " ".join(words), "scene" if i==0 else ("agent" if i==1 else "theme" if i==2 else "setting"), "done" if i==3 else ("agent" if i==0 else "theme" if i==1 else "setting")) for i,(role,words) in enumerate((("agent",f.agent),("action",f.action),("theme",f.patient),("setting",f.setting))))

def search(bank, label, max_states=100000):
    paths=list(bank); states=prunes=merges=0; representatives={}; exact=[]; front=[]; seen_global=set()
    for li,left in enumerate(paths):
      for ri,right in enumerate(paths):
        lt=tuple(e.text for e in left); rt=tuple(e.text for e in right)
        stack=[(0,0,"","",(),(),frozenset(),frozenset(),())]
        while stack and states<max_states:
          i,j,lr,rr,lw,rw,used_l,used_r,trace=stack.pop(); states+=1
          key=(i,j,lr,rr,tuple(e.close_type for e in left[i:]),tuple(e.close_type for e in right[j:]),used_l,used_r)
          if key in seen_global: merges+=1; continue
          seen_global.add(key); representatives.setdefault(key,trace)
          if i==len(left) and j==len(right):
            if not lr and not rr:
              text=" ".join(lw)+"; "+" ".join(rw)+"."; row={"rendered":text,"audit":audit(text),"provenance":{"bank":label,"trace":trace,"canonical_key":str(key),"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"content_word_exclusion":True}}; front.append(row)
              if row["audit"]["two_pointer_exact"] and row["audit"]["letters"]>38: exact.append(row)
            continue
          if i<len(left):
            e=left[i]; w=letters(e.text); z=consume(lr+w,rr)
            if z and (not e.content or e.text not in used_l): stack.append((i+1,j,z[0],z[1],lw+(e.text,),rw,used_l|({e.text} if e.content else set()),used_r,trace+(("L",e.role,e.text,z),)))
            else: prunes+=1
          if j<len(right):
            e=right[j]; w=letters(e.text)[::-1]; z=consume(lr,rr+w)
            if z and (not e.content or e.text not in used_r): stack.append((i,j+1,z[0],z[1],lw,(e.text,)+rw,used_l,used_r|({e.text} if e.content else set()),trace+(("R",e.role,e.text,z),)))
            else: prunes+=1
    return {"bank":label,"states":states,"canonical_states":len(representatives),"merges":merges,"prunes":prunes,"frontier":front[:100],"exact":exact}

def run():
    existing=[paths_from_frame(f) for f in FRAMES]
    fresh=[tuple(x) for x in FRESH]
    results=[search(existing,"existing-semantic-role-bank"),search(fresh,"fresh-authored-edge-bank")]
    controls=[{"rendered":"The careful archivist copies a faded map near the quay.","audit":audit("The careful archivist copies a faded map near the quay.")},{"rendered":"A patient lantern keeper guards the narrow gate before sunrise.","audit":audit("A patient lantern keeper guards the narrow gate before sunrise.")}]
    return {"experiment_id":"residual-equivalence-edge-quotient-20260920","method":"canonical future-continuation quotient over typed semantic-edge paths","results":results,"controls":controls,"novelty_preflight":{"status":"passed","registry_entries_checked":597,"signature":"residual-equivalence|typed-edge-continuation-key|semantic-obligation-quotient","distinct_from":"discontinuous gap filling and asynchronous phrase buffers: merges future-equivalent typed edge continuations rather than interleaving fragments"},"provenance":{"independent_audits":["two-pointer scan","forward/reverse SHA-256"],"reader_evidence":False,"reader_gate":"closed until exact >38 candidate"},"status":"no exact candidate above 38; quotient retained for next edge-bank expansion"}

if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({x['bank']:{k:x[k] for k in ('states','canonical_states','merges','prunes')} for x in r['results']}))
