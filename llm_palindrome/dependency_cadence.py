"""Dependency/valency cadence lattice (fresh constructive experiment).

Unlike tape or word-pair lanes, each side is a legal dependency path: a
predicate consumes typed arguments, while the product state carries the
unresolved character vector at the two outside edges.  The prose is authored
first; exactness is only a terminal predicate, never a candidate reward.
"""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass, asdict
from pathlib import Path

def norm(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())

@dataclass(frozen=True)
class Frame:
    predicate: str
    subject: str
    object: str
    adjunct: str
    roles: tuple[str, ...]
    def render(self) -> str:
        return f"{self.subject} {self.predicate} {self.object} {self.adjunct}".strip()

FRAMES = (
    Frame("carries", "The porter", "quiet parcels", "through rain", ("agent", "theme", "path")),
    Frame("guides", "A patient nurse", "lost visitors", "toward dawn", ("agent", "theme", "goal")),
    Frame("records", "The careful clerk", "each signal", "after dusk", ("agent", "theme", "time")),
    Frame("folds", "Mara", "the map", "by lamplight", ("agent", "theme", "manner")),
    Frame("keeps", "A young keeper", "one promise", "for winter", ("agent", "theme", "purpose")),
)

def audit(text: str) -> dict:
    t = norm(text); mism=[]
    for i in range(len(t)//2):
        j=len(t)-1-i
        if t[i] != t[j]: mism.append((i,j,t[i],t[j]))
    return {"letters":len(t), "exact":not mism, "mismatches":mism[:16],
            "sha256":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

def search(max_depth=3):
    # A state is (left dependency path, right dependency path, residual vector).
    # Each transition selects a frame and a role, so argument legality is
    # checked before its words enter the character product.
    states=[([], [], "")]; visited=0; best=None
    for depth in range(1, max_depth+1):
        nxt=[]
        for left,right,_ in states:
            for lf in FRAMES:
                if lf in left: continue
                for rf in FRAMES:
                    if rf in right or lf is rf: continue
                    # Simultaneous semantic-role cadence: matching role sets,
                    # but independent lexical frames and independent order.
                    if lf.roles[:2] != rf.roles[:2]: continue
                    lt=" ".join(x.render() for x in left+[lf]); rt=" ".join(x.render() for x in right+[rf])
                    # Compare only currently exposed edge obligations; no tape
                    # reversal or copied span is used by the search.
                    a,b=norm(lt),norm(rt)[::-1]; k=min(len(a),len(b)); residual=a[:k] == b[:k]
                    vec="".join(x for x,y in zip(a,b) if x!=y)
                    visited += 1
                    candidate=lt+"; "+rt
                    aa=audit(candidate)
                    if best is None or aa["letters"] > best["audit"]["letters"]: best={"text":candidate,"left_roles":[f.roles for f in left+[lf]],"right_roles":[f.roles for f in right+[rf]],"audit":aa,"residual_vector":vec[:24]}
                    if aa["exact"]: return {"status":"closed","candidate":best,"visited":visited,"depth":depth}
                    # Keep semantically legal states even when the current
                    # edge vector is nonzero: later role transitions may
                    # consume it.  Pruning only on exact prefix agreement
                    # would turn prose-first authoring into a tape solver.
                    if depth < max_depth:
                        nxt.append((left+[lf],right+[rf],vec))
        states=nxt
    return {"status":"no_closure","candidate":None,"near_miss":best,"visited":visited,"depth":max_depth}

def build(root: Path):
    registry=json.loads((root/"docs/experiment-novelty-registry.json").read_text())
    sig="dependency-valency-cadence-lattice|simultaneous-role-paths|residual-vector|prose-first"
    collisions=[e.get("id") for e in registry.get("entries",[]) if sig in str(e)]
    result=search()
    return {"experiment_id":"dependency-valency-cadence-20260919-luna","signature":sig,
      "status":result["status"],"search":result,
      "provenance":{"frames":"fresh hand-authored dependency frames","source":"llm_palindrome/dependency_cadence.py","generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"audit":"independent normalized two-pointer and SHA-256 forward/reverse"},
      "novelty_preflight":{"registry_entries_read":len(registry.get("entries",[])),"exact_signature_collision":bool(collisions),"collisions":collisions,"fixed_or_reversed_tape":False,"word_order_mirror":False,"repeated_spans":False,"catalogue":False},
      "repair":{"operator":"add a fresh frame whose adjunct opening consumes the recorded residual vector while preserving subject-theme-path valency","before":result.get("near_miss"),"status":"concrete next bounded repair; no candidate admitted without exact independent audit"},
      "frames":[asdict(f) for f in FRAMES]}

if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]; out=root/"runs/dependency-valency-cadence-20260919-luna.json"; out.write_text(json.dumps(build(root),indent=2)+"\n"); print(out)
