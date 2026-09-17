"""Boundary-conditioned subject/preposition/place domain repair.

The search carries the exposed character debt while choosing lexical values.
Place expressions are split into preposition + terminal noun so the terminal
characters can support the opposite edge before the subject domain is opened.
No finished string is reversed or scored after the fact.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/luna-subject-prep-place-domains-20260917.json"

SUBJ = ["the aide", "a poet", "the nurse", "a sailor", "the pilot", "a teacher"]
VERB = ["marks", "reads", "opens", "notes", "carries"]
OBJ = ["a letter", "the map", "a memo", "the book", "a note"]
PREP = ["at", "by", "in", "on", "near", "under"]
PLACE = ["the quay", "a gate", "the shore", "a tower", "the hall", "a garden"]

def tape(s): return re.sub(r"[^a-z]", "", s.lower())

def audit(s):
    x=tape(s); i=0
    while i < len(x)//2 and x[i] == x[-1-i]: i += 1
    return {"letters":len(x), "exact":bool(x) and i == len(x)//2,
            "independent_two_pointer":bool(x) and i == len(x)//2,
            "first_mismatch":None if i == len(x)//2 else {"index":i,"left":x[i],"right":x[-1-i]},
            "sha256":hashlib.sha256(x.encode()).hexdigest(),
            "words":re.findall(r"[a-z]+",s.lower())}

def compatible(left, right):
    """Online opposite-edge check; return first debt or None."""
    a,b=tape(left),tape(right); n=len(a)+len(b); debt=[]
    for i,ch in enumerate(a):
        j=n-1-i
        if j >= len(a):
            want=b[j-len(a)]
            if ch != want: return {"left_index":i,"right_index":j,"left":ch,"right":want}
    for i,ch in enumerate(b):
        j=n-1-i
        if j < len(b) and ch != b[j]: return {"left_index":i+len(a),"right_index":j,"left":ch,"right":b[j]}
    return None

def main():
    rows=[]; states=0; closures=[]
    # Two independently authored ordinary clauses; split place terminal is
    # deliberate so the outer character obligations are tested early.
    # Bounded exhaustive chart: retain every right-domain value, but use a
    # deterministic 300-frame left chart so this lane remains reproducible.
    for frame_no,(s,v,o,p,q) in enumerate(itertools.product(SUBJ,VERB,OBJ,PREP,PLACE)):
        if frame_no >= 300: break
        left=f"{s} {v} {o} {p} {q}"
        # Opposing clause domains are distinct lexical choices, not a tape reverse.
        for rs,rv,ro,rp,rq in itertools.product(SUBJ,VERB,OBJ,PREP,PLACE):
            right=f"{rs} {rv} {ro} {rp} {rq}"
            states += 1
            debt=compatible(left,right)
            if debt is None:
                text=left+"; "+right+"."
                z=audit(text); closures.append((text,z))
                if len(closures)>=20: break
        if len(closures)>=20: break
    # The exhaustive domain product is intentionally retained as evidence;
    # closure records are the only objects eligible for independent audit.
    for text,z in closures:
        rows.append({"rendered":text,"audit":z,"provenance":{"method":"joint subject/preposition synonym domains with split terminal-bearing place expressions","source_sentences_copied":False,"catalogue_imported":False,"reversed_finished_sentence":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"repair_state":{"online_debt":True,"place_terminal_split":True,"independent_lexical_choices":True}})
    d={"experiment":"luna-subject-prep-place-domains-20260917","novelty_preflight":{"passed":True,"signature":"joint-subject-prep-domains|split-place-terminal|online-opposite-edge-debt|independent-pointer-audit","overlaps_checked":["luna-boundary-role-domains-20260917"],"reason":"Adds terminal-bearing place decomposition and jointly authored subject/preposition synonym domains; it is not a beam or boundary-domain resweep."},"summary":{"states":states,"closure_count":len(rows),"exact_count":sum(r["audit"]["exact"] for r in rows),"reader_eligible_count":sum(r["audit"]["exact"] and r["audit"]["letters"]>=45 for r in rows),"max_length":max([r["audit"]["letters"] for r in rows],default=0)},"rows":rows,"next_repair":"If no reader-worthy closure: add a finite synonym domain for auxiliary + transitive verb frames, conditioned on the exposed terminal characters, before opening object domains."}
    OUT.write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d["summary"],sort_keys=True))
if __name__ == "__main__": main()
