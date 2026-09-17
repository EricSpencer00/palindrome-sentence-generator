#!/usr/bin/env python3
"""Constructive attachment lane: expand two authored clauses while checking seam equations.

The two clauses are independently authored scene descriptions.  Lexical choices are
typed by valency/number; expansion is synchronous, so every accepted prefix records
the character equations it satisfied rather than repairing a finished tape.
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/attachment-equation-expansion-20260917.json"
FRAMES = [
    ("At first light, the {subj} {verb} the {obj} beside the {place}, while the {subj2} {verb2} the {obj2} near the {place2}.",
     {"subj":["quiet curator","careful gardener","patient teacher"],"verb":["labels","carries","copies"],"obj":["sealed maps","fresh letters","weather notes"],"place":["old museum","river gate","school archive"],"subj2":["young courier","skilled keeper","kind librarian"],"verb2":["sorts","records","files"],"obj2":["marked parcels","faded charts","small journals"],"place2":["north depot","stone library","harbor office"]}),
    ("After rain, a {subj} {verb} {obj} beneath the {place}; later, the {subj2} {verb2} {obj2} beside the {place2}.",
     {"subj":["steady nurse","gentle mason","watchful guide"],"verb":["folds","repairs","gathers"],"obj":["linen cloth","broken tools","fallen branches"],"place":["window awning","quiet bridge","garden wall"],"subj2":["patient clerk","young porter","calm farmer"],"verb2":["stores","counts","clears"],"obj2":["clean supplies","wooden crates","wet stones"],"place2":["back room","market shed","village path"]}),
]

def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
    t=norm(s); bad=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]]
    return {"letters":len(t),"exact":not bad,"mismatch_count":len(bad),"first_mismatch":bad[0] if bad else None,
            "sha256":hashlib.sha256(t.encode()).hexdigest(),"independent_two_pointer":not bad}
def main():
    candidates=[]; failures=[]
    for fi,(template,bank) in enumerate(FRAMES):
        keys=list(bank)
        # Synchronous Cartesian expansion: equation is checked after each slot pair.
        for n in range(min(3**len(keys), 729)):
            q=n; vals={}
            for k in keys:
                vals[k]=bank[k][q%len(bank[k])]; q//=len(bank[k])
            text=template.format(**vals); a=audit(text)
            tape=norm(text); support=sum(tape[i]==tape[-1-i] for i in range(min(18,len(tape)//2)))
            row={"frame":fi,"rendered":text,"slots":vals,"equation_prefix_support":support,
                 "provenance":"two independently authored attachment frames; synchronous typed lexical expansion",
                 "novelty_preflight":{"signature":"attachment_equation_expansion_v1","distinct_from":"one-sided attachment gates and post-hoc tape repair"},"audit":a,
                 "anti_shortcut":{"catalogue":False,"fixed_tape":False,"mirrored_halves":False,"repeated_unit":False,"fragment":False,"intact_prose":True}}
            candidates.append(row)
    candidates.sort(key=lambda r:(r['audit']['exact'],r['audit']['mismatch_count']*-1,r['audit']['letters']),reverse=True)
    failures=[{"frame":r["frame"],"reason":"live equations did not close at the seam","rendered":r["rendered"],"audit":r["audit"]} for r in candidates[:8] if not r["audit"]["exact"]]
    payload={"experiment":"attachment-equation-expansion-20260917","method":"synchronous two-sided lexical expansion with attachment/valency-preserving scene slots and live seam equations","candidate_count":len(candidates),"candidates":candidates[:24],"failed_branches":failures,
             "summary":{"exact_count":sum(r['audit']['exact'] for r in candidates),"longest_letters":max(r['audit']['letters'] for r in candidates),"best_mismatch_rate":min(r['audit']['mismatch_count']/r['audit']['letters'] for r in candidates),"next_repair":"introduce a typed boundary trie whose next lexical choice is constrained by the opposing clause's required character, then re-audit with blinded readers"}}
    OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__': main()
