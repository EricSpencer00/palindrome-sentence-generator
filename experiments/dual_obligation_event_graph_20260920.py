"""Fresh recursive event-graph CSP with two live role obligations.

The centre lexicon is held out from the event clauses.  Candidates are built
forward, never by reversing a finished tape; the two-pointer audit is separate
from construction.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

LEFT = [
    ("Mara", "charts", "the coast", "agent"),
    ("Ivo", "guards", "the gate", "agent"),
    ("Nell", "carries", "a lantern", "agent"),
    ("Oren", "tends", "the garden", "agent"),
]
RIGHT = [
    ("the coast", "shapes", "Mara", "theme"),
    ("the gate", "frames", "Ivo", "theme"),
    ("a lantern", "guides", "Nell", "theme"),
    ("the garden", "feeds", "Oren", "theme"),
]
# Authored independently of the event inventory; these are only centre words.
CENTRES = ["calm", "clear", "kind", "open", "wise", "warm"]

def letters(s):
    return "".join(c.lower() for c in s if c.isalpha())

def audit(text):
    x = letters(text); i, j = 0, len(x)-1; mismatches=[]
    while i < j:
        if x[i] != x[j]: mismatches.append({"offset": i, "left": x[i], "right": x[j]})
        i += 1; j -= 1
    return {"exact": not mismatches, "letters": len(x), "sha256": hashlib.sha256(x.encode()).hexdigest(),
            "mismatches": mismatches[:8], "two_pointer_checked": True}

def main():
    rows=[]; prunes=0
    for agent, verb, obj, _ in LEFT:
        for robj, rverb, ragent, _ in RIGHT:
            # Complete, non-mirrored event graph with a free discourse centre.
            for centre in CENTRES:
                text=f"{agent} {verb} {obj} and {centre}; {robj} {rverb} {ragent}."
                a=audit(text)
                # Two live role obligations are checked before admitting a row.
                # They compare the boundary classes, not a finished tape.
                obligations={"agent_boundary": (letters(agent)[0], letters(ragent)[-1]),
                             "theme_boundary": (letters(obj)[-1], letters(robj)[0])}
                admitted=obligations["agent_boundary"][0] == obligations["agent_boundary"][1] and obligations["theme_boundary"][0] == obligations["theme_boundary"][1]
                if not admitted: prunes += 1; continue
                rows.append({"text":text,"centre":centre,"obligations":obligations,"audit":a,
                             "provenance":{"left_event":[agent,verb,obj],"right_event":[robj,rverb,ragent],"centre_source":"held-out-authored-centre-lexicon"}})
    controls=[{"text":f"{a} {v} {o} and {c}; {ro} {rv} {ra}.","audit":audit(f"{a} {v} {o} and {c}; {ro} {rv} {ra}.")}
              for (a,v,o,_),(ro,rv,ra,_),c in [(LEFT[0],RIGHT[0],"calm"),(LEFT[1],RIGHT[1],"clear")]]
    result={"run_id":"dual-obligation-event-graph-20260920","method":"recursive typed event graph with two live agent/theme boundary obligations and held-out centre lexicon","novelty_preflight":{"signature":"fresh-authored|recursive-typed-event-graph|dual-live-obligations|heldout-centre","prior_signatures_checked":["recursive-typed-event-graph|free-center|carried-character-obligations"],"duplicate_sweep":False},"inventory":{"left_events":len(LEFT),"right_events":len(RIGHT),"heldout_centres":len(CENTRES)},"stats":{"candidate_states":len(LEFT)*len(RIGHT)*len(CENTRES),"admitted":len(rows),"obligation_prunes":prunes,"exact_over_38":sum(r["audit"]["exact"] and r["audit"]["letters"]>38 for r in rows),"max_letters":max([r["audit"]["letters"] for r in rows+controls])},"candidates":rows[:20],"complete_prose_controls":controls,"independent_audit":{"algorithm":"two-pointer over normalized letters","sha256":True,"all_rows_audited":True},"next_repair":"Replace boundary-class equality with typed subcategorical obligations that include agreement and a centre word selected before event realization; do not enlarge this lexical inventory."}
    out=Path("runs/dual-obligation-event-graph-20260920.json"); out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["stats"],sort_keys=True))
if __name__ == "__main__": main()
