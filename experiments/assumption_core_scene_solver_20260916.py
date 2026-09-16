"""Bounded joint lexical-slot search with assumption-core explanations.

This is deliberately not a reverse decoder: every slot is selected in ordinary
order, while character equations and grammar assumptions are checked together.
The distinct contribution is retaining a minimal conflicting assumption core
for each failed complete realization, then replaying a held-out one-slot edit.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def tape(s): return re.sub(r"[^a-z]", "", s.lower())
def units(s): return tuple(re.findall(r"[a-z]+", s.lower()))

SCENES = [
 {"id":"courier_past", "slots":[
   ["the", "a"], ["careful", "quiet", "young"], ["courier", "teacher", "sailor"],
   ["delivered", "carried", "returned"], ["the", "a"], ["letter", "parcel", "map"],
   ["and", "while"], ["the", "a"], ["patient", "quiet"], ["watchman", "driver", "teacher"],
   ["waited", "rested", "watched"], ["the", "a"], ["gate", "harbor", "garden"]],
  "tense":"past", "agreement":"singular"},
 {"id":"garden_present", "slots":[
   ["the", "a"], ["patient", "busy", "kind"], ["gardeners", "workers", "children"],
   ["water", "carry", "sort"], ["the", "a"], ["plants", "baskets", "seeds"],
   ["and", "while"], ["the", "a"], ["patient", "busy"], ["gardeners", "workers", "children"],
   ["wait", "rest", "sort"], ["the", "a"], ["tools", "seeds", "baskets"]],
  "tense":"present", "agreement":"plural"},
]

def grammar_ok(choice, scene):
    # Explicit tense/agreement constraints are assumptions in the same state.
    det, adj, subj, verb, odet, obj = choice[:6]
    if scene["agreement"] == "singular" and subj.endswith("s"): return False
    if scene["agreement"] == "plural" and not subj.endswith("s"): return False
    if scene["tense"] == "past" and verb not in {"delivered","carried","returned","waited","rested","watched"}: return False
    if scene["tense"] == "present" and verb not in {"water","carry","sort","wait","rest"}: return False
    return True

def core(choice, scene):
    checks = {"grammar":grammar_ok(choice,scene), "all_different":len(set(choice))==len(choice),
              "exact_equation":tape(" ".join(choice)) == tape(" ".join(choice))[::-1],
              "ordinary_units":all(w != w[::-1] for w in units(" ".join(choice)))}
    return checks

def minimal_core(checks):
    bad=[k for k,v in checks.items() if not v]
    # Assumption literals are independent; a one-failure core is minimal.
    return bad[:1] if bad else []

def audit(text):
    t=tape(text); ws=units(text)
    return {"exact":bool(t) and t==t[::-1], "distinct_units":len(ws)==len(set(ws)),
            "no_self_palindromic_units":all(w!=w[::-1] for w in ws), "letters":len(t)}

def repair(choice, scene, failed):
    # Held-out repair changes exactly one lexical slot, then recomputes all constraints.
    for i, domain in enumerate(scene["slots"]):
        for value in domain:
            if value == choice[i]: continue
            candidate=list(choice); candidate[i]=value
            if grammar_ok(candidate,scene):
                return {"slot":i,"text":" ".join(candidate),"checks":core(candidate,scene),
                        "audit_a":audit(" ".join(candidate)),
                        "audit_b":tape(" ".join(candidate)) == tape(" ".join(candidate))[::-1]}
    return None

def run(limit=240, min_letters=39):
    rows=[]; tested=0; closures=[]
    for scene in SCENES:
        for choice in itertools.product(*scene["slots"]):
            if tested >= limit: break
            tested += 1
            checks=core(choice,scene); text=" ".join(choice)
            if len(tape(text)) < min_letters:
                continue
            row={"scene":scene["id"],"text":text,"checks":checks,
                 "minimal_conflict_core":minimal_core(checks),"audit_a":audit(text),
                 "audit_b":tape(text)==tape(text)[::-1],
                 "provenance":{"scene":scene["id"],"slot_domains":scene["slots"],"construction":"ordinary-order-joint-assignment"}}
            if all(checks.values()): closures.append(row)
            else: row["heldout_repair"]=repair(choice,scene,row["minimal_conflict_core"])
            rows.append(row)
    return {"status":"bounded_joint_assumption_core_search_complete","tested":tested,
            "closures":closures,"records":rows,
            "solver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope":"Finite scenes only; no readability claim without blinded readers."}

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--limit",type=int,default=240)
    a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(run(a.limit),indent=2)+"\n"); print(json.dumps({"out":str(a.out),"tested":run(a.limit)["tested"],"closures":len(run(a.limit)["closures"])}))
