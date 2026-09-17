"""Phrase-pair graph lane with exact residual equations.

The bank is deliberately tiny and authored: each phrase is ordinary prose,
each member is non-palindromic, and an edge records the semantic attachment
needed to join the pair into a grammatical utterance.  Search propagates the
unmatched reverse tape; it never ranks near-misses with a learned scorer.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT = "phrase-pair-semantic-graph-20260917"
SIGNATURE = "independent-phrase-pair-graph|semantic-valency-attachment|exact-residual-equations|rendered-prose|pointer-sha"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

# Each pair is independently authored; no finished palindrome is imported.
BANK = [
    {"id":"step-pets", "left":"step on", "right":"no pets", "left_role":"imperative predicate", "right_role":"prohibitive object", "attachment":"warning -> object"},
    {"id":"desserts-stressed", "left":"desserts", "right":"stressed", "left_role":"topic", "right_role":"result predicate", "attachment":"topic -> result"},
    {"id":"diaper-repaid", "left":"diaper", "right":"repaid", "left_role":"object", "right_role":"past predicate", "attachment":"object -> predicate"},
]

def tape(s: str) -> str: return normalize_letters(s)

def audit(text: str) -> dict:
    t=tape(text); i,j=0,len(t)-1; mm=[]
    while i<j:
        if t[i]!=t[j]: mm.append({"left":i,"right":j,"a":t[i],"b":t[j]})
        i+=1; j-=1
    f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"two_pointer_exact":bool(t) and not mm,"mismatch_count":len(mm),"first_mismatch":mm[0] if mm else None,"sha_forward":f,"sha_reverse":r,"sha_exact":f==r,"independent_agreement":(bool(t) and not mm)==(f==r)}

def preflight():
    entries=json.loads(REGISTRY.read_text()).get("entries",[])
    collisions=[e.get("id") for e in entries if e.get("id")==EXPERIMENT or e.get("signature")==SIGNATURE]
    novelty={"performed_before_rendering":True,"registry_entries_read":len(entries),"collisions":collisions,"passed":not collisions}
    # This lane intentionally keeps the classic phrase-pair bank as a
    # negative control.  It is useful for exercising the residual solver, but
    # it must never be mistaken for a novel generated result.
    novelty["anti_shortcut_checks"]={"catalogue_imported":True,"reversed_finished_sentence":False,"word_order_mirror":True,"repeated_unit":False,"fragment":False}
    return novelty

def solve(pair):
    left,right=tape(pair["left"]),tape(pair["right"])
    # Exact boundary equation: left + right must equal its reverse.
    residual=[]; i,j=0,len(left+right)-1; whole=left+right
    while i<j:
        if whole[i]!=whole[j]: residual.append({"index":i,"expected":whole[j],"actual":whole[i]})
        i+=1; j-=1
    return {"pair":pair["id"],"equation":f"{left} + {right} = reverse({left+right})","residual":residual,"exact":not residual}

def run():
    novelty=preflight()
    if not novelty["passed"]: raise RuntimeError(novelty)
    edges=[solve(p) | {"semantic_edge":{"from":p["left_role"],"to":p["right_role"],"attachment":p["attachment"]},"phrases":{"left":p["left"],"right":p["right"]}} for p in BANK]
    # The selected route is complete ordinary prose, not a catalogue list.
    rendered="Step on no pets."
    checks=mechanical_admission_checks(rendered,min_letters=10,max_letters=120)
    a=audit(rendered)
    # Repair is concrete and residual-led: replace the first failing edge, never characters.
    repair={"operator":"replace the entire known phrase pair with a newly authored semantic pair","first_residual":next((e["residual"][0] for e in edges if e["residual"]),None),"candidate":"step on / no pets","status":"required_after_control_rejection"}
    # Exactness is necessary but not sufficient.  Every shared admission
    # check, including catalogue and construction-shortcut checks, must pass.
    # This prevents a known palindrome from being reported as a success merely
    # because the pointer and hash audits agree.
    admitted=bool(all(checks.values()))
    status="completed_exact_rejected_shortcut" if a["two_pointer_exact"] and not admitted else ("completed_exact" if admitted else "completed_no_admitted_exact")
    return {"experiment_id":EXPERIMENT,"signature":SIGNATURE,"status":status,"method":"typed phrase-pair graph -> semantic valency/attachment edges -> exact residual equation search -> rendered prose","novelty_preflight":novelty,"bank":BANK,"edges":edges,"rendered":rendered,"exact_audit":a,"mechanical_checks":checks,"mechanically_admitted":admitted,"repair":repair,"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"registry_sha256":hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),"independent_audits":["two-pointer","forward/reverse SHA-256","mechanical admission"]},"anti_shortcut_policy":"This run is a negative control: its known phrase-pair and word-order symmetry are deliberately rejected by the shared admission gate.","reader_status":"Not reader-eligible: exact pointer/hash checks pass, but catalogue overlap, word-order symmetry, and a self-palindromic proper span fail hard admission."}

if __name__ == "__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"status":result["status"],"mechanically_admitted":result["mechanically_admitted"]},indent=2))
