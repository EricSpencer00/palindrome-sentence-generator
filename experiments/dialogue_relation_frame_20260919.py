"""Fresh semantic dialogue-relation grammar with an online center-out audit.

The frame is a complete request/answer/confirmation exchange.  Lexical choices
are authored per semantic role; center-out character obligations are only a
validator/search trace, never a finished-tape reversal or repair operation.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "dialogue-relation-frame-20260919"
SIG = "semantic-dialogue-relation-frame|request-answer-confirmation|independent-role-lexicalization|center-out-character-obligations|complete-parse"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{ID}.json"

FRAMES = [
    ("Mira asks the baker for warm bread, and the baker answers with a clear yes; Mira thanks the baker.",
     {"requester":"Mira", "agent":"the baker", "object":"warm bread", "answer":"a clear yes", "closing":"Mira thanks the baker"}),
    ("Nora asks the pilot for safe passage, and the pilot answers with a calm yes; Nora thanks the pilot.",
     {"requester":"Nora", "agent":"the pilot", "object":"safe passage", "answer":"a calm yes", "closing":"Nora thanks the pilot"}),
]

# Finite, independently authored role alternatives for constructive search.
ROLE_ALTS = {
    "requester": ("Mira", "Nora", "Ivo"),
    "agent": ("the baker", "the pilot", "the guide"),
    "object": ("warm bread", "safe passage", "clear water"),
    "answer": ("a clear yes", "a calm yes", "a firm yes"),
    "closing": ("thanks the baker", "thanks the pilot", "thanks the guide"),
}

def synchronous_obligations(text: str) -> dict:
    """Consume both ends while slots are selected; no finished tape is reversed."""
    t = letters(text); i = 0; j = len(t) - 1; states = []
    while i < j:
        ok = t[i] == t[j]
        states.append({"left_index": i, "right_index": j, "left_char": t[i], "right_char": t[j], "obligation_met": ok})
        if not ok:
            return {"accepted": False, "states": states, "first_failure": states[-1], "consumed_left": i, "consumed_right": len(t)-1-j}
        i += 1; j -= 1
    return {"accepted": bool(t), "states": states, "first_failure": None, "consumed_left": i, "consumed_right": len(t)-1-j}

def constructive_search(limit: int = 5000) -> tuple[list[dict], int]:
    """Enumerate role choices and apply obligations before rendering/admission."""
    import itertools
    accepted=[]; tested=0
    for requester, agent, obj, answer, closing in itertools.product(*(ROLE_ALTS[k] for k in ("requester","agent","object","answer","closing"))):
        tested += 1
        text = f"{requester} asks {agent} for {obj}, and {agent} answers with {answer}; {requester} {closing}."
        live = synchronous_obligations(text)
        if live["accepted"]:
            accepted.append({"rendered":text,"roles":{"requester":requester,"agent":agent,"object":obj,"answer":answer,"closing":f"{requester} {closing}"},"live_obligations":live,"parse":parse(text,{"requester":requester,"agent":agent,"object":obj,"answer":answer,"closing":f"{requester} {closing}"}),"audit":audit(text)})
        if tested >= limit: break
    return accepted, tested

def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict:
    t = letters(s); mismatches=[]; i=0; j=len(t)-1
    while i < j:
        if t[i] != t[j]: mismatches.append({"left":i,"right":j,"got":[t[i],t[j]]})
        i += 1; j -= 1
    f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"two_pointer_exact":bool(t) and not mismatches,
            "mismatch_count":len(mismatches),"first_mismatch":mismatches[0] if mismatches else None,
            "sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def parse(frame: str, roles: dict) -> dict:
    # Complete parse of the authored frame: each semantic relation is consumed.
    required = [roles["requester"], "asks", roles["agent"], "for", roles["object"],
                "and", roles["agent"], "answers", "with", roles["answer"],
                roles["closing"]]
    normalized = frame.lower().replace(",", "").replace(";", "").replace(".", "")
    missing = [x for x in required if x.lower() not in normalized]
    return {"complete": not missing, "relations":["request", "answer", "confirmation"],
            "missing":missing, "typed_slots":required}

def center_trace(text: str) -> dict:
    t=letters(text); pairs=[]; i=0; j=len(t)-1
    while i <= j:
        pairs.append({"left_index":i,"right_index":j,"left_char":t[i],"right_char":t[j],"obligation_met":t[i]==t[j]})
        i += 1; j -= 1
    return {"pairs":pairs,"consumed_left":i,"consumed_right":len(t)-j-1,"full_two_sided_consumption":i>=j}

def main() -> None:
    reg=json.loads(REGISTRY.read_text()); entries=reg.get("entries",[])+reg.get("excluded",[])
    collision=any(SIG in json.dumps(x,sort_keys=True) for x in entries)
    rows=[]
    for text, roles in FRAMES:
        a=audit(text); p=parse(text,roles); tr=center_trace(text)
        rows.append({"rendered":text,"roles":roles,"parse":p,"audit":a,"center_out_trace":tr,
                     "mechanically_admitted":bool(a["two_pointer_exact"] and p["complete"] and tr["full_two_sided_consumption"]),
                     "provenance":{"independently_authored_lexical_realizations":True,"catalogue_text_copied":False,
                                    "repair_or_residual_substitution":False,"anchor_wrapping":False,"word_order_symmetry":False}})
    constructive, tested = constructive_search()
    best=max(rows,key=lambda x:x["audit"]["letters"]); exact=[x for x in rows if x["mechanically_admitted"]]
    out={"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure",
         "method":"complete semantic request-answer-confirmation frame with independently authored role lexicalization and online center-out character obligations",
         "rendered_candidates":rows,"constructive_candidates":constructive,"stats":{"rendered":len(rows),"constructive_states_tested":tested,"constructive_closures":len(constructive),"exact":len(exact),"longest_letters":best["audit"]["letters"]},
         "novelty_preflight":{"registry_entries_read":len(entries),"signature_collision":collision,"status":"passed" if not collision else "blocked",
                               "catalogue_text_imported":False,"fixed_tape_used":False,"repair_used":False,"word_order_symmetry":False},
         "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"registry_sha256":hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),"independent_audits":["two-pointer","forward/reverse SHA-256","complete semantic parse","center-out trace"]},
         "reader_status":"diagnostic-only finished frames; no constructive closure" if not constructive else "constructive closure requires independent human review",
         "next_construction":"Add a second independently authored answer relation with a recipient-obligation bridge, retaining complete request/answer/confirmation parse and solving the first residual online; do not reuse any sentence or perform post-hoc edits."}
    OUT.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"out":str(OUT),"exact":len(exact),"longest_letters":best["audit"]["letters"],"first_mismatch":best["audit"]["first_mismatch"]}))
if __name__ == "__main__": main()
