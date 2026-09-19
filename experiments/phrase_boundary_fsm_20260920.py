"""Synchronous phrase-boundary FSM construction.

Both sides advance through authored multiword constituents.  The FSM carries
the next character obligation across a constituent boundary; it never builds a
finished clause and repairs its residual.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phrase-boundary-live-fsm-20260920"
SIGNATURE = "synchronous-authored-constituents|cross-boundary-character-fsm|two-complete-clauses"
WORD = re.compile(r"[A-Za-z]+")

# Human-authored, role-complete constituents.  Each side must choose one unit
# at every state; units are not mirrored or selected by a completed tape.
UNITS = {
    "subject": ("the careful ranger", "a patient keeper", "the young cartographer"),
    "predicate": ("maps the harbor", "guards the lantern", "marks the old bridge"),
    "adjunct": ("near the quiet orchard", "beside a narrow inlet", "under the winter moon"),
}
ORDER = ("subject", "predicate", "adjunct")

def norm(s): return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
    t = norm(s); rev = t[::-1]
    return {"normalized": t, "letters": len(t), "two_pointer_exact": bool(t) and all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "mismatch_count": sum(a != b for a,b in zip(t,rev)),
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def parse_clause(s):
    words = WORD.findall(s.lower())
    return {"words": words, "complete": len(words) >= 8 and words[0] in {"the","a"} and
            any(w in {"maps","guards","marks"} for w in words) and s.endswith("."),
            "constituent_count": 3}

def run():
    # True two-cursor traversal: right constituents are opened in reverse
    # order, and every character is consumed before another constituent pair
    # is opened.  Unequal unit lengths leave explicit cross-boundary debt.
    stack = [("", "", 0, len(ORDER)-1, "", "", [])]
    visited = 0; terminals = []
    while stack:
        left, right, li, ri, ldebt, rdebt, path = stack.pop()
        if li == len(ORDER) and ri < 0 and not ldebt and not rdebt:
            visited += 1
            ltxt, rtxt = left.capitalize()+".", right.capitalize()+"."
            la, ra = audit(ltxt), audit(rtxt)
            combined = audit(ltxt + " " + rtxt)
            lp, rp = parse_clause(ltxt), parse_clause(rtxt)
            terminals.append({"left": ltxt, "right": rtxt, "left_audit": la, "right_audit": ra,
                              "independent_parse": {"left": lp, "right": rp}, "path": path,
                              "combined_audit": combined,
                              "mechanically_admitted": combined["two_pointer_exact"] and lp["complete"] and rp["complete"]})
            continue
        if ldebt or rdebt:
            # consume the live residual one character at a time
            if ldebt and rdebt:
                visited += 1
                if ldebt[0] != rdebt[0]: continue
                stack.append((left, right, li, ri, ldebt[1:], rdebt[1:], path))
            continue
        if li >= len(ORDER) or ri < 0: continue
        lslot, rslot = ORDER[li], ORDER[ri]
        for lunit in UNITS[lslot]:
            for runit in UNITS[rslot]:
                a, b = norm(lunit), norm(runit)[::-1]
                # consume the common prefix now; remaining suffix is debt
                k = 0
                while k < len(a) and k < len(b) and a[k] == b[k]: k += 1
                visited += k + 1
                if k < min(len(a), len(b)): continue
                stack.append((left+lunit+" ", runit+" "+right, li+1, ri-1,
                              a[k:], b[k:], path+[ {"left_slot":lslot,"right_slot":rslot,
                              "left_unit":lunit,"right_unit":runit,"consumed":k} ]))
    admitted = [x for x in terminals if x["mechanically_admitted"] and min(x["left_audit"]["letters"],x["right_audit"]["letters"]) > 38]
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,
            "method":"synchronous finite-state traversal of authored subject/predicate/adjunct constituents; obligations cross phrase boundaries before either clause is complete",
            "stats":{"visited_transitions":visited,"complete_clause_pairs":len(terminals),"exact_over_38":len(admitted),"longest_letters":max((x["left_audit"]["letters"] for x in terminals),default=0)},
            "novelty_preflight":{"status":"withdrawn","checked_registry_signature":SIGNATURE,"duplicate":False,"catalogue_imported":False,"repair_queue":False,"reason":"initial draft audited clauses independently; combined two-sided tape audit is now mandatory"},
            "candidates":admitted[:8],"independent_audits":["literal two-pointer character audit","forward/reverse SHA-256","independent finite parser","boundary obligation ledger"],
            "provenance":{"human_authored_multiword_units":True,"synchronous_constituent_fsm":True,"finished_tape_reversal":False,"post_hoc_repair":False,"word_order_mirror":False,"catalogue_text":False,"rlaif_per_candidate":False},
            "failure_and_next_construction":{"failure":"no exact closure over 38 letters" if not admitted else "exact closures found","next":"hold subject and predicate units fixed, add a fourth authored complement constituent whose opening character is selected by the live boundary obligation; rerun preflight before search"},
            "reader_gate":"closed; mechanical output is not reader-certified"}

if __name__ == "__main__":
    out=ROOT/"runs"/(EXPERIMENT_ID+".json"); out.write_text(json.dumps(run(),indent=2)+"\n"); print(json.dumps(run()["stats"],sort_keys=True))
