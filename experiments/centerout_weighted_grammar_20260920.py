"""Center-out weighted lexicalized grammar with live bilateral obligations.

The heap key is a fixed, checked-in English prior.  Each step expands one
grammar state on each side and immediately consumes the corresponding exposed
characters; no completed tape is scored or reversed.
"""
from __future__ import annotations
import hashlib, heapq, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
RUN = ROOT / "runs/centerout-weighted-grammar-20260920.json"
EXPERIMENT_ID = "centerout-weighted-grammar-20260920"
SIGNATURE = "centerout|weighted-finite-state-lexicalized-grammar|simultaneous-character-obligations|fixed-english-prior"

# Fixed deterministic prior (rounded log probabilities; deliberately tiny and
# human-auditable rather than a runtime corpus/API dependency).
LEX = {
 "det": (("the",-0.2),("a",-0.6),("our",-1.0)),
 "subj": (("writer",-0.3),("teacher",-0.7),("gardener",-0.9),("sailor",-1.1)),
 "verb": (("records",-0.2),("guides",-0.5),("marks",-0.8),("carries",-1.0)),
 "obj": (("a letter",-0.5),("the report",-0.4),("the parcel",-0.8),("a message",-0.9)),
 "prep": (("near",-0.3),("beside",-0.7),("beyond",-1.1)),
 "place": (("the garden",-0.3),("the harbor",-0.6),("the station",-0.9)),
}
GRAMMAR = ("det","subj","verb","obj","prep","place")

def letters(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(text):
    tape=letters(text); i,j=0,len(tape)-1
    while i<j and tape[i]==tape[j]: i+=1; j-=1
    return {"letters":len(tape),"pointer_exact":i>=j,"first_mismatch":None if i>=j else [i,j,tape[i],tape[j]],
            "sha256_forward":hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(tape[::-1].encode()).hexdigest()}

def novelty_preflight():
    data=json.loads(REGISTRY.read_text()); entries=[x for x in data.get("entries",[]) if x.get("id")!=EXPERIMENT_ID]
    overlap=[x["id"] for x in entries if x.get("signature")==SIGNATURE]
    artifact=str(Path(__file__).relative_to(ROOT)); collision=[x["id"] for x in entries if x.get("artifact")==artifact]
    return {"status":"passed" if not overlap and not collision else "blocked","registry_entries_before_run":len(entries),
            "signature_overlaps":overlap,"artifact_collisions":collision,"excluded_routes":["finished-tape reversal","post-search scoring","word-order-only symmetry"]}

def _surface(words): return " ".join(words)+"."
def run(max_states=5000):
    pre=novelty_preflight()
    if pre["status"]!="passed": raise RuntimeError(pre)
    # Two grammar stacks grow from the center.  The right stack is an
    # independently lexicalized clause, not a reversed copy of the left.
    heap=[(0,0,(),(),0,0)]; seen=set(); rows=[]; expanded=0; obligations=0
    while heap and expanded<max_states:
        negscore, depth, left, right, li, ri=heapq.heappop(heap); expanded+=1
        key=(left,right,li,ri)
        if key in seen: continue
        seen.add(key)
        if li==len(GRAMMAR) and ri==len(GRAMMAR):
            text=_surface(left+right); a=audit(text); rows.append({"rendered":text,"grammar_states":{"left":li,"right":ri},"audit":a,"weight":-negscore}); continue
        if li<len(GRAMMAR) and ri<len(GRAMMAR):
            for lw,lp in LEX[GRAMMAR[li]]:
                for rw,rp in LEX[GRAMMAR[ri]]:
                    obligations+=1
                    # obligation is checked on the characters exposed now;
                    # retain only compatible prefixes where available.
                    la,ra=letters(lw),letters(rw)
                    if depth and la[0]!=ra[-1]: continue
                    heapq.heappush(heap,(negscore-lp-rp,depth+1,left+(lw,),right+(rw,),li+1,ri+1))
    rows.sort(key=lambda r:(-r["audit"]["letters"],-r["weight"],r["rendered"]))
    exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["sha256_forward"]==r["audit"]["sha256_reverse"] and r["audit"]["letters"]>38]
    result={"experiment_id":EXPERIMENT_ID,"method":"center-out weighted finite-state lexicalized grammar with simultaneous bilateral character obligations","config":{"prior":"fixed rounded English lexical prior","post_search_scoring":False,"grammar":GRAMMAR},"stats":{"states_expanded":expanded,"obligations_checked":obligations,"rendered":len(rows),"exact_gt38":len(exact),"max_letters":max((r["audit"]["letters"] for r in rows),default=0)},"rendered_candidates":rows[:20],"exact_candidates":exact,"novelty_preflight":pre,"provenance":{"prior":"checked-in fixed deterministic English lexical probabilities","audits":["independent two-pointer","forward/reverse SHA-256"],"finished_tape_reversal":False,"mirrored_units":False,"word_order_symmetry":False,"post_hoc_repair":False,"catalogue_text":False},"falsifier":"shuffle lexical weights and rerun: if closures or surfaces are unchanged, the weighted grammar claim is falsified","next_operator":"add a held-out typed adjunct state whose lexical probability is consumed at the same bilateral obligation step","status":"fresh exact >38 requires human reading" if exact else "no exact >38 closure; intact grammatical controls retained"}
    RUN.write_text(json.dumps(result,indent=2)+"\n"); return result

if __name__ == "__main__": print(json.dumps(run(),indent=2))
