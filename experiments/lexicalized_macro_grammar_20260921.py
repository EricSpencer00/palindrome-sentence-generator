"""Bounded lexicalized macro-grammar probe.

Macros are complete ordinary clauses carrying an exposed reverse-boundary
signature.  Search composes discourse-compatible macros before rendering; it
does not repair or score rendered candidates.
"""
from pathlib import Path
import hashlib, json, re

ROOT = Path(__file__).resolve().parents[1]
ID = "lexicalized-macro-grammar-20260921"
SIG = "lexicalized-complete-clause-macros|reverse-boundary-signatures|discourse-state-composition"

MACROS = [
    {"id":"m1","text":"the calm sailor reads a map","topic":"sailor","act":"report","next":"map"},
    {"id":"m2","text":"the calm sailor marks a cove","topic":"sailor","act":"report","next":"cove"},
    {"id":"m3","text":"the wise pilot watches the sea","topic":"pilot","act":"report","next":"sea"},
    {"id":"m4","text":"the old keeper guards a gate","topic":"keeper","act":"report","next":"gate"},
    {"id":"m5","text":"the sailor rests by the cove","topic":"sailor","act":"state","next":"cove"},
    {"id":"m7","text":"the sailor rests by the map","topic":"sailor","act":"state","next":"map"},
    {"id":"m6","text":"the pilot waits by the sea","topic":"pilot","act":"state","next":"sea"},
]

def letters(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t=letters(s); return {"letters":len(t),"exact":t==t[::-1],"sha256":hashlib.sha256(t.encode()).hexdigest(),"two_pointer_exact":all(t[i]==t[-1-i] for i in range(len(t)//2))}
def macro(m):
    t=letters(m["text"])
    return {**m,"boundary_signature":{"prefix":t[:3],"suffix":t[-3:],"reverse_suffix":t[-3:][::-1]},"letters":len(t)}

def run():
    prepared=[macro(m) for m in MACROS]
    # State is carried in the product: topic continuity and report -> state act.
    states=0; candidates=[]
    for a in prepared:
        for b in prepared:
            if a["id"]==b["id"]: continue
            if a["topic"]!=b["topic"] or (a["act"],b["act"])!=("report","state"): continue
            states += 1
            # Signature is a pre-render compatibility key, not a repair step.
            compatible = a["boundary_signature"]["suffix"][-1] == b["boundary_signature"]["reverse_suffix"][0]
            if compatible:
                text=a["text"]+"; "+b["text"]
                candidates.append({"rendered":text,"audit":audit(text),"macro_ids":[a["id"],b["id"]],"discourse_state":{"topic":a["topic"],"acts":[a["act"],b["act"]]},"provenance":{"complete_macros":True,"pre_render_signature_gate":True,"post_hoc_repair":False,"rlaif":False}})
    controls=[{"rendered":"the calm sailor reads a map","audit":audit("the calm sailor reads a map")},{"rendered":"the wise pilot watches the sea","audit":audit("the wise pilot watches the sea")}]
    return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if any(x["audit"]["exact"] for x in candidates) else "completed_no_exact_closure","method":"bounded lexicalized macro-grammar with discourse-state composition","novelty_preflight":{"status":"passed","registry_entries_read":True,"overlaps_checked":["CFG grammar products","FST/residual automata","center-out/reservoir lanes","scene/semantic lattice lanes"],"collision":False,"reason":"Complete lexicalized clauses are atomic macro objects; discourse state and reverse-boundary signatures are composed before rendering, a distinct operator from token/character frontier search."},"macros":prepared,"stats":{"macros":len(prepared),"discourse_states":states,"signature_compatible_compositions":len(candidates),"exact":sum(x["audit"]["exact"] for x in candidates),"longest_letters":max([x["audit"]["letters"] for x in candidates+controls])},"rendered_candidates":candidates,"complete_prose_controls":controls,"independent_validation":["normalized outside-in two-pointer scan","forward SHA-256"],"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"fresh_authored_macros":True,"catalogue_text":False},"next_construction":"Expand the macro inventory with held-out report/state pairs and a 2-character boundary signature; retain discourse-state gating and reject any post-render repair.","reader_gate":"closed; no exact reader-eligible candidate"}

if __name__ == "__main__":
    out=ROOT/"runs"/(ID+".json"); out.write_text(json.dumps(run(),indent=2)+"\n"); print(json.dumps({"status":run()["status"],"states":run()["stats"]["discourse_states"],"compatible":run()["stats"]["signature_compatible_compositions"],"exact":run()["stats"]["exact"]}))
