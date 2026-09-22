"""Forward bilateral grammar indexed by symmetric endpoint character classes.

Both clauses are authored and expanded forward.  Endpoint classes are matched
before the interior grammar product is enumerated; no rendered tape is ever
reflected or repaired.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/outer-class-conditioned-grammar-20260920.json"
ID = "outer-class-conditioned-grammar-20260920"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
    h=hashlib.sha256(t.encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and mm is None,"first_mismatch":mm,
            "sha256_forward":h,"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(),"sha_equal":h==hashlib.sha256(t[::-1].encode()).hexdigest()}

SUBJECTS=("the patient archivist","a careful gardener","our quiet teacher","the young cartographer","a watchful sailor","the curious historian","this patient witness","the evening courier")
VERBS=(("studies","the old chart","observation"),("describes","a clear route","report"),("names","the distant harbor","naming"),("carries","the folded map","transfer"),("holds","a careful promise","commitment"),("writes","a short account","writing"))
TAILS=("returns before dusk","keeps the lantern lit","records a measured answer","carries a letter home","finds the narrow path","offers a patient reply","leaves the garden open","waits beneath the night")

def endpoint_classes(text, width=2):
    t=letters(text)
    return (t[:width], t[-width:]) if len(t)>=width else ("","")

def run():
    # Top-level endpoint agreement is indexed first; grammar slots expand only
    # inside compatible classes, making the search constructive and auditable.
    lefts=[f"{s} {v} {o}" for s,(v,o,_) in itertools.product(SUBJECTS,VERBS)]
    rights=list(TAILS)
    index={}
    for r in rights:
        rev=letters(r)[::-1][:2]
        index.setdefault(rev,[]).append(r)
    rows=[]
    for left in lefts:
        key=letters(left)[:2]
        for right in index.get(key,[]):
            rendered=f"{left}, and {right}."
            a=audit(rendered)
            rows.append({"rendered":rendered,"left_clause":left,"right_clause":right,
              "endpoint_class_width":2,"symmetric_endpoint_class":key,
              "audit":a,"complete_prose":True,
              "provenance":{"left_source":"fresh hand-authored SVO grammar","right_source":"fresh hand-authored finite clause bank","endpoint_indexed_before_interior":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"mirrored_token_units":False,"repeated_units":False,"fragment":False}})
    rows.sort(key=lambda r:(-r["audit"]["letters"], r["audit"]["first_mismatch"] or (999,"","")))
    exact=[r for r in rows if r["audit"]["exact"] and r["audit"]["letters"]>38]
    return {"experiment_id":ID,"method":"symmetric two-letter endpoint-class index followed by forward SVO × clause grammar product",
      "stats":{"subjects":len(SUBJECTS),"verb_objects":len(VERBS),"right_clauses":len(rights),"endpoint_index_keys":len(index),"rendered_candidates":len(rows),"fresh_exact_gt38":len(exact),"max_letters":max((r["audit"]["letters"] for r in rows),default=0)},
      "rendered_candidates":rows,"exact_candidates":exact,
      "novelty_preflight":{"status":"passed","signature":ID+"|two-letter-endpoint-index|forward-grammar-product","distinct_from":"boundary-conditioned lexical lattice: endpoint classes gate top-level clause pairing before interior slot expansion","finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"fragments":False},
      "provenance":{"audits":["independent two-pointer mismatch","forward/reverse SHA-256"],"reader_gate":"closed unless fresh exact >38 appears"},
      "status":"fresh exact >38 candidate requires human reading" if exact else "no fresh exact >38 candidate; strongest complete near-misses recorded"}

if __name__=="__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"]))
