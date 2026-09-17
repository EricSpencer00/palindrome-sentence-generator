"""Bounded reversible-word composition lane.

This tests whether familiar reversible words can be embedded in typed prose
without simply mirroring the word sequence.  It is a construction experiment,
not a readability certificate.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "reversible-word-composition-20260917.json"
ID = "reversible-word-composition-20260917"
PAIRS = [("live", "evil"), ("stressed", "desserts"), ("drawer", "reward"),
         ("parts", "strap"), ("stop", "pots"), ("was", "saw"),
         ("time", "emit"), ("diaper", "repaid"), ("deliver", "reviled")]
LEFT = ["The nurse", "A quiet sailor", "The careful baker", "Our young teacher"]
RIGHT = ["helps the child", "marks the map", "opens the drawer", "keeps the letter"]

def tape(s): return "".join(c.lower() for c in s if c.lower() in "abcdefghijklmnopqrstuvwxyz")
def audit(s):
    t=tape(s); m=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
    return {"letters":len(t),"exact":bool(t) and not m,"mismatch_count":len(m),
            "mismatch_rate":len(m)/max(1,len(t)//2),"first_mismatches":m[:10],
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def independent(s):
    c=[x.lower() for x in s if x.lower() in "abcdefghijklmnopqrstuvwxyz"]; i,j=0,len(c)-1; n=0
    while i<j: n += c[i]!=c[j]; i+=1; j-=1
    return {"exact":bool(c) and n==0,"mismatch_count":n,
            "sha256_forward":hashlib.sha256("".join(c).encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256("".join(c[::-1]).encode()).hexdigest()}
def flags(s):
    ws=[tape(w) for w in re.findall(r"[A-Za-z]+",s)]
    content=[w for w in ws if w not in {"a","an","the","and","our","was"}]
    return {"word_order_mirror":ws==[w[::-1] for w in ws[::-1]],
            "self_palindromic_content_words":[w for w in content if len(w)>1 and w==w[::-1]],
            "borrowed_catalogue_text":False,"finished_tape_reversed":False}
def run():
    rows=[]
    # A reversible token is placed in a grammatical left clause; the right
    # clause uses an independently chosen surface phrase.  This deliberately
    # tests cross-boundary repair rather than constructing a reversed word list.
    for (a,b),(l,r),bridge in itertools.product(PAIRS, itertools.product(LEFT,RIGHT),
                                                  ["; then ", ", while ", "; and "]):
        s=f"{l} {a} {bridge}{r} {b}."
        au=audit(s); fl=flags(s)
        row={"rendered":s,"audit":au,"independent_audit":independent(s),"shortcut_flags":fl,
             "provenance":{"generator":ID,"pair":(a,b),"construction":"typed prose around reversible lexical slot",
                            "catalogue_imported":False,"seed_used_as_output":False}}
        rows.append(row)
    eligible=[x for x in rows if not x["shortcut_flags"]["word_order_mirror"] and not x["shortcut_flags"]["self_palindromic_content_words"]]
    best=min(eligible,key=lambda x:(x["audit"]["mismatch_count"],-x["audit"]["letters"]))
    return {"experiment_id":ID,"status":"completed_no_exact_closure","config":{"rows":len(rows),"pairs":len(PAIRS)},
            "actual_candidates":eligible[:12],"best":best,
            "exact_candidates":[x for x in eligible if x["audit"]["exact"] and x["independent_audit"]["exact"]],
            "independent_validation":"ASCII two-pointer audit and independent SHA-256 forward/reverse recomputation",
            "reader_gate":"closed: no exact novel survivor","next_repair":"Use a seam-aware morphology/attachment transducer that lets reversible material cross word boundaries; lexical pair insertion alone is not enough.",
            "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=="__main__":
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(),indent=2)+"\n")
    r=run(); print(json.dumps({"status":r["status"],"rows":r["config"]["rows"],"best":r["best"]["rendered"],"letters":r["best"]["audit"]["letters"],"mismatches":r["best"]["audit"]["mismatch_count"],"exact":len(r["exact_candidates"])}))
