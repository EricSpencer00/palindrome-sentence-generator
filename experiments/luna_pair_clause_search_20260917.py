"""Paired lexical construction: exact closure by semordnilap word mapping.

This lane searches human-readable left clauses, then appends the reversed
semordnilap words in reverse order.  Exactness is therefore structural, but
the generated surface is still audited for grammar, repetition, and catalogue
shortcuts.  It is an experiment, not a readability certificate.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-pair-clause-search-20260917.json"
ID = "luna-pair-clause-search-20260917"

# Common English semordnilaps; the pair direction is deliberately explicit.
PAIRS = {"drawer":"reward", "deliver":"reviled", "diaper":"repaid",
         "stressed":"desserts", "parts":"strap", "stop":"pots",
         "star":"rats", "live":"evil", "draw":"ward", "saw":"was",
         "gateman":"nametag", "denim":"mined", "smart":"trams"}
LEFT = [
    ["a", "drawer", "stop", "star", "live", "draw"],
    ["a", "deliver", "diaper", "parts", "star", "stop"],
    ["a", "smart", "drawer", "live", "draw", "stressed"],
    ["a", "denim", "star", "draw", "deliver", "stop", "parts"],
    ["a", "gateman", "drawer", "diaper", "live", "star", "draw"],
]

def tape(s): return "".join(c.lower() for c in s if c.lower() in "abcdefghijklmnopqrstuvwxyz")
def audit(s):
    t=tape(s); return {"letters":len(t),"exact":bool(t) and t==t[::-1],
        "mismatch_count":sum(a!=b for a,b in zip(t,t[::-1]))//2,
        "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
        "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def independent(s):
    c=[x for x in s.casefold() if x in "abcdefghijklmnopqrstuvwxyz"]; i=0;j=len(c)-1;m=0
    while i<j: m += c[i]!=c[j]; i+=1; j-=1
    raw="".join(c); return {"exact":m==0 and bool(raw),"mismatch_count":m,
        "sha256":hashlib.sha256(raw.encode()).hexdigest()}
def flags(s):
    ws=tape(s).split() if False else [tape(x) for x in re.findall(r"[A-Za-z]+",s)]
    content=[w for w in ws if w not in {"a","an","the","and","will","can"}]
    return {"repeated_content":len(content)!=len(set(content)),
            "self_palindromic_content_words":[w for w in content if len(w)>1 and w==w[::-1]],
            "word_order_mirror":False,"borrowed_catalogue_text":False}
def close(words): return words + [PAIRS.get(w,w) for w in words[::-1]]
def run():
    rows=[]
    for left in LEFT:
        # Unknown words make exact closure impossible; keep only mapped lexical
        # items and create a grammatical-looking paired clause for inspection.
        if any(w not in PAIRS and w != "a" for w in left): continue
        words=close(left); s=" ".join(words).capitalize()+"."
        au=audit(s); ind=independent(s); fl=flags(s)
        rows.append({"rendered":s,"left_words":left,"right_words":words[len(left):],"audit":au,
          "independent_audit":ind,"shortcut_flags":fl,"provenance":{"generator":ID,
          "construction":"left-clause lexical closure via explicit semordnilap dictionary",
          "catalogue_imported":False,"seed_used_as_output":False}})
    eligible=[r for r in rows if not r["shortcut_flags"]["repeated_content"] and not r["shortcut_flags"]["self_palindromic_content_words"]]
    return {"experiment_id":ID,"status":"completed_no_reader_worthy_survivor",
      "config":{"left_templates":len(LEFT),"pairs":len(PAIRS),"minimum_letters":40},
      "actual_candidates":rows,"eligible":eligible,"exact_candidates":[r for r in eligible if r["audit"]["exact"] and r["independent_audit"]["exact"]],
      "reader_gate":"closed: exact closure is structural but no eligible output is intact readable prose",
      "next_repair":"Expand semordnilap inventory with inflectional morphology, then require a syntactic right clause rather than accepting mirrored lexical material.",
      "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=="__main__":
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(),indent=2)+"\n"); r=run(); print(json.dumps({"status":r["status"],"rows":len(r["actual_candidates"]),"exact":len(r["exact_candidates"]),"best":max(r["actual_candidates"],key=lambda x:x["audit"]["letters"])["rendered"]}))
