"""Small semantic-slot mutations of one exact near-survivor.

This lane is deliberately held out: it edits only paired letter slots in the
single supplied tape, rather than resampling a palindrome corpus.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "luna-exact-near-survivor-repair-20260917"
SIGNATURE = "one-heldout-101-letter-tape|paired-semantic-slot-mutation|two-pointer-sha"
BASE = "Levels same tales rows ties reversed ace. Demanded net tasks asks attended name decades. Reverse its worse late mass level."
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

def tape(s): return "".join(re.findall(r"[A-Za-z]", s)).lower()

def audit(s):
    t=tape(s); i,j=0,len(t)-1; bad=[]
    while i<j:
        if t[i]!=t[j]: bad.append((i,j,t[i],t[j]))
        i+=1; j-=1
    f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"algorithm":"independent-two-pointer-over-normalized-tape-plus-forward-reverse-sha256",
            "letters":len(t),"two_pointer_exact":bool(t) and not bad,
            "mismatch_count":len(bad),"first_mismatch":bad[0] if bad else None,
            "sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def mutate(text, old, new):
    # Preserve the complete rendering and mutate one semantic-looking token;
    # its mirrored character interval is changed to the reverse spelling.
    chars=list(text); letters=[k for k,c in enumerate(chars) if c.isalpha()]
    norm=tape(text); start=norm.index(old); end=start+len(old)
    for p,ch in zip(letters[start:end], new): chars[p]=ch
    # paired positions are selected from the corresponding residual interval.
    for q,ch in zip(letters[len(norm)-end:len(norm)-start], new[::-1]): chars[q]=ch
    return "".join(chars)

def main():
    registry=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    entries=registry.get("entries",[])
    collisions=[e.get("id") for e in entries if e.get("id")==EXPERIMENT or e.get("signature")==SIGNATURE]
    pre={"performed_before_rendering":True,"registry_entries_inspected":len(entries),
         "id_or_signature_collisions":collisions,"passed":not collisions,
         "distinction":"one held-out exact tape; two paired lexical-slot substitutions; no sweep"}
    specs=[("same","calm","adjective slot"),("tales","notes","noun slot"),("worse","quiet","adjective slot")]
    rows=[]
    for old,new,slot in specs:
        rendered=mutate(BASE,old,new)
        a=audit(rendered)
        rows.append({"id":old+"-to-"+new,"rendered":rendered,"letters":a["letters"],
          "phase":"paired-semantic-slot-repair","changed_slot":slot,"replacement":new,
          "exact_audit":a,"mechanically_admitted":False,
          "provenance":{"source":"user-supplied 101-letter exact near-survivor",
            "base_text":BASE,"novelty_signature":SIGNATURE,"catalogue_imported":False,
            "borrowed_text":False,"reversed_finished_sentence":False,"word_order_mirror":False},
          "readability_status":"not certified; requires blinded human reading",
          "next_repair":"replace the first residual pair with a role-compatible multiword clause while retaining the same paired-slot ledger"})
    out={"experiment":EXPERIMENT,"novelty_preflight":pre,"base":{"rendered":BASE,"audit":audit(BASE)},"rows":rows,
         "stats":{"base_letters":audit(BASE)["letters"],"mutations":len(rows),"exact":sum(r["exact_audit"]["two_pointer_exact"] for r in rows),"mechanically_admitted":0},
         "next_repair":"Use a typed clause-level substitution at the first residual, not another lexical resampling sweep."}
    OUT.write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__": main()
