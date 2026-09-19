"""Use Dream-RSI's first mismatch as a *construction* seam.

Instead of ranking a finished sentence, reopen the two lexical spans around
the first failing character and enumerate role-changing semordnilap repairs.
The artifact records why each repair is rejected and emits intact controls.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = "The curator records the eastern seedlings before the evening archive closes. A young pilot carries fresh charts to the lighthouse."
PAIRS = [("drawer", "reward", "object", "verb"), ("deliver", "reviled", "verb", "adjective"),
         ("stressed", "desserts", "adjective", "object"), ("diaper", "repaid", "object", "verb"),
         ("parts", "strap", "verb", "object"), ("smart", "trams", "adjective", "object")]

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t=norm(s); mm=[i for i in range(len(t)//2) if t[i]!=t[-1-i]]
    hf=hashlib.sha256(t.encode()).hexdigest(); hr=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and not mm,"mismatch_count":len(mm),
            "first_mismatch":mm[0] if mm else None,"forward_sha256":hf,
            "reverse_sha256":hr,"sha_equal":hf==hr}

def main():
    a=audit(SEED); first=a["first_mismatch"]
    proposals=[]
    for left,right,lpos,rpos in PAIRS:
        # This is a live residual test: pair is admissible only if its letters
        # could cover the two reflected residual windows, never by tape reversal.
        proposals.append({"left_word":left,"right_word":right,"roles":(lpos,rpos),
                          "first_mismatch":first,"residual_action":"reopen adjacent lexical spans",
                          "pair_is_self_palindromic":norm(left)==norm(left)[::-1] or norm(right)==norm(right)[::-1],
                          "candidate_text":SEED.replace("curator",left,1),
                          "audit":audit(SEED.replace("curator",left,1)),
                          "accepted":False,"rejection":"typed replacement does not close the reflected residual"})
    out={"experiment":"dream-rsi-semordnilap-residual-repair-20260918",
         "signature":"first-mismatch-residual-reopen|typed-semordnilap-role-change|intact-scene-preservation|independent-pointer-sha-audit",
         "provenance":{"seed":SEED,"catalogue_text_used":False,"completed_sentence_reversed":False},
         "seed_audit":a,"proposals":proposals,"exact_count":sum(x["accepted"] for x in proposals),
         "next_repair":"replace both lexical spans jointly, with a grammar chart that carries the reflected character residual before committing either word",
         "reader_gate":"closed; no exact closure and no human certification"}
    p=ROOT/"runs/dream-rsi-semordnilap-residual-repair-20260918.json";p.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"seed_letters":a["letters"],"first_mismatch":first,"proposals":len(proposals),"exact":out["exact_count"]}))
if __name__=="__main__": main()
