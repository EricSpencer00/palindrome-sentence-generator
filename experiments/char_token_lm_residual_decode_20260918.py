"""Joint character-residual decoding with a transparent token LM.

The decoder expands grammatical token chunks on both outside edges.  Each
new character is checked against its mate immediately; no completed tape is
reversed and no catalogue sentence is imported.  The tiny character model is
only a search prior: exactness is audited independently and readability is
left to readers.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

VOCAB = {
    "DET": ("a", "the", "our", "one"),
    "SUBJ": ("aide", "baker", "curator", "farmer", "poet", "teacher", "writer"),
    "VERB": ("reads", "marks", "writes", "carries", "inspires", "guides"),
    "OBJ": ("a note", "the map", "nine memos", "a poem", "the letter", "some prose"),
    "ADV": ("at dawn", "in town", "by sea", "near home", "today"),
}
SLOTS = ("DET", "SUBJ", "VERB", "OBJ", "ADV")
CORPUS = "the careful poet reads a letter at dawn a teacher marks the map near home our writer carries some prose today"

def norm(s): return normalize_letters(s)
def lm_counts():
    c = Counter(zip("^" + norm(CORPUS), norm(CORPUS) + "$")); total = sum(c.values())
    return {k: math.log((v + 1) / (total + 28)) for k, v in c.items()}
LM = lm_counts()
def score(s):
    t = norm(s); return sum(LM.get((a,b), -math.log(100)) for a,b in zip("^"+t,t+"$"))
def audit(s):
    t=norm(s); r=t[::-1]; hf=hashlib.sha256(t.encode()).hexdigest(); hr=hashlib.sha256(r.encode()).hexdigest()
    return {"letters":len(t), "two_pointer_exact": bool(t) and all(a==b for a,b in zip(t,r)),
            "mismatch_count":sum(a!=b for a,b in zip(t,r)), "sha256_forward":hf,
            "sha256_reverse":hr, "sha_equal_under_reversal":hf==hr}
def choices(slot):
    return VOCAB[slot]
def consume(residual, left, right):
    """Compare newly emitted left chars with newly emitted reversed-right chars."""
    a=residual + norm(left); b=norm(right)[::-1]
    k=min(len(a),len(b))
    if a[:k] != b[:k]: return None
    return a[k:] if len(a)>len(b) else (b[k:] if len(b)>len(a) else "")
def run(max_states=50000):
    states={("",(),(),0.0)}; counts=[]; pruned=0
    # Both sides use independent grammatical slot choices; right is built
    # from its outside edge, so its token order is reversed at rendering.
    for slot in SLOTS:
        nxt={}
        for residual,lw,rw,sc in states:
            for left in choices(slot):
                for right in choices(slot):
                    nr=consume(residual,left,right)
                    if nr is None: pruned+=1; continue
                    key=(nr,lw+(left,),rw+(right,))
                    old=nxt.get(key)
                    val=sc+score(left)+score(right)
                    if old is None or val>old[3]: nxt[key]=(nr,lw+(left,),rw+(right,),val)
                    if len(nxt)>=max_states: break
                if len(nxt)>=max_states: break
            if len(nxt)>=max_states: break
        states=set(nxt.values()); counts.append(len(states))
        if not states: break
    rows=[]
    for residual,lw,rw,sc in states:
        if residual: continue
        text=" ".join(lw+tuple(reversed(rw)))+"."
        a=audit(text); checks=mechanical_admission_checks(text,min_letters=30,max_letters=220)
        rows.append({"rendered":text,"letters":a["letters"],"normalized_tape":norm(text),
                     "provenance":{"left_tokens":lw,"right_build_tokens":rw,"lm_score":sc},
                     "audit":a,"mechanical_checks":checks,
                     "mechanically_admitted":a["two_pointer_exact"] and all(checks.values()),
                     "reader_status":"not_run; programmatic scores never certify readability"})
    rows.sort(key=lambda x:(-x["mechanically_admitted"],-x["letters"],-x["provenance"]["lm_score"]))
    controls=["The careful poet reads a letter at dawn.","A teacher marks the map near home."]
    return {"experiment_id":"char-token-lm-residual-decode-20260918",
            "signature":"joint-token-expansion|character-residual|transparent-bigram-prior|independent-audit",
            "config":{"slots":SLOTS,"max_states":max_states,"model":"add-one character bigram over embedded prose"},
            "stats":{"state_counts":counts,"pruned_character_conflicts":pruned,"exact_closures":len(rows),
                     "mechanically_admitted":sum(x["mechanically_admitted"] for x in rows)},
            "rendered_candidates_and_controls":rows+[{"rendered":c,"control":True,"audit":audit(c),"reader_status":"intact prose control; not a palindrome"} for c in controls],
            "provenance":{"finished_tape_reversed":False,"catalogue_text_imported":False,
                          "exact_validator":"two-pointer plus forward/reverse SHA-256","readability_certified":False},
            "next_repair":"retain the live residual but allow heterogeneous slot pairings and a character trie over inflected phrase tokens; then run blinded intact-prose controls."}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True,type=Path); ap.add_argument("--max-states",type=int,default=50000); a=ap.parse_args()
    if a.out.exists(): ap.error("refusing to overwrite existing output")
    x=run(a.max_states); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(x,indent=2)+"\n"); print(json.dumps(x["stats"],indent=2))
if __name__=="__main__": main()
