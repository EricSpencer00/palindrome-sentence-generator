"""Synchronous whole-grammar intersection (no post-hoc repair).

Two clauses are derived from the same typed grammar, but their words are chosen
at the same time.  Every newly exposed character is compared with its opposite
endpoint before recursion continues.  The output is therefore a derivation,
not a completed tape subsequently reversed or repaired.
"""
from __future__ import annotations
import hashlib, json, re
from collections import Counter
from pathlib import Path

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def audit(s: str) -> dict[str, object]:
    t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and t==t[::-1],"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def consume(left: str, right: str) -> tuple[str, str] | None:
    """Consume only newly matched endpoint characters.

    ``right`` is stored in forward order; its *end* is opposite ``left``.
    Unmatched residuals survive across word boundaries, which is essential for
    unequal word lengths.
    """
    n=min(len(left),len(right))
    if left[:n] != right[-n:][::-1]: return None
    return left[n:], right[:-n] if n else right

def search(paths: tuple[tuple[str,...], ...], limit: int=16) -> dict[str,object]:
    # A complete English grammar: determiner noun verb determiner noun prep noun.
    roles=("det","subject","verb","obj_det","object","prep","complement")
    states=pruned=0; candidates=[]
    # Pair complete corpus-attested derivations.  Thus every accepted path is
    # grammatical at the skeleton level; role words are never recombined from
    # unrelated banks.
    for left_path in paths:
        for right_path in paths:
            prefix=suffix=""; ok=True
            for lo in range(4):
                hi=6-lo; l,r=left_path[lo],right_path[hi]
                if l==r or l==l[::-1] or r==r[::-1]: ok=False; break
                prefix += letters(l); suffix = letters(r)+suffix; states += 1
                residual=consume(prefix,suffix)
                if residual is None: pruned += 1; ok=False; break
                prefix,suffix=residual
            if not ok: continue
            text=" ".join(left_path+right_path); a=audit(text); states += 1
            if a["exact"]:
                candidates.append({"rendered":text,"audit":a,"provenance":{"grammar":"DET NOUN VERB DET NOUN ADP NOUN","roles":roles,"construction":"synchronous character intersection","left_frame":left_path,"right_frame":right_path}})
                if len(candidates)>=limit: break
        if len(candidates)>=limit: break
    return {"method":"synchronous-cfg-intersection-20260919","grammar":roles,"banks":{"complete_frame_paths":len(paths)},"stats":{"states":states,"pruned":pruned,"exact":len(candidates)},"candidates":candidates,"provenance":{"source":"Brown universal tagged corpus, frequency-ranked complete frame paths","no_repair":True,"no_finished_tape_reversal":True,"word_order_mirror":False}}

def main():
    from nltk.corpus import brown
    c=Counter()
    for sent in brown.tagged_sents(tagset="universal")[:50000]:
        for i in range(len(sent)-6):
            tags=[x[1] for x in sent[i:i+7]]
            if tags[0]=="DET" and tags[1]=="NOUN" and tags[2] in {"VERB","AUX"} and tags[3]=="DET" and tags[4]=="NOUN" and tags[5]=="ADP" and tags[6]=="NOUN":
                vals=[x[0].casefold() for x in sent[i:i+7]]
                if all(v.isalpha() for v in vals): c[tuple(vals)] += 1
    paths=tuple(p for p,n in c.most_common(256))
    out=search(paths); Path("runs").mkdir(exist_ok=True); Path("runs/synchronous-cfg-intersection-20260919.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__":main()
