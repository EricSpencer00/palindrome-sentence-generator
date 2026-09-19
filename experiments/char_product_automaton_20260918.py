"""Bounded product search: character obligations x a typed phrase automaton.

Unlike the residual decoder, the two sides may be in different grammatical
states.  A small DFA tracks clause shape while a character-level product edge
is consumed immediately; no finished tape is reversed.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path

VOCAB = {
    "DET": ("a", "the", "our", "one"),
    "SUBJ": ("poet", "teacher", "writer", "farmer", "baker", "aide"),
    "VERB": ("reads", "marks", "writes", "guides", "carries"),
    "OBJ": ("a note", "the map", "a poem", "the letter", "some prose"),
    "ADV": ("at dawn", "in town", "by sea", "near home", "today"),
}
GRAMMARS = {
    "clause": (("DET", "SUBJ", "VERB", "OBJ", "ADV"), ("DET", "SUBJ", "VERB", "OBJ"),
               ("SUBJ", "VERB", "OBJ"), ("DET", "SUBJ", "VERB", "ADV")),
}
CORPUS = "the careful poet reads a letter at dawn a teacher marks the map near home our writer carries some prose today"

def norm(s): return "".join(c for c in s.lower() if c.isalpha())
def audit(s):
    t, r = norm(s), norm(s)[::-1]
    hf, hr = hashlib.sha256(t.encode()).hexdigest(), hashlib.sha256(r.encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": all(a == b for a,b in zip(t,r)) and len(t)==len(r),
            "sha256_forward": hf, "sha256_reverse": hr, "sha_equal_under_reversal": hf == hr}
def lm():
    t=norm(CORPUS); c=Counter(zip("^"+t,t+"$")); z=sum(c.values())
    return {k:math.log((v+1)/(z+28)) for k,v in c.items()}
LM=lm()
def score(s):
    t=norm(s); return sum(LM.get((a,b),-math.log(100)) for a,b in zip("^"+t,t+"$"))

def consume(residual, left, right):
    a=residual+norm(left); b=norm(right)[::-1]; k=min(len(a),len(b))
    return None if a[:k] != b[:k] else (a[k:] if len(a)>len(b) else b[k:])

def run(max_states=100000):
    # Product state: (left DFA position, right DFA position, char residual,
    # token histories, score). Positions advance independently, so heterogeneous
    # grammatical slot pairs are explored rather than same-slot matching.
    seqs=GRAMMARS["clause"]; states={(0,0,"",(),(),0.0)}; counts=[]; conflicts=0
    for depth in range(5):
        nxt={}
        for lp,rp,res,lw,rw,sc in states:
            for ls in seqs:
                if lp>=len(ls): continue
                for rs in seqs:
                    if rp>=len(rs): continue
                    for left in VOCAB[ls[lp]]:
                        for right in VOCAB[rs[rp]]:
                            nr=consume(res,left,right)
                            if nr is None: conflicts+=1; continue
                            key=(lp+1,rp+1,nr,lw+(left,),rw+(right,))
                            val=sc+score(left)+score(right)
                            if key not in nxt or val>nxt[key][-1]: nxt[key]=key+(val,)
                            if len(nxt)>=max_states: break
                        if len(nxt)>=max_states: break
                    if len(nxt)>=max_states: break
                if len(nxt)>=max_states: break
            if len(nxt)>=max_states: break
        states=set(nxt.values()); counts.append(len(states))
        if not states: break
    rows=[]
    for lp,rp,res,lw,rw,sc in states:
        if res or lp<3 or rp<3: continue
        text=" ".join(lw+tuple(reversed(rw)))+"."
        rows.append({"rendered":text,"audit":audit(text),"provenance":{"left_tokens":lw,"right_build_tokens":rw,"lm_score":sc},"reader_status":"not_run"})
    controls=["The careful poet reads a letter at dawn.","A teacher marks the map near home."]
    return {"experiment_id":"char-product-automaton-20260918","signature":"typed-clause-DFA-x-character-obligation-product|heterogeneous-slot-pairs|add-one-char-bigram","config":{"grammar_states":len(seqs),"max_states":max_states,"depth":5,"model":"transparent add-one character bigram"},"stats":{"state_counts":counts,"character_conflicts":conflicts,"exact_closures":sum(r['audit']['two_pointer_exact'] for r in rows),"candidate_rows":len(rows)},"rendered_candidates":rows,"prose_controls":[{"rendered":c,"audit":audit(c),"provenance":"hand-authored intact English control; not imported as a palindrome"} for c in controls],"provenance":{"finished_tape_reversed":False,"catalogue_text_imported":False,"validator":"independent two-pointer plus forward/reverse SHA-256","readability_certified":False,"novelty_preflight":"candidate normalized tapes must be checked against data/known_palindromes.json before admission"},"next_repair":"replace slot words with a character trie mined from held-out non-palindromic prose and add a center-closure state; do not treat LM score as readability."}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path); ap.add_argument('--max-states',type=int,default=100000); a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(run(a.max_states),indent=2)+'\n'); print(json.dumps(run(a.max_states)['stats'],indent=2))
if __name__=='__main__': main()
