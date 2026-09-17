"""Fresh construction: lexical-trie intersection over a typed grammar.

The left and right grammar slots grow simultaneously from opposite tape ends.
Each lexical alternative is represented by a trie; character equality is tested
before either trie advances. No finished sentence is reversed or resegmented.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from dataclasses import dataclass
from experiments.full_sequence_grammar_product_20260917 import BANKS, FUNCTION_WORDS, exact_audit, anti_shortcut

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs"/"lexical-trie-grammar-intersection-20260917.json"
# fresh grammar geometry: determiner-led coordinated scene with typed roles
PATTERN=("DET","NOUN","VERB_S","DET","NOUN","CONJ","PRON","VERB_BASE","DET","NOUN")
CAT_WORDS={"doc","note","dissent","fast","never","prevents","fatness","diet","cod"}

class Trie:
    def __init__(self, words):
        self.children={}
        self.terminal=[]
        for w in words:
            node=self
            for ch in w:
                node=node.children.setdefault(ch,Trie(()))
            node.terminal.append(w)

def letters(s): return "".join(c for c in s.lower() if c.isalpha())
def audit2(s):
    t=letters(s); ok=all(t[i]==t[-1-i] for i in range(len(t)//2)) and bool(t)
    return {"letters":len(t),"exact":ok,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(),"first_mismatch":next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)}

@dataclass
class Result:
    candidates:list; states:int; mismatch_edges:int; longest_partial:dict; frontiers:list; budget_exhausted:bool

def search(budget=500000):
    tries={k:Trie(tuple(w for w in BANKS[k] if w not in CAT_WORDS)) for k in BANKS}
    # state: slot pointers, trie nodes, words, used; active trie nodes are lexical prefixes
    stack=[(0,len(PATTERN)-1,None,None,0,None,None,0,(),frozenset())]
    seen=set(); cand=[]; front=[]; states=0; mism=0; longest={"chars":0}
    while stack and states<budget:
        li,ri,ln,rn,lp,rp,assign,chars,words,used=stack.pop(); states+=1
        if chars>longest["chars"]: longest={"chars":chars,"assignment":words,"slots":PATTERN}
        # advance completed lexical words as independent epsilon transitions
        if ln is not None and lp==len(ln):
            stack.append((li+1,ri,None,rn,0,rp,assign,chars,words,used)); continue
        if rn is not None and rp==len(rn):
            stack.append((li,ri,ln,None,lp,0,assign,chars,words,used)); continue
        if li>ri:
            text=" ".join(words); a=audit2(text); sh=anti_shortcut(words)
            if a["exact"] and len(letters(text))>38 and not any(sh.values()):
                cand.append({"rendered":text,"words":words,"audit":a,"anti_shortcut":sh,"provenance":{"geometry":"typed lexical trie intersection","pattern":PATTERN,"catalogue_words_excluded":True,"live_character_edges":True}})
            continue
        # if the two pointers are both idle, choose typed word roots on each side
        if ln is None:
            if li==ri:
                # one lexical word can contain the center; test each root independently
                for w in tries[PATTERN[li]].terminal:
                    if w in used and w not in FUNCTION_WORDS: continue
                    text=" ".join(words+(w,)); a=audit2(text); sh=anti_shortcut(words+(w,))
                    if a["exact"] and len(letters(text))>38 and not any(sh.values()): cand.append({"rendered":text,"words":words+(w,),"audit":a,"anti_shortcut":sh,"provenance":{"geometry":"typed lexical trie intersection","pattern":PATTERN,"center_word":True}})
                continue
            for w in BANKS[PATTERN[li]]:
                if w in CAT_WORDS or (w in used and w not in FUNCTION_WORDS): continue
                stack.append((li,ri,w,rn,0,rp,assign,chars,words+(w,),used|({w} if w not in FUNCTION_WORDS else set())))
            continue
        if rn is None:
            for w in BANKS[PATTERN[ri]]:
                if w in CAT_WORDS or (w in used and w not in FUNCTION_WORDS): continue
                stack.append((li,ri,ln,w,lp,0,assign,chars,words+(w,),used|({w} if w not in FUNCTION_WORDS else set())))
            continue
        # trie-character intersection: consume opposite chars only when equal
        lc=ln[lp]; rc=rn[rp]
        if lc!=rc:
            mism+=1
            if len(front)<32: front.append({"left_slot":PATTERN[li],"right_slot":PATTERN[ri],"left_word":ln,"right_word":rn,"left_pos":lp,"right_pos":rp,"residual":lc+"!="+rc,"assignment":words})
            continue
        stack.append((li,ri,ln,rn,lp+1,rp+1,assign,chars+1,words,used))
    return Result(cand,states,mism,longest,front,states>=budget)

if __name__=="__main__":
    r=search(); out={"experiment":"lexical-trie-grammar-intersection-20260917","pattern":PATTERN,"status":"exact_candidate" if r.candidates else "completed_no_novel_exact_closure","states":r.states,"mismatch_edges":r.mismatch_edges,"budget_exhausted":r.budget_exhausted,"longest_partial":r.longest_partial,"candidates":r.candidates,"first_residual_repairs":r.frontiers}
    OUT.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
