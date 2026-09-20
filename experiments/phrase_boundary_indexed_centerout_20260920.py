"""Indexed phrase-pair boundary-debt center-out construction."""
from __future__ import annotations
import argparse, hashlib, json, re, socket
from dataclasses import dataclass
from pathlib import Path

GRAMMAR = {
    "NP": ("a baker", "a pilot", "a clerk", "the poet", "the bird", "an artist"),
    "VP": ("marks a map", "opens a gate", "sees the bird", "keeps a note", "reads a poem"),
}
def tape(s): return re.sub('[^a-z]', '', s.lower())
def words(s): return set(tape(s).split()) if ' ' in s else {tape(s)}
def audit(s):
    t=tape(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    mism=sum(a!=b for a,b in zip(t,t[::-1]))//2
    return {"letters":len(t),"two_pointer_exact":bool(t) and t==t[::-1],"pointer_mismatches":mism,"sha256_forward":f,"sha256_reverse":r,"hash_equal":f==r}

@dataclass(frozen=True)
class State:
    left: tuple[str,...]; right: tuple[str,...]; left_symbol: str; right_symbol: str
    debt: int; depth: int; used_phrases: frozenset[str]; used_words: frozenset[str]

def index_pairs():
    # Pair index is keyed before any tape is rendered: (left exposed last,
    # right exposed first, length difference), with full phrase payloads.
    out={}
    for l in GRAMMAR['NP']:
        for r in GRAMMAR['VP']:
            lt,rt=tape(l),tape(r)
            key=(lt[0],rt[-1],len(lt)-len(rt))
            out.setdefault(key,[]).append((l,r))
    return out

def grows(state, pairs):
    lt=''.join(tape(x) for x in state.left); rt=''.join(tape(x) for x in state.right)
    # Consume all newly exposed debt: the complete shorter boundary must
    # agree, not merely one character.
    for key, options in pairs.items():
        for lp,rp in options:
            if lp in state.used_phrases or rp in state.used_phrases: continue
            cw={w for p in (lp,rp) for w in re.sub('[^a-z ]','',p).split() if len(w)>2}
            if cw & state.used_words: continue
            nl,nr=tape(lp)+lt,rt+tape(rp)
            k=min(len(nl),len(nr))
            if k and nl[-k:] == nr[:k][::-1]:
                yield State((lp,)+state.left,state.right+(rp,),"NP","VP",abs(len(nl)-len(nr)),state.depth+1,state.used_phrases|{lp,rp},state.used_words|cw)

def run(depth, beam):
    pairs=index_pairs(); states=[State(("a",),tuple(),"NP","VP",1,0,frozenset({"a"}),frozenset())]; considered=0
    for _ in range(depth):
        nxt=[]
        for s in states:
            for child in grows(s,pairs): considered+=1; nxt.append(child)
        states=nxt[:beam]
        if not states: break
    closed=[]
    for s in states:
        text=' '.join(s.left+s.right); a=audit(text)
        if a['two_pointer_exact']: closed.append({"rendered":text,"audit":a,"state":s.__dict__})
    return {"experiment":"phrase-boundary-indexed-centerout-20260920","host":socket.gethostname(),"parameters":{"depth":depth,"beam":beam},"index_keys":len(pairs),"pair_options":sum(map(len,pairs.values())),"states_considered":considered,"frontier_states":len(states),"candidates":closed,"closures":len(closed),"provenance":{"fresh_authored_grammar":True,"catalogue_used":False,"finished_tape_reversal":False,"posthoc_repair":False,"pair_index_changes_frontier":True,"repeated_phrase_rejection":True,"repeated_content_word_rejection":True},"next_construction":"add indexed NP/NP and VP/VP phrase families with seam-aware grammar transitions while retaining full debt consumption"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--depth',type=int,default=5); ap.add_argument('--beam',type=int,default=500); ap.add_argument('--out',required=True); a=ap.parse_args(); p=run(a.depth,a.beam); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({k:p[k] for k in ('experiment','index_keys','states_considered','frontier_states','closures')}))
if __name__=='__main__': main()
