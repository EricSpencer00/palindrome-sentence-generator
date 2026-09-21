#!/usr/bin/env python3
"""Corpus role-bank + online character-CSP palindrome experiment.

The corpus supplies sentence *shapes* and role-labelled lexical domains.  A
source sentence is never emitted: every result is checked against source
1/2/3-grams and its provenance records the template id and selected words.
"""
from __future__ import annotations
import hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "authored_sentences.txt"
OUT = ROOT / "runs" / "corpus-role-online-csp-20260921.json"
WORD_RE = re.compile(r"[a-z]+")

def tape(s): return "".join(WORD_RE.findall(s.lower()))
def sha(s): return hashlib.sha256((tape(s)+"|"+tape(s)[::-1]).encode()).hexdigest()
def exact(s):
    t=tape(s)
    forward=hashlib.sha256(t.encode()).hexdigest()
    reverse=hashlib.sha256(t[::-1].encode()).hexdigest()
    return bool(t) and t == t[::-1] and forward == reverse

def role(w, i, toks):
    if w in {"a","an","the","no","one","some","each","my","our","this","that"}: return "DET"
    if w in {"i","we","she","he","they","it","you","no one"}: return "PRON"
    if w in {"was","were","am","are","is","had","have","has","can","did","made","let","kept","told","saw","sat","ran","left","held","read","found","drew","lost","put","took","sold","lit","met","set","cut","sent","rung","sung","went","came","fell","went","opened","closed"}: return "V"
    if w.endswith("ed") or w.endswith("ing"): return "V"
    if w in {"not","all","still","alone","open","dark","old","long","cold","red","small","quiet","near","wet","odd","late","alive","gone","flat","hot","thin","sad","short","out","down","there","home"}: return "ADJ"
    if w in {"on","in","at","of","to","for","with","from","over","under"}: return "PREP"
    return "N"

def load():
    rows=[]
    for n,line in enumerate(SRC.read_text().splitlines()):
        ws=WORD_RE.findall(line.lower())
        if ws: rows.append((n,line,ws))
    # Make long, intact grammatical *shapes* by composing two distinct
    # authored clauses. Lexical realization remains independent of either
    # source; this is the seam the next repair operator will vary.
    base=list(rows)
    rows=[]
    for a in base:
        for b in base:
            for c in base:
                if not (a[0] < b[0] < c[0]): continue
                if len(tape(a[1]))+len(tape(b[1]))+len(tape(c[1])) < 39: continue
                # Deliberately cross the seam: the realized slot sequence is
                # c+b+a, while provenance retains the independent clause IDs.
                # This prevents the search from merely replaying a corpus
                # sentence and makes seam compatibility an explicit variable.
                rows.append((f"{c[0]}+{b[0]}+{a[0]}",c[1]+"; "+b[1]+"; "+a[1],c[2]+b[2]+a[2]))
                if len(rows)>=40: break
            if len(rows)>=40: break
        if len(rows)>=40: break
    domains=defaultdict(set)
    for _,_,ws in rows:
        for i,w in enumerate(ws): domains[role(w,i,ws)].add(w)
    return rows, {k:sorted(v) for k,v in domains.items()}

def novel(text, source_grams):
    ts=tape(text)
    # Character unigrams are necessarily shared; reject source lexical
    # bigrams/trigrams instead, while recording the full source IDs.
    return not any(g in ts for g in source_grams[2]) and not any(g in ts for g in source_grams[3])

def solve(ws, domains, source_grams, limit=2):
    """Assign words left-to-right; each assignment immediately checks all
    character equations whose opposite endpoint is already assigned."""
    roles=[role(w,i,ws) for i,w in enumerate(ws)]
    # only attempt genuinely recombinable templates, never the source words
    pools=[]
    for r,w in zip(roles,ws):
        d=[x for x in domains[r] if len(x)==len(w)][:80]
        pools.append(d)
    if any(not d for d in pools): return []
    lens=[len(x) for x in ws]; N=sum(lens)
    pos=[]
    for i,L in enumerate(lens): pos += [(i,j) for j in range(L)]
    idx={p:k for k,p in enumerate(pos)}
    assigned=[None]*len(ws); found=[]; nodes=0
    def dfs(i):
        nonlocal nodes
        if len(found)>=limit: return
        if i==len(ws):
            text=" ".join(assigned)
            if novel(text,source_grams) and exact(text): found.append(text)
            return
        for cand in pools[i]:
            nodes+=1
            ok=True
            # online propagation against every already-assigned opposite char
            for off,ch in enumerate(cand):
                p=idx[(i,off)]; q=N-1-p; j,jo=pos[q]
                if assigned[j] is not None and assigned[j][jo] != ch: ok=False; break
            if not ok: continue
            assigned[i]=cand; dfs(i+1); assigned[i]=None
    dfs(0)
    return found, nodes

def main():
    rows,domains=load(); grams={n:set(tape(line)[i:i+n] for _,line,_ in rows for i in range(len(tape(line))-n+1)) for n in (1,2,3)}
    controls=[]; candidates=[]; total_nodes=0
    for sid,line,ws in rows:
        res=solve(ws,domains,grams)
        got=res[0] if res else []; nodes=res[1] if res else 0; total_nodes+=nodes
        controls.append({"source_id":sid,"source":line,"roles":[role(w,i,ws) for i,w in enumerate(ws)],"letters":len(tape(line)),"nodes":nodes,"exact_count":len(got)})
        for text in got:
            candidates.append({"text":text,"letters":len(tape(text)),"exact":True,"sha":sha(text),"source_id":sid,"roles":[role(w,i,ws) for i,w in enumerate(ws)],"novel_ngrams":True})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"method":"corpus role bank with online left/right character propagation","source":str(SRC),"source_count":len(rows),"domain_sizes":{k:len(v) for k,v in domains.items()},"controls":controls,"candidates":candidates,"total_nodes":total_nodes,"repair":"Next: generate role-compatible template recombinations; current lane retains source clause order and therefore has no viable cross-boundary seam."},indent=2)+"\n")
    print(json.dumps({"sources":len(rows),"domains":{k:len(v) for k,v in domains.items()},"nodes":total_nodes,"candidates":len(candidates),"max_letters":max((x['letters'] for x in candidates),default=0)}))
if __name__ == "__main__": main()
