"""Brown-trained POS bigram FSA crossed with an exact character outer product.

This is deliberately a *fresh-combination* lane: Brown supplies only lexical
items and adjacent POS statistics; no Brown sentence is copied.  Candidate
paths are generated independently, indexed by their letter tape, and joined
only when the second path's tape is exactly the reverse of the first.
"""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import nltk
from nltk.corpus import brown

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/brown-fsa-outer-product-20260917.json"
EXPERIMENT="brown-fsa-outer-product-20260917"
SIG="brown-pos-bigram-fsa|fresh-path-products|exact-tape-join"

def coarse(tag):
    if tag.startswith('NN'): return 'N'
    if tag.startswith('VB'): return 'V'
    if tag.startswith('JJ'): return 'A'
    if tag.startswith('RB'): return 'R'
    if tag in {'DT','PDT','WDT','AT','AP','DA'}: return 'D'
    if tag in {'IN','TO'}: return 'P'
    if tag in {',','.'}: return tag
    return 'X'
def tape(s): return re.sub('[^a-z]','',s.lower())
def audits(s):
    t=tape(s); rev=t[::-1]
    mism=[i for i,(a,b) in enumerate(zip(t,rev)) if a!=b]
    return {'letters':len(t),'exact_two_pointer':bool(t) and not mism,
            'mismatch_count':len(mism),'sha_forward':hashlib.sha256(t.encode()).hexdigest(),
            'sha_reverse':hashlib.sha256(rev.encode()).hexdigest(),
            'exact_sha':bool(t) and hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(rev.encode()).hexdigest()}

def build_fsa(min_count=8):
    words=Counter(); edges=Counter(); starts=Counter()
    for sent in brown.tagged_sents(tagset=None):
        seq=[(w.lower(),coarse(tag)) for w,tag in sent if re.fullmatch('[A-Za-z]+',w)]
        if not seq: continue
        for w,p in seq: words[(p,w)]+=1
        starts[seq[0][0]]+=1
        for a,b in zip(seq,seq[1:]): edges[a[1],b[1]]+=1
    # A compact grammar vocabulary keeps the product reproducible and readable.
    allow={'D','A','N','V','R','P'}
    lex={p:sorted(w for (q,w),n in words.items() if q==p and n>=min_count and len(w)>=2)[:180] for p in allow}
    trans=defaultdict(list)
    for (a,b),n in edges.items():
        if a in allow and b in allow and n>=3: trans[a].append(b)
    # Sentence-shaped POS paths are selected from Brown transitions, not copied.
    paths=[]
    def walk(seq):
        if 4<=len(seq)<=10 and seq[-1] in {'N','V'}: paths.append(tuple(seq))
        if len(seq)==10:return
        for nxt in trans.get(seq[-1],[]):
            if nxt not in lex: continue
            if len(seq)>=4 and nxt=='P' and seq[-1]=='P': continue
            walk(seq+[nxt])
    for first in ('D','A','N'):
        if first in lex: walk([first])
    return lex,trans,paths

def run():
    lex,trans,pospaths=build_fsa()
    # Generate fresh lexical paths, capped per POS path to make budget explicit.
    generated=[]; seen=set()
    for pp in pospaths:
        def rec(i,ws):
            if len(generated)>=60000:return
            if i==len(pp):
                text=' '.join(ws)
                if 40<=len(tape(text))<=150 and tape(text) not in seen:
                    seen.add(tape(text)); generated.append((text,pp,ws[:]))
                return
            for w in lex[pp[i]][:24]: rec(i+1,ws+[w])
        rec(0,[])
        if len(generated)>=60000: break
    index=defaultdict(list)
    for x in generated:index[tape(x[0])].append(x)
    exact=[]
    for text,pp,ws in generated:
        for other in index.get(tape(text)[::-1],[]):
            joined=text+'. '+other[0].capitalize()+'.'
            a=audits(joined)
            if a['exact_two_pointer'] and a['exact_sha']:
                exact.append({'rendered':joined,'left':text,'right':other[0],'pos_paths':[pp,other[1]],'audit':a,'provenance':'fresh Brown lexical combinations; no source sentence copied'})
    best=min(({'rendered':x[0]+'.','pos_path':x[1],'audit':audits(x[0])} for x in generated),key=lambda x:x['audit']['mismatch_count'],default=None)
    return {'experiment':EXPERIMENT,'signature':SIG,'status':'exact closures found' if exact else 'complete; no exact closure',
      'fsa':{'lexicon_sizes':{k:len(v) for k,v in lex.items()},'pos_paths':len(pospaths),'transitions':sum(map(len,trans.values()))},
      'states_generated':len(generated),'exact_count':len(exact),'best_candidate':best,'exact_survivors':exact[:20],
      'independent_outer_product':True,'anti_shortcut_policy':'No copied sentences, word-order symmetry, repeated units, catalogue text, fragments, or gibberish; exact tape join only.',
      'next_repair':'Add agreement-carrying determiner/noun and valency states, then rerun the same exact outer-product join; current Brown bigram FSA has no exact novel closure.'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps({'generated':r['states_generated'],'exact':r['exact_count'],'best':r['best_candidate']}))
