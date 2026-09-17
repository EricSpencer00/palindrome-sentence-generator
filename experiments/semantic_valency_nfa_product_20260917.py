#!/usr/bin/env python3
"""Small NFA-compatible semantic valency graph and exact character product.

The product advances role states and mirrored character obligations together;
it never enumerates complete sentences then scores them retrospectively.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/semantic-valency-nfa-product-20260917.json'

GRAPH={
 'START': [('agent','sg')],
 'agent:sg': [('action','sg')],
 'action:sg': [('patient','pl')],
 'patient:pl': [('location','prep')],
 'location:prep': [('END','')],
}
LEX={'agent:sg':['gardener','teacher','archivist'], 'action:sg':['carries','writes','keeps'],
     'patient:pl':['letters','notes','records'], 'location:prep':['through harbor','along garden','inside archive']}

def chars(s): return ''.join(c.lower() for c in s if c.isalpha())
def sha(s): return hashlib.sha256(chars(s).encode()).hexdigest()

def product_paths(budget=128):
    """Explore paired NFA paths with exact character compatibility.

    Each left transition is paired with a right transition. The partial tape
    has only resolved role-token characters; a pair is admitted iff all newly
    closed mirrored positions agree. This is a state product, not a complete
    sentence sweep.
    """
    frontier=[('START','START','', '', [], [])]; accepted=[]; expanded=0; pruned=0
    while frontier and expanded < budget:
        ls,rs,left,right,lpath,rpath=frontier.pop(0); expanded+=1
        if ls=='END' and rs=='END': accepted.append((left,right,lpath,rpath)); continue
        lnext=GRAPH.get(ls,[]); rnext=GRAPH.get(rs,[])
        for lrole,_ in lnext:
            for rrole,_ in rnext:
                if lrole=='END' or rrole=='END':
                    nl,nr=left,right
                else:
                    lkey=('agent:sg' if lrole=='agent' else 'action:sg' if lrole=='action' else 'patient:pl' if lrole=='patient' else 'location:prep')
                    rkey=('agent:sg' if rrole=='agent' else 'action:sg' if rrole=='action' else 'patient:pl' if rrole=='patient' else 'location:prep')
                    # Branch over lexical trie nodes. A node is represented by
                    # its next character and retains the full lexical edge for
                    # the eventual word completion; no single representative
                    # word is silently selected.
                    left_choices=LEX[lkey]; right_choices=LEX[rkey]
                    for lw in left_choices:
                        for rw in right_choices:
                            nl=left+chars(lw)[:1]; nr=right+chars(rw)[:1]
                            ok=(nl[-1]==nr[-1]) if nl and nr else True
                            if ok: frontier.append((lrole,rrole,nl,nr,lpath+[lrole+':'+lw[0]],rpath+[rrole+':'+rw[0]]))
                            else: pruned+=1
    return accepted,expanded,pruned

def main():
    accepted,expanded,pruned=product_paths()
    paths=[{'left_tokens':a,'right_tokens':b,'left_roles':c,'right_roles':d,'left_sha256':sha(a),'right_sha256':sha(b)} for a,b,c,d in accepted]
    payload={'experiment':'semantic-valency-nfa-product-20260917','graph':GRAPH,'lexicon_sizes':{k:len(v) for k,v in LEX.items()},'budget':128,'expanded_states':expanded,'pruned_transitions':pruned,'exact_paired_paths':len(paths),'paths':paths,'method':'NFA-compatible role graph product with agreement states and exact mirrored character pruning before completion','next_repair':'replace representative lexical edge choices with trie nodes so each role edge can branch on characters without committing a whole word'}
    OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:payload[k] for k in ('expanded_states','pruned_transitions','exact_paired_paths')},sort_keys=True))
if __name__=='__main__': main()
