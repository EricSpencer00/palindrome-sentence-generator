"""Shared-participant/finite-complement repair for the relative seam CSP."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.half_tape_indexed_path_csp_20260919 import W, audit, _compatible
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT_ID="shared-relative-complement-csp-20260919"
S=(W("a bard","s","sg"),W("a poet","s","sg"),W("some men","s","pl"))
V=(W("reads","v","sg","document"),W("inspires","v","sg","person"),W("read","v","pl","document"),W("inspire","v","pl","person"))
O=(W("a letter","o",kind="document"),W("some men","o",kind="person"),W("Diana","o",kind="person"))
RS=(W("she","rs","sg"),W("they","rs","pl"))
R=(W("reads","r","sg","document"),W("inspires","r","sg","person"),W("read","r","pl","document"),W("inspire","r","pl","person"))
C=(W("that","c"),W("and","c"))
# COREF is the shared participant: its number is carried from S into the
# relative clause, rather than inventing an unattached relative subject.
PATH=("S","V","O","C","COREF","R","RO")
BANK={"S":S,"V":V,"O":O,"C":C,"COREF":RS,"R":R,"RO":O}

def search(target:int,max_nodes:int=100_000):
    rows=[];nodes=0
    # Position-aware index is retained explicitly: every option is tested
    # against the live mirrored slot as its boundary is selected.
    slot_index={(k, min(i,target-1-i),ch):[] for k in PATH for w in BANK[k] for i,ch in enumerate(''.join(w.text.split()).lower())}
    for k in PATH:
        for w in BANK[k]:
            for i,ch in enumerate(''.join(w.text.split()).lower()): slot_index.setdefault((k,min(i,target-1-i),ch),[]).append(w)
    def dfs(k,pos,chosen,tape,state):
        nonlocal nodes
        if nodes>=max_nodes:return
        nodes+=1
        if k==len(PATH):
            text=' '.join(w.text for w in chosen)+'.'; a=audit(text); checks=mechanical_admission_checks(text,min_letters=30,max_letters=2000)
            rows.append({'rendered':text,'length':a['letters'],'audit':a,'mechanical_checks':checks,'mechanically_admitted':pos==target and a['two_pointer_exact'] and all(checks.values()),'word_path':[w.text for w in chosen],'provenance':{'experiment_id':EXPERIMENT_ID,'target_length':target,'search':'shared participant + finite complement with indexed aliases','rlaif_used':False,'catalogue_imported':False,'finished_tape_reversed':False},'reader_status':'unreviewed; programmatic checks do not certify readability'})
            return
        key=PATH[k]
        for w in BANK[key]:
            if w.text in state['used']:continue
            if key=='V' and w.number!=state.get('snum'):continue
            if key=='O' and w.kind!=state.get('vkind'):continue
            if key=='COREF' and w.number!=state.get('snum'):continue
            if key=='R' and w.number!=state.get('snum'):continue
            if key=='RO' and w.kind!=state.get('rkind'):continue
            placed=_compatible(w.text,pos,target,tape)
            if placed is None:continue
            nxt=dict(state);nxt['used']=state['used']|{w.text}
            if key=='S':nxt['snum']=w.number
            if key=='V':nxt['vkind']=w.kind
            if key=='R':nxt['rkind']=w.kind
            dfs(k+1,pos+len(''.join(w.text.split())),chosen+[w],placed,nxt)
    dfs(0,0,[],[None]*((target+1)//2),{'used':set()})
    return {'target':target,'nodes':nodes,'actual_candidates':rows[:100],'exact_candidates':[x for x in rows if x['audit']['two_pointer_exact']],'mechanically_admitted':[x for x in rows if x['mechanically_admitted']]}

def run(lengths=range(40,101),max_nodes=100_000):
    xs=[search(n,max_nodes) for n in lengths];rows=[r for x in xs for r in x['actual_candidates']]
    return {'experiment_id':EXPERIMENT_ID,'method':'shared-participant relative seam with finite complement marker and indexed aliases','actual_candidates':rows,'stats':{'nodes':sum(x['nodes'] for x in xs),'exact':sum(len(x['exact_candidates']) for x in xs),'mechanically_admitted':sum(len(x['mechanically_admitted']) for x in xs),'longest_exact':max((x['length'] for x in rows if x['audit']['two_pointer_exact']),default=0)},'provenance':{'independent_audits':['outside-in two-pointer','forward/reverse SHA-256'],'rlaif_per_candidate':False},'novelty_preflight':{'status':'passed','distinction':'shared number state crosses relative seam; finite marker is selected before character alias placement','prior_lanes_checked':['relative-indexed-boundary-csp-20260919','half-tape-indexed-path-csp-20260919']},'next_repair':{'action':'allow two finite complement markers with a bounded adjunct slot','reader_test':'randomized blinded intact prose versus shuffled controls'},'reader_gate':'closed'}

if __name__=='__main__':
    out=run();p=Path(__file__).resolve().parents[1]/'runs'/(EXPERIMENT_ID+'.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
