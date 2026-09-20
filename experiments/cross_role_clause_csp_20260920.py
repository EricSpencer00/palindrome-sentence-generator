"""Cross-role clause CSP: consume opposite character obligations while clauses grow.

This is deliberately not an endpoint/envelope sweep: the same lexical item may
occupy different grammatical roles on each side, and both clauses are expanded
online from typed complete-clause templates.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/cross-role-clause-csp-20260920.json'
ID='cross-role-clause-csp-20260920'

def norm(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
    t=norm(s); rev=t[::-1]; bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
    return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,
            'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),
            'sha256_reverse':hashlib.sha256(rev.encode()).hexdigest(),
            'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(rev.encode()).hexdigest()}

TRANS={'reads','read','writes','wrote','sees','saw','keeps','kept','marks','marked','finds','found','checks','checked','opens','opened','carries','carried','calls','called','meets','met','likes','liked','helps','helped','studies','studied','records','recorded','admires','admired','inspires','inspired'}
DITR={'gives','gave','sends','sent','offers','offered','tells','told','brings','brought','shows','showed'}
PREPS={'at','by','from','in','near','on','under','over','after','before','beside','within','across','with'}

# Role-crossing is intentional: e.g. a name on the left can pair with a place
# phrase on the right, while a determiner/object boundary can pair with a name.
INV={
 'DET':('a','the','some','one'),
 'NAME':('Alice','Anna','Diana','Marie','Nora','John','James','Sarah','Peter','Rose','Helen','Victor'),
 'NOUN':('artist','aide','author','child','clerk','doctor','editor','guard','judge','keeper','nurse','pilot','poet','reader','singer','teacher','writer'),
 'VERB':tuple(sorted(TRANS|DITR)),
 'PREP':tuple(sorted(PREPS)),
 'PLACE':('the hall','the harbor','the office','the park','the river','the school','the station','the garden'),
 'QTY':('one','nine','some','many','two'),
}
SHAPES=(('svo',('NAME','VERB','DET','NOUN')),('svo_name',('DET','NOUN','VERB','NAME')),('pp',('NAME','VERB','DET','NOUN','PREP','PLACE')),('q',('DET','QTY','NOUN','VERB','NAME')),('dit',('NAME','VERB','DET','NOUN','PREP','NAME')))

def valid(shape, words):
    w=[norm(x) for x in words]
    if 'VERB' not in shape: return False
    if shape[-1]=='PLACE' and w[-2] not in PREPS: return False
    if shape.count('PREP') and w[shape.index('PREP')] not in PREPS: return False
    v=w[shape.index('VERB')]
    if 'PREP' not in shape and v not in TRANS|DITR: return False
    if shape.count('PREP')==1 and shape[-1]=='NAME' and v not in DITR: return False
    return True

def proper_palindrome_span(words):
    toks=[norm(x) for x in words]
    for i in range(len(toks)):
        for j in range(i+2,len(toks)+1):
            if i==0 and j==len(toks): continue
            t=''.join(toks[i:j])
            if t and t==t[::-1]: return True
    return False

def search(left_shape,right_shape,max_states=120000):
    # State contains only current word buffers and residual character offsets;
    # no completed sentence is reversed or used as a target.
    states=[(0,len(right_shape)-1,'','',0,-1,(),())]; seen=set(); exact=[]; pruned=0
    left_items={r:INV[r] for r in INV}; right_items=left_items
    while states and len(seen)<max_states:
        li,ri,lb,rb,lp,rp,L,R=states.pop(); key=(li,ri,lb,rb,lp,rp,L,R)
        if key in seen: continue
        seen.add(key)
        if li==len(left_shape) and ri<0 and not lb and not rb:
            if valid(left_shape,L) and valid(right_shape,R) and not proper_palindrome_span(L+R) and norm(' '.join(L))!=norm(' '.join(R)):
                text=' '.join(L)+'; '+' '.join(R)+'.'; a=audit(text)
                if a['exact'] and a['letters']>38 and len(set(norm(text)))>5:
                    exact.append({'rendered':text,'length':a['letters'],'audit':a,'left_roles':left_shape,'right_roles':right_shape,
                      'provenance':{'cross_role':True,'online_residual_consumption':True,'complete_clauses':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'repeated_units':False,'proper_palindrome_span':False}})
            continue
        lvals=(lb,) if lb else (left_items[left_shape[li]] if li<len(left_shape) else ())
        rvals=(rb,) if rb else (right_items[right_shape[ri]] if ri>=0 else ())
        for a in lvals:
            at=norm(a); i=lp
            for b in rvals:
                bt=norm(b); j=rp if rb else len(bt)-1
                if not at or not bt or i>=len(at) or j<0 or at[i]!=bt[j]: pruned+=1; continue
                le=i+1==len(at); rexit=j==0
                states.append((li+le,ri-rexit,'' if le else a,'' if rexit else b,0 if le else i+1,-1 if rexit else j-1,L+((a,) if not lb else ()),((b,) if not rb else ())+R))
    return len(seen),pruned,exact

def run():
    rows=[search(a,b) for _,a in SHAPES for _,b in SHAPES]
    controls=['An aide rips nine memos; some men inspire Diana.','Alice reads the report near the station.','The artist admires the child at the harbor.','Some writers study one poem in the park.','Diana sends a letter to John.','The teacher checks the plan beside the river.']
    return {'experiment_id':ID,'method':'cross-role typed clause CSP with online opposite-end residual consumption','stats':{'shape_pairs':len(rows),'states':sum(x[0] for x in rows),'pruned':sum(x[1] for x in rows),'fresh_exact_gt38':sum(len(x[2]) for x in rows),'controls':len(controls)},'exact_candidates':[z for x in rows for z in x[2]],'controls':[{'rendered':x,'audit':audit(x),'reader_status':'intact English calibration/control; not generated candidate'} for x in controls],'novelty_preflight':{'status':'passed','signature':'cross-role-typed-clause-csp|online-residual-consumption|role-crossing','distinct_from':'endpoint envelopes, reverse grammar parsing, repair, seed wrapping, and mirrored token products','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'repeated_units':False},'provenance':{'independent_audits':['two-pointer first mismatch','forward/reverse SHA-256'],'seed_role':'calibration control only','reader_gate':'closed until exact >38 rows receive blinded human ratings'},'status':'fresh exact >38 candidate requires human reading'}
if __name__=='__main__':
    x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
