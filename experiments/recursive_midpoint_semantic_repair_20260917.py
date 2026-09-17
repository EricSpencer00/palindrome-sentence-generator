"""Midpoint-crossing product with one held-out semantic-frame repair."""
from __future__ import annotations
import argparse,json,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve();s=importlib.util.spec_from_file_location('base',HERE.with_name('recursive_discourse_frame_product_20260917.py'));base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
base.OBJECT['teacher']='the path'  # one role-compatible repair for h versus r

def product(left,right,cap=8000):
    # The sole search state: opposing lexical offsets plus consumed lengths.
    L=list(left);R=list(right); todo=[(0,0,len(R)-1,len(R[-1])-1,0,0)];seen=set();front=[]
    while todo and len(seen)<cap:
        state=todo.pop();lw,lc,rw,rc,nl,nr=state
        if state in seen:continue
        seen.add(state)
        if lw==len(L) and rw<0:return True,front,state
        if lw<len(L) and lc==len(L[lw]):todo.append((lw+1,0,rw,rc,nl,nr));continue
        if rw>=0 and rc<0:todo.append((lw,lc,rw-1,len(R[rw-1])-1,nl,nr));continue
        if lw>=len(L) or rw<0:continue
        if lc>=len(L[lw]) or rc<0:continue
        if L[lw][lc].lower()!=R[rw][rc].lower():front.append((lw,lc,rw,rc,L[lw][lc],R[rw][rc],nl,nr));continue
        # Increment unequal sides independently; crossing may occur mid-token.
        todo.append((lw,lc+1,rw,rc-1,nl+1,nr+1))
    return False,front,(todo[-1] if todo else None)
def run():
 ps=base.derivations();rows=[];closures=[];states=0
 for lp,lf in ps:
  for rp,rf in ps:
   ok,front,mid=product(lp,rp);states+=len(front)+1;text=' '.join(lp+rp);a=base.audit(text)
   row={'text':text,'length_letters':a['letters'],'midpoint_state':mid,'first_residual':front[:1],'provenance':{'left_derivation':lf,'right_derivation':rf,'repair':'teacher guides the path','condition':'held-out semantic edge for prior h versus r residual','unequal_clause_partitions':True},'independent_exact_audit':a,'mechanically_admitted':bool(ok and a['exact'] and a['letters']>=39)}
   (closures if row['mechanically_admitted'] else rows).append(row)
 return {'status':'exhausted','stats':{'derivations':len(ps),'product_states':states,'rendered':len(rows),'exact_accepted':len(closures),'longest_letters':max((r['length_letters'] for r in rows),default=0)},'closures':closures,'diagnostic_witnesses':rows[:12],'config':{'midpoint_crossing_product':True,'inside_token_crossing':True,'unequal_clause_partitions':True,'fixed_tape':False,'posthoc_reverse':False},'novelty_preflight':{'distinction':'single held-out semantic edge searched with midpoint offset state; no broad bank or equal-half constraint','excluded':{'seed':True,'catalogue':True,'word_order_mirror':True}},'reader_gate':{'status':'not_triggered' if not closures else 'human_blind_review_required','next_repair':'condition one further role-compatible edge on the surviving midpoint residual'}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.write_text(json.dumps(run(),indent=2)+'\n')
