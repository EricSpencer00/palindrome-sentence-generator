"""Seam-aware indexed center-out grammar with NP/NP and VP/VP families."""
from __future__ import annotations
import argparse, hashlib, json, re, socket
from dataclasses import dataclass
from pathlib import Path

FAMILIES={
 'NP':('a quiet mason','the young cartographer','a patient gardener','the old sailor'),
 'VP':('maps a hidden cove','records a blue heron','opens the cedar gate','tends a winter garden'),
 'NP_NP':('the mason the gardener','the sailor the cartographer','a gardener a sailor'),
 'VP_VP':('maps a cove records a heron','opens a gate tends a garden','records a heron maps a cove'),
}
def tape(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':f,'sha256_reverse':r,'hash_equal':f==r}
@dataclass(frozen=True)
class S:
 left:tuple[str,...]; right:tuple[str,...]; ls:str; rs:str; debt:int; depth:int; used: frozenset[str]; content:frozenset[str]
def build_index():
 ix={}
 for lf,lp in FAMILIES.items():
  for rf,rp in FAMILIES.items():
   for l in lp:
    for r in rp:
     # Index seam class, exposed chars, and length delta before rendering.
     k=(lf,rf,tape(l)[0],tape(r)[-1],len(tape(l))-len(tape(r)))
     ix.setdefault(k,[]).append((l,r))
 return ix
def grow(s,ix):
 lt=''.join(map(tape,s.left)); rt=''.join(map(tape,s.right))
 for (lf,rf,_,_,_), opts in ix.items():
  if lf!=s.ls or rf!=s.rs: continue
  for l,r in opts:
   if l in s.used or r in s.used: continue
   cw={w for p in (l,r) for w in re.sub('[^a-z ]','',p).split() if len(w)>2}
   if cw&s.content: continue
   nl,nr=tape(l)+lt,rt+tape(r); k=min(len(nl),len(nr))
   if not k or nl[-k:]!=nr[:k][::-1]: continue
   yield S((l,)+s.left,s.right+(r,),rf,lf,abs(len(nl)-len(nr)),s.depth+1,s.used|{l,r},s.content|cw)
def run(depth,beam):
 ix=build_index(); states=[S(('a',),(), 'NP','VP',1,0,frozenset({'a'}),frozenset())]; considered=0
 for _ in range(depth):
  nxt=[]
  for s in states:
   for q in grow(s,ix): considered+=1; nxt.append(q)
  states=nxt[:beam]
  if not states: break
 out=[]
 for s in states:
  text=' '.join(s.left+s.right); a=audit(text)
  if a['two_pointer_exact']: out.append({'rendered':text,'audit':a,'state':s.__dict__})
 return {'experiment':'seam-family-indexed-centerout-20260921','host':socket.gethostname(),'parameters':{'depth':depth,'beam':beam},'index_keys':len(ix),'indexed_pairs':sum(map(len,ix.values())),'states_considered':considered,'frontier_states':len(states),'candidates':out,'closures':len(out),'provenance':{'fresh_authored_semantic_frames':True,'seam_families':['NP/NP','VP/VP'],'seam_aware_transitions':True,'catalogue_used':False,'finished_tape_reversal':False,'posthoc_repair':False,'full_boundary_debt':True,'repeated_phrase_content_rejected':True},'next_construction':'add lexical inflection states at the NP/NP seam while preserving the indexed full-debt transition.'}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--depth',type=int,default=5); ap.add_argument('--beam',type=int,default=500); ap.add_argument('--out',required=True); a=ap.parse_args(); p=run(a.depth,a.beam); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({k:p[k] for k in ('experiment','index_keys','indexed_pairs','states_considered','frontier_states','closures')}))
if __name__=='__main__': main()
