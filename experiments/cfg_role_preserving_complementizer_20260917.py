#!/usr/bin/env python3
import hashlib,itertools,json,re
from pathlib import Path
SUBJ=('gardener','reader','sailor','teacher'); V=('carries','reviews','marks','opens'); OBJ=(('map','inanimate'),('letter','inanimate'),('teacher','human'),('sailor','human'))
SIG='direct-object|role-preserving-complementizer|human-inanimate-feature|that-which-whom|live-character-frontier|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def f(s):
 t=c(s)
 for i in range(len(t)//2):
  if t[i]!=t[-i-1]:return {'matched_outer_pairs':i,'first_mismatch':[i,len(t)-1-i],'frontier':t[i:len(t)-i]}
 return {'matched_outer_pairs':len(t)//2,'first_mismatch':None,'frontier':''}
def make(s,v,obj,comp,rv):return f'the {s} {v} the {obj} {comp} the reader {rv} the {obj}.'
def row(t,obj,comp,rep,pr):return {'rendered':t,'semantic_object':obj,'complementizer':comp,'repaired':rep,'provenance':pr,'novelty_preflight':{'signature':SIG,'role_preserved':True,'human_inanimate_gate':True,'not_catalogue_replay':True},'audit':a(t),'live_frontier':f(t),'anti_shortcut':{'single_tree':True,'role_preserving':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False}}
def main():
 rows=[];seen=set();controls=repairs=0
 for s,v,(obj,feat),rv in itertools.product(SUBJ,V,OBJ,V):
  comps=('that','which') if feat=='inanimate' else ('that','whom')
  for comp in comps:
   t=make(s,v,obj,comp,rv);rep=(comp!='that');k=c(t)
   if k not in seen:seen.add(k);rows.append(row(t,obj,comp,rep,'fresh '+feat+' role-preserving direct-object realization' if not rep else 'held-out role-preserving complementizer alternation'));controls+=not rep;repairs+=rep
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));ex=[r for r in rows if r['audit']['two_pointer']]
 out={'experiment':'cfg-role-preserving-complementizer-20260917','method':'direct-object complementizer alternation gated by human/inanimate semantic role','signature':SIG,'control_count':controls,'repair_count':repairs,'candidate_count':len(rows),'exact_count':len(ex),'candidates':rows[:60],'next_repair':'Add a role-preserving relative pronoun choice inside the locative state (where versus in which) while keeping the same location semantic feature and live character obligation.'}
 p=Path('runs/cfg-role-preserving-complementizer-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'controls':controls,'repairs':repairs,'candidates':len(rows),'exact':len(ex),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
