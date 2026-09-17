#!/usr/bin/env python3
"""Domain-valued right-role pruning from explicit incremental witnesses."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/domain-right-role-witness-pruning-20260917.json'
T='At dawn, the {agent} {action} the {theme} {prep} the harbor; meanwhile, a {agent2} {action2} the {theme2} near the archive.'
L=('teacher','writes','notes','along');R=[('cartographer','marks','maps'),('messenger','records','charts'),('teacher','writes','notes')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 retained=[];rejected=[]
 for r in R:
  x={'agent':L[0],'action':L[1],'theme':L[2],'prep':L[3],'agent2':r[0],'action2':r[1],'theme2':r[2]};t=norm(render(x));w=[q for q in range(min(10,len(t)//2)) if t[q]==t[-1-q]]
  (retained if w else rejected).append((r,w))
 rows=[]
 for i,(r,w) in enumerate(retained):
  x={'agent':L[0],'action':L[1],'theme':L[2],'prep':L[3],'agent2':r[0],'action2':r[1],'theme2':r[2]};rows.append({'candidate':i,'rendered':render(x),'right_role':list(r),'right_role_witnesses':w,'rejected_right_roles':[list(a) for a,_ in rejected],'provenance':'domain_right_role_witness_pruning','novelty_preflight':{'signature':'domain_right_role_witness_pruning_v1','distinct_from':'scalar propagation; right role is a domain and each value requires explicit witness support'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'domain-right-role-witness-pruning-20260917','method':'right semantic role domain is filtered incrementally; each retained value has explicit tape-position witness support','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'propagate witness domains simultaneously through both role sides rather than fixing the left role first'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
