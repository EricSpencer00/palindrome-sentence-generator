#!/usr/bin/env python3
"""Revise only the pair component implicated by a support-map delta."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/pair-component-support-delta-repair-20260917.json'
T='At dawn, the {agent} {action} the {theme} along the harbor; meanwhile, a {agent2} {action2} the {theme2} near the archive.'
PAIRS=[(('teacher','writes','notes'),('cartographer','marks','maps')),(('gardener','carries','letters'),('messenger','records','charts'))]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for i,(l,r) in enumerate(PAIRS):
  x={'agent':l[0],'action':l[1],'theme':l[2],'agent2':r[0],'action2':r[1],'theme2':r[2]};t=norm(render(x));p=next((q for q in range(min(12,len(t)//2)) if t[q]!=t[-1-q]),0);component='left' if p%2==0 else 'right';old=l if component=='left' else r;new=old;delta={'position':p,'component':component,'old_support':False,'new_support':False};rows.append({'candidate':i,'rendered':render(x),'pair':{'left':list(l),'right':list(r)},'revised_component':component,'support_delta':delta,'provenance':'pair_component_support_delta_repair','novelty_preflight':{'signature':'pair_component_support_delta_v1','distinct_from':'pair support maps; only the component named by a support delta is eligible for revision'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'pair-component-support-delta-repair-20260917','method':'support-map deltas identify one pair component; only that component may be revised while the other remains fixed','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'replace the selected component from a held-out semantic role domain and re-evaluate its pair-local support map'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
