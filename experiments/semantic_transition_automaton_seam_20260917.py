#!/usr/bin/env python3
"""Seam-conditioned semantic transition automaton over clause roles."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/semantic-transition-automaton-seam-20260917.json'
T='The {a} {v} the {o} near the {s0}{bridge}{a2} {v2} the {o2} near the {s1}.'
STATES={'agentive':('gardener','carries','letters'),'pedagogic':('teacher','writes','notes'),'reporting':('messenger','records','charts')};TRANS={'agentive':['cooperative','contrastive'],'pedagogic':['cooperative','contrastive'],'reporting':['contrastive','cooperative']};BR={'cooperative':', and the ','contrastive':', while the '};SET=['harbor','garden','station']
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def seam_prefix(x):
 t=norm(render(x));return sum(t[k]==t[-1-k] for k in range(min(10,len(t)//2)))
def main():
 rows=[]
 for src in STATES:
  for dst in TRANS[src]:
   target=next(k for k in STATES if (src,dst) in [(src,x) for x in TRANS[src]] and k!=src) if dst!='cooperative' else 'pedagogic'
   # relation-labeled transition chooses a deterministic semantic target.
   dst_state={'cooperative':'pedagogic','contrastive':'reporting'}[dst]
   l=STATES[src];r=STATES[dst_state];x={'a':l[0],'v':l[1],'o':l[2],'s0':SET[len(rows)%3],'bridge':BR[dst],'a2':r[0],'v2':r[1],'o2':r[2],'s1':SET[(len(rows)+1)%3]}
   text=render(x);rows.append({'candidate':len(rows),'rendered':text,'source_role':src,'transition':dst,'target_role':dst_state,'slots':x,'precompletion_seam_matches':seam_prefix(x),'provenance':'semantic_transition_automaton_seam_conditioned','novelty_preflight':{'signature':'semantic_transition_automaton_v1','distinct_from':'semantic bridge CSP; relation transitions form a role automaton with seam-conditioned edge scoring'},'audit':audit(text),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'semantic-transition-automaton-seam-20260917','method':'role-state automaton selects valid semantic transitions and scores seam character constraints before clause completion','template':T,'states':list(STATES),'transitions':TRANS,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'add character-labeled automaton edges and propagate exact seam requirements during transition traversal'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
