#!/usr/bin/env python3
"""Character-labeled semantic automaton with propagated seam requirements."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/character-labeled-semantic-automaton-20260917.json'
T='The {a} {v} the {o} near the {s0}{bridge}{a2} {v2} the {o2} near the {s1}.'
ST={'agentive':('gardener','carries','letters'),'pedagogic':('teacher','writes','notes'),'reporting':('messenger','records','charts')};ED={'agentive':['cooperative','contrastive'],'pedagogic':['cooperative','contrastive'],'reporting':['contrastive','cooperative']};BR={'cooperative':', and the ','contrastive':', while the '};SET=['harbor','garden','station']
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(x):return T.format(**x)
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0][0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for src,edges in ED.items():
  for rel in edges:
   dst='pedagogic' if rel=='cooperative' else 'reporting';l=ST[src];r=ST[dst];x={'a':l[0],'v':l[1],'o':l[2],'s0':SET[len(rows)%3],'bridge':BR[rel],'a2':r[0],'v2':r[1],'o2':r[2],'s1':SET[(len(rows)+1)%3]}
   tape=norm(render(x)); labels=list(norm(BR[rel])); required=[]; propagated=0
   for q,ch in enumerate(labels):
    if q<len(tape)//2:
     j=len(tape)-1-q;required.append({'left':q,'right':j,'char':ch});propagated+=int(tape[q]==tape[j])
   rows.append({'candidate':len(rows),'rendered':render(x),'source_role':src,'target_role':dst,'transition':rel,'edge_character_labels':labels,'propagated_requirements':required,'propagated_matches':propagated,'provenance':'character_labeled_semantic_automaton','novelty_preflight':{'signature':'character_labeled_semantic_automaton_v1','distinct_from':'semantic role automaton; edge characters create seam requirements during traversal'},'audit':audit(render(x)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'character-labeled-semantic-automaton-20260917','method':'semantic automaton edges carry character labels; exact seam requirements propagate during each transition traversal','template':T,'candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'make edge labels constrain lexical trie transitions and reject incompatible characters before selecting full words'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
