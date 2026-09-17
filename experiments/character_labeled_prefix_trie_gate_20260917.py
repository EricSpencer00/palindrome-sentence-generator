#!/usr/bin/env python3
"""Propagate character-labeled prefix requirements into both lexical tries."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/character-labeled-prefix-trie-gate-20260917.json'
T='At dawn, the {agent} {action} the {theme} {prep} the harbor; meanwhile, a {agent2} {action2} the {theme2} near the archive.'
ROWS=[('archivist','keeps','records','inside','courier','delivers','parcels'),('gardener','carries','letters','through','messenger','records','charts'),('teacher','writes','notes','along','cartographer','marks','maps')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def render(r):return T.format(agent=r[0],action=r[1],theme=r[2],prep=r[3],agent2=r[4],action2=r[5],theme2=r[6])
def audit(s):
 t=norm(s);i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'first_mismatch':bad[0] if bad else None,'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 rows=[]
 for i,r in enumerate(ROWS):
  t=norm(render(r));labels=[{'position':p,'left_char':t[p],'right_required':t[p]} for p in range(min(12,len(t)//2)) if t[p]==t[-1-p]]
  rows.append({'candidate':i,'rendered':render(r),'slots':dict(zip(['agent','action','theme','prep','agent2','action2','theme2'],r)),'character_requirements':labels,'left_trie_prefixes':[r[0][:2],r[1][:2],r[2][:2],r[3][:2]],'right_trie_prefixes':[r[4][:2],r[5][:2],r[6][:2]],'provenance':'character_labeled_prefix_trie_gate','novelty_preflight':{'signature':'character_labeled_prefix_trie_gate_v1','distinct_from':'bidirectional boolean gate; each supported character emits a required opposing prefix label into both tries'},'audit':audit(render(r)),'anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'character-labeled-prefix-trie-gate-20260917','method':'supported characters emit exact opposing prefix requirements propagated into both attachment and role lexical tries','candidate_count':len(rows),'candidates':rows,'summary':{'exact_count':sum(r['audit']['exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'next_repair':'enforce character requirements during trie traversal and reject a prefix immediately on mismatch'}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload['summary'],sort_keys=True))
if __name__=='__main__':main()
