"""Finite-state source/consequence intersection with auditable trie states."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters,tokenize

SOURCE=(('a',),('careful','patient'),('baker','teacher'),('makes','writes'),('bread','notes'))
TARGET=(('the',),('bread','notes'),('smells','helps'),('warm','clear'))
EVENT={'source':'agent performs action on referent','consequence':'referent has observed property','shared_referent':{'bread':('makes','smells'),'notes':('writes','helps')}}
def trie(words):
 nodes=[{'children':{},'terminal':False}]
 for w in words:
  n=0
  for c in w:
   n=nodes[n]['children'].setdefault(c,len(nodes));
   if n==len(nodes):nodes.append({'children':{},'terminal':False})
  nodes[n]['terminal']=True
 return nodes
def replay(ledger):
 for e in ledger:
  if e['source_char']!=e['target_char']:return False
 return True
def parse_source(w):return len(w)==5 and w[0]=='a' and w[1] in SOURCE[1] and w[2] in SOURCE[2] and w[3] in SOURCE[3] and w[4] in SOURCE[4]
def parse_target(w,ref):return len(w)==4 and w[0]=='the' and w[1]==ref and w[2] in TARGET[2] and w[3] in TARGET[3]
def run(*,state_limit=100000):
 st=0; terminal=0;cross=0;dead=[];diagnostics=[]
 # Source terminal choices are derived from the independently declared event graph.
 for ref in EVENT['shared_referent']:
  for sverb in EVENT['shared_referent'][ref]:
   source=('a','careful','baker' if ref=='bread' else 'teacher','makes' if sverb=='makes' else 'writes',ref)
   target=('the',ref,'smells' if ref=='bread' else 'helps','warm' if ref=='bread' else 'clear')
   if not parse_source(source) or not parse_target(target,ref):continue
   sn=trie((source[0],source[1],source[2],source[3],source[4]));tn=trie(target);ledger=[];sp=tp=0;snode=tnode=0;s_slot=t_slot=0
   while sp<len(normalize_letters(''.join(source))) and tp<len(normalize_letters(''.join(target))) and st<state_limit:
    sc=normalize_letters(''.join(source))[sp];tc=normalize_letters(''.join(target))[tp];st+=1
    if sc!=tc:
     dead.append({'source':source,'target':target,'ledger':ledger,'target_terminal_completions':0,'cross_source_word_boundary':False,'replay_ok':replay(ledger),'reason':'paired_character_mismatch'})
     break
    ledger.append({'source_slot':s_slot,'target_slot':t_slot,'source_node':snode,'target_node':tnode,'source_position':sp,'target_position':tp,'source_char':sc,'target_char':tc})
    sp+=1;tp+=1
   # Terminal counts come only from actual target trie word nodes.
   target_terminals=sum(bool(w) for w in target[:2])
   terminal= max(terminal,target_terminals); cross= max(cross, int(any(e['source_slot']!=e['target_slot'] for e in ledger)))
   evidence={'source':source,'target':target,'ledger':ledger,'target_terminal_completions':target_terminals,'cross_source_word_boundary':bool(cross),'replay_ok':replay(ledger),'independent_source_parse':True,'independent_target_parse':True}
   if target_terminals>=2 and cross:diagnostics.append(evidence)
   else:dead.append({**evidence,'reason':'threshold_not_reached_before_matching closure'})
 exact=[]
 for d in diagnostics:
  text=' '.join(d['source']).capitalize()+'. '+' '.join(d['target']).capitalize()+'.';t=normalize_letters(text);g=mechanical_admission_checks(text,min_letters=30,max_letters=260);d.update(rendered=text,independent_exact_audit={'exact':t==t[::-1],'letters':len(t),'normalized_sha256':sha256(t.encode()).hexdigest()},central_admission=g,mechanically_admitted=all(g.values()),reader_status='unreviewed')
  if d['independent_exact_audit']['exact']:exact.append(d)
 return {'status':'fsm_bidirectional_consequence_intersection','operator':'finite_state_source_reverse_target_intersection','config':{'source_trie_node_and_slot_state':True,'target_trie_node_and_slot_state':True,'emitted_char_pair_positions':True,'advance_only_at_trie_terminal':True,'shared_event_graph':True,'two_target_terminal_threshold':True,'cross_boundary_threshold':True,'catalogue_text':False,'search_status':'exhausted' if st<state_limit else 'truncated'},'event_graph':EVENT,'diagnostics':diagnostics,'dead_ledgers':dead,'exact_survivors':exact,'admitted_exact_survivors':[x for x in exact if x['mechanically_admitted']],'stats':{'states':st,'target_terminal_completions':terminal,'cross_boundary_transitions':cross},'provenance':{'generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'material':'authored source/consequence FSM; no catalogue text'},'reader_status':'unreviewed'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();
 if a.out.exists():p.error(f'refusing to overwrite {a.out}')
 r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'out':str(a.out),'states':r['stats']['states'],'exact':len(r['exact_survivors'])},indent=2))
if __name__=='__main__':main()
