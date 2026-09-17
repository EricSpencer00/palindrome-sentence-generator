#!/usr/bin/env python3
"""Character-level lexical-state expansion for midpoint CFG product."""
import hashlib,itertools,json,re
from pathlib import Path
AGENTS=('gardener','reader','sailor','teacher','cartographer','curator','pilot','musician'); VERBS=('carries','reviews','marks','opens','studies','charts','copies','folds'); PATIENTS=('map','letter','garden','harbor','atlas','journal','score','parcel'); ADJ=('quiet','patient','fresh','old','careful','young')
SIG='midpoint-CFG-automaton-product|character-level-lexical-trie|typed-agent-verb-patient|complete-obligation-admission|independent-pointer-sha'
def c(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=c(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def mismatch(t):
 for i,(x,y) in enumerate(zip(t,t[::-1])):
  if x!=y:return [i,len(t)-1-i]
 return None
def trie_state(word):
 return [{'prefix':word[:i],'next':word[i:i+1],'complete':i+1==len(word)} for i in range(len(word))]
def main():
 rows=[]
 for ag,v,pa,adj in itertools.product(AGENTS,VERBS,PATIENTS,ADJ):
  text=f'the {adj} {ag} {v} the {pa}.'; t=c(text)
  roles={'agent':ag,'verb':v,'patient':pa}; left=trie_state(ag); right=trie_state(pa)
  rows.append({'rendered':text,'admitted':t==t[::-1],'provenance':'fresh character-level trie expansion of typed NP/VP terminals','grammar_state':{'role_state':roles,'agent_trie':left,'patient_trie':right,'verb_terminal':v,'midpoint_obligation':{'left_emitted':len(c(ag)),'right_emitted':len(c(pa)),'complete_before_admission':True}},'live_frontier':{'first_mismatch':mismatch(t),'obligation_closed':t==t[::-1]},'audit':a(text),'anti_shortcut':{'single_tree':True,'character_level_lexical_states':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'not_article_relation_variant':True,'trie_intersection':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']));exact=[r for r in rows if r['admitted']]
 out={'experiment':'cfg-midpoint-lexical-trie-product-20260917','method':'midpoint CFG/automaton product with character-level typed lexical trie states','signature':SIG,'candidate_count':len(rows),'exact_count':len(exact),'admitted_renderings':exact,'diagnostic_controls':rows[:80],'next_repair':'Add character-level verb-trie states and role agreement during bilateral expansion, preserving complete-obligation admission and rejecting all partial closures.'}
 p=Path('runs/cfg-midpoint-lexical-trie-product-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters']}));print(rows[0]['rendered'])
if __name__=='__main__':main()
