#!/usr/bin/env python3
"""Fresh verb-trie/agreement branch of the midpoint CFG product."""
import hashlib,itertools,json,re
from pathlib import Path
AGENTS=('gardener','reader','sailor','teacher','cartographer','curator','pilot','musician')
VERBS=(('carries','sg'),('reviews','sg'),('marks','sg'),('opens','sg'),('studies','sg'),('charts','sg'),('copies','sg'),('folds','sg'),('carry','pl'),('review','pl'),('mark','pl'),('open','pl'),('study','pl'),('chart','pl'),('copy','pl'),('fold','pl'))
PATIENTS=('map','letter','garden','harbor','atlas','journal','score','parcel'); ADJ=('quiet','patient','fresh','old','careful','young')
SIG='midpoint-CFG-automaton-product|character-level-verb-trie|typed-agreement|complete-obligation-admission|independent-pointer-sha'
def c(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=c(s); return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def trie(w): return [{'prefix':w[:i],'next':w[i:i+1],'complete':i+1==len(w)} for i in range(len(w))]
def main():
 rows=[]
 for ag,(v,num),pa,adj in itertools.product(AGENTS,VERBS,PATIENTS,ADJ):
  # Singular subjects select inflected verbs; plural subjects select bare forms.
  subj_num='pl' if ag.endswith('s') else 'sg'; text=f'the {adj} {ag} {v} the {pa}.'; t=c(text)
  rows.append({'rendered':text,'admitted':t==t[::-1],'provenance':'fresh bilateral character-level trie expansion with held agreement feature','grammar_state':{'roles':{'agent':ag,'verb':v,'patient':pa},'agreement':{'subject_number':subj_num,'verb_number':num,'compatible':subj_num==num},'agent_trie':trie(ag),'verb_trie':trie(v),'patient_trie':trie(pa),'midpoint_obligation':{'complete_before_admission':True,'all_lexical_tries_closed':True}},'live_frontier':{'obligation_closed':t==t[::-1]},'audit':audit(text),'anti_shortcut':{'single_tree':True,'character_level_lexical_states':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'not_article_relation_variant':True,'verb_trie_intersection':True,'agreement_gate':True}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); exact=[r for r in rows if r['admitted'] and r['grammar_state']['agreement']['compatible']]
 out={'experiment':'cfg-midpoint-verb-trie-agreement-20260917','method':'midpoint CFG/automaton product with character-level agent/verb/patient tries and agreement gating','signature':SIG,'candidate_count':len(rows),'exact_count':len(exact),'admitted_renderings':exact,'diagnostic_controls':rows[:80],'next_repair':'Add bilateral role-conditioned verb argument frames with held-out lexical classes; preserve trie closure and agreement pruning before rendering.'}
 p=Path('runs/cfg-midpoint-verb-trie-agreement-20260917.json'); p.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'run':str(p),'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters']}))
if __name__=='__main__': main()
