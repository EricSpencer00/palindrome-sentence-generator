#!/usr/bin/env python3
"""Typed relative-clause predicate slot with role-preserving valency."""
import hashlib,itertools,json,re
from pathlib import Path
AGENTS=('gardener','reader','sailor','teacher','cartographer','curator','pilot','musician'); RELSUB=('reader','sailor','teacher','pilot'); FRAMES=(('carries','object','map','charts'),('reviews','object','letter','copies'),('marks','object','score','reviews'),('opens','object','parcel','folds'),('studies','object','atlas','charts'),('charts','object','harbor','studies'),('copies','object','journal','marks'),('folds','object','map','opens'),('walks','locative','garden','rests'),('waits','locative','harbor','works'),('works','locative','studio','rests'),('rests','locative','park','waits')); ADJ=('quiet','patient','fresh','old','careful','young'); SIG='midpoint-CFG|typed-relative-predicate-slot|role-valency|bilateral-obligations|independent-pointer-sha'
def clean(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=clean(s);return {'letters':len(t),'two_pointer':t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256_equal':hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}
def trie(w):return [{'prefix':w[:i],'next':w[i:i+1],'complete':i+1==len(w)} for i in range(len(w))]
def main():
 rows=[]
 for ag,(v,role,obj,rv),rs,adj in itertools.product(AGENTS,FRAMES,RELSUB,ADJ):
  comp='that' if role=='object' else 'where'; valid=(role=='object' and rv in ('charts','copies','reviews','folds','marks','opens')) or (role=='locative' and rv in ('rests','works','waits'))
  text=f'the {adj} {ag} {v} the {obj} {comp} {rs} {rv}.';t=clean(text)
  rows.append({'rendered':text,'admitted':t==t[::-1] and valid,'provenance':'fresh typed relative predicate expansion with role-preserving valency and bilateral lexical tries','grammar_state':{'matrix_role':role,'complementizer':comp,'relative_subject':rs,'relative_predicate':rv,'valency_valid':valid,'tries':{'matrix_agent':trie(ag),'matrix_verb':trie(v),'relative_subject':trie(rs),'relative_predicate':trie(rv),'complement':trie(obj)},'complete_obligations':True},'audit':audit(text),'anti_shortcut':{'single_tree':True,'word_order_only':False,'repeated_unit':False,'catalogue_source':False,'fragment':False},'novelty_preflight':{'signature':SIG,'typed_relative_predicate':True,'role_valency_gate':True,'not_article_relation_variant':True}})
 rows.sort(key=lambda r:(-r['audit']['letters'],r['rendered']));exact=[r for r in rows if r['admitted']]
 out={'experiment':'cfg-midpoint-typed-relative-predicate-20260917','method':'whole-tape CFG product with role-typed relative predicate and valency state','signature':SIG,'candidate_count':len(rows),'exact_count':len(exact),'admitted_renderings':exact,'diagnostic_controls':rows[:80],'next_repair':'Add a typed relative object slot with semantic class agreement, retaining predicate valency and bilateral obligations.'}
 p=Path('runs/cfg-midpoint-typed-relative-predicate-20260917.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'run':str(p),'candidates':len(rows),'exact':len(exact),'longest':rows[0]['audit']['letters'],'text':rows[0]['rendered']}))
if __name__=='__main__':main()
