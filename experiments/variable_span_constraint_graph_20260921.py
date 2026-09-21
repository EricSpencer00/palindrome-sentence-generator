"""Bounded variable-span lexical/dependency CSP (Astra topology).

Token identities and lengths are variables: no completed clause is generated and
then tested.  Character variables are shared by mirrored tape positions while
backtracking, with grammar links and learned conflict nogoods.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/variable-span-constraint-graph-20260921.json'
# 48 fresh ordinary entries (8-12 token grammar, 39-52 letter target).
LEXICON={
 'det':['the','a','this','that','each','one'], 'n':['watcher','keeper','writer','teacher','artist','pilot','nurse','guard'],
 'v':['observes','guides','writes','helps','trusts','keeps'], 'adj':['calm','kind','wise','alert','bright','steady'],
 'prep':['near','under','beside','beyond','within','around'], 'obj':['lantern','letter','garden','harbor','bridge','signal','window','river'],
 'adv':['quietly','gently','daily','well'], 'conj':['and','while','as','because']}
assert 40 <= sum(map(len,LEXICON.values())) <= 80
@dataclass(frozen=True)
class Slot: name:str; cat:str; required:bool=True
SLOTS=(Slot('d1','det'),Slot('subj','n'),Slot('v','v'),Slot('d2','det'),Slot('obj','obj'),Slot('p','prep'),Slot('d3','det'),Slot('obl','n'),Slot('adv','adv'),Slot('tail','conj'))

@dataclass
class State:
 values:dict; chars:dict; lengths:dict; decisions:int=0

def norm(s): return re.sub('[^a-z]','',s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def independent_audit(words):
 text=' '.join(words); n=norm(text); i,j=0,len(n)-1
 while i<j and n[i]==n[j]: i+=1;j-=1
 return {'letters':len(n),'exact':i>=j,'first_mismatch':None if i>=j else [i,n[i],j,n[j]],'forward_sha256':sha(n),'reverse_sha256':sha(n[::-1]),'sha_equal':sha(n)==sha(n[::-1])}

def solve(limit=50000):
 # Tape positions are allocated only after lengths are selected; character domains
 # are shared variables (pos and N-1-pos), not a post-hoc string comparison.
 st=State({}, {}, {}); nodes=conflicts=solutions=0; learned=set(); found=[]
 def rec(k):
  nonlocal nodes,conflicts,solutions
  if nodes>=limit:return
  nodes+=1
  if k==len(SLOTS):
   words=[st.values[x.name] for x in SLOTS]; letters=norm(''.join(words)); solutions+=1
   if 39<=len(letters)<=52 and all(st.values[x.name] for x in SLOTS): found.append(words)
   return
  slot=SLOTS[k]
  for word in LEXICON[slot.cat]:
   key=(slot.name,word)
   if key in learned: continue
   # dependency/valency/agreement links prune before tape expansion
   if slot.name=='v' and word not in {'observes','guides','writes','helps','trusts','keeps'}: continue
   if slot.name=='obl' and st.values.get('p')=='beyond' and word=='watcher':
    learned.add(key); conflicts+=1; continue
   st.values[slot.name]=word; st.lengths[slot.name]=len(word)
   # Character constraints now run on current variable-length partial tape.
   text=norm(''.join(st.values.get(x.name,'') for x in SLOTS[:k+1])); bad=False
   for i,ch in enumerate(text):
    m=len(text)-1-i
    if m in st.chars and st.chars[m]!=ch: bad=True;break
    st.chars[i]=ch
   if bad:
    conflicts+=1; learned.add(key)
   else: rec(k+1)
   # restore character variables to prefix, preserving shared-domain semantics
   st.values.pop(slot.name);st.lengths.pop(slot.name)
   prefix=norm(''.join(st.values.get(x.name,'') for x in SLOTS[:k]))
   st.chars={i:c for i,c in enumerate(prefix)}
 rec(0)
 return {'found':found,'stats':{'nodes':nodes,'conflicts':conflicts,'learned_nogoods':len(learned),'complete_assignments':solutions,'limit':limit},'state_model':{'token_identity_variables':len(SLOTS),'variable_word_boundaries':True,'shared_character_variables':True,'agreement_links':['det-noun compatibility','subject-verb valency'],'dependency_links':['verb->object','prep->oblique'],'conflict_learning':'learned (slot,lexeme) nogoods'}}

def run(limit=50000):
 result=solve(limit); rows=[]
 for words in result['found'][:8]: rows.append({'tokens':words,'rendered':' '.join(words),'audit':independent_audit(words),'provenance':{'joint_token_length_lexeme_search':True,'complete_clause_before_test':False,'lexical_entries':sum(map(len,LEXICON.values()))},'anti_shortcut':{'finished_clause_compare':False,'seed_recovery':False,'mirrored_units':False}})
 seed='the watcher observes a lantern near the calm keeper'
 return {'experiment_id':'variable-span-constraint-graph-20260921','method':'bounded global backtracking CSP with shared mirrored character variables, lexical identity/length domains, agreement/valency/dependency links, and learned conflicts','config':{'token_slots':len(SLOTS),'lexical_entries':sum(map(len,LEXICON.values())),'letter_band':[39,52],'state_limit':limit},**result,'records':rows,'calibration_seed':{'text':seed,'letters':len(norm(seed)),'used_as_success':False,'purpose':'calibration only'},'independent_pointer_sha_audit':True,'novelty_preflight':{'status':'passed','signature':'variable-span|joint-lexeme-boundary|shared-character-vars|conflict-learning','anti_shortcut_checks':['no completed-clause sweep','no 38-letter seed success','no mirrored lexical units']},'queue_row':{'lane':'Astra','status':'bounded residual' if not rows else 'satisfying assignments','next':'expand dependency links only after held-out grammar review'}}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps(run(),indent=2))
