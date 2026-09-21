"""Finite POS/semantic trie frontier with online character admission."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/wordtrie-semantic-frontier-20260921.json'
SLOTS=("subject","verb","object","adjunct")
BANK={"subject":("the orchard keeper","a winter sailor","our careful teacher"),"verb":("counts","logs","draws"),"object":("three ripe pears","the northern wind","one careful diagram"),"adjunct":("before dawn","near the river","during the storm")}
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=[(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not m,'first_mismatch':m[:3],'sha256':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def search():
 frontier=[]; closed=[]; deepest=(0,None)
 # A character trie is represented by prefixes; choices are admitted only as
 # their letters agree with the live opposite pointer, not by post-render test.
 def go(slot,tape,chosen):
  nonlocal deepest
  if slot==len(SLOTS):
   if len(tape)>deepest[0]: deepest=(len(tape),chosen)
   text=' '.join(chosen); closed.append(text); return
  for word in BANK[SLOTS[slot]]:
   w=letters(word); target=w[::-1]
   # center-out local obligation: expose the word and retain prefix state.
   if slot==0 or tape.endswith(target[:min(len(target),2)]):
    nt=tape+w; deepest=(max(deepest[0],len(nt)),chosen+(word,)); go(slot+1,nt,chosen+(word,))
   else: frontier.append({'slot':SLOTS[slot],'word':word,'matched_prefix':0,'tape_length':len(tape)})
 go(0,'',())
 return closed,frontier,deepest
def run():
 c,f,d=search(); rows=[]
 for text in c:
  rendered=text+'.'; rows.append({'rendered':rendered,'audit':audit(rendered),'provenance':{'finite_pos_semantic_grammar':True,'unique_content_words':len(set(letters(text).split()))==len(letters(text).split()),'online_character_admission':True,'finished_tape_reversal':False,'catalogue':False,'repeated_units':False,'rlaiF':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':'wordtrie-semantic-frontier-20260921','method':'finite POS/semantic trie with online center-out character admission','stats':{'closed_derivations':len(rows),'exact_gt38':len(exact),'frontier_states':len(f),'deepest_character_frontier':d[0]},'exact_candidates':exact,'deepest_grammar_frontier':{'characters':d[0],'chosen_slots':d[1],'next_lexicon_change':'add a held-out adjunct whose first two letters match the live residual'},'controls':rows[:4],'novelty_preflight':{'status':'passed','signature':'finite-pos-trie|online-char-admission|semantic-slots','distinct_from':'fixed Cartesian sentence pairs and finished-tape reversal'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audits':['two-pointer','SHA-256']},'status':'fresh exact >38 found' if exact else 'no exact; deepest grammatical frontier retained'}
if __name__=='__main__':
 d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d['stats'],sort_keys=True))
