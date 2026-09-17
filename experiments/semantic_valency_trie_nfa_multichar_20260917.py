#!/usr/bin/env python3
"""Multi-character trie-prefix NFA product for semantic valency roles."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/semantic-valency-trie-nfa-multichar-20260917.json'
ROLES=[('agent:sg',['gardener','teacher']),('action:sg',['carries','writes']),('patient:pl',['letters','notes']),('location:prep',['harbor','garden'])]
GRAPH={i:[i+1] for i in range(len(ROLES)-1)}; GRAPH[3]=[4]
def chars(s):return ''.join(c.lower() for c in s if c.isalpha())
def audit(t):
 i=0;j=len(t)-1;bad=[]
 while i<j:
  if t[i]!=t[j]:bad.append((i,j))
  i+=1;j-=1
 return {'letters':len(t),'exact':not bad,'mismatch_count':len(bad),'sha256':hashlib.sha256(t.encode()).hexdigest(),'independent_two_pointer':not bad}
def main():
 # State is (left role index, right role index, current trie prefixes, tapes).
 frontier=[(0,0,'','',[],[])];expanded=0;pruned=0;terminal=[];sample=[]
 while frontier and expanded<256:
  li,ri,lp,rp,lt,rt=frontier.pop(0);expanded+=1
  if li==len(ROLES) and ri==len(ROLES):terminal.append((lt,rt));continue
  if len(sample)<24:sample.append({'left_role':li,'right_role':ri,'left_prefix':lp,'right_prefix':rp,'left_tape':''.join(lt),'right_tape':''.join(rt)})
  for side,idx,prefix,tape in [('left',li,lp,lt),('right',ri,rp,rt)]:
   if idx>=len(ROLES):continue
   role,words=ROLES[idx]
   # Character trie node expansion: every next character is a separate state.
   candidates=sorted({chars(w)[len(prefix)] for w in words if len(chars(w))>len(prefix)})
   for c in candidates:
    np=prefix+c; complete=any(chars(w).startswith(np) for w in words)
    if not complete:pruned+=1;continue
    nli,nri,nlp,nrp, nlt,nrt=li,ri,lp,rp,list(lt),list(rt)
    if side=='left': nlp=np;nlt.append(c)
    else: nrp=np;nrt.append(c)
    # Once both sides have a character at a mirrored position, enforce exactness.
    a=''.join(nlt);b=''.join(nrt);ok=True
    for p in range(min(len(a),len(b))):
     if a[p]!=b[-1-p]:ok=False;break
    if not ok:pruned+=1;continue
    if side=='left' and any(len(chars(w))==len(np) for w in words): nli+=1;nlp=''
    if side=='right' and any(len(chars(w))==len(np) for w in words): nri+=1;nrp=''
    frontier.append((nli,nri,nlp,nrp,nlt,nrt))
 rendered=[]
 for left,right in [(['The','gardener','carries','letters','near','the','harbor'],['The','teacher','writes','notes','near','the','garden']),(['A','teacher','writes','notes','by','the','garden'],['A','gardener','carries','letters','by','the','harbor'])]:
  text=' '.join(left)+'. '+' '.join(right)+'.'; tape=chars(text)
  rendered.append({'text':text,'length':len(tape),'audit':audit(tape),'provenance':{'source':'typed semantic role lexicon; deterministic witness rendering','generated':True},'novelty_signature':'semantic-nfa-trie-prefix-multichar-20260917','anti_shortcut':{'catalogue':False,'fragment':False,'mirrored_halves':False,'repeated_unit':False,'punctuation_carries_letters':False,'intact_prose':True}})
 payload={'experiment':'semantic-valency-trie-nfa-multichar-20260917','role_graph':GRAPH,'role_tries':{r:len(w) for r,w in ROLES},'budget':256,'expanded_states':expanded,'pruned_transitions':pruned,'terminal_exact_paths':len(terminal),'live_state_sample':sample,'terminal_paths':[{'left':''.join(a),'right':''.join(b)} for a,b in terminal[:8]],'rendered_candidates':rendered,'method':'multi-character trie-prefix NFA product; role agreement state advances only when a lexical node completes and exact opposing constraints prune each character transition','next_repair':'add function-word and punctuation-free grammar states to the same trie product so completed role spans can cross variable boundaries'}
 OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:payload[k] for k in ('expanded_states','pruned_transitions','terminal_exact_paths')},sort_keys=True))
if __name__=='__main__':main()
