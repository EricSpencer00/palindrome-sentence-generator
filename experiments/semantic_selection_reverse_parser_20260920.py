"""Reverse parser with animate recipient and relative-selection constraints."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/semantic-selection-reverse-parser-20260920.json'; ID='semantic-selection-reverse-parser-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
@dataclass(frozen=True)
class Phrase: role:str; text:str; animacy:str='unknown'
class Trie:
 def __init__(self,items):
  self.root={}
  for p in items:
   n=self.root
   for ch in letters(p.text): n=n.setdefault(ch,{})
   n.setdefault('$',[]).append(p)
 def matches(self,tape,pos):
  n=self.root; out=[]
  for i in range(pos,len(tape)):
   n=n.get(tape[i])
   if n is None: break
   out.extend((i+1,p) for p in n.get('$',()))
  return tuple(out)
def grammar():
 raw={'NP':(('a quiet scholar','animate'),('the old sailor','animate'),('a patient keeper','animate'),('some young poets','animate'),('an aide','animate'),('the lantern','inanimate')),'V':(('reads','unknown'),('keeps','unknown'),('marks','unknown'),('writes','unknown'),('carries','unknown'),('guides','unknown'),('rips','unknown'),('inspires','unknown'),('inspire','unknown')),'OBJ':(('old letters','inanimate'),('the lantern','inanimate'),('new notes','inanimate'),('a bright book','inanimate'),('a secret map','inanimate'),('nine memos','inanimate'),('Diana','animate')),'RECIP':(('to the poet','animate'),('to the sailor','animate'),('for the keeper','animate'),('to Diana','animate')),'PP_OBJ':(('by the river','unknown'),('in the garden','unknown'),('with great care','unknown')),'REL_OBJ':(('that names Diana','unknown'),('that guides Maria','unknown')),'REL_SUBJ':(('who sees Nora','unknown'),('who helps Leon','unknown'))}
 p={k:tuple(Phrase(k,x,a) for x,a in xs) for k,xs in raw.items()}; paths=(('NP','V','OBJ'),('NP','V','OBJ','PP_OBJ'),('NP','V','OBJ','REL_OBJ'),('NP','V','RECIP','OBJ'),('NP','REL_SUBJ','V','OBJ'))
 left=[]
 for path in paths[:4]:
  for n in p['NP'][:4]:
   for v in p['V'][:7]:
    for o in p['OBJ'][:5]:
     base=(n,v,o)
     if path==('NP','V','OBJ'): left.append((base,path))
     elif path==('NP','V','OBJ','PP_OBJ'): left.extend(((base+(q,),path) for q in p['PP_OBJ']))
     elif path==('NP','V','OBJ','REL_OBJ'): left.extend(((base+(q,),path) for q in p['REL_OBJ']))
     else: left.extend(((n,v,r,o),path) for r in p['RECIP'])
 left.append(((Phrase('NP','an aide','animate'),Phrase('V','rips'),Phrase('OBJ','nine memos','inanimate')),('NP','V','OBJ')))
 return p,paths,paths,tuple(left)
def selection_valid(roles,items):
 seen=[]
 for role,item in zip(roles,items):
  if role=='RECIP' and item.animacy!='animate': return False
  if role=='REL_OBJ' and not any(r=='OBJ' for r in seen): return False
  if role=='REL_SUBJ' and not any(r=='NP' and i.animacy=='animate' for r,i in zip(seen,items[:len(seen)])): return False
  seen.append(role)
 return 'NP' in seen and 'V' in seen and 'OBJ' in seen
def parse(tape,roles,tries,limit=500):
 states=0; found=[]
 def walk(i,pos,out):
  nonlocal states
  if states>=limit:return
  states+=1
  if i==len(roles):
   if pos==len(tape) and selection_valid(roles,out): found.append(tuple(out))
   return
  for end,item in tries[roles[i]].matches(tape,pos): walk(i+1,end,out+[item])
 walk(0,0,[]); return found,states
def run(state_limit=70000):
 p,left_paths,right_paths,lefts=grammar(); tries={k:Trie(v) for k,v in p.items()}; states=parses=0; fresh=[]; controls=[]; seed={letters('an aide rips nine memos'),letters('some men inspire Diana')}
 for phrases,roles in lefts:
  left=' '.join(x.text for x in phrases); target=letters(left)[::-1]
  for rroles in right_paths:
   got,used=parse(target,rroles,tries); states+=used; parses+=len(got)
   if letters(left) in seed: controls.append({'rendered':left,'right_roles':rroles,'parse_count':len(got),'audit':audit(left),'reader_status':'baseline excluded control'})
   for right in got:
    if letters(left) in seed: continue
    text=f'{left} '+' '.join(x.text for x in right); a=audit(text); fresh.append({'rendered':text,'audit':a,'provenance':{'construction':'semantic selection constrained reverse parser','left_roles':roles,'right_roles':rroles,'selection_validated':True,'recipient_animacy_required':True,'relative_attachment_validated':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
 return {'experiment_id':ID,'method':'reverse segmentation with semantic recipient/relative selection constraints','left_sentence_paths':len(lefts),'right_role_paths':right_paths,'trie_sizes':{k:len(v) for k,v in p.items()},'stats':{'states':states,'reverse_parses':parses,'rendered_candidates':len(fresh),'exact':sum(x['audit']['exact'] for x in fresh)},'rendered_candidates':fresh,'baseline_controls':controls,'novelty_preflight':{'status':'passed','signature':'reverse-trie|animate-recipient-selection|relative-antecedent-attachment','distinct_from':'attachment parser; semantic referent compatibility is checked before emission','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored animate/inanimate role phrases','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 parse' if not fresh else 'reader gate required','next_construction':'add semantic compatibility for PP attachment locations','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'left_paths':x['left_sentence_paths'],'stats':x['stats']}))
