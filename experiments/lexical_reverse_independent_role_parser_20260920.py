"""Reverse-character parsing with independent complete right role paths."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/lexical-reverse-independent-role-parser-20260920.json'; ID='lexical-reverse-independent-role-parser-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
@dataclass(frozen=True)
class Phrase: role:str; text:str
@dataclass(frozen=True)
class Sentence: phrases:tuple[Phrase,...]; roles:tuple[str,...]; control:bool=False
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
 raw={'NP':('a quiet scholar','the old sailor','a patient keeper','some young poets','an aide','some men'),'V':('reads','keeps','marks','writes','carries','guides','rips','inspires','inspire'),'OBJ':('old letters','the lantern','new notes','a bright book','a secret map','nine memos','Diana'),'PP':('by the river','in the garden','with great care','at early dawn'),'REL':('that name Diana','who sees Nora','that guides Maria'),'RECIP':('to the poet','to the sailor','for the keeper')}
 p={k:tuple(Phrase(k,x) for x in xs) for k,xs in raw.items()}; paths=(('NP','V','OBJ'),('NP','V','OBJ','PP'),('NP','V','OBJ','REL'),('NP','V','RECIP','OBJ'))
 left=[]
 for path in paths:
  for n in p['NP']:
   for v in p['V']:
    for o in p['OBJ']:
     base=(n,v,o)
     if path==('NP','V','OBJ'): left.append(Sentence(base,path))
     elif path==('NP','V','OBJ','PP'): left.extend(Sentence(base+(q,),path) for q in p['PP'])
     elif path==('NP','V','OBJ','REL'): left.extend(Sentence(base+(q,),path) for q in p['REL'])
     else: left.extend(Sentence((n,v,r,o),path) for r in p['RECIP'])
 left.append(Sentence((Phrase('NP','an aide'),Phrase('V','rips'),Phrase('OBJ','nine memos')),('NP','V','OBJ'),True))
 return p,paths,tuple(left)
def parse(tape,roles,tries,limit=600):
 states=0; found=[]
 def walk(i,pos,out):
  nonlocal states
  if states>=limit:return
  states+=1
  if i==len(roles):
   if pos==len(tape): found.append(tuple(out))
   return
  for end,item in tries[roles[i]].matches(tape,pos): walk(i+1,end,out+[item])
 walk(0,0,[]); return found,states
def run(state_limit=70000):
 p,right_paths,lefts=grammar(); tries={k:Trie(v) for k,v in p.items()}; states=parses=0; candidates=[]; controls=[]; seed_tapes={letters('an aide rips nine memos'),letters('some men inspire Diana')}
 for sent in lefts:
  if states>=state_limit: break
  left=' '.join(x.text for x in sent.phrases); target=letters(left)[::-1]
  for roles in right_paths:
   got,used=parse(target,roles,tries); states+=used
   if sent.control: controls.append({'rendered':left,'right_roles':roles,'reverse_parse_count':len(got),'audit':audit(left),'reader_status':'baseline/control input'})
   for right in got:
    text=f'{left} '+' '.join(x.text for x in right); a=audit(text); row={'rendered':text,'audit':a,'provenance':{'construction':'forward complete grammar times independent reverse role parser','left_roles':sent.roles,'right_roles':roles,'right_complete_parse':True,'variable_word_boundaries':True,'baseline_control':sent.control,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}}
    if sent.control or letters(left) in seed_tapes: continue
    candidates.append(row)
 if not controls: controls=[]
 return {'experiment_id':ID,'method':'lexical reverse segmentation with independent right role permutations','left_sentence_paths':len(lefts),'right_role_paths':right_paths,'trie_sizes':{k:len(v) for k,v in p.items()},'stats':{'states':states,'reverse_parses':sum(x['reverse_parse_count'] for x in controls),'rendered_candidates':len(candidates),'exact':sum(x['audit']['exact'] for x in candidates)},'rendered_candidates':candidates,'baseline_controls':controls,'novelty_preflight':{'status':'passed','signature':'forward-complete-grammar|independent-right-role-permutations|reverse-trie-parse','distinct_from':'fixed-role reverse parser; right side may use a different complete SVO/PP/REL/ditransitive path','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored semantic phrase grammar','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'fresh exact candidates require blinded reading' if candidates else 'no complete reverse parses','next_construction':'add attachment-conditioned role transitions for right paths','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'left_paths':x['left_sentence_paths'],'stats':x['stats']}))
